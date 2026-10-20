"""Three/six-input gradient probe: causality, full gradient chain, frozen weights."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('link_three_step_probe', ROOT / 'scripts/link_three_step_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft'))


class ProbePlanTests(unittest.TestCase):
    def test_cli_accepts_six_inputs_and_rejects_other_lengths(self):
        import subprocess
        import sys
        script = str(ROOT / 'scripts/link_three_step_probe.py')
        result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--link-steps {3,6}', result.stdout)
        result = subprocess.run([sys.executable, script, '--pilot', 'unused', '--model', 'unused',
                                 '--calibration', 'unused', '--output-dir', 'unused', '--link-steps', '5'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid choice', result.stderr)

    def test_first_training_record_is_fixed_and_no_optimizer_or_export(self):
        import ast
        rows = [{'id': str(i)} for i in range(64)]
        self.assertEqual(probe.select_record(rows), rows[0])
        for bad in (rows[:63], rows + [rows[0]], rows[:63] + [rows[0]]):
            with self.assertRaises(ValueError):
                probe.select_record(bad)
        tree = ast.parse((ROOT / 'scripts/link_three_step_probe.py').read_text())
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('step', calls)
        self.assertNotIn('save_pretrained', calls)
        self.assertNotIn('AdamW', calls)


@unittest.skipUnless(AVAILABLE, 'requires torch, transformers and peft')
class ThreeStepRuntimeTests(unittest.TestCase):
    def test_three_input_gradients_and_checkpoint_equivalence(self):
        self.check_rollout(3)

    def test_six_input_gradients_and_checkpoint_equivalence(self):
        self.check_rollout(6)

    def check_rollout(self, steps):
        import torch
        from unittest.mock import patch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hybrid_agent.lora import enable_format_lora
        from hybrid_agent.compat import text_decoder
        from hybrid_sft.model import HybridSFTModel, LINK_NEXT_TOKEN

        class Tokenizer:
            unk_token_id = None
            def __len__(self):
                return 34
            def encode(self, text, **kwargs):
                return [{'<latent>': 32, '</latent>': 33}[text]]

        torch.manual_seed(19)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
        with TemporaryDirectory() as tmp:
            Qwen3ForCausalLM(config).save_pretrained(tmp)
            model = HybridSFTModel.from_pretrained(tmp, dtype=torch.float32)
            enable_format_lora(model, Tokenizer(), rank=2, alpha=4)
            model.output_scale = .02
            model.set_trainable(LINK_NEXT_TOKEN)
            model.train()
            self.assertFalse(model.backbone.training)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            ids = torch.tensor([[1, 32] + [5] + list(range(7, 7 + steps)) + [33, 2]])
            batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                     'latent_mask': torch.tensor([[0] + [1] * (steps + 2) + [0, 0]])}
            decoder = text_decoder(model.backbone)
            original = decoder.forward
            calls = []
            def checked_forward(*args, **kwargs):
                self.assertFalse(kwargs['use_cache'])
                self.assertNotIn('past_key_values', kwargs)
                emb, mask = kwargs['inputs_embeds'], kwargs['attention_mask']
                self.assertEqual(emb.shape[1], mask.shape[1])
                self.assertLessEqual(emb.shape[1], steps + 2)
                calls.append(emb.detach().clone())
                return original(*args, **kwargs)
            with patch.object(decoder, 'forward', side_effect=checked_forward):
                out = model(**batch, stage=LINK_NEXT_TOKEN, link_steps=steps)
                self.assertEqual([c.shape[1] for c in calls], list(range(2, steps + 3)))
                self.assertEqual(out['substitution_positions'], list(range(2, steps + 2)))
                self.assertEqual(out['target_positions'], list(range(3, steps + 3)))
                self.assertEqual(out['target_ids'].tolist(), ids[:, 3:steps + 3].tolist())
                torch.testing.assert_close(out['loss'], out['step_losses'].mean())
                # Loss after the final input must reach every earlier vector.
                last_grads = torch.autograd.grad(out['step_losses'][-1], out['rollout_vectors'], retain_graph=True)
                self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in last_grads))
                out['loss'].backward()
            self.assertGreater(len(calls), steps + 1)  # checkpoint recomputation really happened
            saved_grads = {n: p.grad.clone() for n, p in model.hybridlink.named_parameters()}
            probe.gradient_report(model.hybridlink.named_parameters())
            for n, p in model.named_parameters():
                self.assertTrue(torch.equal(p, before[n]), n)  # backward only
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.backbone.parameters()))

            # Independent uncheckpointed rollout: predictions depend on the
            # previous prediction's resulting hidden state, never teacher content.
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                prefix = model.backbone.get_input_embeddings()(ids[:, :2])
                hidden = decoder(inputs_embeds=prefix, attention_mask=torch.ones(1, 2, dtype=torch.long),
                                 use_cache=False, return_dict=True).last_hidden_state[:, -1:]
            vectors, losses = [], []
            for i in range(steps):
                vectors.append(model.run_hybridlink(hidden, output_dtype=torch.float32))
                embeddings = torch.cat((prefix, *vectors), dim=1)
                torch.testing.assert_close(embeddings, calls[i + 1])
                hidden = decoder(inputs_embeds=embeddings,
                                 attention_mask=torch.ones(1, 3 + i, dtype=torch.long),
                                 use_cache=False, return_dict=True).last_hidden_state[:, -1:]
                logits = model.backbone.get_output_embeddings()(hidden).float()
                losses.append(torch.nn.functional.cross_entropy(logits[:, -1], ids[:, 3 + i]))
            expected = torch.stack(losses)
            torch.testing.assert_close(expected, out['step_losses'])
            expected.mean().backward()
            for n, p in model.hybridlink.named_parameters():
                torch.testing.assert_close(p.grad, saved_grads[n], rtol=1e-4, atol=1e-6)

            # All latent text, including supervised target IDs, can change
            # without changing the decoder inputs or generated vectors.
            changed = ids.clone()
            changed[0, 2:] = 12
            with torch.no_grad():
                other = model(**{**batch, 'input_ids': changed}, stage=LINK_NEXT_TOKEN, link_steps=steps)
                for a, b in zip(out['rollout_vectors'], other['rollout_vectors']):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                suffix = ids.clone()
                suffix[0, steps + 3:] = 13
                same = model(**{**batch, 'input_ids': suffix}, stage=LINK_NEXT_TOKEN, link_steps=steps)
                torch.testing.assert_close(same['loss'], out['loss'], rtol=0, atol=0)
                self.assertEqual(same['predicted_ids'].shape, (1, steps))
                # Tiny attention-only Qwen is a control where both execution
                # paths should agree. This does NOT assume Qwen3.8 will agree.
                if steps == 3:
                    from link_execution_probe import cached_rollout
                    cached = cached_rollout(model, {'input_ids': ids.tolist()[0], 'prompt_len': 1}, 32, 33)
                    torch.testing.assert_close(cached['step_losses'], out['step_losses'], rtol=1e-4, atol=1e-5)
                    self.assertEqual(cached['predicted_ids'].tolist(), out['predicted_ids'].tolist())
                    for a, b in zip(cached['rollout_vectors'], out['rollout_vectors']):
                        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
            for invalid in (0, 2, 4, 5, 7):
                with self.assertRaisesRegex(ValueError, 'link_steps'):
                    model(**batch, stage=LINK_NEXT_TOKEN, link_steps=invalid)
            with self.assertRaisesRegex(ValueError, 'content tokens'):
                model(**{**batch, 'latent_mask': torch.zeros_like(ids)}, stage=LINK_NEXT_TOKEN, link_steps=steps)
            short_mask = batch['latent_mask'].clone()
            short_mask[0, steps + 1:] = 0
            with self.assertRaisesRegex(ValueError, 'content tokens'):
                model(**{**batch, 'latent_mask': short_mask}, stage=LINK_NEXT_TOKEN, link_steps=steps)
            gap_mask = batch['latent_mask'].clone()
            gap_mask[0, 2] = 0
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                model(**{**batch, 'latent_mask': gap_mask}, stage=LINK_NEXT_TOKEN, link_steps=steps)
            with torch.no_grad():
                exit_ids = ids.clone()
                exit_ids[0, steps + 2] = 33
                exit_mask = batch['latent_mask'].clone()
                exit_mask[0, steps + 2:] = 0
                exit_out = model(**{**batch, 'input_ids': exit_ids, 'latent_mask': exit_mask},
                                 stage=LINK_NEXT_TOKEN, link_steps=steps)
                self.assertEqual(exit_out['target_ids'][0, -1].item(), 33)
            with self.assertRaisesRegex(ValueError, 'following target'):
                model(**{k: v[:, :steps + 2] for k, v in batch.items()},
                      stage=LINK_NEXT_TOKEN, link_steps=steps)
            padded = batch['attention_mask'].clone()
            padded[0, 0] = 0
            with self.assertRaisesRegex(ValueError, 'unpadded'):
                model(**{**batch, 'attention_mask': padded}, stage=LINK_NEXT_TOKEN, link_steps=steps)
            with self.assertRaisesRegex(ValueError, 'batch size one'):
                model(**{k: v.repeat(2, 1) for k, v in batch.items()}, stage=LINK_NEXT_TOKEN, link_steps=steps)
            model.backbone.get_input_embeddings().requires_grad_(True)
            with self.assertRaisesRegex(ValueError, 'fully frozen'):
                model(**batch, stage=LINK_NEXT_TOKEN, link_steps=steps)


if __name__ == '__main__':
    unittest.main()
