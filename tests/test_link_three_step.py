"""Three-input gradient probe: causality, full gradient chain, frozen weights."""
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
    def test_recursive_gradients_no_future_inputs_and_checkpoint_equivalence(self):
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
            ids = torch.tensor([[1, 32, 5, 7, 8, 9, 33, 2]])
            batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                     'latent_mask': torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0]])}
            decoder = text_decoder(model.backbone)
            original = decoder.forward
            calls = []
            def checked_forward(*args, **kwargs):
                self.assertFalse(kwargs['use_cache'])
                self.assertNotIn('past_key_values', kwargs)
                emb, mask = kwargs['inputs_embeds'], kwargs['attention_mask']
                self.assertEqual(emb.shape[1], mask.shape[1])
                self.assertLessEqual(emb.shape[1], 5)
                calls.append(emb.detach().clone())
                return original(*args, **kwargs)
            with patch.object(decoder, 'forward', side_effect=checked_forward):
                out = model(**batch, stage=LINK_NEXT_TOKEN, link_steps=3)
                self.assertEqual([c.shape[1] for c in calls], [2, 3, 4, 5])
                self.assertEqual(out['substitution_positions'], [2, 3, 4])
                self.assertEqual(out['target_positions'], [3, 4, 5])
                self.assertEqual(out['target_ids'].tolist(), [[7, 8, 9]])
                torch.testing.assert_close(out['loss'], out['step_losses'].mean())
                # Loss after the THIRD input must reach every earlier vector.
                last_grads = torch.autograd.grad(out['step_losses'][-1], out['rollout_vectors'], retain_graph=True)
                self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in last_grads))
                out['loss'].backward()
            self.assertGreater(len(calls), 4)  # checkpoint recomputation really happened
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
            for i in range(3):
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
                other = model(**{**batch, 'input_ids': changed}, stage=LINK_NEXT_TOKEN, link_steps=3)
                for a, b in zip(out['rollout_vectors'], other['rollout_vectors']):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                suffix = ids.clone()
                suffix[0, 6:] = 13
                same = model(**{**batch, 'input_ids': suffix}, stage=LINK_NEXT_TOKEN, link_steps=3)
                torch.testing.assert_close(same['loss'], out['loss'], rtol=0, atol=0)
                self.assertEqual(same['predicted_ids'].shape, (1, 3))
                # Tiny attention-only Qwen is a control where both execution
                # paths should agree. This does NOT assume Qwen3.8 will agree.
                from link_execution_probe import cached_rollout
                cached = cached_rollout(model, {'input_ids': ids.tolist()[0], 'prompt_len': 1}, 32, 33)
                torch.testing.assert_close(cached['step_losses'], out['step_losses'], rtol=1e-4, atol=1e-5)
                self.assertEqual(cached['predicted_ids'].tolist(), out['predicted_ids'].tolist())
                for a, b in zip(cached['rollout_vectors'], out['rollout_vectors']):
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
            for steps in (0, 2, 4):
                with self.assertRaisesRegex(ValueError, 'link_steps'):
                    model(**batch, stage=LINK_NEXT_TOKEN, link_steps=steps)
            with self.assertRaisesRegex(ValueError, 'three content'):
                model(**{**batch, 'latent_mask': torch.zeros_like(ids)}, stage=LINK_NEXT_TOKEN, link_steps=3)
            padded = batch['attention_mask'].clone()
            padded[0, 0] = 0
            with self.assertRaisesRegex(ValueError, 'unpadded'):
                model(**{**batch, 'attention_mask': padded}, stage=LINK_NEXT_TOKEN, link_steps=3)
            with self.assertRaisesRegex(ValueError, 'batch size one'):
                model(**{k: v.repeat(2, 1) for k, v in batch.items()}, stage=LINK_NEXT_TOKEN, link_steps=3)
            model.backbone.get_input_embeddings().requires_grad_(True)
            with self.assertRaisesRegex(ValueError, 'fully frozen'):
                model(**batch, stage=LINK_NEXT_TOKEN, link_steps=3)


if __name__ == '__main__':
    unittest.main()
