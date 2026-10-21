"""Bounded two-pass LoRA probe: frozen base/anchors, gradient chain and no updates."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lora_self_conditioned_probe', ROOT / 'scripts/lora_self_conditioned_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft'))


class ProbeSafetyTests(unittest.TestCase):
    def test_no_optimizer_export_or_training_options(self):
        script = ROOT / 'scripts/lora_self_conditioned_probe.py'
        calls = [n.func.attr for n in ast.walk(ast.parse(script.read_text()))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        for forbidden in ('step', 'save_pretrained', 'AdamW', 'load_state_dict'):
            self.assertNotIn(forbidden, calls)
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--calibration', result.stdout)
        for option in ('--resume', '--max-steps', '--lr', '--all-train'):
            self.assertNotIn(option, result.stdout)

    def test_existing_and_overlapping_outputs_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, pilot = root / 'source', root / 'pilot'
            source.mkdir()
            pilot.mkdir()
            probe.validate_paths(source, pilot, root / 'new')
            for output in (root, source, source / 'child', pilot, pilot / 'child'):
                with self.assertRaises(ValueError):
                    probe.validate_paths(source, pilot, output)
            existing = root / 'existing'
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                probe.validate_paths(source, pilot, existing)


class Tokenizer:
    unk_token_id = None
    def __len__(self):
        return 34
    def encode(self, text, **kwargs):
        return [{'<latent>': 32, '</latent>': 33}[text]]
    def save_pretrained(self, directory):
        (Path(directory) / 'test-tokenizer.txt').write_text('anchors=32,33')


@unittest.skipUnless(AVAILABLE, 'requires torch, transformers and peft')
class SelfConditionedRuntimeTests(unittest.TestCase):
    def make_model(self, root):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hybrid_agent.lora import enable_format_lora
        from hybrid_sft.model import HybridSFTModel
        torch.manual_seed(37)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
        Qwen3ForCausalLM(config).save_pretrained(root / 'base')
        model = HybridSFTModel.from_pretrained(root / 'base', dtype=torch.float32)
        enable_format_lora(model, Tokenizer(), rank=2, alpha=4)
        # Match the real case: nonzero trained LoRA and anchor deltas, then reload
        # via is_trainable=False rather than relying on initial PEFT flags.
        with torch.no_grad():
            for p in model.backbone.parameters():
                if p.requires_grad:
                    p.add_(0.003)
        model.output_scale = .02
        model.save_pretrained(root / 'source', Tokenizer())
        return HybridSFTModel.from_pretrained(root / 'source', dtype=torch.float32)

    def test_explicit_opt_in_frozen_base_anchors_and_roundtrip(self):
        import torch
        from hybrid_sft.model import HybridSFTModel, SELF_CONDITIONED, FORMAT_ACTIVATION, LINK_NEXT_TOKEN
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self.make_model(root)
            with self.assertRaisesRegex(ValueError, 'format_activation'):
                model.set_trainable(SELF_CONDITIONED)
            with self.assertRaisesRegex(ValueError, 'requires self_conditioned'):
                model.set_trainable(FORMAT_ACTIVATION, lora_self_conditioned=True)
            model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
            model.train()
            self.assertTrue(model.backbone.training)
            trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
            self.assertTrue(any('.lora_A.' in n for n in trainable))
            self.assertTrue(any('.lora_B.' in n for n in trainable))
            self.assertTrue(any(n.startswith('hybridlink.') for n in trainable))
            self.assertTrue(all(n.startswith('hybridlink.') or '.lora_A.' in n or '.lora_B.' in n
                                for n in trainable))
            self.assertTrue(all(p.dtype == torch.float32 for p in trainable.values()))
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            ids = torch.tensor([[1, 32, 5, 7, 33, 6, 2]])
            batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                     'latent_mask': torch.tensor([[0, 1, 1, 1, 0, 0, 0]]),
                     'loss_mask': torch.tensor([[0, 1, 0, 0, 2, 1, 1]])}
            out = model(**batch, stage=SELF_CONDITIONED)
            out['loss'].backward()
            probe.gradient_report((n, p) for n, p in trainable.items() if n.startswith('backbone.'))
            probe.gradient_report(model.hybridlink.named_parameters())
            for n, p in model.named_parameters():
                torch.testing.assert_close(p, before[n], rtol=0, atol=0)
                if n not in trainable:
                    self.assertIsNone(p.grad, n)
            # Tiny CPU-only update validates the intended parameter boundary and
            # compact persistence. The production feasibility probe never updates.
            optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3, weight_decay=0)
            optimizer.step()
            for n, p in model.named_parameters():
                if n not in trainable:
                    torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            for prefix in ('backbone.', 'hybridlink.'):
                self.assertTrue(any(not torch.equal(p, before[n]) for n, p in trainable.items()
                                    if n.startswith(prefix)))
            model.eval()
            with torch.no_grad():
                expected = model(**batch, stage=SELF_CONDITIONED)['loss']
            model.save_pretrained(root / 'updated', Tokenizer())
            restored = HybridSFTModel.from_pretrained(root / 'updated', dtype=torch.float32).eval()
            self.assertEqual(restored.output_scale, model.output_scale)
            with torch.no_grad():
                torch.testing.assert_close(restored(**batch, stage=SELF_CONDITIONED)['loss'], expected,
                                           rtol=0, atol=0)
            # Returning to the existing link-only path must clear adapter grads.
            model.set_trainable(LINK_NEXT_TOKEN)
            self.assertFalse(any(p.requires_grad or p.grad is not None for p in model.backbone.parameters()))

    def test_two_pass_inputs_loss_and_checkpointed_gradients_match_reference(self):
        import torch
        import torch.nn.functional as F
        from unittest.mock import patch
        from hybrid_agent.compat import text_decoder
        from hybrid_sft.model import SELF_CONDITIONED
        with TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp))
            model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
            model.train()
            ids = torch.tensor([[1, 32, 5, 7, 33, 6, 2]])
            attn = torch.ones_like(ids)
            lat = torch.tensor([[0, 1, 1, 1, 0, 0, 0]])
            loss_mask = torch.tensor([[0, 1, 0, 0, 2, 1, 1]])
            batch = dict(input_ids=ids, attention_mask=attn, latent_mask=lat, loss_mask=loss_mask)
            embedding = model.backbone.get_input_embeddings()
            decoder = text_decoder(model.backbone)
            head = model.backbone.get_output_embeddings()
            trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
            # Independent two-pass reference without checkpointing: replacement
            # includes the exit input but never the following answer token.
            targets = embedding(ids).detach()
            first = decoder(inputs_embeds=embedding(ids), attention_mask=attn,
                            use_cache=False, return_dict=True).last_hidden_state
            predicted = model.run_hybridlink(first[:, :-1], output_dtype=targets.dtype)
            modified = torch.cat([targets[:, :1], torch.where(lat[:, :-1, None].bool(),
                                                             predicted, targets[:, 1:])], dim=1)
            second = decoder(inputs_embeds=modified, attention_mask=attn,
                             use_cache=False, return_dict=True).last_hidden_state
            logits = head(second[:, :-1]).float()
            ce = F.cross_entropy(logits.reshape(-1, 34), ids[:, 1:].reshape(-1), reduction='none')
            ce = (ce * loss_mask[:, 1:].reshape(-1)).sum() / loss_mask[:, 1:].sum()
            preds = model.run_hybridlink(second[:, :-1], output_dtype=targets.dtype).float()
            pair_loss = 1 - F.cosine_similarity(preds, targets[:, 1:].float(), dim=-1)
            pair_loss = pair_loss + .2 * (preds - targets[:, 1:].float()).square().mean(-1)
            # Three latent sources, with last source -> exit additionally weighted.
            reference = ce + pair_loss[0, 1:4].mean() + pair_loss[0, 3]
            ce_to_first = torch.autograd.grad(ce, first, retain_graph=True)[0]
            self.assertGreater(ce_to_first.abs().sum().item(), 0)
            first_link = torch.autograd.grad(ce, predicted, retain_graph=True)[0]
            self.assertGreater(first_link[:, 1:4].abs().sum().item(), 0)
            torch.testing.assert_close(first_link[:, 0], torch.zeros_like(first_link[:, 0]))
            reference.backward()
            expected_grads = {n: p.grad.detach().clone() for n, p in trainable.items()}
            model.zero_grad(set_to_none=True)
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            calls, generated = [], []
            original_decoder, original_link = decoder.forward, model.run_hybridlink
            def capture_decoder(*args, **kwargs):
                self.assertFalse(kwargs['use_cache'])
                self.assertNotIn('past_key_values', kwargs)
                calls.append(kwargs['inputs_embeds'].detach().clone())
                return original_decoder(*args, **kwargs)
            def capture_link(*args, **kwargs):
                result = original_link(*args, **kwargs)
                result.retain_grad()
                generated.append(result)
                return result
            with patch.object(decoder, 'forward', side_effect=capture_decoder), \
                    patch.object(model, 'run_hybridlink', side_effect=capture_link):
                actual = model(**batch, stage=SELF_CONDITIONED)
                self.assertEqual(len(calls), 2)
                torch.testing.assert_close(calls[0], targets, rtol=0, atol=0)
                torch.testing.assert_close(calls[1], modified.detach())
                torch.testing.assert_close(calls[1][:, 5:], targets[:, 5:], rtol=0, atol=0)
                # Causal LM logits for EXIT come from position 3, before the
                # substituted exit input at position 4. Answer starts at 5.
                torch.testing.assert_close(actual['loss'], reference.detach())
                actual['loss'].backward()
            self.assertGreater(generated[0].grad[:, 1:4].abs().sum().item(), 0)
            for n, p in trainable.items():
                self.assertIsNotNone(p.grad, n)
                torch.testing.assert_close(p.grad, expected_grads[n], rtol=1e-4, atol=1e-6, msg=n)


if __name__ == '__main__':
    unittest.main()
