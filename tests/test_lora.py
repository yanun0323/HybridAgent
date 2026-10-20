"""CPU regression tests for anchor-row training and portable adapter checkpoints."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "peft"))


@unittest.skipUnless(AVAILABLE, "requires torch, transformers and peft")
class LoraTests(unittest.TestCase):
    def test_frozen_base_anchor_gradients_and_roundtrip(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hybrid_agent.model import HybridAgentModel
        from hybrid_agent.lora import enable_format_lora
        from hybrid_agent.compat import text_forward
        from hybrid_sft.model import HybridSFTModel, FORMAT_ACTIVATION, SELF_CONDITIONED

        class Tokenizer:
            unk_token_id = None
            def __len__(self):
                return 34
            def encode(self, text, **kwargs):
                return [{"<latent>": 32, "</latent>": 33}[text]]
            def save_pretrained(self, directory):
                # This test loads model weights directly, not AutoTokenizer.
                (Path(directory) / "test-tokenizer.txt").write_text("anchors=32,33")

        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
        with TemporaryDirectory() as temp:
            base_dir, saved = Path(temp) / "base", Path(temp) / "trained"
            Qwen3ForCausalLM(config).save_pretrained(base_dir)
            model = HybridSFTModel.from_pretrained(base_dir, dtype=torch.float32)
            enable_format_lora(model, Tokenizer(), rank=2, alpha=4)
            model.set_trainable(FORMAT_ACTIVATION)
            model.eval()
            trainable = {name: p for name, p in model.backbone.named_parameters() if p.requires_grad}
            self.assertTrue(trainable)
            self.assertTrue(all("lora_" in name or "trainable_tokens_delta" in name for name in trainable))
            anchor_params = {name: p for name, p in trainable.items() if "trainable_tokens_delta" in name}
            self.assertEqual(len(anchor_params), 2)  # input AND untied output head
            for p in anchor_params.values():
                self.assertEqual(tuple(p.shape), (2, 16))
            frozen = {name: p.detach().clone() for name, p in model.backbone.named_parameters()
                      if not p.requires_grad}
            anchors_before = {name: p.detach().clone() for name, p in anchor_params.items()}
            ids = torch.tensor([[1, 32, 5, 33, 6, 2]])
            mask = torch.ones_like(ids)
            optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3, weight_decay=0)
            out = model(ids, mask, loss_mask=mask, latent_mask=torch.zeros_like(ids), stage=FORMAT_ACTIVATION)
            out["loss"].backward()
            for name, p in anchor_params.items():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(p.grad.abs().sum().item(), 0, name)
            optimizer.step()
            for name, p in anchor_params.items():
                self.assertFalse(torch.equal(p, anchors_before[name]), name)
            for name, p in model.backbone.named_parameters():
                if name in frozen:
                    self.assertIsNone(p.grad)
                    self.assertTrue(torch.equal(p, frozen[name]), name)
            self.assertTrue(all(p.grad is None for p in model.hybridlink.parameters()))
            with self.assertRaisesRegex(ValueError, "format_activation"):
                model.set_trainable(SELF_CONDITIONED)

            model.output_scale = 0.012
            with torch.no_grad():
                expected = text_forward(model.backbone, input_ids=ids, use_cache=True, return_dict=True).logits
            model.save_pretrained(saved, Tokenizer())
            self.assertTrue((saved / "hybrid_lora.json").is_file())
            self.assertTrue((saved / "lora/adapter_config.json").is_file())
            self.assertFalse((saved / "adapter_config.json").exists())
            restored = HybridAgentModel.from_pretrained(saved, dtype=torch.float32, require_hybridlink=True).eval()
            self.assertEqual(restored.output_scale, model.output_scale)
            with torch.no_grad():
                actual = text_forward(restored.backbone, input_ids=ids, use_cache=True, return_dict=True).logits
            torch.testing.assert_close(actual, expected)
            restored.save_pretrained(Path(temp) / "resaved", Tokenizer())

    def test_frozen_link_activation_calibration_and_checkpoint_roundtrip(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hybrid_agent.model import HybridAgentModel
        from hybrid_agent.lora import enable_format_lora
        from hybrid_agent.compat import text_forward
        from hybrid_sft.model import HybridSFTModel, HYBRIDLINK_ACTIVATION

        class Tokenizer:
            unk_token_id = None
            def __len__(self):
                return 34
            def encode(self, text, **kwargs):
                return [{"<latent>": 32, "</latent>": 33}[text]]
            def save_pretrained(self, directory):
                (Path(directory) / "test-tokenizer.txt").write_text("anchors=32,33")

        torch.manual_seed(11)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            Qwen3ForCausalLM(config).save_pretrained(root / "base")
            initial = HybridSFTModel.from_pretrained(root / "base", dtype=torch.float32)
            enable_format_lora(initial, Tokenizer(), rank=2, alpha=4)
            # Nonzero deltas ensure reloading preserves learned format parameters.
            with torch.no_grad():
                for p in initial.backbone.parameters():
                    if p.requires_grad:
                        p.add_(0.001)
            initial.save_pretrained(root / "format", Tokenizer())
            model = HybridSFTModel.from_pretrained(root / "format", dtype=torch.float32)
            model.set_trainable(HYBRIDLINK_ACTIVATION)
            model.train()
            self.assertFalse(model.backbone.training)
            self.assertTrue(model.hybridlink.training)
            self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
            self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in model.hybridlink.parameters()))
            ids = torch.tensor([[1, 32, 5, 7, 33, 6, 2]])
            batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                     "latent_mask": torch.tensor([[0, 1, 1, 1, 0, 0, 0]])}
            hidden, targets, exits = model._link_pairs(**batch)
            with torch.no_grad():
                embeddings = model.backbone.get_input_embeddings()(ids)
            torch.testing.assert_close(targets, embeddings[:, 2:5].reshape(3, 16))
            self.assertEqual(exits.tolist(), [False, False, True])
            self.assertFalse(hidden.requires_grad)
            self.assertFalse(targets.requires_grad)
            # Padding cannot create an extra regression target, even if its latent mask is set.
            padded = {k: torch.nn.functional.pad(v, (0, 2)) for k, v in batch.items()}
            padded["latent_mask"][0, -2:] = 1
            _, padded_targets, padded_exits = model._link_pairs(**padded)
            torch.testing.assert_close(padded_targets, targets)
            self.assertEqual(padded_exits.tolist(), exits.tolist())
            report = model.calibrate_link_scale([batch])
            self.assertEqual(report["pairs"], 3)
            predicted = model.run_hybridlink(hidden)
            torch.testing.assert_close(predicted.square().mean().sqrt(), targets.square().mean().sqrt())
            before = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
            link_before = {n: p.detach().clone() for n, p in model.hybridlink.named_parameters()}
            with torch.no_grad():
                logits_before = text_forward(model.backbone, input_ids=ids, return_dict=True).logits
            optimizer = torch.optim.AdamW(model.hybridlink.parameters(), lr=1e-3)
            from unittest.mock import patch
            with patch.object(model, "_chunked_ce", side_effect=AssertionError("frozen text CE must be skipped")):
                out = model(**batch, stage=HYBRIDLINK_ACTIVATION)
            out["loss"].backward()
            self.assertEqual(out["backbone_loss"].item(), 0)
            self.assertTrue(torch.isfinite(out["loss"]))
            for p in model.hybridlink.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()
            self.assertTrue(any(not torch.equal(p, link_before[n]) for n, p in model.hybridlink.named_parameters()))
            for n, p in model.backbone.named_parameters():
                self.assertIsNone(p.grad)
                self.assertTrue(torch.equal(p, before[n]), n)
            with torch.no_grad():
                logits_after = text_forward(model.backbone, input_ids=ids, return_dict=True).logits
                link_after = model.run_hybridlink(hidden)
            torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)
            model.save_pretrained(root / "link", Tokenizer())
            restored = HybridAgentModel.from_pretrained(root / "link", dtype=torch.float32).eval()
            self.assertEqual(restored.output_scale, model.output_scale)
            torch.testing.assert_close(restored.run_hybridlink(hidden), link_after, rtol=0, atol=0)
            with torch.no_grad():
                logits_restored = text_forward(restored.backbone, input_ids=ids, return_dict=True).logits
            torch.testing.assert_close(logits_restored, logits_before, rtol=0, atol=0)
            self.assertFalse((root / "link/model.safetensors").exists())
            # FP32 link tensors must not silently be rounded by BF16 backbone loading.
            mixed = HybridAgentModel.from_pretrained(root / "link", dtype=torch.bfloat16)
            for n, p in mixed.hybridlink.named_parameters():
                self.assertEqual(p.dtype, torch.float32)
                self.assertTrue(torch.equal(p, model.hybridlink.state_dict()[n]))
            with self.assertRaisesRegex(ValueError, "nonempty latent"):
                model(**{**batch, "latent_mask": torch.zeros_like(ids)}, stage=HYBRIDLINK_ACTIVATION)
            with self.assertRaisesRegex(ValueError, "nonzero training"):
                model.calibrate_link_scale([])
            with self.assertRaisesRegex(ValueError, "fully frozen"):
                model.backbone.get_input_embeddings().requires_grad_(True)
                model(**batch, stage=HYBRIDLINK_ACTIVATION)

    def test_next_token_ce_reaches_link_only_and_excludes_future_tokens(self):
        import torch
        from unittest.mock import patch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hybrid_agent.lora import enable_format_lora
        from hybrid_agent.compat import text_decoder, text_forward
        from hybrid_sft.model import HybridSFTModel, LINK_NEXT_TOKEN

        class Tokenizer:
            unk_token_id = None
            def __len__(self):
                return 34
            def encode(self, text, **kwargs):
                return [{"<latent>": 32, "</latent>": 33}[text]]
            def save_pretrained(self, directory):
                (Path(directory) / "test-tokenizer.txt").write_text("anchors=32,33")

        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            Qwen3ForCausalLM(config).save_pretrained(root / "base")
            model = HybridSFTModel.from_pretrained(root / "base", dtype=torch.float32)
            enable_format_lora(model, Tokenizer(), rank=2, alpha=4)
            model.output_scale = 0.02
            model.set_trainable(LINK_NEXT_TOKEN)
            model.train()
            self.assertFalse(model.backbone.training)
            ids = torch.tensor([[1, 32, 5, 7, 33, 6, 2]])
            batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                     "latent_mask": torch.tensor([[0, 1, 1, 1, 0, 0, 0]])}
            before = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
            link_before = {n: p.detach().clone() for n, p in model.hybridlink.named_parameters()}
            with torch.no_grad():
                text_before = text_forward(model.backbone, input_ids=ids, return_dict=True).logits
            decoder = text_decoder(model.backbone)
            calls = []
            original = decoder.forward
            def checked_forward(*args, **kwargs):
                self.assertFalse(kwargs["use_cache"])
                self.assertNotIn("past_key_values", kwargs)
                calls.append(kwargs["inputs_embeds"].detach().clone())
                self.assertLessEqual(kwargs["inputs_embeds"].shape[1], 3)
                return original(*args, **kwargs)
            with patch.object(decoder, "forward", side_effect=checked_forward):
                out = model(**batch, stage=LINK_NEXT_TOKEN)
                self.assertTrue(out["loss"].requires_grad)
                self.assertEqual(out["substitution_position"], 2)
                self.assertEqual(out["target_position"], 3)
                self.assertEqual(out["target_id"].item(), 7)
                out["loss"].backward()
            self.assertGreaterEqual(len(calls), 3)  # includes checkpoint recomputation
            self.assertEqual(calls[0].shape[1], 2)
            expected_input = model.backbone.get_input_embeddings()(ids[:, :2]).detach()
            torch.testing.assert_close(calls[0], expected_input)
            torch.testing.assert_close(calls[1][:, :2], expected_input)
            torch.testing.assert_close(calls[1], calls[-1])
            self.assertFalse(torch.equal(calls[1][:, 2:], model.backbone.get_input_embeddings()(ids[:, 2:3])))
            gradients = [p.grad for p in model.hybridlink.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
            # The target and future tokens cannot leak into the decoder's inputs.
            with torch.no_grad():
                altered = ids.clone()
                altered[0, 2] = 12  # replaced input is never read
                altered[0, 4:] = 13  # future suffix is never read
                other = model(**{**batch, "input_ids": altered}, stage=LINK_NEXT_TOKEN)
                torch.testing.assert_close(other["loss"], out["loss"], rtol=0, atol=0)
                # Independent CE from the captured second-pass inputs, no checkpoint.
                hidden = decoder(inputs_embeds=calls[1], attention_mask=torch.ones(1, 3, dtype=torch.long),
                                 use_cache=False, return_dict=True).last_hidden_state[:, -1:]
                logits = model.backbone.get_output_embeddings()(hidden).float()
                expected_ce = torch.nn.functional.cross_entropy(logits[:, -1], torch.tensor([7]))
                torch.testing.assert_close(expected_ce, out["loss"])
            optimizer = torch.optim.AdamW(model.hybridlink.parameters(), lr=1e-3)
            optimizer.step()
            self.assertTrue(any(not torch.equal(p, link_before[n]) for n, p in model.hybridlink.named_parameters()))
            for n, p in model.backbone.named_parameters():
                self.assertIsNone(p.grad)
                self.assertFalse(p.requires_grad)
                self.assertTrue(torch.equal(p, before[n]))
            with torch.no_grad():
                text_after = text_forward(model.backbone, input_ids=ids, return_dict=True).logits
                loss_after = model(**batch, stage=LINK_NEXT_TOKEN)["loss"]
            torch.testing.assert_close(text_after, text_before, rtol=0, atol=0)
            model.save_pretrained(root / "saved", Tokenizer())
            restored = HybridSFTModel.from_pretrained(root / "saved", dtype=torch.float32)
            restored.set_trainable(LINK_NEXT_TOKEN)
            self.assertEqual(restored.output_scale, model.output_scale)
            with torch.no_grad():
                torch.testing.assert_close(restored(**batch, stage=LINK_NEXT_TOKEN)["loss"], loss_after, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                model(**{**batch, "latent_mask": torch.zeros_like(ids)}, stage=LINK_NEXT_TOKEN)
            with self.assertRaisesRegex(ValueError, "batch size one"):
                model(**{k: v.repeat(2, 1) for k, v in batch.items()}, stage=LINK_NEXT_TOKEN)
            model.backbone.get_input_embeddings().requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "fully frozen"):
                model(**batch, stage=LINK_NEXT_TOKEN)

    def test_one_step_schedule_has_nonzero_first_update(self):
        import torch
        from hybrid_sft.train import cosine_schedule_with_warmup
        p = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([p], lr=0.01)
        cosine_schedule_with_warmup(optimizer, 0, 1)
        self.assertGreater(optimizer.param_groups[0]["lr"], 0)


if __name__ == "__main__":
    unittest.main()
