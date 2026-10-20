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

    def test_one_step_schedule_has_nonzero_first_update(self):
        import torch
        from hybrid_sft.train import cosine_schedule_with_warmup
        p = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([p], lr=0.01)
        cosine_schedule_with_warmup(optimizer, 0, 1)
        self.assertGreater(optimizer.param_groups[0]["lr"], 0)


if __name__ == "__main__":
    unittest.main()
