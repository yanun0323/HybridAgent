"""Small CPU tests; no model downloads. Requires project dependencies."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "requires torch and transformers")
class RuntimeTests(unittest.TestCase):
    def test_scale_checkpoint_roundtrip_and_legacy_config(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hybrid_agent.model import HybridAgentModel
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8)
        model = HybridAgentModel(Qwen3ForCausalLM(config)).eval()
        model.output_scale = 0.012
        hidden = torch.randn(1, 16)
        expected = model.run_hybridlink(hidden)
        with TemporaryDirectory() as temp:
            model.save_pretrained(temp)
            self.assertFalse((Path(temp) / "adapter_config.json").exists())
            restored = HybridAgentModel.from_pretrained(temp, dtype=torch.float32, require_hybridlink=True)
            torch.testing.assert_close(restored.run_hybridlink(hidden), expected)
            # A legacy config must also load when PEFT is installed.
            (Path(temp) / "hybridlink_config.json").rename(Path(temp) / "adapter_config.json")
            restored = HybridAgentModel.from_pretrained(temp, dtype=torch.float32, require_hybridlink=True)
            torch.testing.assert_close(restored.run_hybridlink(hidden), expected)

    def test_generated_experiment_record_passes_real_dataset_loader(self):
        import importlib.util
        import json
        from hybrid_sft.data import HybridSFTDataset
        script = Path(__file__).resolve().parents[1] / "scripts/format_experiment.py"
        spec = importlib.util.spec_from_file_location("runtime_experiment", script)
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        class Tokenizer:
            eos_token_id = 999
            def apply_chat_template(self, *args, **kwargs):
                return "assistant\n<think>\n"
            def __call__(self, text, **kwargs):
                special = {"<latent>": 1000, "</latent>": 1001}
                return {"input_ids": [special[text]] if text in special else list(map(ord, text))}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            path.write_text(json.dumps(experiment.record(50, 3, 1)) + "\n")
            dataset = HybridSFTDataset(Tokenizer(), path, 1024, text_only=True)
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            start = sample["prompt_len"]
            self.assertEqual(sample["input_ids"][start:start + 8], list(map(ord, "Multiply")))

    def test_entry_exit_answer_and_observation_masks(self):
        from hybrid_sft.data import tokenize_trajectory
        class Tokenizer:
            eos_token_id = 999
            def apply_chat_template(self, *args, **kwargs):
                return "assistant\n<think>\n"
            def __call__(self, text, **kwargs):
                special = {"<latent>": 1000, "</latent>": 1001}
                return {"input_ids": [special[text]] if text in special else list(map(ord, text))}
        tokenizer = Tokenizer()
        rounds = [
            {"assistant": "<think>\nPlan.<latent>xy</latent>Done.</think><python>1</python>",
             "tool_result": "<result>7</result>"},
            {"assistant": "<answer>169</answer>"},
        ]
        sample = tokenize_trajectory(tokenizer, "", "q", rounds, 4096)
        ids, lat, loss = sample["input_ids"], sample["latent_mask"], sample["loss_mask"]
        entry, exit = ids.index(1000), ids.index(1001)
        self.assertEqual(lat[entry:exit + 1], [1, 1, 1, 0])
        self.assertEqual(loss[entry:exit + 1], [1, 0, 0, 2])
        # Sources at the entry and final latent token predict the first latent token
        # and EXIT respectively, never the first digit of the visible answer.
        text = ''.join(chr(i) if i < 1000 else '?' for i in ids)
        answer = text.index('<answer>169') + len('<answer>')
        self.assertEqual(lat[answer - 1:answer + 3], [0, 0, 0, 0])
        self.assertEqual(loss[answer:answer + 3], [1, 1, 1])
        observation = text.index('<result>')
        self.assertTrue(all(v == 0 for v in loss[observation:observation + len('<result>7</result>')]))


if __name__ == "__main__":
    unittest.main()
