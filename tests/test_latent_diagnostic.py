"""Controlled substitutions test mechanics, not learned reasoning."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("latent_diagnostic", ROOT / "scripts/latent_diagnostic.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DiagnosticPlanTests(unittest.TestCase):
    def test_ignores_system_anchors_and_rejects_empty_reference(self):
        sample = {"input_ids": [2, 3, 9, 2, 4, 5, 3, 6, 1], "prompt_len": 3}
        self.assertEqual(diagnostic.span_bounds(sample, 2, 3), (3, 6))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            diagnostic.span_bounds({"input_ids": [9, 2, 3], "prompt_len": 1}, 2, 3)

    def test_substitution_counts_are_fixed_and_bounded(self):
        self.assertEqual([diagnostic.replacement_count(c, 10) for c in
                          ("teacher", "first-1", "first-4", "all")], [0, 1, 4, 10])
        self.assertEqual(diagnostic.replacement_count("first-4", 2), 2)
        with self.assertRaises(ValueError):
            diagnostic.replacement_count("unknown", 10)


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("torch", "transformers")),
                     "requires torch and transformers")
class DiagnosticRuntimeTests(unittest.TestCase):
    def test_same_segmentation_correct_substitutions_and_real_exit(self):
        import torch
        from unittest.mock import patch
        embedding = torch.nn.Embedding(12, 4)
        with torch.no_grad():
            embedding.weight.copy_(torch.arange(48).reshape(12, 4).float())
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = embedding
                self.generation_config = SimpleNamespace(eos_token_id=1)
            def get_input_embeddings(self):
                return self.embedding
        class Model:
            backbone = Backbone()
            def run_hybridlink(self, hidden, output_dtype=None):
                return torch.full_like(hidden, -7, dtype=output_dtype)
        class Tokenizer:
            eos_token_id = 1
            def decode(self, ids, **kwargs):
                return ",".join(map(str, ids))
        sample = {"input_ids": [2, 3, 9, 6, 2, 4, 5, 7, 8, 10, 3, 11, 1], "prompt_len": 3}
        for condition, count in (("teacher", 0), ("first-1", 1), ("first-4", 4), ("all", 5)):
            with self.subTest(condition=condition):
                calls, cache = [], None
                def forward(backbone, **kwargs):
                    nonlocal cache
                    if "input_ids" in kwargs:
                        self.assertIsNone(cache)
                        self.assertEqual(kwargs["input_ids"].tolist(), [[2, 3, 9]])
                        cache = SimpleNamespace(length=3)
                        cache.get_seq_length = lambda: cache.length
                        next_token = 6
                    else:
                        self.assertIs(kwargs["past_key_values"], cache)
                        self.assertEqual(kwargs["position_ids"].item(), cache.length)
                        self.assertEqual(kwargs["attention_mask"].shape[1], cache.length + 1)
                        cache.length += 1
                        calls.append(kwargs["inputs_embeds"].clone())
                        # Plan, entry, five reference positions, forced exit, continuation.
                        next_token = [2, 4, 5, 7, 8, 10, 3, 11, 1][len(calls) - 1]
                    logits = torch.full((1, 1, 12), -20.0)
                    logits[0, 0, next_token] = 20
                    return SimpleNamespace(logits=logits, past_key_values=cache,
                                           hidden_states=(torch.zeros(1, 1, 4),))
                with patch("hybrid_agent.compat.text_forward", side_effect=forward), torch.inference_mode():
                    result = diagnostic.run_condition(Model(), Tokenizer(), sample, 2, 3, condition)
                self.assertEqual(len(calls), 9)
                for i, token in enumerate((4, 5, 7, 8, 10)):
                    expected = torch.full((4,), -7.0) if i < count else embedding.weight[token]
                    torch.testing.assert_close(calls[i + 2][0, 0], expected)
                torch.testing.assert_close(calls[7][0, 0], embedding.weight[3])
                self.assertEqual(result["replaced_tokens"], count)
                self.assertEqual(result["steps"][-1]["target_id"], 3)
                self.assertEqual(result["steps"][-1]["fed"], "stop-before-exit")
                self.assertEqual(result["steps"][-1]["exit_rank"], 1)
                self.assertEqual(result["continuation_after_forced_exit"], "11")
                self.assertEqual(result["continuation_end_reason"], "eos")


if __name__ == "__main__":
    unittest.main()
