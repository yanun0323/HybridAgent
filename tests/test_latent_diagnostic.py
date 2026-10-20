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
        self.assertEqual(diagnostic.condition_names("substitution"), ("teacher", "first-1", "first-4", "all"))
        self.assertEqual(diagnostic.condition_names("direction-size"),
                         ("teacher", "all", "pred-direction", "target-direction"))
        for name in ("pred-direction", "target-direction"):
            self.assertEqual(diagnostic.replacement_count(name, 10), 10)
        with self.assertRaises(ValueError):
            diagnostic.condition_names("unknown")
        with self.assertRaises(ValueError):
            diagnostic.replacement_count("unknown", 10)


    def test_third_input_plan_has_four_fixed_masks_and_only_ten_percent(self):
        self.assertEqual(diagnostic.condition_names("third-input"),
                         ("teacher", "first-2", "third-only", "first-3"))
        expected = ((), (0, 1), (2,), (0, 1, 2))
        for condition, positions in zip(diagnostic.condition_names("third-input"), expected):
            self.assertEqual(diagnostic.replacement_positions(condition, 24), positions)
            self.assertEqual(diagnostic.replacement_count(condition, 24), len(positions))
        self.assertEqual(diagnostic.diagnostic_fractions("third-input"), (.1,))
        self.assertEqual(diagnostic.diagnostic_fractions("substitution"), (0., .1))
        with self.assertRaisesRegex(ValueError, "three"):
            diagnostic.replacement_positions("third-only", 2)


class PairedDiagnosticTests(unittest.TestCase):
    def test_teacher_replay_ignores_link_metrics_but_rejects_behavior_changes(self):
        import copy
        before = {"reference_latent_tokens": 1, "replaced_tokens": 0,
                  "continuation_after_forced_exit": "answer", "continuation_end_reason": "eos",
                  "expected_text_continuation": "answer",
                  "steps": [{"step": 0, "target_id": 5, "predicted_id": 5,
                             "target_probability": .9, "exit_probability": .01,
                             "exit_rank": 3, "exit_argmax": False, "cosine": .5}]}
        after = copy.deepcopy(before)
        after["steps"][0]["cosine"] = .8
        diagnostic.check_teacher_replay(before, after)
        for key in ("predicted_id", "target_probability", "exit_probability", "exit_rank", "exit_argmax"):
            bad = copy.deepcopy(after)
            bad["steps"][0][key] = 99
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "teacher control changed"):
                diagnostic.check_teacher_replay(before, bad)
        for key in ("continuation_after_forced_exit", "continuation_end_reason"):
            with self.assertRaises(RuntimeError):
                diagnostic.check_teacher_replay(before, {**after, key: "changed"})
        with self.assertRaisesRegex(RuntimeError, "step count"):
            diagnostic.check_teacher_replay(before, {**after, "steps": []})

    def test_pair_provenance_binds_exact_saved_update_and_scale(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, joint, step = root / "source", root / "joint", root / "step"
            def save(path, obj):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(obj))
            save(source / "adapter.pt", "source")
            save(joint / "checkpoint/adapter.pt", "updated")
            save(source / "lora/adapter_model.safetensors", "lora")
            config = {"output_scale": .012}
            for directory in (source, joint / "checkpoint"):
                save(directory / "hybridlink_config.json", config)
            run = {"source_link_sha256": diagnostic.sha256(source / "adapter.pt"),
                   "source_lora_sha256": diagnostic.sha256(source / "lora/adapter_model.safetensors"),
                   "calibration": config, "data_sha256": "data", "update_ids": ["a"], "observation_ids": ["b"]}
            save(joint / "run.json", run)
            save(joint / "summary.json", {})
            prior = {"source_link_sha256": run["source_link_sha256"],
                     "updated_link_sha256": diagnostic.sha256(joint / "checkpoint/adapter.pt"),
                     "joint_run_sha256": diagnostic.sha256(joint / "run.json"),
                     "joint_summary_sha256": diagnostic.sha256(joint / "summary.json"),
                     "fractions_in_execution_order": [0., 1., .01, .1], "scale": .012,
                     "data_sha256": "data", "update_ids": ["a"], "observation_ids": ["b"]}
            save(step / "run.json", prior)
            save(step / "summary.json", {"results": [{"fraction": f} for f in [0., 1., .01, .1]]})
            result = diagnostic.check_pair_provenance(source, joint, step, .012)
            self.assertEqual(result["source_link_sha256"], run["source_link_sha256"])
            with self.assertRaisesRegex(ValueError, "scale"):
                diagnostic.check_pair_provenance(source, joint, step, .013)
            save(joint / "checkpoint/adapter.pt", "different update")
            with self.assertRaisesRegex(ValueError, "provenance"):
                diagnostic.check_pair_provenance(source, joint, step, .012)


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("torch", "transformers")),
                     "requires torch and transformers")
class DiagnosticRuntimeTests(unittest.TestCase):
    def test_direction_size_controls_preserve_selected_direction_and_magnitude(self):
        import torch
        import torch.nn.functional as F
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                # Equal-sized dimensions but different directions AND magnitudes.
                p = torch.tensor([[[3., 4., 0., 0.]]], dtype=dtype)
                t = torch.tensor([[[0., 0., 6., 8.]]], dtype=dtype)
                before_p, before_t = p.clone(), t.clone()
                pd = diagnostic.mixed_vector(p, t, "pred-direction")
                td = diagnostic.mixed_vector(p, t, "target-direction")
                self.assertEqual(pd.dtype, dtype)
                self.assertEqual(td.dtype, dtype)
                torch.testing.assert_close(pd, p * 2)
                torch.testing.assert_close(td, t / 2)
                for vector, direction, magnitude in ((pd, p, t), (td, t, p)):
                    self.assertAlmostEqual(F.cosine_similarity(vector.float().flatten(),
                                                              direction.float().flatten(), dim=0).item(), 1, places=6)
                    torch.testing.assert_close(vector.float().square().mean().sqrt(),
                                               magnitude.float().square().mean().sqrt())
                self.assertTrue(torch.equal(p, before_p))
                self.assertTrue(torch.equal(t, before_t))
        for bad in (torch.zeros_like(p), torch.full_like(p, float("nan")), torch.full_like(p, float("inf"))):
            with self.assertRaises(ValueError):
                diagnostic.mixed_vector(bad, t, "pred-direction")
            with self.assertRaises(ValueError):
                diagnostic.mixed_vector(p, bad, "target-direction")
        with self.assertRaises(ValueError):
            diagnostic.mixed_vector(p, t, "unknown")

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
        for condition, count in (("teacher", 0), ("first-1", 1), ("first-4", 4), ("all", 5),
                                 ("pred-direction", 5), ("target-direction", 5),
                                 ("first-2", 2), ("third-only", 1), ("first-3", 3)):
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
                    target = embedding.weight[token]
                    substituted = i == 2 if condition == "third-only" else i < count
                    expected = torch.full((4,), -7.0) if substituted else target
                    self.assertEqual(result["steps"][i]["fed"], condition if condition in
                                     ("pred-direction", "target-direction") else
                                     "predicted" if substituted else "teacher")
                    if condition == "pred-direction":
                        expected = -torch.ones(4) * target.square().mean().sqrt()
                    elif condition == "target-direction":
                        expected = target * (7 / target.square().mean().sqrt())
                    actual = calls[i + 2][0, 0]
                    torch.testing.assert_close(actual, expected)
                    self.assertAlmostEqual(result["steps"][i]["fed_rms"], actual.square().mean().sqrt().item())
                torch.testing.assert_close(calls[7][0, 0], embedding.weight[3])
                self.assertEqual(result["replaced_tokens"], count)
                self.assertEqual(result["replacement_positions"], [2] if condition == "third-only" else list(range(count)))
                self.assertEqual(result["steps"][-1]["target_id"], 3)
                self.assertEqual(result["steps"][-1]["fed"], "stop-before-exit")
                self.assertEqual(result["steps"][-1]["exit_rank"], 1)
                self.assertEqual(result["continuation_after_forced_exit"], "11")
                self.assertEqual(result["continuation_end_reason"], "eos")


if __name__ == "__main__":
    unittest.main()
