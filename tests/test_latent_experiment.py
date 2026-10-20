"""Dataset isolation and honest latent success criteria, without ML dependencies."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("latent_experiment", ROOT / "scripts/latent_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
import format_experiment as format_pilot


class LatentExperimentTests(unittest.TestCase):
    def test_reference_config_ignores_only_local_source_flag(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, saved = root / "source", root / "saved"
            source.mkdir()
            saved.mkdir()
            configs = {
                "hybrid_lora.json": {"base_revision": "fixed", "anchor_ids": [32, 33]},
                "tokenizer.json": {"model": {"vocab": {"a": 0}}},
                "tokenizer_config.json": {"is_local": False, "eos_token": "EOS",
                                          "padding_side": "left", "chat_template": "original"},
            }
            for name, value in configs.items():
                format_pilot.write_json(source / name, value)
                format_pilot.write_json(saved / name, value)
            changed = {**configs["tokenizer_config.json"], "is_local": True}
            format_pilot.write_json(saved / "tokenizer_config.json", changed)
            before = {str(p): p.read_bytes() for d in (source, saved) for p in d.iterdir()}
            experiment.check_reference_config(saved, source)
            self.assertEqual(before, {p: Path(p).read_bytes() for p in before})
            for key in ("eos_token", "padding_side", "chat_template", "unknown_future_setting"):
                with self.subTest(key=key):
                    format_pilot.write_json(saved / "tokenizer_config.json", {**changed, key: "different"})
                    with self.assertRaisesRegex(ValueError, "tokenizer_config.json"):
                        experiment.check_reference_config(saved, source)
            format_pilot.write_json(saved / "tokenizer_config.json", changed)
            for name in ("hybrid_lora.json", "tokenizer.json"):
                with self.subTest(name=name):
                    format_pilot.write_json(saved / name, {**configs[name], "is_local": True})
                    with self.assertRaises(ValueError):
                        experiment.check_reference_config(saved, source)
                    format_pilot.write_json(saved / name, configs[name])

    def test_fixed_new_splits_preserve_old_data_and_training_only_scale(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            pilot, general, first, second = [root / n for n in ("pilot", "general", "first", "second")]
            with contextlib.redirect_stdout(io.StringIO()):
                format_pilot.prepare(pilot, 6060)
                format_pilot.prepare_generalization(general, pilot)
                cal = root / "calibration.json"
                format_pilot.write_json(cal, {"output_scale": 0.012, "data_sha256": format_pilot.sha256(pilot / "train.jsonl")})
                old = {str(p): p.read_bytes() for folder in (pilot, general) for p in folder.iterdir()}
                experiment.prepare(first, pilot, general, cal)
                experiment.prepare(second, pilot, general, cal)
            self.assertEqual(old, {p: Path(p).read_bytes() for p in old})
            seen = {tuple(json.loads(line)["operands"]) for p, data in old.items()
                    if p.endswith(".jsonl") for line in data.splitlines()}
            for split, count in (("latent-dev", 8), ("latent-test", 16)):
                path, manifest = experiment.verify_data(first, split)
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
                rows = [json.loads(s) for s in path.read_text().splitlines()]
                self.assertEqual(len(rows), count)
                for r in rows:
                    key = tuple(r["operands"])
                    self.assertNotIn(key, seen)
                    seen.add(key)
                    a, b, c = key
                    self.assertEqual(r["expected"], a*b-c)
                    self.assertEqual(r["system"], format_pilot.SYSTEM)
                self.assertEqual(manifest["output_scale"], 0.012)
                self.assertEqual(manifest["hybrid"]["min_latent_steps"], 0)
            with self.assertRaises(FileExistsError):
                experiment.prepare(first, pilot, general, cal)
            format_pilot.write_json(cal, {"output_scale": 0.012, "data_sha256": "not-training"})
            with self.assertRaisesRegex(ValueError, "training data"):
                experiment.prepare(root / "bad", pilot, general, cal)
            self.assertFalse((root / "bad").exists())

    def test_success_requires_real_vectors_natural_exit_and_complete_answer(self):
        text = "<think>Plan.<latent></latent>Done.</think><answer>\\boxed{117}</answer>"
        good = {"end_reason": "eos", "latent_spans": 1, "latent_steps": 2,
                "anchor_exits": 1, "forced_exits": 0, "tool_requests": 0}
        self.assertTrue(experiment.score_hybrid(text, 117, good)["success"])
        for patch in ({"forced_exits": 1, "anchor_exits": 0}, {"latent_steps": 0},
                      {"latent_spans": 0}, {"latent_spans": 2}, {"anchor_exits": 0},
                      {"end_reason": "budget"}, {"end_reason": "tool_cap"}, {"tool_requests": 1}):
            with self.subTest(patch=patch):
                score = experiment.score_hybrid(text, 117, {**good, **patch})
                self.assertTrue(score["answer_correct"])
                self.assertFalse(score["success"])
        for broken in (text.replace("117", "17"), text.replace("</latent>", ""),
                       text.replace("<latent></latent>", "<latent>fake text</latent>"),
                       text + "<answer>\\boxed{117}</answer>"):
            self.assertFalse(experiment.score_hybrid(broken, 117, good)["success"])
        self.assertFalse(experiment.score_hybrid(text, 117, {**good, "end_reason": "tool_cap"})["truncated"])
        self.assertTrue(experiment.score_hybrid(text, 117, {**good, "end_reason": "budget"})["truncated"])


if __name__ == "__main__":
    unittest.main()
