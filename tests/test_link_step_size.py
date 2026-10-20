"""Endpoint replay and immutable fixed-fraction interpolation."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("link_step_size", ROOT / "scripts/link_step_size.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class EndpointTests(unittest.TestCase):
    def test_endpoint_checks_positions_ids_and_ce_before_interpreting_fractions(self):
        self.assertEqual(probe.FRACTIONS, (0., 1., .01, .1))
        rows = [{"id": "a"}]
        meta = {"target_id": 15, "target_position": 124, "substitution_position": 123}
        measured = [{**meta, "ce": 0.6}]
        saved = [{**meta, "id": "a", "before_ce": 0.6, "after_ce": 2.5}]
        self.assertEqual(probe.verify_endpoint(rows, measured, saved, "before_ce"), 0)
        with self.assertRaisesRegex(ValueError, "CE mismatch"):
            probe.verify_endpoint(rows, measured, saved, "after_ce")
        for field in ("target_id", "target_position", "substitution_position"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "IDs or target"):
                probe.verify_endpoint(rows, [{**measured[0], field: -1}], saved, "before_ce")
        with self.assertRaisesRegex(ValueError, "IDs or target"):
            probe.verify_endpoint([{"id": "b"}], measured, saved, "before_ce")
        with self.assertRaisesRegex(ValueError, "counts"):
            probe.verify_endpoint([], measured, saved, "before_ce")
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                probe.verify_endpoint(rows, [{**measured[0], "ce": value}], saved, "before_ce")
        near = [{**measured[0], "ce": .600001}]
        self.assertGreater(probe.verify_endpoint(rows, near, saved, "before_ce"), 0)


class RemainingTrainTests(unittest.TestCase):
    def inputs(self):
        rows = [{"id": str(i), "question": f"q{i}"} for i in range(64)]
        run = {"update_ids": [r["id"] for r in rows[:8]],
               "observation_ids": [r["id"] for r in rows[8:16]],
               "calibration": {"output_scale": .012}}
        hashes = {"source_link_sha256": "a", "updated_link_sha256": "b",
                  "data_sha256": "c", "joint_run_sha256": "d", "joint_summary_sha256": "e"}
        prior = {**hashes, "update_ids": run["update_ids"], "observation_ids": run["observation_ids"],
                 "scale": .012, "fractions_in_execution_order": list(probe.FRACTIONS)}
        summary = {"results": [{"fraction": f, "update": {"rows": rows[:8]},
                                "observation": {"rows": rows[8:16]}} for f in probe.FRACTIONS]}
        return rows, run, prior, summary, hashes

    def test_fixed_remaining_rows_and_separate_reporting(self):
        rows, run, prior, summary, hashes = self.inputs()
        remaining, references = probe.remaining_plan(rows, run, prior, summary, hashes)
        self.assertEqual(remaining, rows[16:])
        self.assertEqual(len(remaining), 48)
        self.assertFalse({r["id"] for r in remaining} & set(run["update_ids"] + run["observation_ids"]))
        self.assertEqual(set(references), {0., .1})
        self.assertEqual(references[.1], rows[:16])
        meta = {"target_id": 15, "target_position": 124, "substitution_position": 123}
        before = [{**meta, "ce": 1.} for _ in remaining]
        after = [{**meta, "ce": 0. if i < 24 else 2.} for i in range(48)]
        report = probe.comparison(remaining, before, after)
        self.assertEqual((report["lower_ce"], report["higher_ce"]), (24, 24))
        self.assertEqual([r["id"] for r in report["rows"]], [r["id"] for r in remaining])

    def test_rejects_changed_data_weights_selection_and_reference(self):
        import copy
        rows, run, prior, summary, hashes = self.inputs()
        for bad_rows in (rows[:63], rows + [rows[0]], rows[:63] + [rows[0]]):
            with self.assertRaisesRegex(ValueError, "64 unique"):
                probe.remaining_plan(bad_rows, run, prior, summary, hashes)
        for key in hashes:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "provenance"):
                probe.remaining_plan(rows, run, {**prior, key: "changed"}, summary, hashes)
        for key, value in (("scale", .02), ("update_ids", []), ("observation_ids", []),
                           ("fractions_in_execution_order", [0., .1])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.remaining_plan(rows, run, {**prior, key: value}, summary, hashes)
        for bad in ({"results": summary["results"][:-1]},
                    {"results": summary["results"] + [summary["results"][-1]]}):
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                probe.remaining_plan(rows, run, prior, bad, hashes)
        bad = copy.deepcopy(summary)
        bad["results"][-1]["observation"]["rows"][0]["id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            probe.remaining_plan(rows, run, prior, bad, hashes)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class InterpolationTests(unittest.TestCase):
    def test_exact_endpoints_fractions_no_alias_or_mutation(self):
        import torch
        before = {"weight": torch.tensor([1e10, 1., -2.])}
        after = {"weight": torch.tensor([1., 3., 2.])}
        original = {k: v.clone() for k, v in before.items()}
        original_after = {k: v.clone() for k, v in after.items()}
        for fraction in probe.FRACTIONS:
            mixed = probe.interpolate_state(before, after, fraction)
            expected = before["weight"] if fraction == 0 else after["weight"] if fraction == 1 else (
                before["weight"] + fraction * (after["weight"] - before["weight"]))
            self.assertTrue(torch.equal(mixed["weight"], expected))
            self.assertEqual(mixed["weight"].dtype, torch.float32)
            mixed["weight"].zero_()
            self.assertTrue(torch.equal(before["weight"], original["weight"]))
            self.assertTrue(torch.equal(after["weight"], original_after["weight"]))
        for value in (-1, 2, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "fraction"):
                probe.interpolate_state(before, after, value)
        for bad in ({"other": after["weight"]}, {"weight": torch.ones(4)},
                    {"weight": after["weight"].bfloat16()},
                    {"weight": torch.full((3,), float("nan"))}):
            with self.assertRaises(ValueError):
                probe.interpolate_state(before, bad, .1)


if __name__ == "__main__":
    unittest.main()
