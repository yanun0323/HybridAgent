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
