"""Fixed selection, per-record reporting and mean-gradient accumulation."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("link_joint_step", ROOT / "scripts/link_joint_step.py")
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)


class JointPlanTests(unittest.TestCase):
    def test_fixed_disjoint_selection_never_uses_outcome_to_choose_rows(self):
        rows = [{"id": str(i), "question": f"q{i}"} for i in range(64)]
        update, observation = joint.select_rows(rows)
        self.assertEqual(update, rows[:8])
        self.assertEqual(observation, rows[8:16])
        with self.assertRaises(ValueError):
            joint.select_rows(rows[:15])
        with self.assertRaises(ValueError):
            joint.select_rows([rows[0]] * 16)

    def test_report_keeps_improvements_and_regressions_separate(self):
        rows = [{"id": str(i), "question": f"q{i}"} for i in range(3)]
        metadata = {"target_id": 15, "target_position": 124, "substitution_position": 123}
        before = [{**metadata, "ce": v} for v in (1., 1., 1.)]
        after = [{**metadata, "ce": v} for v in (0., 2., 1.)]
        result = joint.comparison(rows, before, after)
        self.assertEqual(result["mean_before_ce"], result["mean_after_ce"])
        self.assertEqual((result["lower_ce"], result["higher_ce"], result["unchanged_within_1e-6"]), (1, 1, 1))
        self.assertEqual([r["delta_ce"] for r in result["rows"]], [-1, 1, 0])
        with self.assertRaisesRegex(ValueError, "counts"):
            joint.comparison(rows, before[:1], after)
        after[0]["target_id"] = 16
        with self.assertRaisesRegex(ValueError, "positions"):
            joint.comparison(rows, before, after)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class JointGradientTests(unittest.TestCase):
    def test_average_backward_matches_mean_loss_and_never_updates_early(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.link = torch.nn.Linear(2, 1, bias=False)
            def forward(self, x, y, stage):
                loss = (self.link(x) - y).square().mean()
                return {"loss": loss, "target_id": torch.tensor([15]),
                        "target_position": 4, "substitution_position": 3}
        torch.manual_seed(42)
        actual, expected = Model(), Model()
        expected.load_state_dict(actual.state_dict())
        batches = [{"x": torch.tensor([[float(i), 1.]]), "y": torch.tensor([[i / 3.]])} for i in range(8)]
        original = actual.link.weight.detach().clone()
        calls = []
        def backward(loss):
            self.assertTrue(torch.equal(actual.link.weight, original))
            calls.append(loss.detach().item())
            loss.backward()
        metrics = joint.average_backward(actual, batches, "unused", backward)
        reference = torch.stack([expected(**b, stage="unused")["loss"] for b in batches]).mean()
        reference.backward()
        torch.testing.assert_close(actual.link.weight.grad, expected.link.weight.grad)
        self.assertEqual(len(calls), 8)
        self.assertAlmostEqual(sum(calls), sum(r["ce"] for r in metrics) / 8)
        for model in (actual, expected):
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        torch.testing.assert_close(actual.link.weight, expected.link.weight)
        self.assertFalse(torch.equal(actual.link.weight, original))
        with self.assertRaises(ValueError):
            joint.average_backward(actual, [], "unused", backward)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            joint.average_backward(actual, [{"x": torch.full((1, 2), float("nan")), "y": torch.ones(1, 1)}], "unused", backward)


if __name__ == "__main__":
    unittest.main()
