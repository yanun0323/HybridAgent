"""Three/six-step joint averaging and reports must expose individual regressions."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('link_three_step_joint', ROOT / 'scripts/link_three_step_joint.py')
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)


class JointReportTests(unittest.TestCase):
    def test_fixed_64_record_plan_has_eight_disjoint_ordered_groups(self):
        rows = [{'id': str(i)} for i in range(64)]
        update, observe, groups = joint.training_plan(rows, all_train=True)
        self.assertEqual(update, rows)
        self.assertEqual(observe, [])
        self.assertEqual([r for group in groups for r in group], rows)
        self.assertEqual([len(group) for group in groups], [8] * 8)
        self.assertEqual(joint.training_plan(rows), (rows[:8], rows[8:16], [rows[:8]]))
        for bad in (rows[:63], rows + [rows[0]], rows[:63] + [rows[0]]):
            with self.assertRaisesRegex(ValueError, '64 unique'):
                joint.training_plan(bad, all_train=True)

    def test_six_step_report_preserves_last_position_regression(self):
        rows = [{'id': 'a', 'question': 'a'}, {'id': 'b', 'question': 'b'}]
        meta = {'target_ids': list(range(6)), 'predicted_ids': list(range(6)),
                'target_positions': list(range(10, 16)), 'substitution_positions': list(range(9, 15))}
        before = [{**meta, 'step_ce': [1.] * 6}] * 2
        after = [{**meta, 'step_ce': [0.] * 5 + [2.], 'predicted_ids': list(range(5)) + [99]},
                 {**meta, 'step_ce': [2.] * 6}]
        result = joint.compare_group(rows, before, after, link_steps=6)
        self.assertEqual(result['mean_ce']['lower_ce'], 1)
        self.assertEqual(result['mean_ce']['higher_ce'], 1)
        self.assertAlmostEqual(result['mean_ce']['mean_after_ce'], (2 / 6 + 2) / 2)
        self.assertEqual(len(result['steps']), 6)
        self.assertEqual(result['steps'][5]['higher_ce'], 2)
        self.assertEqual(result['all_six_correct_before'], 2)
        self.assertEqual(result['all_six_correct_after'], 1)
        self.assertNotIn('all_three_correct_after', result)
        self.assertEqual(result['rows'][0]['delta_step_ce'], [-1.] * 5 + [1.])
        with self.assertRaisesRegex(ValueError, 'positions'):
            joint.compare_group(rows, before, after)  # three-input default must reject six
        with self.assertRaisesRegex(ValueError, 'positions'):
            joint.compare_group(rows, before, [{**after[0], 'step_ce': [0.] * 3}, after[1]], link_steps=6)

    def test_cli_limits_six_inputs_to_one_update(self):
        import subprocess
        import sys
        script = str(ROOT / 'scripts/link_three_step_joint.py')
        result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--link-steps {3,6}', result.stdout)
        base = [sys.executable, script, '--pilot', 'unused', '--model', 'unused',
                '--calibration', 'unused', '--output-dir', 'unused']
        result = subprocess.run(base + ['--link-steps', '6', '--all-train'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('one eight-record update only', result.stderr)
        result = subprocess.run(base + ['--link-steps', '5'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid choice', result.stderr)

    def test_mean_does_not_hide_position_or_record_regressions(self):
        rows = [{'id': 'a', 'question': 'a'}, {'id': 'b', 'question': 'b'}]
        meta = {'target_ids': [1, 2, 3], 'target_positions': [4, 5, 6],
                'substitution_positions': [3, 4, 5], 'predicted_ids': [1, 2, 3]}
        before = [{**meta, 'step_ce': [1., 1., 1.]}, {**meta, 'step_ce': [1., 1., 1.]}]
        after = [{**meta, 'step_ce': [0., 0., 2.]},
                 {**meta, 'step_ce': [2., 1., 1.], 'predicted_ids': [4, 2, 3]}]
        result = joint.compare_group(rows, before, after)
        self.assertEqual(result['mean_ce']['lower_ce'], 1)
        self.assertEqual(result['mean_ce']['higher_ce'], 1)
        self.assertEqual(result['steps'][2]['higher_ce'], 1)
        self.assertEqual(result['all_three_correct_before'], 2)
        self.assertEqual(result['all_three_correct_after'], 1)
        self.assertEqual(result['rows'][0]['delta_step_ce'], [-1., -1., 1.])
        with self.assertRaises(ValueError):
            joint.compare_group(rows, before[:1], after)
        with self.assertRaisesRegex(ValueError, 'positions'):
            joint.compare_group(rows, before, [{**after[0], 'target_ids': [7, 2, 3]}, after[1]])
        self.assertEqual(joint.LR, 5e-6)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
class JointAccumulationTests(unittest.TestCase):
    def test_eight_updates_keep_adam_state_and_clear_gradients(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hybridlink = torch.nn.Linear(1, 1, bias=False)
        class Accelerator:
            backward = staticmethod(lambda loss: loss.backward())
            unwrap_model = staticmethod(lambda model: model)
            clip_grad_norm_ = staticmethod(torch.nn.utils.clip_grad_norm_)
        actual, expected = Model(), Model()
        expected.load_state_dict(actual.state_dict())
        opts = [torch.optim.AdamW(m.parameters(), lr=joint.LR, betas=(.9, .95), weight_decay=.01)
                for m in (actual, expected)]
        rows = [{'id': str(i)} for i in range(64)]
        groups = joint.training_plan(rows, True)[2]
        seen = []
        def accumulate(model, group, stage, backward, tokenizer, link_steps=3):
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            initial = model.hybridlink.weight.detach().clone()
            for row in group:
                seen.append(row['id'])
                x = torch.tensor([[float(int(row['id']) + 1)]])
                backward(model.hybridlink(x).square().mean() / 8)
                self.assertTrue(torch.equal(initial, model.hybridlink.weight))
            return []
        with patch.object(joint, 'accumulate', side_effect=accumulate):
            for step, group in enumerate(groups, 1):
                _, report = joint.train_group(actual, group, None, Accelerator(), opts[0],
                                               None, lambda: None, 1.)
                opts[1].zero_grad(set_to_none=True)
                x = torch.tensor([[float(int(r['id']) + 1)] for r in group])
                expected.hybridlink(x).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.)
                opts[1].step()
                torch.testing.assert_close(actual.hybridlink.weight, expected.hybridlink.weight)
                self.assertTrue(all(p.grad is None for p in actual.parameters()))
                self.assertEqual(opts[0].state[actual.hybridlink.weight]['step'].item(), step)
                self.assertEqual(report['learning_rate'], joint.LR)
        self.assertEqual(seen, [r['id'] for r in rows])

    def test_six_input_group_routes_length_clips_once_and_steps_once(self):
        import torch
        from unittest.mock import Mock
        model = torch.nn.Module()
        model.hybridlink = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=joint.LR)
        accelerator = Mock()
        accelerator.unwrap_model.return_value = model
        accelerator.clip_grad_norm_.side_effect = torch.nn.utils.clip_grad_norm_
        frozen = Mock()
        def accumulate(model, batches, stage, backward, tokenizer, link_steps):
            self.assertEqual(link_steps, 6)
            self.assertEqual(len(batches), 8)
            self.assertIsNone(model.hybridlink.weight.grad)
            self.assertEqual(len(optimizer.state), 0)  # no early update
            model.hybridlink.weight.grad = torch.ones_like(model.hybridlink.weight)
            return [{'mean_ce': 1.}] * 8
        with patch.object(joint, 'accumulate', side_effect=accumulate) as accumulation:
            metrics, report = joint.train_group(model, [{}] * 8, None, accelerator, optimizer,
                                                None, frozen, 1., link_steps=6)
        accumulation.assert_called_once()
        accelerator.clip_grad_norm_.assert_called_once()
        self.assertEqual(frozen.call_count, 2)
        self.assertEqual(optimizer.state[model.hybridlink.weight]['step'].item(), 1)
        self.assertIsNone(model.hybridlink.weight.grad)
        self.assertEqual(len(metrics), 8)
        self.assertEqual(report['learning_rate'], joint.LR)

    def test_bad_gradients_stop_before_optimizer_step(self):
        import torch
        from unittest.mock import Mock
        model = torch.nn.Module()
        model.hybridlink = torch.nn.Linear(1, 1)
        accelerator = Mock()
        accelerator.unwrap_model.return_value = model
        optimizer = Mock()
        with patch.object(joint, 'accumulate', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'gradient'):
                joint.train_group(model, [{}] * 8, None, accelerator, optimizer, None, lambda: None, 1.)
        optimizer.step.assert_not_called()

    def test_eight_three_loss_means_match_one_24_loss_mean(self):
        self.check_average(3)

    def test_eight_six_loss_means_match_one_48_loss_mean(self):
        self.check_average(6)

    def check_average(self, steps):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.link = torch.nn.Linear(2, steps, bias=False)
            def forward(self, x, stage, link_steps):
                self.assert_steps = link_steps
                v = self.link(x)
                losses = v.square().flatten()
                return {'loss': losses.mean(), 'step_losses': losses,
                        'rollout_vectors': tuple(v[:, i:i+1] for i in range(steps)),
                        'target_ids': torch.tensor([list(range(1, steps + 1))]),
                        'predicted_ids': torch.tensor([list(range(1, steps + 1))]),
                        'target_positions': list(range(4, 4 + steps)), 'substitution_positions': list(range(3, 3 + steps))}
        class Tokenizer:
            def decode(self, ids):
                return str(ids[0])
        torch.manual_seed(29)
        actual, expected = Model(), Model()
        expected.load_state_dict(actual.state_dict())
        initial = actual.link.weight.detach().clone()
        batches = [{'x': torch.tensor([[float(i), 1.]])} for i in range(8)]
        calls = []
        def backward(loss):
            self.assertTrue(torch.equal(initial, actual.link.weight))
            calls.append(loss.item())
            loss.backward()
        metrics = joint.accumulate(actual, batches, 'unused', backward, Tokenizer(), link_steps=steps)
        all_losses = torch.cat([expected(**b, stage='unused', link_steps=steps)['step_losses'] for b in batches])
        all_losses.mean().backward()
        torch.testing.assert_close(actual.link.weight.grad, expected.link.weight.grad)
        self.assertEqual(actual.assert_steps, steps)
        self.assertEqual(len(calls), 8)
        self.assertAlmostEqual(sum(calls), sum(m['mean_ce'] for m in metrics) / 8, places=5)
        for model in (actual, expected):
            opt = torch.optim.AdamW(model.parameters(), lr=joint.LR, betas=(.9, .95), weight_decay=0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        torch.testing.assert_close(actual.link.weight, expected.link.weight)
        self.assertFalse(torch.equal(initial, actual.link.weight))
        with self.assertRaises(ValueError):
            joint.accumulate(actual, batches[:7], 'unused', backward, Tokenizer(), link_steps=steps)
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            joint.accumulate(actual, [{'x': torch.full((1, 2), float('nan'))}] * 8,
                             'unused', backward, Tokenizer(), link_steps=steps)


if __name__ == '__main__':
    unittest.main()
