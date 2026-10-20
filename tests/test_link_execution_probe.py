"""Cached three/six-input execution: segmentation, target alignment, no content leakage."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('link_execution_probe', ROOT / 'scripts/link_execution_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ExecutionPlanTests(unittest.TestCase):
    def test_cli_lengths(self):
        import subprocess
        import sys
        script = str(ROOT / 'scripts/link_execution_probe.py')
        result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--link-steps {3,6}', result.stdout)
        result = subprocess.run([sys.executable, script, '--pilot', 'unused', '--model', 'unused',
                                 '--calibration', 'unused', '--output-dir', 'unused', '--link-steps', '5'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid choice', result.stderr)

    def test_six_step_totals_keep_failures_visible(self):
        paths = ('full_grad', 'full_no_grad', 'cached_no_grad')
        a = {'step_ce': [0., 0., 0., 0., 0., 6.], 'target_ids': [1] * 6,
             'predicted_ids': [1, 1, 1, 1, 1, 2]}
        b = {'step_ce': [0.] * 6, 'target_ids': [1] * 6, 'predicted_ids': [1] * 6}
        rows = [{'id': 'a', **{p: a for p in paths}}, {'id': 'b', **{p: b for p in paths}}]
        totals = probe.path_totals(rows, 6)
        for p in paths:
            self.assertEqual(totals[p]['mean_step_ce'], [0., 0., 0., 0., 0., 3.])
            self.assertEqual(totals[p]['correct_by_step'], [2, 2, 2, 2, 2, 1])
            self.assertEqual(totals[p]['all_steps_correct'], 1)
            self.assertEqual(totals[p]['first_mismatch_by_record'],
                             [{'id': 'a', 'step': 6}, {'id': 'b', 'step': None}])
        with self.assertRaises(ValueError):
            probe.path_totals(rows, 3)
        with self.assertRaises(ValueError):
            probe.path_totals([], 6)

    def test_no_backward_optimizer_or_export(self):
        import ast
        tree = ast.parse((ROOT / 'scripts/link_execution_probe.py').read_text())
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        for forbidden in ('backward', 'step', 'save_pretrained', 'AdamW'):
            self.assertNotIn(forbidden, calls)
        rows = [{'id': str(i)} for i in range(64)]
        self.assertEqual(probe.select_record(rows), rows[0])
        with self.assertRaises(ValueError):
            probe.select_record(rows[:8])


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch', 'transformers')), 'requires torch and transformers')
class CachedExecutionTests(unittest.TestCase):
    def test_three_input_segmentation_and_fresh_cache(self):
        self.check_rollout(3)

    def test_six_input_segmentation_and_fresh_cache(self):
        self.check_rollout(6)

    def check_rollout(self, steps):
        import torch
        from unittest.mock import patch
        embedding = torch.nn.Embedding(16, 4)
        head = torch.nn.Linear(4, 16, bias=False)
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = embedding
                self.head = head
            def get_input_embeddings(self):
                return self.embedding
            def get_output_embeddings(self):
                return self.head
        class Model:
            backbone = Backbone()
            def run_hybridlink(self, hidden, output_dtype=None):
                return (hidden + .1).to(output_dtype)
        class Tokenizer:
            def decode(self, ids):
                return str(ids[0])
        sample = {'input_ids': [2, 3, 9, 6, 2] + list(range(4, 4 + steps + 1)) + [3, 1], 'prompt_len': 3}
        runs = []
        for changed in (False, True):
            calls, cache = [], None
            current = {**sample, 'input_ids': sample['input_ids'].copy()}
            if changed:
                current['input_ids'][5:6 + steps] = [14] * (steps + 1)
            def decoder(**kwargs):
                nonlocal cache
                self.assertTrue(kwargs['use_cache'])
                if 'input_ids' in kwargs:
                    self.assertIsNone(cache)
                    self.assertEqual(kwargs['input_ids'].tolist(), [[2, 3, 9]])
                    cache = SimpleNamespace(length=3)
                    cache.get_seq_length = lambda: cache.length
                    hidden = torch.ones(1, 1, 4)
                else:
                    self.assertIs(kwargs['past_key_values'], cache)
                    self.assertEqual(kwargs['position_ids'].item(), cache.length)
                    self.assertEqual(kwargs['attention_mask'].shape[1], cache.length + 1)
                    cache.length += 1
                    calls.append(kwargs['inputs_embeds'].clone())
                    hidden = kwargs['inputs_embeds'] + .2
                return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)
            with patch('hybrid_agent.compat.text_decoder', return_value=decoder), torch.no_grad():
                out = probe.cached_rollout(Model(), current, 2, 3, link_steps=steps)
                self.assertEqual(len(calls), steps + 2)  # plan, entry, predicted inputs
                for i, token in enumerate((6, 2)):
                    torch.testing.assert_close(calls[i], embedding(torch.tensor([[token]])))
                for i in range(steps):
                    torch.testing.assert_close(calls[i + 2], out['rollout_vectors'][i])
                self.assertEqual(out['target_positions'], list(range(6, 6 + steps)))
                self.assertEqual(out['substitution_positions'], list(range(5, 5 + steps)))
                self.assertEqual(out['target_ids'].tolist(), [current['input_ids'][6:6 + steps]])
                torch.testing.assert_close(out['loss'], out['step_losses'].mean())
                metrics, vectors = probe.measurements(out, Tokenizer())
                self.assertEqual(len(metrics['predicted_ids']), steps)
                self.assertAlmostEqual(metrics['mean_ce'], out['step_losses'].double().mean().item())
                self.assertEqual(metrics['target_probabilities'], [__import__('math').exp(-c) for c in metrics['step_ce']])
                report = probe.compare_paths(metrics, metrics, vectors, vectors)
                self.assertEqual(report['delta_ce_b_minus_a'], [0.] * steps)
                self.assertEqual(report['argmax_matches'], [True] * steps)
                self.assertEqual(report['vector_max_abs_differences'], [0.] * steps)
                with self.assertRaises(ValueError):
                    probe.compare_paths(metrics, {**metrics, 'target_positions': [0, 1, 2]}, vectors, vectors)
                with self.assertRaisesRegex(ValueError, 'lengths'):
                    probe.compare_paths(metrics, metrics, vectors[:-1], vectors)
                with self.assertRaisesRegex(ValueError, 'measurements'):
                    probe.measurements({**out, 'step_losses': out['step_losses'][:-1]}, Tokenizer())
                for invalid in (0, 1, 5, 7):
                    with self.assertRaisesRegex(ValueError, 'link_steps'):
                        probe.cached_rollout(Model(), current, 2, 3, link_steps=invalid)
                short = {**sample, 'input_ids': [2, 3, 9, 6, 2] + [4] * (steps - 1) + [3, 1]}
                with self.assertRaisesRegex(ValueError, 'content tokens'):
                    probe.cached_rollout(Model(), short, 2, 3, link_steps=steps)
                runs.append(out)
        # Reference content is used only as CE labels, never as decoder inputs.
        self.assertEqual(runs[0]['predicted_ids'].tolist(), runs[1]['predicted_ids'].tolist())
        for a, b in zip(runs[0]['rollout_vectors'], runs[1]['rollout_vectors']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
