"""Fixed 48-record extension and strict replay before new measurements."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('link_three_step_remaining', ROOT / 'scripts/link_three_step_remaining.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class RemainingTests(unittest.TestCase):
    def inputs(self):
        rows = [{'id': str(i)} for i in range(64)]
        settings = {'link_steps': 3, 'gradient_average_count': 8, 'learning_rate': 5e-6}
        run = {**settings, 'update_ids': [r['id'] for r in rows[:8]],
               'observation_ids': [r['id'] for r in rows[8:16]]}
        summary = {**settings, 'optimizer_steps': 1, 'saved_link_replay_exact': True}
        for group, selected in (('update', rows[:8]), ('observation', rows[8:16])):
            summary[group] = {path: {'rows': [{**r, 'before': {'step_ce': [1., 2., 3.]},
                                                       'after': {'step_ce': [0., 1., 2.]}} for r in selected]}
                              for path in ('full', 'cached')}
        return rows, run, summary

    def test_selection_and_both_state_path_references(self):
        rows, run, summary = self.inputs()
        remaining, refs = probe.fixed_plan(rows, run, summary)
        self.assertEqual(remaining, rows[16:])
        self.assertFalse(set(r['id'] for r in remaining) & set(run['update_ids'] + run['observation_ids']))
        self.assertEqual(set(refs), {'before', 'after'})
        for state in refs:
            self.assertEqual(set(refs[state]), {'full', 'cached'})
            for path in refs[state]:
                self.assertEqual(len(refs[state][path]), 16)
        for bad in (rows[:63], rows[:63] + [rows[0]]):
            with self.assertRaises(ValueError):
                probe.fixed_plan(bad, run, summary)
        with self.assertRaisesRegex(ValueError, 'selection'):
            probe.fixed_plan(rows, {**run, 'update_ids': []}, summary)
        for key, value in (('link_steps', 1), ('optimizer_steps', 2), ('saved_link_replay_exact', False),
                           ('learning_rate', 5e-5)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.fixed_plan(rows, run, {**summary, key: value})
        bad = copy.deepcopy(summary)
        bad['observation']['cached']['rows'][0]['id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'control IDs'):
            probe.fixed_plan(rows, run, bad)

    def test_exact_control_checks_ce_ids_and_positions(self):
        expected = {'step_ce': [1., 2., 3.], 'predicted_ids': [4, 5, 6],
                    'target_positions': [12, 13, 14]}
        probe.verify_control(copy.deepcopy(expected), expected, 'test')
        for key in expected:
            changed = copy.deepcopy(expected)
            changed[key][0] += .000001
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'control replay mismatch'):
                probe.verify_control(changed, expected, 'test')

    def test_no_backward_optimizer_or_model_export(self):
        tree = ast.parse((ROOT / 'scripts/link_three_step_remaining.py').read_text())
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        for forbidden in ('backward', 'step', 'AdamW', 'save_pretrained'):
            self.assertNotIn(forbidden, calls)


if __name__ == '__main__':
    unittest.main()
