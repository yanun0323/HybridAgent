"""Cached three-input execution: segmentation, target alignment, no content leakage."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('link_execution_probe', ROOT / 'scripts/link_execution_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ExecutionPlanTests(unittest.TestCase):
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
    def test_prompt_then_assistant_then_three_vectors_and_fresh_cache(self):
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
        sample = {'input_ids': [2, 3, 9, 6, 2, 4, 5, 7, 8, 3, 1], 'prompt_len': 3}
        runs = []
        for changed in (False, True):
            calls, cache = [], None
            current = {**sample, 'input_ids': sample['input_ids'].copy()}
            if changed:
                current['input_ids'][5:9] = [11, 12, 13, 14]
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
                out = probe.cached_rollout(Model(), current, 2, 3)
                self.assertEqual(len(calls), 5)  # plan, entry, three predictions
                for i, token in enumerate((6, 2)):
                    torch.testing.assert_close(calls[i], embedding(torch.tensor([[token]])))
                for i in range(3):
                    torch.testing.assert_close(calls[i + 2], out['rollout_vectors'][i])
                self.assertEqual(out['target_positions'], [6, 7, 8])
                self.assertEqual(out['substitution_positions'], [5, 6, 7])
                self.assertEqual(out['target_ids'].tolist(), [current['input_ids'][6:9]])
                torch.testing.assert_close(out['loss'], out['step_losses'].mean())
                metrics, vectors = probe.measurements(out, Tokenizer())
                self.assertEqual(len(metrics['predicted_ids']), 3)
                self.assertEqual(metrics['target_probabilities'], [__import__('math').exp(-c) for c in metrics['step_ce']])
                report = probe.compare_paths(metrics, metrics, vectors, vectors)
                self.assertEqual(report['delta_ce_b_minus_a'], [0., 0., 0.])
                self.assertEqual(report['argmax_matches'], [True] * 3)
                self.assertEqual(report['vector_max_abs_differences'], [0.] * 3)
                with self.assertRaises(ValueError):
                    probe.compare_paths(metrics, {**metrics, 'target_positions': [0, 1, 2]}, vectors, vectors)
                runs.append(out)
        # Reference content is used only as CE labels, never as decoder inputs.
        self.assertEqual(runs[0]['predicted_ids'].tolist(), runs[1]['predicted_ids'].tolist())
        for a, b in zip(runs[0]['rollout_vectors'], runs[1]['rollout_vectors']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
