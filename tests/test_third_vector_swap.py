"""History/third-vector interventions: controls first, no continuation or training."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import third_vector_swap as probe
import fixed_vector_execution as fixed
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft'))


def metric(ce=1.):
    return {'mean_ce': ce, 'step_ce': [ce] * 6, 'target_probabilities': [.5] * 6,
            'target_ids': [220] * 6, 'predicted_ids': [220] * 6,
            'target_texts': [' '] * 6, 'predicted_texts': [' '] * 6,
            'substitution_positions': list(range(10, 16)), 'target_positions': list(range(11, 17))}


class ThirdPlanTests(unittest.TestCase):
    def test_prefix_metrics_and_conditional_effects(self):
        m = metric()
        m['step_ce'] = [1., 2., 3., 100., 100., 100.]
        truncated = probe.prefix_metrics(m)
        self.assertEqual(truncated['mean_ce'], 2.)
        self.assertTrue(all(len(v) == 3 for k, v in truncated.items() if k != 'mean_ce'))
        results = {}
        for name, ce in [('full/full', 10.), ('full/cached', 2.),
                         ('cached/full', 3.), ('cached/cached', 1.)]:
            ms = probe.prefix_metrics(metric(ce))
            results[name] = {'metrics': {p: copy.deepcopy(ms) for p in ('full', 'cached')}}
        report = probe.third_effects(results)
        self.assertEqual(report['effects']['third_with_full_history']['paths']['full']['delta_third_ce'], -8.)
        self.assertEqual(report['effects']['history_with_full_third']['paths']['cached']['delta_third_ce'], -7.)
        self.assertEqual(report['third_ce_interaction_cached_minus_full_history'], {'full': 6., 'cached': 6.})

    def test_historical_failures_gate_every_three_input_run(self):
        expected = {'fixed': {d: {'metrics': {p: metric() for p in ('full', 'cached')},
                                  'comparison': {'exact': True}} for d in ('full', 'cached')}}
        for bad_donor, bad_path in [('full', 'full'), ('full', 'cached'), ('cached', 'full'), ('cached', 'cached')]:
            calls = []
            def evaluate(history, third, steps):
                calls.append((history, third, steps))
                metrics = {p: metric() for p in ('full', 'cached')}
                if history == bad_donor:
                    metrics[bad_path]['step_ce'][2] += 1e-12
                return {'full': {}, 'cached': {}}, metrics
            with patch.object(probe, 'compare_fixed', return_value={'exact': True}), self.assertRaises(RuntimeError):
                probe.execute_combinations(evaluate, expected, lambda item: None)
            self.assertTrue(all(c[2] == 6 for c in calls))
        with patch.object(probe, 'compare_fixed', return_value={'exact': False}), self.assertRaises(RuntimeError):
            probe.execute_combinations(lambda *args: ({'full': {}, 'cached': {}},
                                      {p: metric() for p in ('full', 'cached')}), expected, lambda item: None)

    def test_saved_run_provenance_and_vector_hash(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            expected = [{'id': probe.CASE_ID if i == 14 else str(i), 'full': metric(), 'cached': metric()}
                        for i in range(16)]
            plan = {**dict.fromkeys(probe.VERSIONS, 'version'), 'calibration': {'scale': 1.},
                    'seed': 6064, 'tokenization_sha256': ['t'] * 16}
            run = {'code_commit': probe.FIXED_COMMIT, 'script_sha256': probe.FIXED_SCRIPT_SHA256,
                   'pair': 'A', 'optimizer_steps': 0, 'weights_saved': False,
                   'sample_ids': [r['id'] for r in expected], 'protected_hashes': {}, **plan}
            rows, controls = [], []
            for r in expected:
                vector = path / f'vectors-{r["id"]}.pt'
                vector.write_bytes(b'fixture bytes; no torch loading in provenance check')
                controls.append(r)
                rows.append({'id': r['id'], 'recursive': {k: r[k] for k in ('full', 'cached')},
                             'vectors_sha256': probe.sha256(vector)})
            summary = {'count': 16, 'pair': 'A', 'optimizer_steps': 0, 'weights_saved': False,
                       'results': rows, **dict.fromkeys(('historical_32_path_controls_exact',
                       'fixed_self_replay_exact', 'fixed_repeats_hidden_logits_exact', 'same_cast_inputs_exact',
                       'parameters_versions_no_grad_unchanged', 'anchors_values_unchanged',
                       'protected_files_unchanged', 'vectors_saved'), True)}
            (path / 'run.json').write_text(json.dumps(run))
            (path / 'summary.json').write_text(json.dumps(summary))
            for name, data in [('measurements', rows), ('controls', controls)]:
                (path / f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in data))
            result = probe.verify_fixed_run(path, {}, plan, expected)
            self.assertEqual(result['row']['id'], probe.CASE_ID)
            for key in ('code_commit', 'script_sha256', 'seed', 'sample_ids', 'protected_hashes'):
                bad = {**run, key: 'bad'}
                (path / 'run.json').write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError):
                    probe.verify_fixed_run(path, {}, plan, expected)
            (path / 'run.json').write_text(json.dumps(run))
            result['vectors_path'].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                probe.verify_fixed_run(path, {}, plan, expected)


@unittest.skipUnless(AVAILABLE, 'requires torch, transformers and peft')
class ThirdRuntimeTests(unittest.TestCase):
    def test_mixing_uses_only_two_history_vectors_and_one_third(self):
        import torch
        saved = {k: [torch.full((1, 1, 4), float(offset + i)) for i in range(6)]
                 for k, offset in [('full', 0), ('cached', 10)]}
        for history, third in probe.COMBINATIONS:
            values = probe.mix_vectors(saved, history, third)
            self.assertEqual(len(values), 3)
            for a, b in zip(values, [*saved[history][:2], saved[third][2]]):
                self.assertTrue(torch.equal(a, b))
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
        with self.assertRaises(ValueError):
            probe.mix_vectors({'full': saved['full']}, 'full', 'full')

    def test_three_steps_match_six_prefix_and_do_not_run_fourth_input(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hybrid_sft.model import HybridSFTModel
        from hybrid_agent.compat import text_decoder
        torch.manual_seed(7)
        model = HybridSFTModel(Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8))).eval().requires_grad_(False)
        sample = {'input_ids': [1, 2, 30, 4, 5, 6, 7, 8, 9, 10, 31], 'prompt_len': 1}
        saved = {k: [torch.randn(1, 1, 16) for _ in range(6)] for k in ('full', 'cached')}
        class Tokenizer:
            def decode(self, ids): return str(ids[0])
        tok = Tokenizer()
        expected = {'fixed': {}}
        with torch.no_grad():
            for donor in ('full', 'cached'):
                outputs = {p: fixed.fixed_rollout(model, sample, 30, 31, saved[donor], p) for p in ('full', 'cached')}
                ms = {p: fixed.measurements(o, tok)[0] for p, o in outputs.items()}
                expected['fixed'][donor] = {'metrics': ms, 'comparison': fixed.compare_fixed(
                    outputs['full'], outputs['cached'], ms['full'], ms['cached'])}
            calls = []
            def evaluate(history, third, steps):
                calls.append((history, third, steps))
                vectors = saved[history] if steps == 6 else probe.mix_vectors(saved, history, third)
                outputs = {}
                for p in ('full', 'cached'):
                    decoder = text_decoder(model.backbone)
                    with patch.object(decoder, 'forward', wraps=decoder.forward) as forward:
                        out = fixed.fixed_rollout(model, sample, 30, 31, vectors, p, steps=steps)
                    if steps == 3:
                        self.assertEqual(out['target_positions'], [4, 5, 6])
                        self.assertEqual(len(out['hidden']), 4)
                        if p == 'full':
                            self.assertEqual([c.kwargs['inputs_embeds'].shape[1] for c in forward.call_args_list], [3, 4, 5, 6])
                        else:
                            self.assertEqual(forward.call_args_list[-1].kwargs['position_ids'].item(), 5)
                    repeat = fixed.fixed_rollout(model, sample, 30, 31, vectors, p, steps=steps)
                    fixed.require_fixed_repeat(out, repeat, tok)
                    outputs[p] = out
                return outputs, {p: fixed.measurements(o, tok)[0] for p, o in outputs.items()}
            emitted = []
            with patch.object(model, 'run_hybridlink', side_effect=AssertionError('Link called')):
                result = probe.execute_combinations(evaluate, expected, emitted.append)
            self.assertEqual(calls, [('full', 'full', 6), ('cached', 'cached', 6)] +
                             [(h, t, 3) for h, t in probe.COMBINATIONS])
            self.assertEqual(len(result['combinations']), 4)
            self.assertEqual(len(emitted), 6)
            a = {'hidden': [torch.zeros(1)] * 4, 'logits': [torch.zeros(1)] * 4}
            b = copy.deepcopy(a)
            b['hidden'][0] = torch.ones(1)
            with self.assertRaisesRegex(RuntimeError, 'history changed'):
                probe.require_history(a, b)
            for invalid in (0, 2, 4, 7):
                with self.assertRaises(ValueError):
                    fixed.fixed_rollout(model, sample, 30, 31, saved['full'], 'full', steps=invalid)


if __name__ == '__main__':
    unittest.main()
