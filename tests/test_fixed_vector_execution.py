"""Fixed-vector diagnostic: strict controls, no reference leakage, fresh caches."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fixed_vector_execution as probe
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft'))


class FixedPlanTests(unittest.TestCase):
    def test_cli_and_no_training_or_export(self):
        script = ROOT / 'scripts/fixed_vector_execution.py'
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--swap-run', result.stdout)
        tree = ast.parse(script.read_text())
        names = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, (ast.Attribute, ast.Name))]
        for forbidden in ('backward', 'AdamW', 'step', 'save_pretrained', 'train_once'):
            self.assertNotIn(forbidden, names)
        fixed = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fixed_rollout')
        self.assertNotIn('run_hybridlink', ast.unparse(fixed))
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'HybridSFTModel'
                             and n.func.attr == 'from_pretrained' for n in ast.walk(tree)), 1)

    def test_all_controls_and_mismatch_stop(self):
        rows = [{'id': str(i)} for i in range(16)]
        expected = [dict(r, full={'mean_ce': 1.}, cached={'mean_ce': 2.}) for r in rows]
        seen = []
        def measure(row):
            r = expected[int(row['id'])]
            return {k: copy.deepcopy(r[k]) for k in ('full', 'cached')}, {'full': [], 'cached': []}
        results = probe.collect_controls(rows, expected, measure, seen.append)
        self.assertEqual(len(results), 16)
        self.assertEqual(seen, expected)
        for index in (0, 15):
            seen.clear()
            bad = copy.deepcopy(expected)
            bad[index]['cached']['mean_ce'] += 1e-12
            with self.assertRaisesRegex(RuntimeError, 'exact replay mismatch'):
                probe.collect_controls(rows, bad, measure, seen.append)
            self.assertEqual(len(seen), index)
        with self.assertRaises(ValueError):
            probe.collect_controls(rows[:15], expected[:15], measure, seen.append)

    def test_swap_provenance_and_reload_guards(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            rows = [{'id': str(i), 'full': {}, 'cached': {}} for i in range(16)]
            plan = {**dict.fromkeys(probe.VERSIONS, 'v'), 'tokenization_sha256': ['hash'] * 16,
                    'calibration': {'scale': 1.}}
            pairs = {k: list(v) for k, v in probe.PAIRS.items()}
            run = {'code_commit': probe.SWAP_COMMIT, 'pairs': pairs, 'original_plan': plan,
                   'protected_hashes': {str(path / 'source'): {'file': 'hash'}},
                   'tokenization_sha256': plan['tokenization_sha256'], 'calibration': plan['calibration'],
                   'data_sha256': probe.TRAIN_SHA256, 'optimizer_steps': 0, 'weights_saved': False,
                   **dict.fromkeys(probe.VERSIONS, 'v')}
            run.update(sequence=['source', 'B', 'C', 'A', 'A_reloaded', 'D', 'D_reloaded'],
                       lora_sha256={'source': probe.SOURCE_LORA_SHA256, 'joint': probe.SAVED_LORA_SHA256,
                                    'control': probe.SOURCE_LORA_SHA256},
                       link_sha256={'source': probe.SOURCE_LINK_SHA256, 'joint': probe.SAVED_LINK_SHA256,
                                    'control': probe.CONTROL_LINK_SHA256})
            summary = {'pairs': pairs, 'optimizer_steps': 0, 'weights_saved': False,
                       **dict.fromkeys(('original_and_B_C_controls_exact', 'A_D_reload_exact',
                       'anchors_values_unchanged', 'original_backbone_versions_no_grad_unchanged',
                       'protected_files_unchanged', 'text_only_same_lora_exact'), True)}
            def write(name, value):
                (path / name).write_text(json.dumps(value))
            write('run.json', run)
            write('summary.json', summary)
            for label in ('source', 'B', 'C', 'A', 'A_reloaded', 'D', 'D_reloaded'):
                (path / f'{label}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            args = (path, {path / 'source': {'file': 'hash'}}, plan, [r['id'] for r in rows],
                    {'before': rows, 'after': rows}, rows)
            self.assertEqual(probe.verify_swap(*args), rows)
            for field in ('code_commit', 'data_sha256', 'tokenization_sha256', 'protected_hashes',
                          'sequence', 'lora_sha256', 'link_sha256'):
                bad = copy.deepcopy(run)
                bad[field] = 'wrong'
                write('run.json', bad)
                with self.assertRaises(RuntimeError):
                    probe.verify_swap(*args)
            write('run.json', run)
            write('summary.json', {**summary, 'A_D_reload_exact': False})
            with self.assertRaises(ValueError):
                probe.verify_swap(*args)
            write('summary.json', summary)
            (path / 'A_reloaded.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[::-1]))
            with self.assertRaises(RuntimeError):
                probe.verify_swap(*args)


@unittest.skipUnless(AVAILABLE, 'requires torch, transformers and peft')
class FixedRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hybrid_sft.model import HybridSFTModel
        torch.manual_seed(3)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        Qwen3ForCausalLM(config).save_pretrained(self.tmp.name)
        self.model = HybridSFTModel.from_pretrained(self.tmp.name, dtype=torch.float32)
        self.model.requires_grad_(False).eval()
        self.sample = {'input_ids': [1, 2, 3, 30, 5, 6, 7, 8, 9, 10, 11, 31, 2], 'prompt_len': 2}
        self.vectors = [torch.randn(1, 1, 16) for _ in range(6)]
        class Tokenizer:
            def decode(self, ids):
                return str(ids[0])
        self.tokenizer = Tokenizer()

    def test_exact_self_replay_of_both_recursive_paths_and_repeat(self):
        import torch
        from hybrid_sft.model import LINK_NEXT_TOKEN
        ids = torch.tensor([self.sample['input_ids']])
        mask = torch.tensor([[0, 0, 0] + [1] * 8 + [0, 0]])
        with torch.no_grad():
            full = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), latent_mask=mask,
                              stage=LINK_NEXT_TOKEN, link_steps=6)
            cached = probe.cached_rollout(self.model, self.sample, 30, 31, link_steps=6)
            for path, recursive in (('full', full), ('cached', cached)):
                metrics, vectors = probe.measurements(recursive, self.tokenizer)
                fixed = probe.fixed_rollout(self.model, self.sample, 30, 31, vectors, path)
                self.assertEqual(probe.measurements(fixed, self.tokenizer)[0], metrics)
                again = probe.fixed_rollout(self.model, self.sample, 30, 31, vectors, path)
                probe.require_fixed_repeat(fixed, again, self.tokenizer)
                again['hidden'][3][0, 0, 0] += .001
                with self.assertRaisesRegex(RuntimeError, 'hidden'):
                    probe.require_fixed_repeat(fixed, again, self.tokenizer)

    def test_no_link_calls_no_teacher_content_and_correct_segmentation(self):
        import torch
        from hybrid_agent.compat import text_decoder
        decoder = text_decoder(self.model.backbone)
        forward = decoder.forward
        original = [v.clone() for v in self.vectors]
        for path in ('full', 'cached'):
            runs = []
            for changed in (False, True):
                sample = copy.deepcopy(self.sample)
                if changed:
                    sample['input_ids'][4:11] = [20] * 7
                calls = []
                def checked(**kwargs):
                    calls.append(kwargs)
                    return forward(**kwargs)
                with patch.object(decoder, 'forward', side_effect=checked), patch.object(
                        self.model, 'run_hybridlink', side_effect=AssertionError('Link called')), torch.no_grad():
                    out = probe.fixed_rollout(self.model, sample, 30, 31, self.vectors, path)
                if path == 'full':
                    self.assertEqual([c['inputs_embeds'].shape[1] for c in calls], list(range(4, 11)))
                    for i, call in enumerate(calls[1:]):
                        self.assertFalse(call['use_cache'])
                        self.assertNotIn('past_key_values', call)
                        for k in range(i + 1):
                            self.assertTrue(torch.equal(call['inputs_embeds'][:, 4+k:5+k], original[k]))
                else:
                    self.assertEqual(calls[0]['input_ids'].tolist(), [[1, 2]])
                    self.assertNotIn('past_key_values', calls[0])
                    self.assertEqual([c['position_ids'].item() for c in calls[1:]], list(range(2, 10)))
                    self.assertEqual([c['attention_mask'].shape[1] for c in calls[1:]], list(range(3, 11)))
                    for call, vector in zip(calls[3:], original):
                        self.assertTrue(torch.equal(call['inputs_embeds'], vector))
                self.assertEqual(out['target_positions'], list(range(5, 11)))
                self.assertEqual(out['substitution_positions'], list(range(4, 10)))
                self.assertEqual(out['target_ids'].tolist(), [sample['input_ids'][5:11]])
                runs.append(out)
            for key in ('hidden', 'logits', 'fed_vectors'):
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(runs[0][key], runs[1][key])))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.vectors, original)))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_guards_before_decoder_and_comparison_detects_differences(self):
        import torch
        from hybrid_agent.compat import text_decoder
        decoder = text_decoder(self.model.backbone)
        with patch.object(decoder, 'forward', side_effect=AssertionError('decoder called')):
            with self.assertRaises(ValueError):
                probe.fixed_rollout(self.model, self.sample, 30, 31, self.vectors, 'full')
            with torch.no_grad():
                for vectors in (self.vectors[:5], [v.double() for v in self.vectors],
                                [torch.zeros(1, 2, 16)] * 6, [torch.full((1, 1, 16), float('nan'))] * 6):
                    with self.assertRaises(ValueError):
                        probe.fixed_rollout(self.model, self.sample, 30, 31, vectors, 'cached')
                self.model.train()
                with self.assertRaises(ValueError):
                    probe.fixed_rollout(self.model, self.sample, 30, 31, self.vectors, 'full')
                self.model.eval()
        with torch.no_grad():
            out = probe.fixed_rollout(self.model, self.sample, 30, 31, self.vectors, 'full')
        metrics, _ = probe.measurements(out, self.tokenizer)
        report = probe.compare_fixed(out, out, metrics, metrics)
        self.assertIsNone(report['first_hidden_difference'])
        self.assertIsNone(report['first_logit_difference'])
        malformed = dict(out, hidden=out['hidden'][:6])
        with self.assertRaises(ValueError):
            probe.compare_fixed(out, malformed, metrics, metrics)
        altered = copy.deepcopy(out)
        altered['hidden'][0][0, 0, 0] += .001
        altered['logits'][2][0, 0] += 100.
        report = probe.compare_fixed(out, altered, metrics, metrics)
        self.assertEqual(report['first_hidden_difference'], 0)
        self.assertEqual(report['first_logit_difference'], 2)
        altered['fed_vectors'][0][0, 0, 0] += .001
        with self.assertRaisesRegex(RuntimeError, 'inputs differ'):
            probe.compare_fixed(out, altered, metrics, metrics)


if __name__ == '__main__':
    unittest.main()
