"""Fixed saved-weight swaps: no optimizer, exact controls, protected anchors."""
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
import self_conditioned_swap as swap
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft'))


def records():
    six = dict(step_ce=[1.] * 6, target_ids=list(range(6)), predicted_ids=list(range(6)),
               target_positions=list(range(10, 16)), substitution_positions=list(range(9, 15)))
    generation = dict(answer_correct=True, format_valid=True, success=True, truncated=False,
                      natural_exit=False, forced_exit=True, used_vectors=True, terminated=True,
                      scored_text='42')
    from lora_self_conditioned_joint import LOSS_KEYS
    return [{'id': str(i), 'two_pass': dict.fromkeys(LOSS_KEYS, 1.),
             'full': copy.deepcopy(six), 'cached': copy.deepcopy(six),
             **{k: copy.deepcopy(generation) for k in ('plain', 'format', 'hybrid')}} for i in range(16)]


class SwapPlanTests(unittest.TestCase):
    def test_cli_no_training_export_or_second_base_model(self):
        script = ROOT / 'scripts/self_conditioned_swap.py'
        tree = ast.parse(script.read_text())
        names = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, (ast.Attribute, ast.Name))]
        for name in ('backward', 'AdamW', 'step', 'save_pretrained', 'train_once'):
            self.assertNotIn(name, names)
        bases = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'HybridSFTModel'
                 and n.func.attr == 'from_pretrained']
        self.assertEqual(len(bases), 1)
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--control-run', result.stdout)
        for arg in ('--resume', '--lr', '--max-steps', '--combinations'):
            self.assertNotIn(arg, result.stdout)

    def test_controls_first_and_failure_never_reaches_new_combinations(self):
        controls = {k: [{'id': k}] for k in ('source', 'B', 'C')}
        calls = []
        def install(label):
            calls.append(('install', label))
        def measure(label, expected):
            calls.append(('measure', label))
            return copy.deepcopy(expected) if expected is not None else [{'id': label}]
        results = swap.run_pairs(install, measure, controls)
        self.assertEqual(list(results), ['source', 'B', 'C', 'A', 'D'])
        self.assertEqual(calls, [(action, label) for label in ('source', 'B', 'C', 'A')
                                for action in ('install', 'measure')] +
                         [('install', 'A'), ('measure', 'A_reloaded'), ('install', 'D'),
                          ('measure', 'D'), ('install', 'D'), ('measure', 'D_reloaded')])
        for bad in ('source', 'B', 'C', 'A_reloaded', 'D_reloaded'):
            calls.clear()
            def fail(label, expected):
                value = measure(label, expected)
                return [{'id': 'wrong'}] if label == bad else value
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, 'exact replay mismatch'):
                swap.run_pairs(install, fail, controls)
            if bad in controls:
                self.assertNotIn(('install', 'A'), calls)
            if bad == 'A_reloaded':
                self.assertNotIn(('install', 'D'), calls)

    def test_comparisons_keep_labels_rows_and_position_regressions(self):
        rows = [{'id': str(i), 'question': 'q'} for i in range(16)]
        results = {k: records() for k in swap.PAIRS}
        results['D'][0]['cached']['step_ce'][-1] = 8.
        results['D'][0]['cached']['predicted_ids'][-1] = 99
        pairs = swap.comparisons(rows[:8], rows[8:], results)
        self.assertEqual({k: (v['before_label'], v['after_label']) for k, v in pairs.items()}, {
            'lora_with_joint_link': ('A', 'C'), 'lora_with_control_link': ('B', 'D'),
            'link_with_original_lora': ('A', 'B'), 'link_with_joint_lora': ('C', 'D')})
        report = pairs['lora_with_control_link']['update']
        self.assertEqual(report['cached']['steps'][5]['higher_ce'], 1)
        self.assertEqual(report['cached']['all_six_correct_after'], 7)
        self.assertEqual(report['rows'][0]['after'], results['D'][0])

    def test_control_provenance_reports_order_and_saved_replay(self):
        rows = [{'id': str(i), 'question': 'q'} for i in range(16)]
        before, after = records(), records()
        after[0]['two_pass']['loss'] = .8
        keys = ('data_sha256', 'source_hashes', 'update_ids', 'observation_ids', 'tokenization_sha256',
                'calibration', 'seed', 'planned_optimizer_steps', 'gradient_average_count',
                'learning_rate_link', 'betas', 'weight_decay', 'max_grad_norm', 'shuffle', 'scheduler',
                'anchors_frozen', 'hybrid_limits', 'text_budgets', 'recursive_steps', 'precision', *swap.VERSIONS)
        original = dict.fromkeys(keys, 'same')
        original['source_hashes'] = {'adapter.pt': 'source'}
        reference = {'run': original, 'before': before, 'after': records()}
        report = {'optimizer_steps': 1, 'lora_gradient_norms': {},
                  'pre_update_losses': [r['two_pass'] for r in before[:8]]}
        summary = {'optimizer_steps': 1, 'link_only': True, 'learning_rate_lora': None,
                   'learning_rate_link': swap.LR, 'trainable_parameters': 52459520,
                   'saved_link_sha256': swap.CONTROL_LINK_SHA256, 'saved_lora_sha256': swap.SOURCE_LORA_SHA256,
                   **dict.fromkeys(('original_16_controls_exact', 'frozen_lora_values_unchanged',
                       'protected_files_unchanged', 'anchors_values_unchanged',
                       'frozen_original_weights_no_grad_versions_unchanged',
                       'saved_lora_link_replay_exact', 'source_hashes_unchanged'), True),
                   'joint_vs_link_only': {}}
        for group, sl in (('update', slice(0, 8)), ('observation', slice(8, 16))):
            summary[group] = swap.compare_records(rows[sl], before[sl], after[sl])
            summary['joint_vs_link_only'][group] = swap.compare_records(rows[sl], reference['after'][sl], after[sl])
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            control, joint, evaluation = [root / k for k in ('control', 'joint', 'evaluation')]
            for p in (control, joint, evaluation):
                p.mkdir()
            run = {**original, 'code_commit': swap.CONTROL_COMMIT, 'link_only': True, 'learning_rate_lora': None,
                   'joint_run': str(joint), 'joint_evaluation': str(evaluation),
                   'protected_run_hashes': {str(joint): {}, str(evaluation): {}}}
            documents = {'run.json': run, 'update.json': report, 'summary.json': summary}
            for name, value in documents.items():
                (control / name).write_text(json.dumps(value))
            for name, value in (('before', before), ('after', after), ('reloaded', after)):
                (control / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in value))
            def verify():
                return swap.verify_control(control, reference, original['source_hashes'], rows[:8], rows[8:], joint, evaluation)
            self.assertEqual(verify(), after)
            for name, key, value in (('run.json', 'precision', 'changed'), ('run.json', 'code_commit', 'changed'),
                                     ('summary.json', 'saved_lora_link_replay_exact', False),
                                     ('update.json', 'optimizer_steps', 2)):
                path = control / name
                path.write_text(json.dumps({**documents[name], key: value}))
                with self.assertRaises(ValueError):
                    verify()
                path.write_text(json.dumps(documents[name]))
            (evaluation / 'extra').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'reference files'):
                verify()
            (evaluation / 'extra').unlink()
            path = control / 'reloaded.jsonl'
            path.write_text(path.read_text().replace('42', '43'))
            with self.assertRaisesRegex(RuntimeError, 'exact replay mismatch'):
                verify()


@unittest.skipUnless(AVAILABLE, 'requires torch, transformers and peft')
class SwapRuntimeTests(unittest.TestCase):
    def test_all_pairs_restore_disk_values_preserve_base_and_repeat_forward(self):
        import torch
        import test_lora_self_conditioned as fixtures
        from hybrid_sft.model import SELF_CONDITIONED
        from lora_self_conditioned_joint import scalar_losses
        from peft import get_peft_model_state_dict
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = fixtures.SelfConditionedRuntimeTests().make_model(root)
            model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
            snapshots = {}
            for label, lora_delta, link_delta in (('source', 0., 0.), ('joint', .003, .002), ('control', 0., -.001)):
                candidate = copy.deepcopy(model)
                with torch.no_grad():
                    for n, p in candidate.backbone.named_parameters():
                        if '.lora_A.' in n or '.lora_B.' in n:
                            p.add_(lora_delta)
                    for p in candidate.hybridlink.parameters():
                        p.add_(link_delta)
                candidate.save_pretrained(root / label, fixtures.Tokenizer())
                snapshots[label] = candidate
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            base_versions = {n: p._version for n, p in model.named_parameters()
                             if n in frozen and 'trainable_tokens_delta' not in n}
            ids = torch.tensor([[1, 32, 5, 7, 33, 6, 2]])
            batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                         latent_mask=torch.tensor([[0, 1, 1, 1, 0, 0, 0]]),
                         loss_mask=torch.tensor([[0, 1, 0, 0, 2, 1, 1]]))
            for label in ('source', 'B', 'C', 'A', 'D', 'A', 'D'):
                lora, link = swap.PAIRS[label]
                lp, hp = root / lora / 'lora/adapter_model.safetensors', root / link / 'adapter.pt'
                expected_model = copy.deepcopy(snapshots[lora])
                expected_model.hybridlink.load_state_dict(snapshots[link].hybridlink.state_dict())
                expected_model.train()
                expected = scalar_losses(expected_model(**batch, stage=SELF_CONDITIONED))
                for _ in range(2):
                    swap.install_pair(model, lp, hp, swap.sha256(lp), swap.sha256(hp))
                    model.train()
                    self.assertEqual(scalar_losses(model(**batch, stage=SELF_CONDITIONED)), expected)
                    for n, p in model.named_parameters():
                        self.assertIsNone(p.grad)
                        if n in frozen:
                            torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
                        if n in base_versions:
                            self.assertEqual(p._version, base_versions[n])
                    live = get_peft_model_state_dict(model.backbone, save_embedding_layers=False)
                    target = get_peft_model_state_dict(snapshots[lora].backbone, save_embedding_layers=False)
                    for k in live:
                        torch.testing.assert_close(live[k], target[k], rtol=0, atol=0)

    def test_invalid_hash_anchor_shape_dtype_or_nan_fails_before_any_mutation(self):
        import torch
        import test_lora_self_conditioned as fixtures
        from hybrid_sft.model import SELF_CONDITIONED
        from safetensors.torch import load_file, save_file
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = fixtures.SelfConditionedRuntimeTests().make_model(root)
            model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
            model.save_pretrained(root / 'saved', fixtures.Tokenizer())
            lp, hp = root / 'saved/lora/adapter_model.safetensors', root / 'saved/adapter.pt'
            lora, link = load_file(str(lp)), torch.load(hp, weights_only=True)
            values = {n: p.detach().clone() for n, p in model.named_parameters()}
            versions = {n: p._version for n, p in model.named_parameters()}
            for fault in ('hash', 'anchor', 'shape', 'dtype', 'nan', 'keys', 'gradient'):
                a, b = copy.deepcopy(lora), copy.deepcopy(link)
                if fault == 'anchor':
                    a[next(k for k in a if 'trainable_tokens_delta' in k)].add_(1)
                key = next(iter(b))
                if fault == 'shape':
                    b[key] = b[key].reshape(-1)[:1]
                if fault == 'dtype':
                    b[key] = b[key].double()
                if fault == 'nan':
                    b[key].fill_(float('nan'))
                if fault == 'keys':
                    del b[key]
                if fault == 'gradient':
                    next(model.hybridlink.parameters()).grad = torch.ones_like(next(model.hybridlink.parameters()))
                save_file(a, str(lp))
                torch.save(b, hp)
                with self.subTest(fault=fault), patch('peft.set_peft_model_state_dict') as install:
                    with self.assertRaises((ValueError, RuntimeError)):
                        swap.install_pair(model, lp, hp, 'wrong' if fault == 'hash' else swap.sha256(lp), swap.sha256(hp))
                    install.assert_not_called()
                model.zero_grad(set_to_none=True)
                for n, p in model.named_parameters():
                    self.assertEqual(p._version, versions[n], n)
                    torch.testing.assert_close(p, values[n], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
