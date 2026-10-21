"""Interrupted-run evaluation must never repeat the optimizer update."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lora_self_conditioned_evaluate', ROOT / 'scripts/lora_self_conditioned_evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft'))


class EvaluationPlanTests(unittest.TestCase):
    def test_no_backward_optimizer_export_or_resume_and_controls_first(self):
        script = ROOT / 'scripts/lora_self_conditioned_evaluate.py'
        tree = ast.parse(script.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                 for n in calls if isinstance(n.func, (ast.Attribute, ast.Name))]
        for forbidden in ('backward', 'AdamW', 'step', 'save_pretrained', 'train_once'):
            self.assertNotIn(forbidden, names)
        control = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and
                       n.func.id == 'measure_all' and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'control')
        install = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'install_saved')
        self.assertLess(control, install)
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--joint-run', result.stdout)
        for option in ('--resume', '--lr', '--max-steps'):
            self.assertNotIn(option, result.stdout)

    def test_plan_requires_one_update_and_exact_ordered_controls(self):
        update = [{'id': str(i)} for i in range(8)]
        observe = [{'id': str(i)} for i in range(8, 16)]
        before = [{'id': str(i), 'two_pass': {'loss': float(i)}} for i in range(16)]
        run = {'code_commit': evaluation.FAILED_COMMIT, 'data_sha256': evaluation.TRAIN_SHA256,
               'planned_optimizer_steps': 1, 'gradient_average_count': 8,
               'learning_rate_lora': evaluation.LR, 'learning_rate_link': evaluation.LR,
               'betas': [.9, .95], 'weight_decay': 0., 'max_grad_norm': 1., 'shuffle': False,
               'scheduler': None, 'anchors_frozen': True, 'hybrid_limits': evaluation.HYBRID_LIMITS,
               'text_budgets': {'plain': 32, 'format': 256}, 'recursive_steps': 6, 'seed': 6064,
               'update_ids': [r['id'] for r in update], 'observation_ids': [r['id'] for r in observe]}
        report = {'optimizer_steps': 1, 'pre_update_losses': [r['two_pass'] for r in before[:8]]}
        evaluation.verify_plan(run, report, before, update, observe)
        for key, value in (('gradient_average_count', 16), ('learning_rate_link', 1e-4),
                           ('code_commit', 'other'), ('recursive_steps', 3)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.verify_plan({**run, key: value}, report, before, update, observe)
        with self.assertRaises(ValueError):
            evaluation.verify_plan(run, {**report, 'optimizer_steps': 2}, before, update, observe)
        with self.assertRaises(ValueError):
            evaluation.verify_plan(run, report, before[::-1], update, observe)
        changed = copy.deepcopy(before)
        changed[0]['two_pass']['loss'] += .1
        with self.assertRaises(ValueError):
            evaluation.verify_plan(run, report, changed, update, observe)

    def test_replay_rejects_any_metric_or_generated_output_change(self):
        expected = {'id': 'a', 'two_pass': {'loss': .5}, 'hybrid': {'scored_text': '42'}}
        evaluation.require_replay(copy.deepcopy(expected), expected, 'test')
        for changed in ({**expected, 'id': 'b'}, {**expected, 'two_pass': {'loss': .50000001}},
                        {**expected, 'hybrid': {'scored_text': '43'}}):
            with self.assertRaisesRegex(RuntimeError, 'exact replay mismatch'):
                evaluation.require_replay(changed, expected, 'test')


@unittest.skipUnless(AVAILABLE, 'requires torch, transformers and peft')
class SavedAdapterTests(unittest.TestCase):
    def test_install_preserves_original_and_anchors_and_replays_saved_model(self):
        import torch
        import test_lora_self_conditioned as fixtures
        from hybrid_sft.model import SELF_CONDITIONED
        from lora_self_conditioned_joint import scalar_losses, reload_adapters
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file, save_file
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = fixtures.SelfConditionedRuntimeTests().make_model(root)
            source.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
            trained = copy.deepcopy(source)
            with torch.no_grad():
                for p in trained.parameters():
                    if p.requires_grad:
                        p.add_(.001)
            checkpoint = root / 'updated'
            trained.save_pretrained(checkpoint, fixtures.Tokenizer())
            frozen = {n: p.detach().clone() for n, p in source.named_parameters() if not p.requires_grad}
            ids = torch.tensor([[1, 32, 5, 7, 33, 6, 2]])
            batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                     'latent_mask': torch.tensor([[0, 1, 1, 1, 0, 0, 0]]),
                     'loss_mask': torch.tensor([[0, 1, 0, 0, 2, 1, 1]])}
            trained.eval()
            with torch.no_grad():
                expected = scalar_losses(trained(**batch, stage=SELF_CONDITIONED))
            changed = evaluation.install_saved(source, checkpoint)
            self.assertTrue(changed['lora'])
            self.assertTrue(changed['link'])
            source.eval()
            with torch.no_grad():
                self.assertEqual(scalar_losses(source(**batch, stage=SELF_CONDITIONED)), expected)
            reload_adapters(source, checkpoint)
            for n, p in source.named_parameters():
                self.assertIsNone(p.grad, n)
                if n in frozen:
                    torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
            for k, v in get_peft_model_state_dict(trained.backbone, save_embedding_layers=False).items():
                torch.testing.assert_close(get_peft_model_state_dict(source.backbone, save_embedding_layers=False)[k],
                                           v, rtol=0, atol=0)
            # A modified frozen anchor must fail before installing any weights.
            path = checkpoint / 'lora/adapter_model.safetensors'
            state = load_file(str(path))
            anchor = next(k for k in state if 'trainable_tokens_delta' in k)
            state[anchor].add_(1)
            save_file(state, str(path))
            with patch('peft.set_peft_model_state_dict') as install:
                with self.assertRaisesRegex(ValueError, 'anchor'):
                    evaluation.install_saved(source, checkpoint)
                install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
