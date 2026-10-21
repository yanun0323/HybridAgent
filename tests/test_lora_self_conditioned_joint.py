"""Single joint update safety, regression reports and actual PEFT restoration."""
import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lora_self_conditioned_joint', ROOT / 'scripts/lora_self_conditioned_joint.py')
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft'))


class JointPlanTests(unittest.TestCase):
    def test_peft_comparison_uses_same_embedding_policy_as_save(self):
        import ast
        # Dependency-free guard: auto adds full embeddings after vocabulary
        # resizing, but compact checkpoints deliberately exclude those matrices.
        path = ROOT / 'scripts/lora_self_conditioned_joint.py'
        calls = [n for n in ast.walk(ast.parse(path.read_text()))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'get_peft_model_state_dict']
        self.assertEqual(len(calls), 2)
        for call in calls:
            values = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
            self.assertIs(values.get('save_embedding_layers'), False)

    def test_cli_is_single_update_only(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/lora_self_conditioned_joint.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--probe-run', result.stdout)
        for option in ('--all-train', '--max-steps', '--lr', '--resume'):
            self.assertNotIn(option, result.stdout)
        rows = [{'id': str(i)} for i in range(64)]
        update, observe, groups = joint.training_plan(rows)
        self.assertEqual(update, rows[:8])
        self.assertEqual(observe, rows[8:16])
        self.assertEqual(groups, [rows[:8]])

    def test_probe_validation_rejects_changed_sources_or_failed_probe(self):
        hashes = {'adapter.pt': 'source'}
        run = {'source_hashes': hashes, 'data_sha256': joint.TRAIN_SHA256, 'sample_id': 'mul-sub-54-5-6'}
        summary = {'sample_id': run['sample_id'], 'optimizer_steps': 0, 'weights_saved': False,
                   'frozen_parameters_no_grad': True, 'all_parameter_versions_unchanged': True,
                   'lora_link_anchors_values_unchanged': True, 'source_hashes_unchanged': True,
                   'lora_gradient_norms': {'a': 1.}, 'link_gradient_norms': {'b': 1.}}
        joint.verify_probe(run, summary, hashes)
        for key, value in [('optimizer_steps', 1), ('weights_saved', True),
                           ('source_hashes_unchanged', False), ('lora_gradient_norms', {}),
                           ('link_gradient_norms', {'b': float('nan')})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                joint.verify_probe(run, {**summary, key: value}, hashes)
        with self.assertRaises(ValueError):
            joint.verify_probe(run, summary, {'adapter.pt': 'other'})

    def test_means_cannot_hide_late_or_generation_regressions(self):
        six = {'step_ce': [1.] * 6, 'target_ids': list(range(6)), 'predicted_ids': list(range(6)),
               'target_positions': list(range(10, 16)), 'substitution_positions': list(range(9, 15))}
        generation = {'answer_correct': True, 'format_valid': True, 'success': True, 'truncated': False,
                      'natural_exit': True, 'forced_exit': False, 'used_vectors': True, 'terminated': True}
        before = {'id': 'a', 'two_pass': dict.fromkeys(joint.LOSS_KEYS, 1.),
                  'full': copy.deepcopy(six), 'cached': copy.deepcopy(six),
                  **{k: copy.deepcopy(generation) for k in ('plain', 'format', 'hybrid')}}
        after = copy.deepcopy(before)
        after['two_pass']['loss'] = .5
        for key in ('full', 'cached'):
            after[key]['step_ce'] = [0.] * 5 + [2.]
            after[key]['predicted_ids'][-1] = 99
        after['plain']['answer_correct'] = False
        after['hybrid']['natural_exit'] = False
        after['hybrid']['forced_exit'] = True
        after['hybrid']['success'] = False
        report = joint.compare_records([{'id': 'a', 'question': 'q'}], [before], [after])
        self.assertEqual(report['two_pass']['loss']['lower'], 1)
        self.assertEqual(report['cached']['steps'][5]['higher_ce'], 1)
        self.assertEqual(report['cached']['all_six_correct_after'], 0)
        self.assertEqual(report['generation']['plain']['answer_correct']['true_to_false_ids'], ['a'])
        self.assertEqual(report['generation']['hybrid']['forced_exit']['false_to_true_ids'], ['a'])
        self.assertEqual(report['rows'][0]['after'], after)
        with self.assertRaisesRegex(ValueError, 'IDs'):
            joint.compare_records([{'id': 'other'}], [before], [after])


@unittest.skipUnless(AVAILABLE, 'requires torch, transformers and peft')
class JointRuntimeTests(unittest.TestCase):
    def test_frozen_evaluation_restores_flags_modes_even_on_failure(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.1))
        model[0].bias.requires_grad_(False)
        model.train()
        model[1].eval()
        flags = [p.requires_grad for p in model.parameters()]
        modes = [m.training for m in model.modules()]
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with joint.frozen_evaluation(model):
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                self.assertFalse(any(m.training for m in model.modules()))
                raise RuntimeError('test failure')
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        self.assertEqual([m.training for m in model.modules()], modes)
        model[0].weight.grad = torch.ones_like(model[0].weight)
        with self.assertRaisesRegex(RuntimeError, 'cleared gradients'):
            with joint.frozen_evaluation(model):
                pass
        self.assertIsNotNone(model[0].weight.grad)

    def test_one_eight_record_update_matches_mean_and_saved_replay(self):
        import torch
        import test_lora_self_conditioned as fixtures
        from hybrid_sft.model import SELF_CONDITIONED
        from peft import get_peft_model_state_dict
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = fixtures.SelfConditionedRuntimeTests().make_model(root)
            model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
            model.train()
            reference = copy.deepcopy(model)
            initial = {n: p.detach().clone() for n, p in model.named_parameters()}
            batches = []
            for i in range(8):
                ids = torch.tensor([[1, 32, 5 + i, 7, 33, 6, 2]])
                batches.append({'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                                'latent_mask': torch.tensor([[0, 1, 1, 1, 0, 0, 0]]),
                                'loss_mask': torch.tensor([[0, 1, 0, 0, 2, 1, 1]])})
            controls = [joint.scalar_losses(model(**b, stage=SELF_CONDITIONED)) for b in batches]
            opts = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=joint.LR,
                                     betas=(.9, .95), weight_decay=0., foreach=False)
                    for m in (model, reference)]
            def check_frozen():
                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        self.assertIsNone(p.grad, n)
                        torch.testing.assert_close(p, initial[n], rtol=0, atol=0)
            accelerator = Mock()
            accelerator.unwrap_model.side_effect = lambda m: m
            accelerator.backward.side_effect = lambda loss: loss.backward()
            accelerator.clip_grad_norm_.side_effect = torch.nn.utils.clip_grad_norm_
            with patch.object(opts[0], 'step', wraps=opts[0].step) as step:
                report = joint.train_once(model, batches, accelerator, opts[0], check_frozen, controls)
                self.assertEqual(step.call_count, 1)
            self.assertEqual(accelerator.backward.call_count, 8)
            self.assertEqual(accelerator.clip_grad_norm_.call_count, 1)
            self.assertEqual(report['optimizer_steps'], 1)
            sum(reference(**b, stage=SELF_CONDITIONED)['loss'] for b in batches).div(8).backward()
            torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], 1.)
            opts[1].step()
            for (n, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-7, msg=n)
                self.assertIsNone(p.grad)
            check_frozen()
            with joint.frozen_evaluation(model), torch.no_grad():
                expected = joint.scalar_losses(model(**batches[0], stage=SELF_CONDITIONED))
            model.save_pretrained(root / 'saved', fixtures.Tokenizer())
            from safetensors.torch import load_file
            saved = load_file(str(root / 'saved/lora/adapter_model.safetensors'))
            compact = get_peft_model_state_dict(model.backbone, save_embedding_layers=False)
            self.assertEqual(saved.keys(), compact.keys())
            self.assertEqual(sum('trainable_tokens_delta' in k for k in saved), 2)
            self.assertTrue(all('lora_' in k or 'trainable_tokens_delta' in k for k in saved))
            for k, v in compact.items():
                torch.testing.assert_close(v.detach().cpu(), saved[k], rtol=0, atol=0)
            joint.reload_adapters(model, root / 'saved')
            for k, v in get_peft_model_state_dict(model.backbone, save_embedding_layers=False).items():
                torch.testing.assert_close(v, saved[k], rtol=0, atol=0)
            check_frozen()
            with joint.frozen_evaluation(model), torch.no_grad():
                self.assertEqual(joint.scalar_losses(model(**batches[0], stage=SELF_CONDITIONED)), expected)
            # Check saved equality before clearing: corrupt files cannot silently
            # replace the live model during the replay check.
            link = torch.load(root / 'saved/adapter.pt', weights_only=True)
            link['proj1.bias'].add_(1)
            torch.save(link, root / 'saved/adapter.pt')
            with self.assertRaisesRegex(RuntimeError, 'saved adapter tensors differ'):
                joint.reload_adapters(model, root / 'saved')

    def test_nonfinite_loss_or_control_mismatch_stops_before_update(self):
        import torch
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(1, 1)
        model.hybridlink = torch.nn.Linear(1, 1)
        model.forward = lambda **kw: {k: model.hybridlink.weight.sum() * float('nan') for k in joint.LOSS_KEYS}
        acc, optimizer = Mock(), Mock()
        acc.unwrap_model.return_value = model
        controls = [dict.fromkeys(joint.LOSS_KEYS, 1.)] * 8
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            joint.train_once(model, [{}] * 8, acc, optimizer, lambda: None, controls)
        optimizer.step.assert_not_called()
        acc.backward.assert_not_called()
        model.forward = lambda **kw: {k: model.hybridlink.weight.sum() * 0 for k in joint.LOSS_KEYS}
        with self.assertRaisesRegex(RuntimeError, 'replay mismatch'):
            joint.train_once(model, [{}] * 8, acc, optimizer, lambda: None, controls)
        optimizer.step.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'eight'):
            joint.train_once(model, [{}] * 7, acc, optimizer, lambda: None, controls)
        # Finite forward values do not imply finite gradients. A bad gradient
        # must also stop before clipping or optimizer.step.
        model.forward = lambda **kw: {k: model.backbone.weight.sum() + model.hybridlink.weight.sum()
                                     for k in joint.LOSS_KEYS}
        controls = [joint.scalar_losses(model())] * 8
        acc.backward.side_effect = lambda loss: loss.backward()
        hook = model.backbone.weight.register_hook(lambda grad: grad * float('nan'))
        try:
            with self.assertRaisesRegex(RuntimeError, 'gradient'):
                joint.train_once(model, [{}] * 8, acc, optimizer, lambda: None, controls)
        finally:
            hook.remove()
        optimizer.step.assert_not_called()
        acc.clip_grad_norm_.assert_not_called()

    def test_text_generation_preserves_answer_prefix_and_fresh_cache(self):
        import torch
        from types import SimpleNamespace
        class Tokenizer:
            eos_token_id = 2
            def apply_chat_template(self, *args, **kwargs):
                return 'plain prompt'
            def __call__(self, text, **kwargs):
                return {'input_ids': torch.tensor([[1, 3]])}
            def decode(self, ids, **kwargs):
                return ''.join({4: '4', 5: '2'}[i] for i in ids)
        backbone = torch.nn.Linear(1, 1)
        backbone.generation_config = SimpleNamespace(eos_token_id=2)
        model = SimpleNamespace(backbone=backbone)
        row = {'question': 'q', 'expected': 42}
        calls = []
        def forward(backbone, **kwargs):
            if 'past_key_values' not in kwargs:
                self.assertEqual(kwargs['input_ids'].tolist(), [[1, 3]])
                token, length = 4, 2
            else:
                length = kwargs['past_key_values'].get_seq_length() + 1
                token = 5 if length == 3 else 2
                self.assertEqual(kwargs['position_ids'].item(), length - 1)
                self.assertEqual(kwargs['attention_mask'].shape[-1], length)
            calls.append(length)
            logits = torch.full((1, 1, 6), -10.)
            logits[0, 0, token] = 10
            return SimpleNamespace(logits=logits, past_key_values=SimpleNamespace(get_seq_length=lambda: length))
        with patch('hybrid_agent.compat.text_forward', side_effect=forward):
            a = joint.text_generation(model, Tokenizer(), row, 'plain')
            b = joint.text_generation(model, Tokenizer(), row, 'plain')
        self.assertEqual(a, b)
        self.assertEqual(a['scored_text'], '42')
        self.assertTrue(a['success'])
        self.assertEqual(calls, [2, 3, 4, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
