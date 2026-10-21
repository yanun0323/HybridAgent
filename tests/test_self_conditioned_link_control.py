"""Matched Link-only control: provenance, modes, gradients, one update and reload."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import lora_self_conditioned_joint as joint
import lora_self_conditioned_evaluate as evaluation
AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'transformers', 'peft'))


class ControlPlanTests(unittest.TestCase):
    def test_control_gate_precedes_update_and_save_precedes_after(self):
        import ast
        tree = ast.parse((ROOT / 'scripts/lora_self_conditioned_joint.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        def call(name):
            return next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == name)
        before = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'measure_all'
                      and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'before')
        after = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'measure_all'
                     and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'after')
        save = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'save_pretrained')
        self.assertLess(call('verify_joint_reference').lineno, before.lineno)
        self.assertEqual(len(before.args), 2)  # recorded controls are mandatory in Link-only mode
        self.assertLess(before.lineno, call('train_once').lineno)
        self.assertLess(call('train_once').lineno, save.lineno)
        self.assertLess(save.lineno, after.lineno)

    def test_cli_requires_paired_reference_and_no_extra_updates(self):
        script = str(ROOT / 'scripts/lora_self_conditioned_joint.py')
        result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for arg in ('--link-only', '--joint-run', '--joint-evaluation'):
            self.assertIn(arg, result.stdout)
        for arg in ('--resume', '--max-steps', '--lr', '--all-train'):
            self.assertNotIn(arg, result.stdout)
        base = [sys.executable, script]
        for arg in ('pilot', 'model', 'calibration', 'probe-run', 'output-dir'):
            base += ['--' + arg, '/does-not-exist']
        for extra in (['--link-only'], ['--joint-run', 'j', '--joint-evaluation', 'e'],
                      ['--link-only', '--joint-run', 'j']):
            failed = subprocess.run(base + extra, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertIn('requires both', failed.stderr)

    def test_complete_reference_and_any_changed_evidence_rejected(self):
        update = [{'id': str(i), 'question': 'q'} for i in range(8)]
        observe = [{'id': str(i), 'question': 'q'} for i in range(8, 16)]
        six = {'step_ce': [1.] * 6, 'target_ids': list(range(6)), 'predicted_ids': list(range(6)),
               'target_positions': list(range(10, 16)), 'substitution_positions': list(range(9, 15))}
        generation = dict(answer_correct=True, format_valid=True, success=True, truncated=False,
                          natural_exit=False, forced_exit=True, used_vectors=True, terminated=True,
                          scored_text='42')
        before = [{'id': str(i), 'two_pass': dict.fromkeys(joint.LOSS_KEYS, 1.),
                   'full': copy.deepcopy(six), 'cached': copy.deepcopy(six),
                   **{k: copy.deepcopy(generation) for k in ('plain', 'format', 'hybrid')}} for i in range(16)]
        after = copy.deepcopy(before)
        after[0]['two_pass']['loss'] = .8
        hashes = {'adapter.pt': joint.SOURCE_LINK_SHA256,
                  'lora/adapter_model.safetensors': joint.SOURCE_LORA_SHA256}
        saved_hashes = {'checkpoint/adapter.pt': evaluation.SAVED_LINK_SHA256,
                        'checkpoint/lora/adapter_model.safetensors': evaluation.SAVED_LORA_SHA256}
        run = {'code_commit': evaluation.FAILED_COMMIT, 'data_sha256': joint.TRAIN_SHA256,
               'planned_optimizer_steps': 1, 'gradient_average_count': 8,
               'learning_rate_lora': joint.LR, 'learning_rate_link': joint.LR,
               'betas': [.9, .95], 'weight_decay': 0., 'max_grad_norm': 1., 'shuffle': False,
               'scheduler': None, 'anchors_frozen': True, 'hybrid_limits': joint.HYBRID_LIMITS,
               'text_budgets': {'plain': 32, 'format': 256}, 'recursive_steps': 6, 'seed': 6064,
               'update_ids': [r['id'] for r in update], 'observation_ids': [r['id'] for r in observe],
               'source_hashes': hashes, 'tokenization_sha256': ['token'] * 16,
               'calibration': {'output_scale': .012},
               **dict.fromkeys(('torch', 'transformers', 'peft', 'accelerate', 'tokenizers'), 'version')}
        report = {'optimizer_steps': 1, 'pre_update_losses': [r['two_pass'] for r in before[:8]],
                  'total_norm_before_clip': 2.}
        erun = {'code_commit': '2f37602dcf09b299a0cc9fab43193a7dab71afd0',
                'original_plan': run, 'interrupted_run_hashes': saved_hashes, 'source_hashes': hashes,
                'data_sha256': joint.TRAIN_SHA256, 'tokenization_sha256': run['tokenization_sha256'],
                **{k: run[k] for k in ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers')}}
        summary = {'optimizer_steps_this_run': 0, 'original_optimizer_steps': 1, 'weights_saved': False,
                   'saved_link_sha256': evaluation.SAVED_LINK_SHA256,
                   'saved_lora_sha256': evaluation.SAVED_LORA_SHA256,
                   **dict.fromkeys(('original_16_controls_exact', 'saved_adapter_replay_exact',
                                    'anchors_values_unchanged', 'original_backbone_versions_no_grad_unchanged',
                                    'protected_files_unchanged'), True),
                   'update': joint.compare_records(update, before[:8], after[:8]),
                   'observation': joint.compare_records(observe, before[8:], after[8:])}
        with TemporaryDirectory() as tmp:
            j, e = Path(tmp) / 'joint', Path(tmp) / 'evaluation'
            (j / 'checkpoint').mkdir(parents=True)
            e.mkdir()
            records = {j / 'run.json': run, j / 'update.json': report,
                       j / 'checkpoint/link_calibration.json': run['calibration'],
                       e / 'run.json': erun, e / 'summary.json': summary}
            rows = {j / 'before.jsonl': before, e / 'control.jsonl': before,
                    e / 'after.jsonl': after, e / 'reloaded.jsonl': after}
            for p, value in records.items():
                p.write_text(json.dumps(value))
            for p, value in rows.items():
                p.write_text(''.join(json.dumps(r) + '\n' for r in value))
            originals = {p: p.read_bytes() for p in (*records, *rows)}
            with patch.object(joint, 'file_hashes', return_value=saved_hashes):
                result = joint.verify_joint_reference(j, e, hashes, update, observe)
                self.assertEqual(result['after'], after)
                changes = [
                    (e / 'summary.json', {**summary, 'protected_files_unchanged': False}),
                    (e / 'run.json', {**erun, 'code_commit': 'different'}),
                    (e / 'run.json', {**erun, 'tokenizers': 'different'}),
                    (j / 'run.json', {**run, 'learning_rate_link': 1e-4}),
                    (j / 'update.json', {**report, 'optimizer_steps': 2}),
                    (j / 'checkpoint/link_calibration.json', {'output_scale': .1}),
                ]
                for p, changed in changes:
                    with self.subTest(path=str(p)):
                        p.write_text(json.dumps(changed))
                        with self.assertRaises((ValueError, RuntimeError)):
                            joint.verify_joint_reference(j, e, hashes, update, observe)
                        p.write_bytes(originals[p])
                for p in rows:
                    p.write_text(p.read_text().replace('42', '43'))
                    with self.assertRaises((ValueError, RuntimeError)):
                        joint.verify_joint_reference(j, e, hashes, update, observe)
                    p.write_bytes(originals[p])
            with patch.object(joint, 'file_hashes', return_value={}):
                with self.assertRaisesRegex(ValueError, 'weights differ'):
                    joint.verify_joint_reference(j, e, hashes, update, observe)
            self.assertEqual(originals, {p: p.read_bytes() for p in originals})


@unittest.skipUnless(AVAILABLE, 'requires torch, transformers and peft')
class ControlRuntimeTests(unittest.TestCase):
    def make_pair(self, root):
        import test_lora_self_conditioned as fixtures
        from hybrid_sft.model import SELF_CONDITIONED
        original = fixtures.SelfConditionedRuntimeTests().make_model(root)
        original.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
        control = copy.deepcopy(original)
        control.set_trainable(SELF_CONDITIONED, link_only_self_conditioned=True)
        return original, control

    def batches(self):
        import torch
        return [dict(input_ids=torch.tensor([[1, 32, 5 + i, 7, 33, 6, 2]]),
                     attention_mask=torch.ones(1, 7, dtype=torch.long),
                     latent_mask=torch.tensor([[0, 1, 1, 1, 0, 0, 0]]),
                     loss_mask=torch.tensor([[0, 1, 0, 0, 2, 1, 1]])) for i in range(8)]

    def test_control_rejects_trainable_backbone_before_backward_or_reload(self):
        import torch
        with TemporaryDirectory() as tmp:
            original, _ = self.make_pair(Path(tmp))
            optimizer, accelerator = Mock(), Mock()
            accelerator.unwrap_model.return_value = original
            with self.assertRaisesRegex(RuntimeError, 'trainable backbone'):
                joint.train_once(original, self.batches(), accelerator, optimizer, lambda: None,
                                 [{}] * 8, link_only=True)
            accelerator.backward.assert_not_called()
            optimizer.step.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'frozen backbone'):
                joint.reload_adapters(original, Path(tmp) / 'absent', link_only=True)

    def test_mode_boundary_and_switch_back(self):
        import torch
        from hybrid_sft.model import SELF_CONDITIONED, LINK_NEXT_TOKEN
        with TemporaryDirectory() as tmp:
            original, control = self.make_pair(Path(tmp))
            for model in (original, control):
                model.train()
                self.assertTrue(all(m.training for m in model.modules()))
            for (n, a), (_, b) in zip(original.named_parameters(), control.named_parameters()):
                self.assertEqual(a.dtype, b.dtype, n)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(b.requires_grad, n.startswith('hybridlink.'), n)
            with joint.frozen_evaluation(control):
                self.assertFalse(any(m.training for m in control.modules()))
            self.assertTrue(all(m.training for m in control.modules()))
            with self.assertRaises(ValueError):
                control.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True,
                                      link_only_self_conditioned=True)
            with self.assertRaises(ValueError):
                control.set_trainable(LINK_NEXT_TOKEN, link_only_self_conditioned=True)
            control.set_trainable(LINK_NEXT_TOKEN)
            control.train()
            self.assertFalse(control.backbone.training)

    def test_same_forward_and_link_gradients_through_frozen_decoder(self):
        import torch
        from hybrid_sft.model import SELF_CONDITIONED
        with TemporaryDirectory() as tmp:
            original, control = self.make_pair(Path(tmp))
            batch = self.batches()[0]
            for model in (original, control):
                model.train()
                model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            generated = []
            link = control.run_hybridlink
            def capture(*args, **kwargs):
                result = link(*args, **kwargs)
                result.retain_grad()
                generated.append(result)
                return result
            a = original(**batch, stage=SELF_CONDITIONED)
            with patch.object(control, 'run_hybridlink', side_effect=capture):
                b = control(**batch, stage=SELF_CONDITIONED)
            self.assertEqual(joint.scalar_losses(a), joint.scalar_losses(b))
            # CE alone must reach first-pass Link outputs through frozen decoder.
            from hybrid_agent.compat import text_decoder
            decoder = text_decoder(control.backbone)
            targets = control.backbone.get_input_embeddings()(batch['input_ids']).detach()
            first = decoder(inputs_embeds=targets, attention_mask=batch['attention_mask'],
                            use_cache=False, return_dict=True).last_hidden_state
            predicted = control.run_hybridlink(first[:, :-1], output_dtype=targets.dtype)
            modified = torch.cat([targets[:, :1], torch.where(batch['latent_mask'][:, :-1, None].bool(),
                                                             predicted, targets[:, 1:])], dim=1)
            second = decoder(inputs_embeds=modified, attention_mask=batch['attention_mask'],
                             use_cache=False, return_dict=True).last_hidden_state
            ce = control._chunked_ce(second, batch['input_ids'], batch['loss_mask'])
            grad = torch.autograd.grad(ce, predicted)[0]
            self.assertGreater(grad[:, 1:4].abs().sum().item(), 0.)
            a['loss'].backward()
            b['loss'].backward()
            self.assertGreater(generated[0].grad.abs().sum().item(), 0.)
            for (_, p), (_, q) in zip(original.hybridlink.named_parameters(), control.hybridlink.named_parameters()):
                self.assertIsNotNone(q.grad)
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)
            self.assertTrue(all(p.grad is None for p in control.backbone.parameters()))

    def test_eight_record_update_and_reload_never_write_backbone(self):
        import torch
        import test_lora_self_conditioned as fixtures
        from hybrid_sft.model import SELF_CONDITIONED
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, model = self.make_pair(root)
            model.train()
            reference = copy.deepcopy(model)
            frozen = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
            versions = {n: p._version for n, p in model.backbone.named_parameters()}
            def check():
                for n, p in model.backbone.named_parameters():
                    self.assertFalse(p.requires_grad)
                    self.assertIsNone(p.grad)
                    self.assertEqual(p._version, versions[n], n)
                    torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
            batches = self.batches()
            controls = [joint.scalar_losses(model(**b, stage=SELF_CONDITIONED)) for b in batches]
            opts = [torch.optim.AdamW(m.hybridlink.parameters(), lr=joint.LR, betas=(.9, .95),
                                     weight_decay=0., foreach=False) for m in (model, reference)]
            acc = Mock()
            acc.unwrap_model.side_effect = lambda m: m
            acc.backward.side_effect = lambda loss: loss.backward()
            acc.clip_grad_norm_.side_effect = torch.nn.utils.clip_grad_norm_
            initial = {n: p.detach().clone() for n, p in model.hybridlink.named_parameters()}
            with patch.object(opts[0], 'step', wraps=opts[0].step) as step:
                report = joint.train_once(model, batches, acc, opts[0], check, controls, link_only=True)
                self.assertEqual(step.call_count, 1)
            self.assertEqual(report['lora_gradient_norms'], {})
            self.assertEqual(report['clip_scope'], 'Link only')
            self.assertEqual(acc.backward.call_count, 8)
            self.assertEqual(acc.clip_grad_norm_.call_count, 1)
            sum(reference(**b, stage=SELF_CONDITIONED)['loss'] for b in batches).div(8).backward()
            torch.nn.utils.clip_grad_norm_(reference.hybridlink.parameters(), 1.)
            opts[1].step()
            self.assertTrue(any(not torch.equal(p, initial[n]) for n, p in model.hybridlink.named_parameters()))
            for p, q in zip(model.hybridlink.parameters(), reference.hybridlink.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-7)
            expected = [joint.scalar_losses(model(**b, stage=SELF_CONDITIONED)) for b in batches]
            model.save_pretrained(root / 'saved', fixtures.Tokenizer())
            with patch('peft.set_peft_model_state_dict') as install:
                joint.reload_adapters(model, root / 'saved', link_only=True)
                install.assert_not_called()
            check()
            self.assertEqual([joint.scalar_losses(model(**b, stage=SELF_CONDITIONED)) for b in batches], expected)
            from safetensors.torch import load_file, save_file
            path = root / 'saved/lora/adapter_model.safetensors'
            state = load_file(str(path))
            state[next(k for k in state if 'lora_A' in k)].add_(1)
            save_file(state, str(path))
            with self.assertRaisesRegex(RuntimeError, 'saved adapter tensors differ'):
                joint.reload_adapters(model, root / 'saved', link_only=True)
            check()


if __name__ == '__main__':
    unittest.main()
