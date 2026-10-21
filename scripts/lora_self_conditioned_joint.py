"""One eight-record LoRA + Link update; next eight records observe only.

Two-pass objective with first-pass gradients, frozen original weights and anchors.
Optional matched Link-only control; not an exact upstream reproduction. No resume,
shuffle, scheduler, repeated updates, development/test access or checkpoint search.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_experiment import ROOT, verify_data, sha256, write_json, PLAIN_INSTRUCTION, score_output
from lora_self_conditioned_probe import validate_paths, SOURCE_LINK_SHA256, SOURCE_LORA_SHA256
from link_three_step_probe import gradient_report
from link_three_step_joint import training_plan, compare_group
from link_execution_probe import cached_rollout, measurements
from latent_diagnostic import span_bounds
from latent_experiment import score_hybrid, scored_engine_text, check_reference_config

LR = 5e-6
TRAIN_SHA256 = 'c58b8d73cf5f9c45f12472acb53ce9b718ee5cc179fcce5afffc4b1c68227ae2'
LOSS_KEYS = ('loss', 'backbone_loss', 'hybridlink_loss', 'cos_loss', 'mse_loss')
HYBRID_LIMITS = dict(max_response_tokens=256, min_latent_steps=0, max_latent_steps=64,
                     count_tool_tokens=False, max_turns=1)


def scalar_losses(out):
    values = {k: out[k].detach().float().item() for k in LOSS_KEYS}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError('nonfinite two-pass loss')
    return values


def file_hashes(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_probe(run, summary, hashes):
    if (run['source_hashes'] != hashes or run['data_sha256'] != TRAIN_SHA256
            or run['sample_id'] != 'mul-sub-54-5-6' or summary['sample_id'] != run['sample_id']
            or summary['optimizer_steps'] != 0 or summary['weights_saved'] is not False):
        raise ValueError('probe provenance differs')
    for key in ('frozen_parameters_no_grad', 'all_parameter_versions_unchanged',
                'lora_link_anchors_values_unchanged', 'source_hashes_unchanged'):
        if summary[key] is not True:
            raise ValueError(f'probe did not pass: {key}')
    for key in ('lora_gradient_norms', 'link_gradient_norms'):
        norms = summary[key].values()
        if not norms or not all(math.isfinite(v) and v >= 0 for v in norms) or not any(v > 0 for v in norms):
            raise ValueError('probe gradients invalid')


@contextmanager
def frozen_evaluation(model):
    """Restore exact gradient flags and modes; never discard accumulated gradients."""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    if any(p.grad is not None for p, _ in flags):
        raise RuntimeError('evaluation requires cleared gradients')
    try:
        model.requires_grad_(False)
        model.eval()
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def train_once(model, batches, accelerator, optimizer, check_frozen, expected, *, link_only=False):
    import torch
    from hybrid_sft.model import SELF_CONDITIONED
    if len(batches) != 8 or len(expected) != 8:
        raise ValueError('requires exactly eight update records and controls')
    raw = accelerator.unwrap_model(model)
    raw.train()
    params = [(n, p) for n, p in raw.named_parameters() if p.requires_grad]
    if link_only and (not params or any(not n.startswith('hybridlink.') for n, _ in params)):
        raise RuntimeError('Link-only control has trainable backbone parameters')
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    for i, batch in enumerate(batches):
        out = model(**batch, stage=SELF_CONDITIONED)
        metric = scalar_losses(out)
        if metric != expected[i]:
            raise RuntimeError(f'pre-update two-pass replay mismatch: record {i + 1}')
        accelerator.backward(out['loss'] / 8)
        metrics.append(metric)
        del out
        print(f'Backward {i + 1}/8: loss={metric["loss"]:.8g}', flush=True)
    lora = {} if link_only else gradient_report((n, p) for n, p in params if n.startswith('backbone.'))
    link = gradient_report(raw.hybridlink.named_parameters())
    check_frozen()
    norm = accelerator.clip_grad_norm_([p for _, p in params], 1.)
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite joint clipping norm')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not torch.isfinite(p).all() for _, p in params):
        raise RuntimeError('nonfinite updated parameters')
    check_frozen()
    return {'pre_update_losses': metrics, 'lora_gradient_norms': lora, 'link_gradient_norms': link,
            'total_norm_before_clip': norm.item(),
            'clip_multiplier': min(1., 1. / (norm.item() + 1e-6)),
            'clip_scope': 'Link only' if link_only else 'LoRA + Link', 'optimizer_steps': 1}


def text_generation(model, tokenizer, row, mode):
    """Greedy cached text-only baseline, same prompts/budgets as format_experiment."""
    import torch
    from hybrid_agent.compat import text_forward
    from hybrid_agent.prompts import build_prompt, THINK_PREFILL
    device = next(model.backbone.parameters()).device
    if mode == 'format':
        prompt = build_prompt(tokenizer, row['system'], row['question'])
    else:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['question'] + ' ' + PLAIN_INSTRUCTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
    out = text_forward(model.backbone, input_ids=ids, use_cache=True, return_dict=True)
    eos = model.backbone.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    position, generated, end = ids.shape[1], [], 'limit'
    budget = 32 if mode == 'plain' else 256
    for step in range(budget):
        logits = out.logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError('nonfinite text logits')
        token = logits.argmax().item()
        if token in eos_ids:
            end = 'eos'
            break
        generated.append(token)
        if step + 1 < budget:
            out = text_forward(model.backbone, input_ids=torch.tensor([[token]], device=device),
                               attention_mask=torch.ones(1, position + 1, dtype=torch.long, device=device),
                               position_ids=torch.tensor([[position]], device=device),
                               past_key_values=out.past_key_values, use_cache=True, return_dict=True)
            position += 1
            if out.past_key_values.get_seq_length() != position:
                raise RuntimeError('text cache position mismatch')
    text = tokenizer.decode(generated, skip_special_tokens=False)
    if mode == 'format' and prompt.endswith(THINK_PREFILL):
        text = THINK_PREFILL + text
    return {'scored_text': text, 'generated_ids': generated, 'end_reason': end,
            **score_output(text, row['expected'], mode, end)}


def measure_record(model, raw, accelerator, tokenizer, row, sample, batch, entry, exit):
    import torch
    from hybrid_sft.model import SELF_CONDITIONED, LINK_NEXT_TOKEN
    from hybrid_agent.engine import HybridGenerationEngine
    from hybrid_agent.tools import ToolExecutor
    raw.train()
    # Same grad-enabled training path as the feasibility probe and update.
    out = model(**batch, stage=SELF_CONDITIONED)
    result = {'id': row['id'], 'two_pass': scalar_losses(out)}
    del out
    with frozen_evaluation(raw), torch.no_grad(), accelerator.autocast():
        full = raw(**batch, stage=LINK_NEXT_TOKEN, link_steps=6)
        result['full'], _ = measurements(full, tokenizer)
        del full
        cached = cached_rollout(raw, sample, entry, exit, link_steps=6)
        result['cached'], _ = measurements(cached, tokenizer)
        del cached
        for mode in ('plain', 'format'):
            result[mode] = text_generation(raw, tokenizer, row, mode)
        engine = HybridGenerationEngine(
            raw, tokenizer, accelerator.device, ToolExecutor(), row['system'], row['question'],
            batch_size=1, temperature=0., top_p=1., top_k=0, max_tool_calls=0,
            terminate_on_tool_cap=True, tool_workers=1, **HYBRID_LIMITS)
        engine.run()
        outcome = engine.row_result(engine.rows[0])
        text = scored_engine_text(engine, tokenizer)
        result['hybrid'] = {**outcome, 'scored_text': text,
                            **score_hybrid(text, row['expected'], outcome)}
    return result


def compare_records(rows, before, after):
    if not rows or len(rows) != len(before) or len(rows) != len(after):
        raise ValueError('comparison row counts differ')
    if [r['id'] for r in rows] != [r['id'] for r in before] or [r['id'] for r in rows] != [r['id'] for r in after]:
        raise ValueError('comparison IDs differ')
    result = {'two_pass': {}, 'generation': {}}
    for key in LOSS_KEYS:
        a, b = [r['two_pass'][key] for r in before], [r['two_pass'][key] for r in after]
        result['two_pass'][key] = {
            'before': sum(a) / len(a), 'after': sum(b) / len(b),
            'lower': sum(y < x - 1e-6 for x, y in zip(a, b)),
            'higher': sum(y > x + 1e-6 for x, y in zip(a, b))}
    for path in ('full', 'cached'):
        result[path] = compare_group(rows, [r[path] for r in before], [r[path] for r in after], link_steps=6)
    for mode in ('plain', 'format', 'hybrid'):
        keys = ('answer_correct', 'format_valid', 'success', 'truncated')
        if mode == 'hybrid':
            keys += ('natural_exit', 'forced_exit', 'used_vectors', 'terminated')
        result['generation'][mode] = {
            k: {'before': sum(r[mode][k] for r in before), 'after': sum(r[mode][k] for r in after),
                'true_to_false_ids': [r['id'] for r, a, b in zip(rows, before, after) if a[mode][k] and not b[mode][k]],
                'false_to_true_ids': [r['id'] for r, a, b in zip(rows, before, after) if not a[mode][k] and b[mode][k]]}
            for k in keys}
    result['rows'] = [{'id': r['id'], 'before': a, 'after': b} for r, a, b in zip(rows, before, after)]
    return result


def reload_adapters(raw, checkpoint, *, link_only=False):
    """Read saved PEFT + Link tensors into the SAME backbone, verify exact values."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    if link_only and any(p.requires_grad for p in raw.backbone.parameters()):
        raise RuntimeError('Link-only reload requires frozen backbone')
    saved = load_file(str(checkpoint / 'lora/adapter_model.safetensors'))
    # Match save_pretrained's compact policy; PEFT's default 'auto' may add
    # full embedding/head matrices when the vocabulary has been resized.
    live = get_peft_model_state_dict(raw.backbone, save_embedding_layers=False)
    link = torch.load(checkpoint / 'adapter.pt', map_location='cpu', weights_only=True)
    def equal(a, b):
        return (a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and
                torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a))
    if not equal(saved, live) or not equal(link, raw.hybridlink.state_dict()):
        raise RuntimeError('saved adapter tensors differ from live weights')
    # Clear only trainable tensors, forcing restoration from disk rather than a
    # no-op reload. The Link-only control verifies saved PEFT values but never
    # copies them back: even frozen backbone version counters must stay unchanged.
    with torch.no_grad():
        for p in raw.parameters():
            if p.requires_grad:
                p.zero_()
    if not link_only:
        loaded = set_peft_model_state_dict(raw.backbone, saved, adapter_name='default')
        if loaded.unexpected_keys:
            raise RuntimeError(f'unexpected saved PEFT keys: {loaded.unexpected_keys}')
    raw.hybridlink.load_state_dict(link, strict=True)
    if not equal(saved, get_peft_model_state_dict(raw.backbone, save_embedding_layers=False)) or not equal(link, raw.hybridlink.state_dict()):
        raise RuntimeError('restored adapter tensors differ from saved weights')


def verify_joint_reference(joint, evaluation, source_hashes, update, observe):
    """Bind the control to the preserved update and completed recovery evaluation."""
    from lora_self_conditioned_evaluate import (
        verify_plan, require_replay, SAVED_LINK_SHA256, SAVED_LORA_SHA256,
    )
    def read(directory, name):
        return json.loads((directory / name).read_text())
    def rows(directory, name):
        return [json.loads(s) for s in (directory / name).read_text().splitlines() if s.strip()]
    run, report = read(joint, 'run.json'), read(joint, 'update.json')
    before = rows(joint, 'before.jsonl')
    verify_plan(run, report, before, update, observe)
    saved_hashes = file_hashes(joint)
    if (run['source_hashes'] != source_hashes or
            saved_hashes.get('checkpoint/adapter.pt') != SAVED_LINK_SHA256 or
            saved_hashes.get('checkpoint/lora/adapter_model.safetensors') != SAVED_LORA_SHA256):
        raise ValueError('joint source or saved weights differ')
    erun, summary = read(evaluation, 'run.json'), read(evaluation, 'summary.json')
    expected = {'optimizer_steps_this_run': 0, 'original_optimizer_steps': 1,
                'weights_saved': False, 'saved_link_sha256': SAVED_LINK_SHA256,
                'saved_lora_sha256': SAVED_LORA_SHA256,
                **dict.fromkeys(('original_16_controls_exact', 'saved_adapter_replay_exact',
                     'anchors_values_unchanged', 'original_backbone_versions_no_grad_unchanged',
                     'protected_files_unchanged'), True)}
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError('joint recovery did not pass all checks')
    if (erun['code_commit'] != '2f37602dcf09b299a0cc9fab43193a7dab71afd0' or
            erun['original_plan'] != run or erun['interrupted_run_hashes'] != saved_hashes or
            erun['source_hashes'] != source_hashes or erun['data_sha256'] != TRAIN_SHA256 or
            erun['tokenization_sha256'] != run['tokenization_sha256'] or
            any(erun[k] != run[k] for k in ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers'))):
        raise ValueError('joint recovery provenance differs')
    require_replay(rows(evaluation, 'control.jsonl'), before, 'joint controls')
    after = rows(evaluation, 'after.jsonl')
    require_replay(rows(evaluation, 'reloaded.jsonl'), after, 'joint saved replay')
    # Recompute reports from raw rows, including IDs and generated text.
    require_replay(summary['update'], compare_records(update, before[:8], after[:8]), 'joint update report')
    require_replay(summary['observation'], compare_records(observe, before[8:], after[8:]), 'joint observation report')
    if read(joint / 'checkpoint', 'link_calibration.json') != run['calibration']:
        raise ValueError('saved joint calibration differs')
    return {'run': run, 'report': report, 'before': before, 'after': after}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'model', 'calibration', 'probe-run', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--link-only', action='store_true', help='matched two-pass control; freeze LoRA')
    p.add_argument('--joint-run')
    p.add_argument('--joint-evaluation')
    args = p.parse_args()
    if args.link_only != bool(args.joint_run) or args.link_only != bool(args.joint_evaluation):
        p.error('--link-only requires both --joint-run and --joint-evaluation, and vice versa')
    source, pilot, probe, output = [Path(v).resolve() for v in
                                   (args.model, args.pilot, args.probe_run, args.output_dir)]
    validate_paths(source, pilot, output)
    validate_paths(source, probe, output)
    data, _ = verify_data(pilot, 'train')
    if sha256(data) != TRAIN_SHA256:
        raise ValueError('requires original training file')
    update, observe, _ = training_plan([json.loads(s) for s in data.read_text().splitlines() if s.strip()])
    rows = update + observe
    hashes = file_hashes(source)
    if hashes.get('adapter.pt') != SOURCE_LINK_SHA256 or hashes.get('lora/adapter_model.safetensors') != SOURCE_LORA_SHA256:
        raise ValueError('requires original 64-step Link and LoRA weights')
    probe_run = json.loads((probe / 'run.json').read_text())
    probe_summary = json.loads((probe / 'summary.json').read_text())
    verify_probe(probe_run, probe_summary, hashes)
    reference = None
    protected = {probe: file_hashes(probe)}
    if args.link_only:
        for directory in (args.joint_run, args.joint_evaluation):
            path = Path(directory).resolve()
            validate_paths(source, path, output)
            protected[path] = file_hashes(path)
        reference = verify_joint_reference(Path(args.joint_run), Path(args.joint_evaluation),
                                           hashes, update, observe)

    import torch
    import transformers
    import peft
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoTokenizer
    from hybrid_agent.compat import check_execution
    from hybrid_agent.lora import is_lora_model
    from hybrid_sft.model import HybridSFTModel, SELF_CONDITIONED
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.calibration import read_training_calibration

    calibration = read_training_calibration(args.calibration, data)
    config = json.loads((source / 'hybridlink_config.json').read_text())
    if config['output_scale'] != calibration['output_scale'] or calibration != probe_run['calibration']:
        raise ValueError('calibration differs')
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    if (accelerator.device.type != 'cuda' or accelerator.num_processes != 1
            or str(accelerator.distributed_type).endswith('DEEPSPEED')):
        raise ValueError('requires one CUDA process without DeepSpeed')
    versions = {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}
    if any(versions[k] != probe_run[k] for k in versions):
        raise ValueError('library versions differ from probe')
    if reference is not None:
        from importlib.metadata import version
        if any(version(k) != reference['run'][k] for k in
               ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers')):
            raise ValueError('library versions differ from joint reference')
        if calibration != reference['run']['calibration']:
            raise ValueError('calibration differs from joint reference')
    set_seed(6064)
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    entry, exit = [tokenizer.convert_tokens_to_ids(t) for t in ('<latent>', '</latent>')]
    samples, batches = [], []
    for row in rows:
        sample = tokenize_trajectory(tokenizer, row['system'], row['question'], row['rounds'], 1024)
        if sample is None:
            raise ValueError(f'rejected record: {row["id"]}')
        start, end = span_bounds(sample, entry, exit)
        if end - start - 1 < 6:
            raise ValueError('requires six content tokens')
        samples.append(sample)
        batches.append(collate_batch(tokenizer, [sample], pad_to_multiple=1))
    token_hashes = [hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for s in samples]
    if token_hashes[0] != probe_run['tokenization_sha256']:
        raise ValueError('probe tokenization differs')
    if reference is not None and token_hashes != reference['run']['tokenization_sha256']:
        raise ValueError('tokenization differs from joint reference')
    output.mkdir(parents=True)
    write_json(output / 'run.json', {
        'source': str(source), 'source_hashes': hashes, 'data_sha256': TRAIN_SHA256,
        'probe_run_sha256': sha256(probe / 'run.json'), 'probe_summary_sha256': sha256(probe / 'summary.json'),
        'update_ids': [r['id'] for r in update], 'observation_ids': [r['id'] for r in observe],
        'tokenization_sha256': token_hashes, 'calibration': calibration, 'seed': 6064,
        'planned_optimizer_steps': 1, 'gradient_average_count': 8,
        'learning_rate_lora': None if args.link_only else LR, 'link_only': args.link_only,
        'protected_run_hashes': {str(k): v for k, v in protected.items()},
        'joint_run': args.joint_run, 'joint_evaluation': args.joint_evaluation,
        'learning_rate_link': LR, 'betas': [.9, .95], 'weight_decay': 0., 'max_grad_norm': 1.,
        'shuffle': False, 'scheduler': None, 'anchors_frozen': True, 'hybrid_limits': HYBRID_LIMITS,
        'text_budgets': {'plain': 32, 'format': 256}, 'recursive_steps': 6,
        'precision': 'two-pass BF16 autocast incl. Link operations, FP32 trainable weights; '
                     'six-input full/cached no-grad BF16 autocast with FP32 Link operations; '
                     'text/hybrid no-grad BF16 autocast. Not old inference-mode diagnostics.',
        **versions, 'accelerate': __import__('accelerate').__version__,
        'tokenizers': __import__('tokenizers').__version__,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    })
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model) or model.lora_anchor_ids != [entry, exit]:
        raise ValueError('expected matching LoRA checkpoint')
    check_execution(model.backbone, batch_size=1)
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=not args.link_only,
                        link_only_self_conditioned=args.link_only)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    expected_parameters = 52459520 if args.link_only else probe_summary['trainable_parameters']
    if (not trainable or sum(p.numel() for p in trainable.values()) != expected_parameters
            or (args.link_only and any(not n.startswith('hybridlink.') for n in trainable))
            or any(p.dtype != torch.float32 or not (n.startswith('hybridlink.') or '.lora_A.' in n or '.lora_B.' in n)
                   for n, p in trainable.items())):
        raise RuntimeError('unexpected trainable parameter or dtype')
    optimizer = torch.optim.AdamW(trainable.values(), lr=LR, betas=(.9, .95), weight_decay=0., foreach=False)
    model, optimizer = accelerator.prepare(model, optimizer)
    raw = accelerator.unwrap_model(model)
    batches = [{k: v.to(accelerator.device) for k, v in b.items()} for b in batches]
    frozen = {n: p._version for n, p in raw.named_parameters() if not p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in raw.named_parameters() if p.requires_grad}
    anchors = {n: p.detach().cpu().clone() for n, p in raw.named_parameters() if 'trainable_tokens_delta' in n}
    frozen_adapters = {n: p.detach().cpu().clone() for n, p in raw.named_parameters()
                       if args.link_only and ('.lora_A.' in n or '.lora_B.' in n)}
    if len(anchors) != 2:
        raise RuntimeError('expected two frozen anchor tensors')
    def check_frozen(check_versions=True):
        for n, param in raw.named_parameters():
            if n in frozen and (param.requires_grad or param.grad is not None or
                                (check_versions and param._version != frozen[n])):
                raise RuntimeError(f'frozen parameter changed: {n}')
            if n in anchors and not torch.equal(param.detach().cpu(), anchors[n]):
                raise RuntimeError(f'anchor values changed: {n}')
            if n in frozen_adapters and not torch.equal(param.detach().cpu(), frozen_adapters[n]):
                raise RuntimeError(f'frozen LoRA values changed: {n}')
    def measure_all(phase, controls=None):
        results = []
        with (output / f'{phase}.jsonl').open('x') as stream:
            for i, (row, sample, batch) in enumerate(zip(rows, samples, batches)):
                item = measure_record(model, raw, accelerator, tokenizer, row, sample, batch, entry, exit)
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                stream.flush()
                results.append(item)
                if controls is not None and item != controls[i]:
                    raise RuntimeError(f'exact replay mismatch: {phase}/{row["id"]}; preserve evidence')
                check_frozen()
                if phase == 'before' and len(results) == 1:
                    if item['two_pass'] != {k: probe_summary[k] for k in LOSS_KEYS}:
                        raise RuntimeError('first-record loss differs from completed probe; stop before update')
                print(f'{phase} {row["id"]}: loss={item["two_pass"]["loss"]:.8g} '
                      f'cached6={item["cached"]["mean_ce"]:.8g} '
                      f'natural_exit={item["hybrid"]["natural_exit"]}', flush=True)
        return results
    raw.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    before = measure_all('before', reference['before'] if reference is not None else None)
    check_frozen()
    report = train_once(model, batches[:8], accelerator, optimizer, check_frozen,
                        [r['two_pass'] for r in before[:8]], link_only=args.link_only)
    write_json(output / 'update.json', report)
    changed = [n for n, p in raw.named_parameters() if n in initial and not torch.equal(p.detach().cpu(), initial[n])]
    if (not any(n.startswith('hybridlink.') for n in changed) or
            (not args.link_only and not any(n.startswith('backbone.') for n in changed))):
        raise RuntimeError('expected trainable modules did not change')
    del initial, optimizer
    # Preserve the single update even if a later measurement/replay fails.
    # A checkpoint alone is not a completed run: summary.json is written last.
    check_frozen()
    checkpoint = output / 'checkpoint'
    raw.save_pretrained(checkpoint, tokenizer)
    write_json(checkpoint / 'train_args.json', {'stage': SELF_CONDITIONED, 'lora_self_conditioned': not args.link_only,
               'link_only_self_conditioned': args.link_only,
               'max_steps': 1, 'grad_accum': 8, 'batch_size': 1, 'lr': LR, 'hybridlink_lr': LR, 'source': str(source)})
    write_json(checkpoint / 'link_calibration.json', calibration)
    check_reference_config(checkpoint, source)
    if json.loads((checkpoint / 'hybridlink_config.json').read_text()) != config:
        raise RuntimeError('saved link config differs')
    after = measure_all('after')
    if args.link_only and any(a[mode] != b[mode] for a, b in zip(before, after)
                              for mode in ('plain', 'format')):
        raise RuntimeError('frozen text-only generation changed; preserve the saved update')
    check_frozen()
    reload_adapters(raw, checkpoint, link_only=args.link_only)
    # PEFT reload copies frozen anchor deltas too. Their versions legitimately
    # increment; verify exact anchor values before refreshing ONLY those versions.
    check_frozen(check_versions=False)
    for n, param in raw.named_parameters():
        if n in anchors and not args.link_only:
            frozen[n] = param._version
    check_frozen()
    reloaded = measure_all('reloaded', after)
    if after != reloaded:
        raise RuntimeError('saved LoRA+Link replay differs; preserve all reports')
    check_frozen()
    if (file_hashes(source) != hashes or sha256(data) != TRAIN_SHA256
            or sha256(args.calibration) != calibration['source_calibration_sha256']
            or any(file_hashes(k) != v for k, v in protected.items())):
        raise RuntimeError('source checkpoint or training data changed')
    torch.cuda.synchronize()
    summary = {
        'optimizer_steps': 1, 'learning_rate_lora': None if args.link_only else LR, 'learning_rate_link': LR,
        'link_only': args.link_only, 'trainable_parameters': expected_parameters,
        'original_16_controls_exact': True if args.link_only else None,
        'frozen_lora_values_unchanged': True if args.link_only else None,
        'protected_files_unchanged': True,
        'changed_tensors': changed, 'anchors_values_unchanged': True,
        'frozen_original_weights_no_grad_versions_unchanged': True,
        'saved_lora_link_replay_exact': True, 'source_hashes_unchanged': True,
        'checkpoint': str(checkpoint), 'saved_link_sha256': sha256(checkpoint / 'adapter.pt'),
        'saved_lora_sha256': sha256(checkpoint / 'lora/adapter_model.safetensors'),
        'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'peak_reserved_gpu_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'elapsed_seconds': time.monotonic() - started,
        'update': compare_records(update, before[:8], after[:8]),
        'observation': compare_records(observe, before[8:], after[8:]),
        'warning': 'All records participated in prior training; not held-out generalization. '
                   'Two-pass loss and six-input predictions do not establish autonomous reasoning. '
                   'Hybrid success is a behavioral score, not proof the vector span completed calculation. '
                   'Frozen original weights checked by versions/no gradients, not full bytewise comparison. '
                   'Reload uses one backbone, not a second base-model load; the Link-only control '
                   'checks frozen PEFT values without copying them back.',
    }
    if reference is not None:
        summary['joint_vs_link_only'] = {
            'update': compare_records(update, reference['after'][:8], after[:8]),
            'observation': compare_records(observe, reference['after'][8:], after[8:]),
            'before_label': 'joint_after', 'after_label': 'link_only_after',
            'joint_clip_norm': reference['report']['total_norm_before_clip'],
            'link_only_clip_norm': report['total_norm_before_clip'],
            'warning': 'Same clip limit, different parameter sets; not equal Link displacement. '
                       'This is a single-update comparison, not isolated LoRA causality.',
        }
    write_json(output / 'summary.json', summary)
    # Full per-position and raw generation evidence stays in JSONL/summary.
    for group in ('update', 'observation'):
        metrics = summary[group]
        print(group, json.dumps({k: metrics[k] for k in ('two_pass', 'generation')}, ensure_ascii=False), flush=True)
        for path in ('full', 'cached'):
            result = metrics[path]
            regressions = [r for r in result['rows'] if any(d > 1e-6 for d in r['delta_step_ce']) or
                           any(a == t and b != t for a, b, t in zip(r['before']['predicted_ids'],
                               r['after']['predicted_ids'], r['before']['target_ids']))]
            print(group, path, json.dumps({**{k: v for k, v in result.items() if k != 'rows'},
                                          'regressions': regressions}, ensure_ascii=False), flush=True)
    if reference is not None:
        for group in ('update', 'observation'):
            paired = summary['joint_vs_link_only'][group]
            print('joint_after_vs_link_only_after', group,
                  json.dumps({k: v for k, v in paired.items() if k not in ('full', 'cached', 'rows')},
                             ensure_ascii=False), flush=True)
            for path in ('full', 'cached'):
                print('joint_after_vs_link_only_after', group, path,
                      json.dumps({k: v for k, v in paired[path].items() if k != 'rows'},
                                 ensure_ascii=False), flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in
                     ('update', 'observation', 'changed_tensors', 'joint_vs_link_only')},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
