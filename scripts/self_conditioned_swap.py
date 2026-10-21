"""Four fixed saved LoRA/Link combinations. Evaluate only; never train or export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_experiment import ROOT, verify_data, sha256, write_json
from lora_self_conditioned_probe import validate_paths, SOURCE_LINK_SHA256, SOURCE_LORA_SHA256
from lora_self_conditioned_joint import (
    TRAIN_SHA256, LR, file_hashes, training_plan, verify_joint_reference,
    measure_record, compare_records,
)
from lora_self_conditioned_evaluate import (
    SAVED_LINK_SHA256, SAVED_LORA_SHA256, require_replay, verify_calibration,
)
from latent_experiment import check_reference_config
from latent_diagnostic import span_bounds

CONTROL_COMMIT = '93b0e6aa46b417f0d1445329da5b713a5c0fe8f4'
CONTROL_LINK_SHA256 = 'ffd5f8f402438cf2ce896907478b5ddfe752ccd935f15f585005729892a9185e'
# (LoRA origin, Link origin); never change selection based on outcomes.
PAIRS = {'source': ('source', 'source'), 'B': ('source', 'control'),
         'C': ('joint', 'joint'), 'A': ('source', 'joint'), 'D': ('joint', 'control')}
VERSIONS = ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def verify_control(control, reference, source_hashes, update, observe, joint, evaluation):
    """Accept only the completed, recorded single Link-only control."""
    run, report, summary = [read_json(control / name) for name in ('run.json', 'update.json', 'summary.json')]
    original = reference['run']
    fixed = ('data_sha256', 'source_hashes', 'update_ids', 'observation_ids', 'tokenization_sha256',
             'calibration', 'seed', 'planned_optimizer_steps', 'gradient_average_count',
             'learning_rate_link', 'betas', 'weight_decay', 'max_grad_norm', 'shuffle', 'scheduler',
             'anchors_frozen', 'hybrid_limits', 'text_budgets', 'recursive_steps', 'precision', *VERSIONS)
    if any(run[k] != original[k] for k in fixed):
        raise ValueError('Link-only plan differs from joint reference')
    if (run['code_commit'] != CONTROL_COMMIT or run['link_only'] is not True or
            run['learning_rate_lora'] is not None or run['source_hashes'] != source_hashes):
        raise ValueError('Link-only run provenance differs')
    for path, key in ((joint, 'joint_run'), (evaluation, 'joint_evaluation')):
        if Path(run[key]).resolve() != path.resolve():
            raise ValueError('Link-only reference path differs')
        if run['protected_run_hashes'][str(path.resolve())] != file_hashes(path):
            raise ValueError('Link-only reference files differ')
    expected = {'optimizer_steps': 1, 'link_only': True, 'learning_rate_lora': None,
                'learning_rate_link': LR, 'trainable_parameters': 52459520,
                'saved_link_sha256': CONTROL_LINK_SHA256, 'saved_lora_sha256': SOURCE_LORA_SHA256,
                **dict.fromkeys(('original_16_controls_exact', 'frozen_lora_values_unchanged',
                    'protected_files_unchanged', 'anchors_values_unchanged',
                    'frozen_original_weights_no_grad_versions_unchanged',
                    'saved_lora_link_replay_exact', 'source_hashes_unchanged'), True)}
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError('Link-only summary did not pass all checks')
    before, after, reloaded = [read_rows(control / name) for name in
                               ('before.jsonl', 'after.jsonl', 'reloaded.jsonl')]
    require_replay(before, reference['before'], 'Link-only before')
    require_replay([r['id'] for r in after], [r['id'] for r in before], 'Link-only ordered IDs')
    require_replay(after, reloaded, 'Link-only reload')
    if (report['optimizer_steps'] != 1 or report['lora_gradient_norms'] != {} or
            report['pre_update_losses'] != [r['two_pass'] for r in before[:8]]):
        raise ValueError('Link-only update report differs')
    for group, rows, sl in (('update', update, slice(0, 8)), ('observation', observe, slice(8, 16))):
        require_replay(summary[group], compare_records(rows, before[sl], after[sl]), f'Link-only {group}')
        require_replay(summary['joint_vs_link_only'][group],
                       compare_records(rows, reference['after'][sl], after[sl]), f'paired {group}')
    return after


def install_pair(raw, lora_path, link_path, lora_sha, link_sha):
    """Validate both files before changing adapters; clear/reload even identical pairs."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    if any(p.grad is not None for p in raw.parameters()):
        raise RuntimeError('unexpected pending gradient')
    if sha256(lora_path) != lora_sha or sha256(link_path) != link_sha:
        raise ValueError('selected weight hash differs')
    saved = load_file(str(lora_path))
    link = torch.load(link_path, map_location='cpu', weights_only=True)
    live = get_peft_model_state_dict(raw.backbone, save_embedding_layers=False)
    for disk, current in ((saved, live), (link, raw.hybridlink.state_dict())):
        if disk.keys() != current.keys() or any(
                disk[k].shape != current[k].shape or disk[k].dtype != current[k].dtype or
                not torch.isfinite(disk[k]).all() for k in disk):
            raise ValueError('adapter keys, shapes, dtypes or values invalid')
    if any(not ('lora_' in k or 'trainable_tokens_delta' in k) for k in saved):
        raise ValueError('non-adapter backbone weight in checkpoint')
    anchors = [k for k in saved if 'trainable_tokens_delta' in k]
    if len(anchors) != 2 or any(not torch.equal(saved[k], live[k].detach().cpu()) for k in anchors):
        raise ValueError('anchor values differ')
    adapters = [p for n, p in raw.backbone.named_parameters() if '.lora_A.' in n or '.lora_B.' in n]
    if not adapters:
        raise ValueError('no LoRA tensors found')
    with torch.no_grad():
        for p in adapters:
            p.zero_()
        for p in raw.hybridlink.parameters():
            p.zero_()
    loaded = set_peft_model_state_dict(raw.backbone, saved, adapter_name='default')
    if loaded.unexpected_keys:
        raise RuntimeError(f'unexpected PEFT keys: {loaded.unexpected_keys}')
    raw.hybridlink.load_state_dict(link, strict=True)
    for disk, current in ((saved, get_peft_model_state_dict(raw.backbone, save_embedding_layers=False)),
                          (link, raw.hybridlink.state_dict())):
        if disk.keys() != current.keys() or any(disk[k].dtype != current[k].dtype or
                not torch.equal(disk[k], current[k].detach().cpu()) for k in disk):
            raise RuntimeError('installed adapters differ from disk')


def run_pairs(install, measure, controls):
    results = {}
    # All three controls must pass before measuring either unseen combination.
    for label in ('source', 'B', 'C'):
        install(label)
        results[label] = measure(label, controls[label])
        require_replay(results[label], controls[label], label)
    for label in ('A', 'D'):
        install(label)
        results[label] = measure(label, None)
        install(label)  # clear both small adapters, then read both from disk again
        require_replay(measure(label + '_reloaded', results[label]), results[label], label + ' reload')
    return results


def comparisons(update, observe, results):
    pairs = {'lora_with_joint_link': ('A', 'C'), 'lora_with_control_link': ('B', 'D'),
             'link_with_original_lora': ('A', 'B'), 'link_with_joint_lora': ('C', 'D')}
    return {name: {'before_label': a, 'after_label': b,
                  'update': compare_records(update, results[a][:8], results[b][:8]),
                  'observation': compare_records(observe, results[a][8:], results[b][8:])}
            for name, (a, b) in pairs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'model', 'calibration', 'joint-run', 'joint-evaluation', 'control-run', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source, pilot, joint, evaluation, control, output = [Path(v).resolve() for v in
        (args.model, args.pilot, args.joint_run, args.joint_evaluation, args.control_run, args.output_dir)]
    for path in (pilot, joint, evaluation, control):
        validate_paths(source, path, output)
    data, _ = verify_data(pilot, 'train')
    if sha256(data) != TRAIN_SHA256:
        raise ValueError('requires original training data')
    protected = {p: file_hashes(p) for p in (source, joint, evaluation, control)}
    checkpoints = {'source': source, 'joint': joint / 'checkpoint', 'control': control / 'checkpoint'}
    lora_hashes = {'source': SOURCE_LORA_SHA256, 'joint': SAVED_LORA_SHA256, 'control': SOURCE_LORA_SHA256}
    link_hashes = {'source': SOURCE_LINK_SHA256, 'joint': SAVED_LINK_SHA256, 'control': CONTROL_LINK_SHA256}
    for label, checkpoint in checkpoints.items():
        if (sha256(checkpoint / 'adapter.pt') != link_hashes[label] or
                sha256(checkpoint / 'lora/adapter_model.safetensors') != lora_hashes[label]):
            raise ValueError(f'weight provenance differs: {label}')
        check_reference_config(checkpoint, source)
        if read_json(checkpoint / 'hybridlink_config.json') != read_json(source / 'hybridlink_config.json'):
            raise ValueError('Link configuration differs')
    settings = read_json(control / 'checkpoint/train_args.json')
    for k, v in {'stage': 'self_conditioned', 'lora_self_conditioned': False,
                  'link_only_self_conditioned': True, 'max_steps': 1, 'grad_accum': 8,
                  'batch_size': 1, 'lr': LR, 'hybridlink_lr': LR}.items():
        if settings.get(k) != v:
            raise ValueError(f'Link-only checkpoint settings differ: {k}')
    update, observe, _ = training_plan(read_rows(data))
    rows = update + observe
    reference = verify_joint_reference(joint, evaluation, protected[source], update, observe)
    control_after = verify_control(control, reference, protected[source], update, observe, joint, evaluation)
    plan = reference['run']

    import torch
    from importlib.metadata import version
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoTokenizer
    from hybrid_agent.compat import check_execution
    from hybrid_agent.lora import is_lora_model
    from hybrid_sft.model import HybridSFTModel, SELF_CONDITIONED
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.calibration import read_training_calibration

    versions = {k: version(k) for k in VERSIONS}
    if any(v != plan[k] for k, v in versions.items()):
        raise ValueError('library versions differ')
    calibration = read_training_calibration(args.calibration, data)
    scale = read_json(source / 'hybridlink_config.json')['output_scale']
    for label in ('joint', 'control'):
        verify_calibration(checkpoints[label] / 'link_calibration.json', plan['calibration'], calibration, scale)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    if (accelerator.device.type != 'cuda' or accelerator.num_processes != 1 or
            str(accelerator.distributed_type).endswith('DEEPSPEED')):
        raise ValueError('requires one CUDA process without DeepSpeed')
    set_seed(plan['seed'])
    tokenizers = [AutoTokenizer.from_pretrained(p, local_files_only=True) for p in checkpoints.values()]
    for tok in tokenizers:
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
    tokenizer = tokenizers[0]
    entry, exit = [tokenizer.convert_tokens_to_ids(t) for t in ('<latent>', '</latent>')]
    samples, batches = [], []
    for row in rows:
        encoded = [tokenize_trajectory(tok, row['system'], row['question'], row['rounds'], 1024)
                   for tok in tokenizers]
        sample = encoded[0]
        if sample is None or any(s != sample for s in encoded):
            raise ValueError('saved/source tokenization differs')
        start, end = span_bounds(sample, entry, exit)
        if end - start - 1 < 6:
            raise ValueError('requires six content tokens')
        samples.append(sample)
        batches.append(collate_batch(tokenizer, [sample], pad_to_multiple=1))
    token_hashes = [hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for s in samples]
    require_replay(token_hashes, plan['tokenization_sha256'], 'tokenization')
    output.mkdir(parents=True)
    write_json(output / 'run.json', {
        'pairs': PAIRS, 'protected_hashes': {str(p): v for p, v in protected.items()},
        'lora_sha256': lora_hashes, 'link_sha256': link_hashes,
        'original_plan': plan, 'data_sha256': TRAIN_SHA256, 'calibration': calibration,
        'tokenization_sha256': token_hashes, 'optimizer_steps': 0, 'weights_saved': False,
        'sequence': ['source', 'B', 'C', 'A', 'A_reloaded', 'D', 'D_reloaded'],
        'measurement_mode': 'joint grad-enabled forward for all pairs; no backward; exact B replay required',
        **versions, 'gpu': torch.cuda.get_device_name(accelerator.device),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    })
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model) or model.lora_anchor_ids != [entry, exit]:
        raise ValueError('expected matching LoRA checkpoint')
    check_execution(model.backbone, batch_size=1)
    model.resize_to_tokenizer(tokenizer)
    # Same flags for every combination, so LoRA swaps do not change execution path.
    # These flags preserve the measured forward path only; never call backward.
    model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = accelerator.prepare(model)
    raw = accelerator.unwrap_model(model)
    raw.zero_grad(set_to_none=True)
    batches = [{k: v.to(accelerator.device) for k, v in b.items()} for b in batches]
    anchors = {n: p.detach().cpu().clone() for n, p in raw.named_parameters() if 'trainable_tokens_delta' in n}
    if len(anchors) != 2:
        raise RuntimeError('expected two frozen anchor tensors')
    base_versions = {n: p._version for n, p in raw.named_parameters() if not p.requires_grad and n not in anchors}
    flags = {n: p.requires_grad for n, p in raw.named_parameters()}
    def check_original():
        for n, p in raw.named_parameters():
            if p.grad is not None or p.requires_grad != flags[n]:
                raise RuntimeError(f'gradient or parameter flag changed: {n}')
            if n in base_versions and p._version != base_versions[n]:
                raise RuntimeError(f'original backbone changed: {n}')
            if n in anchors and not torch.equal(p.detach().cpu(), anchors[n]):
                raise RuntimeError(f'anchor changed: {n}')
    def install(label):
        lora, link = PAIRS[label]
        install_pair(raw, checkpoints[lora] / 'lora/adapter_model.safetensors',
                     checkpoints[link] / 'adapter.pt', lora_hashes[lora], link_hashes[link])
        check_original()
    text_controls = {'source': reference['before'], 'A': reference['before'], 'B': reference['before'],
                     'C': reference['after'], 'D': reference['after']}
    def measure(label, controls):
        results = []
        before_versions = {n: p._version for n, p in raw.named_parameters()}
        with (output / f'{label}.jsonl').open('x') as stream:
            for i, (row, sample, batch) in enumerate(zip(rows, samples, batches)):
                item = measure_record(model, raw, accelerator, tokenizer, row, sample, batch, entry, exit)
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                stream.flush()
                if controls is not None:
                    require_replay(item, controls[i], f'{label}/{row["id"]}')
                for mode in ('plain', 'format'):
                    require_replay(item[mode], text_controls[label.removesuffix('_reloaded')][i][mode],
                                   f'{label}/{row["id"]}/{mode}: Link must not affect text-only output')
                check_original()
                if any(p._version != before_versions[n] for n, p in raw.named_parameters()):
                    raise RuntimeError('parameter changed during measurement')
                results.append(item)
                print(f'{label} {row["id"]}: loss={item["two_pass"]["loss"]:.8g} '
                      f'cached6={item["cached"]["mean_ce"]:.8g} '
                      f'natural_exit={item["hybrid"]["natural_exit"]}', flush=True)
        return results
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = run_pairs(install, measure, {'source': reference['before'], 'B': control_after, 'C': reference['after']})
    check_original()
    if (any(file_hashes(p) != v for p, v in protected.items()) or sha256(data) != TRAIN_SHA256 or
            sha256(args.calibration) != calibration['source_calibration_sha256']):
        raise RuntimeError('protected files changed')
    torch.cuda.synchronize()
    paired = comparisons(update, observe, results)
    summary = {
        'optimizer_steps': 0, 'weights_saved': False, 'pairs': PAIRS,
        'original_and_B_C_controls_exact': True, 'A_D_reload_exact': True,
        'anchors_values_unchanged': True, 'original_backbone_versions_no_grad_unchanged': True,
        'protected_files_unchanged': True, 'text_only_same_lora_exact': True,
        'comparisons': paired,
        'versus_source': {label: {'update': compare_records(update, results['source'][:8], results[label][:8]),
                                 'observation': compare_records(observe, results['source'][8:], results[label][8:])}
                          for label in ('A', 'B', 'C', 'D')},
        'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'peak_reserved_gpu_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'elapsed_seconds': time.monotonic() - started,
        'warning': 'Fixed saved-weight substitutions only; not new training, held-out generalization or proof '
                   'of latent reasoning. Each recursive execution path generates its own vectors. '
                   'Original backbone checked by versions/no gradients, not full bytewise comparison.',
    }
    write_json(output / 'summary.json', summary)
    for name, comparison in paired.items():
        print(name, comparison['before_label'], '->', comparison['after_label'], flush=True)
        for group in ('update', 'observation'):
            metrics = comparison[group]
            print(group, json.dumps({k: metrics[k] for k in ('two_pass', 'generation')}, ensure_ascii=False), flush=True)
            for path in ('full', 'cached'):
                result = metrics[path]
                regressions = [r for r in result['rows'] if any(d > 1e-6 for d in r['delta_step_ce']) or
                               any(a == t and b != t for a, b, t in zip(r['before']['predicted_ids'],
                                   r['after']['predicted_ids'], r['before']['target_ids']))]
                print(group, path, json.dumps({**{k: v for k, v in result.items() if k != 'rows'},
                                              'regressions': regressions}, ensure_ascii=False), flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('comparisons', 'versus_source')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
