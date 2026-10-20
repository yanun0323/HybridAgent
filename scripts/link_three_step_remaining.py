"""Compare original and saved three-step update on remaining 48 training records.

Replay the original sixteen controls for BOTH checkpoints before measuring any
remaining record. No backward, optimizer, checkpoint selection or model export.
"""
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
from link_three_step_probe import select_record
from link_three_step_joint import compare_group
from link_execution_probe import cached_rollout, measurements
from latent_diagnostic import span_bounds
from latent_experiment import check_reference_config


def fixed_plan(rows, run, summary):
    select_record(rows)  # exactly 64 unique IDs
    if ([r['id'] for r in rows[:8]] != run['update_ids'] or
            [r['id'] for r in rows[8:16]] != run['observation_ids']):
        raise ValueError('joint-run selection differs from training file order')
    for key, value in (('link_steps', 3), ('gradient_average_count', 8), ('learning_rate', 5e-6)):
        if run[key] != value or summary[key] != value:
            raise ValueError(f'expected fixed three-step joint experiment: {key}')
    if summary['optimizer_steps'] != 1 or not summary['saved_link_replay_exact']:
        raise ValueError('requires a completed, replay-verified single update')
    references = {name: {path: [] for path in ('full', 'cached')} for name in ('before', 'after')}
    for path in ('full', 'cached'):
        saved = summary['update'][path]['rows'] + summary['observation'][path]['rows']
        if [r['id'] for r in saved] != [r['id'] for r in rows[:16]]:
            raise ValueError('saved control IDs differ')
        for name in references:
            references[name][path] = [r[name] for r in saved]
    return rows[16:], references


def verify_control(actual, expected, label):
    # Joint training already verified exact saved-link replay. Keep that same
    # strict requirement here; do not loosen tolerances based on observed losses.
    if actual != expected:
        raise ValueError(f'control replay mismatch: {label}; stop before remaining records')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', required=True)
    p.add_argument('--model', required=True, help='original 64-step checkpoint')
    p.add_argument('--joint-run', required=True, help='completed three-step joint run')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    source, joint, output = (Path(v).resolve() for v in (args.model, args.joint_run, args.output_dir))
    for protected in (source, joint, Path(args.pilot).resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('output must be separate from source artifacts')
    if output.exists():
        raise FileExistsError('output exists; no overwrite or resume')
    updated = joint / 'checkpoint'
    run = json.loads((joint / 'run.json').read_text())
    prior = json.loads((joint / 'summary.json').read_text())
    data, _ = verify_data(args.pilot, 'train')
    if sha256(data) != run['data_sha256']:
        raise ValueError('training data differs from joint experiment')
    rows = [json.loads(s) for s in data.read_text().splitlines() if s.strip()]
    remaining, references = fixed_plan(rows, run, prior)
    hashes = {name: sha256(source / name) for name in run['source_hashes']}
    if hashes != run['source_hashes'] or sha256(updated / 'adapter.pt') != prior['saved_link_sha256']:
        raise ValueError('checkpoint hashes differ from completed joint run')
    settings = json.loads((source / 'train_args.json').read_text())
    if settings['stage'] != 'hybridlink_activation' or settings.get('max_steps') != 64:
        raise ValueError('requires original 64-step activation checkpoint')
    check_reference_config(updated, source)
    config = json.loads((source / 'hybridlink_config.json').read_text())
    if config != json.loads((updated / 'hybridlink_config.json').read_text()):
        raise ValueError('link configs differ')
    scale = config['output_scale']
    if scale != run['calibration']['output_scale'] or scale != prior['scale']:
        raise ValueError('recorded scales differ')

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_agent.lora import is_lora_model
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.model import HybridSFTModel, LINK_NEXT_TOKEN

    a = load_file(str(source / 'lora/adapter_model.safetensors'))
    b = load_file(str(updated / 'lora/adapter_model.safetensors'))
    if a.keys() != b.keys() or any(a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k]) for k in a):
        raise ValueError('LoRA/anchor weights differ')
    del a, b
    states = {name: torch.load(path / 'adapter.pt', map_location='cpu', weights_only=True)
              for name, path in (('before', source), ('after', updated))}
    if states['before'].keys() != states['after'].keys():
        raise ValueError('HybridLink tensor names differ')
    for name, state in states.items():
        for key, value in state.items():
            if (value.dtype != torch.float32 or not torch.isfinite(value).all() or
                    value.shape != states['before'][key].shape):
                raise ValueError(f'invalid FP32 link state: {name}/{key}')
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    if (accelerator.device.type != 'cuda' or accelerator.num_processes != 1
            or str(accelerator.distributed_type).endswith('DEEPSPEED')):
        raise ValueError('requires one CUDA process without DeepSpeed')
    set_seed(run['seed'])
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    entry, exit = [tokenizer.convert_tokens_to_ids(t) for t in ('<latent>', '</latent>')]
    samples, batches = [], []
    for row in rows:
        sample = tokenize_trajectory(tokenizer, row['system'], row['question'], row['rounds'], 1024)
        if sample is None:
            raise ValueError(f'rejected record {row["id"]}')
        start, end = span_bounds(sample, entry, exit)
        if end - start - 1 < 3:
            raise ValueError('requires three content tokens')
        samples.append(sample)
        batches.append(collate_batch(tokenizer, [sample]))
    input_hashes = [hashlib.sha256(json.dumps(s['input_ids']).encode()).hexdigest() for s in samples]
    if input_hashes[:16] != run['input_ids_sha256']:
        raise ValueError('control tokenization differs from joint run')
    output.mkdir(parents=True)
    write_json(output / 'run.json', {
        'source': str(source), 'joint_run': str(joint), 'source_hashes': hashes,
        'updated_link_sha256': prior['saved_link_sha256'],
        'joint_run_sha256': sha256(joint / 'run.json'), 'joint_summary_sha256': sha256(joint / 'summary.json'),
        'data_sha256': sha256(data), 'remaining_ids': [r['id'] for r in remaining],
        'control_ids': [r['id'] for r in rows[:16]], 'input_ids_sha256': input_hashes,
        'scale': scale, 'seed': run['seed'], 'torch': torch.__version__,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'control_policy': 'Exact metrics for both paths and both saved states before remaining rows',
    })
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model) or model.lora_anchor_ids != [entry, exit]:
        raise ValueError('expected matching LoRA anchors')
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(LINK_NEXT_TOKEN)
    model.output_scale = scale
    model = accelerator.prepare(model)
    model.train()
    raw = accelerator.unwrap_model(model)
    versions = {n: p._version for n, p in raw.backbone.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    collected = {name: {path: [] for path in ('full', 'cached')} for name in states}
    with (output / 'measurements.jsonl').open('x', encoding='utf-8') as stream:
        # Both sets of controls complete before any remaining-row measurement.
        for phase, indices in (('controls', range(16)), ('remaining', range(16, 64))):
            for name, state in states.items():
                raw.hybridlink.load_state_dict(state, strict=True)
                for i in indices:
                    batch = {k: v.to(accelerator.device) for k, v in batches[i].items()}
                    out = model(**batch, stage=LINK_NEXT_TOKEN, link_steps=3)
                    full, _ = measurements(out, tokenizer)
                    del out  # free graph before cached path or next record
                    with torch.no_grad(), accelerator.autocast():
                        out = cached_rollout(raw, samples[i], entry, exit)
                        cached, _ = measurements(out, tokenizer)
                        del out
                    item = {'id': rows[i]['id'], 'phase': phase, 'state': name, 'full': full, 'cached': cached}
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                    stream.flush()
                    for path in ('full', 'cached'):
                        if phase == 'controls':
                            verify_control(item[path], references[name][path][i], f'{name}/{path}/{rows[i]["id"]}')
                        else:
                            collected[name][path].append(item[path])
                    if phase == 'remaining':
                        print(f'{name} {i - 15}/48 {rows[i]["id"]}: full={full["mean_ce"]:.8g} cached={cached["mean_ce"]:.8g}', flush=True)
                if any(p.grad is not None for p in raw.parameters()):
                    raise RuntimeError('unexpected gradients')
                if any(not torch.equal(v.detach().cpu(), state[k]) for k, v in raw.hybridlink.state_dict().items()):
                    raise RuntimeError('link changed during measurements')
                print(f'{phase} {name}: complete' + ('; exact replay verified' if phase == 'controls' else ''), flush=True)
    if any(p.requires_grad or p.grad is not None or p._version != versions[n] for n, p in raw.backbone.named_parameters()):
        raise RuntimeError('backbone/LoRA/anchors changed')
    if (any(sha256(source / name) != value for name, value in hashes.items()) or
            sha256(updated / 'adapter.pt') != prior['saved_link_sha256']):
        raise RuntimeError('checkpoint files changed')
    summary = {'count': 48, 'controls_exact': True, 'optimizer_steps': 0, 'weights_saved': False,
               'backbone_unchanged': True, 'source_hashes_unchanged': True,
               'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
               'elapsed_seconds': time.monotonic() - started,
               'remaining_train': {path: compare_group(remaining, collected['before'][path], collected['after'][path])
                                   for path in ('full', 'cached')},
               'warning': 'Three-position training-set comparison only. These 48 records participated in earlier training, '
                          'not the eight-record joint update. Not held-out generalization, complete answers or natural exits.'}
    write_json(output / 'summary.json', summary)
    # Full per-record reports stay on disk; print concise totals plus regressions.
    compact = {k: v for k, v in summary.items() if k != 'remaining_train'}
    compact['remaining_train'] = {}
    for path, result in summary['remaining_train'].items():
        compact['remaining_train'][path] = {k: v for k, v in result.items() if k != 'rows'}
        compact['remaining_train'][path]['regressions'] = [
            r for r in result['rows'] if any(d > 1e-6 for d in r['delta_step_ce']) or
            any(a == t and b != t for a, b, t in zip(r['before']['predicted_ids'], r['after']['predicted_ids'], r['before']['target_ids']))]
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
