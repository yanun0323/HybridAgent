"""One fixed eight-record, three-input update; compare full-prefix and cached CE."""
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
from link_joint_step import select_rows, comparison
from link_three_step_probe import select_record, gradient_report
from link_execution_probe import cached_rollout, measurements
from latent_diagnostic import span_bounds
from latent_experiment import check_reference_config

LR = 5e-6


def accumulate(model, batches, stage, backward, tokenizer):
    if len(batches) != 8:
        raise ValueError('requires exactly eight update records')
    results = []
    for i, batch in enumerate(batches):
        out = model(**batch, stage=stage, link_steps=3)
        metric, _ = measurements(out, tokenizer)
        backward(out['loss'] / 8)
        results.append(metric)
        del out
        print(f'Backward {i + 1}/8: mean_ce={metric["mean_ce"]:.8g}', flush=True)
    return results


def compare_group(rows, before, after):
    if not rows or len(rows) != len(before) or len(rows) != len(after):
        raise ValueError('comparison row counts differ')
    for a, b in zip(before, after):
        for key in ('target_ids', 'target_positions', 'substitution_positions'):
            if a[key] != b[key]:
                raise ValueError('objective positions changed')
        if any(len(x[key]) != 3 for x in (a, b) for key in
               ('step_ce', 'predicted_ids', 'target_ids', 'target_positions', 'substitution_positions')):
            raise ValueError('expected three positions per record')
    def scalar(metric, step=None):
        return {'ce': sum(metric['step_ce']) / 3 if step is None else metric['step_ce'][step],
                'target_id': metric['target_ids'], 'target_position': metric['target_positions'],
                'substitution_position': metric['substitution_positions']}
    mean = comparison(rows, [scalar(x) for x in before], [scalar(x) for x in after])
    steps = [comparison(rows, [scalar(x, i) for x in before], [scalar(x, i) for x in after]) for i in range(3)]
    for result in (mean, *steps):
        result.pop('rows')
    return {'mean_ce': mean, 'steps': steps,
            'all_three_correct_before': sum(a['predicted_ids'] == a['target_ids'] for a in before),
            'all_three_correct_after': sum(a['predicted_ids'] == a['target_ids'] for a in after),
            'rows': [{'id': r['id'], 'question': r['question'], 'before': a, 'after': b,
                      'delta_step_ce': [y - x for x, y in zip(a['step_ce'], b['step_ce'])]}
                     for r, a, b in zip(rows, before, after)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    source, output = Path(args.model).resolve(), Path(args.output_dir).resolve()
    for protected in (source, Path(args.pilot).resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('output must be separate from source and data')
    if output.exists():
        raise FileExistsError('output exists; no resume or overwrite')
    data, _ = verify_data(args.pilot, 'train')
    rows = [json.loads(s) for s in data.read_text().splitlines() if s.strip()]
    select_record(rows)  # validate 64 unique training records
    update, observe = select_rows(rows)
    rows = update + observe
    source_args = json.loads((source / 'train_args.json').read_text())
    if source_args['stage'] != 'hybridlink_activation' or source_args.get('max_steps') != 64:
        raise ValueError('start from original 64-step activation checkpoint')

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_agent.lora import is_lora_model
    from hybrid_sft.calibration import read_training_calibration
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.model import HybridSFTModel, LINK_NEXT_TOKEN
    from hybrid_sft.train import ADAM_BETAS, WEIGHT_DECAY, MAX_GRAD_NORM

    calibration = read_training_calibration(args.calibration, data)
    config = json.loads((source / 'hybridlink_config.json').read_text())
    if config['output_scale'] != calibration['output_scale']:
        raise ValueError('source scale differs from training calibration')
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    if (accelerator.device.type != 'cuda' or accelerator.num_processes != 1
            or str(accelerator.distributed_type).endswith('DEEPSPEED')):
        raise ValueError('requires one CUDA process without DeepSpeed')
    set_seed(6064)
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
    hashes = {name: sha256(source / name) for name in ('adapter.pt', 'lora/adapter_model.safetensors')}
    output.mkdir(parents=True)
    run = {'source': str(source), 'source_hashes': hashes, 'data_sha256': sha256(data),
           'update_ids': [r['id'] for r in update], 'observation_ids': [r['id'] for r in observe],
           'input_ids_sha256': [hashlib.sha256(json.dumps(s['input_ids']).encode()).hexdigest() for s in samples],
           'calibration': calibration, 'seed': 6064, 'learning_rate': LR,
           'gradient_average_count': 8, 'link_steps': 3, 'planned_optimizer_steps': 1,
           'adam_betas': ADAM_BETAS, 'weight_decay': WEIGHT_DECAY, 'max_grad_norm': MAX_GRAD_NORM,
           'torch': torch.__version__,
           'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write_json(output / 'run.json', run)
    print('Update IDs:', run['update_ids'], flush=True)
    print('Observation-only IDs:', run['observation_ids'], flush=True)
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model) or model.lora_anchor_ids != [entry, exit]:
        raise ValueError('expected matching LoRA anchors')
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(LINK_NEXT_TOKEN)
    model.output_scale = calibration['output_scale']
    optimizer = torch.optim.AdamW(model.hybridlink.parameters(), lr=LR,
                                 betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()
    raw = accelerator.unwrap_model(model)
    versions = {n: p._version for n, p in raw.backbone.named_parameters()}
    link_before = {k: v.detach().cpu().clone() for k, v in raw.hybridlink.state_dict().items()}
    batches = [{k: v.to(accelerator.device) for k, v in b.items()} for b in batches]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def measure_full(batch):
        out = model(**batch, stage=LINK_NEXT_TOKEN, link_steps=3)
        result, _ = measurements(out, tokenizer)
        del out  # keep only plain metrics, never retain graphs between rows
        return result

    def measure_cached(sample):
        with torch.no_grad(), accelerator.autocast():
            out = cached_rollout(raw, sample, entry, exit)
            result, _ = measurements(out, tokenizer)
            del out
        return result

    before = {'cached': [measure_cached(s) for s in samples]}
    observed_before = [measure_full(b) for b in batches[8:]]
    optimizer.zero_grad(set_to_none=True)
    update_before = accumulate(model, batches[:8], LINK_NEXT_TOKEN, accelerator.backward, tokenizer)
    before['full'] = update_before + observed_before
    write_json(output / 'before.json', before)
    grads = gradient_report(raw.hybridlink.named_parameters())
    if any(p.requires_grad or p.grad is not None or p._version != versions[n]
           for n, p in raw.backbone.named_parameters()):
        raise RuntimeError('backbone/LoRA/anchors not frozen')
    norm = accelerator.clip_grad_norm_(raw.hybridlink.parameters(), MAX_GRAD_NORM)
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite clipping norm')
    write_json(output / 'gradients.json', {'norms_before_clip': grads, 'total_norm_before_clip': norm.item(),
                                         'learning_rate': optimizer.param_groups[0]['lr']})
    optimizer.step()  # exactly once; no scheduler, no optimizer resume
    optimizer.zero_grad(set_to_none=True)
    updated = [k for k, v in raw.hybridlink.state_dict().items() if not torch.equal(v.detach().cpu(), link_before[k])]
    if not updated or any(not torch.isfinite(p).all() for p in raw.hybridlink.parameters()):
        raise RuntimeError('HybridLink unchanged or nonfinite after update')
    del link_before
    after = {'full': [measure_full(b) for b in batches],
             'cached': [measure_cached(s) for s in samples]}
    write_json(output / 'after.json', after)
    if any(p.requires_grad or p.grad is not None or p._version != versions[n]
           for n, p in raw.backbone.named_parameters()):
        raise RuntimeError('backbone/LoRA/anchors changed')
    checkpoint = output / 'checkpoint'
    raw.save_pretrained(checkpoint, tokenizer)
    write_json(checkpoint / 'train_args.json', {'stage': 'link_next_token', 'link_steps': 3,
               'max_steps': 1, 'grad_accum': 8, 'batch_size': 1, 'hybridlink_lr': LR, 'source': str(source)})
    write_json(checkpoint / 'link_calibration.json', calibration)
    check_reference_config(checkpoint, source)
    if json.loads((checkpoint / 'hybridlink_config.json').read_text()) != config:
        raise RuntimeError('saved link config differs')
    a = load_file(str(source / 'lora/adapter_model.safetensors'))
    b = load_file(str(checkpoint / 'lora/adapter_model.safetensors'))
    if a.keys() != b.keys() or any(a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k]) for k in a):
        raise RuntimeError('saved LoRA/anchor tensors changed')
    del a, b
    saved = torch.load(checkpoint / 'adapter.pt', map_location='cpu', weights_only=True)
    live = raw.hybridlink.state_dict()
    if saved.keys() != live.keys() or any(saved[k].dtype != torch.float32 or not torch.equal(saved[k], live[k].detach().cpu()) for k in saved):
        raise RuntimeError('saved link tensor mismatch')
    raw.hybridlink.load_state_dict(saved, strict=True)
    del saved, live
    # Replay all sixteen records after reloading saved link tensors; backbone is
    # not reloaded here. Saved LoRA/config equality was checked above.
    reloaded = {'full': [measure_full(b) for b in batches],
                'cached': [measure_cached(s) for s in samples]}
    if reloaded != after:
        write_json(output / 'reload-mismatch.json', reloaded)
        raise RuntimeError('saved-link replay differs from post-update metrics')
    if any(sha256(source / name) != value for name, value in hashes.items()):
        raise RuntimeError('source files changed')
    summary = {'optimizer_steps': 1, 'gradient_average_count': 8, 'link_steps': 3,
               'learning_rate': LR, 'scale': raw.output_scale, 'updated_link_tensors': updated,
               'lora_anchors_unchanged': True, 'saved_link_replay_exact': True,
               'checkpoint': str(checkpoint), 'saved_link_sha256': sha256(checkpoint / 'adapter.pt'),
               'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
               'elapsed_seconds': time.monotonic() - started,
               'update': {path: compare_group(update, before[path][:8], after[path][:8]) for path in before},
               'observation': {path: compare_group(observe, before[path][8:], after[path][8:]) for path in before},
               'warning': 'Three-position training-set CE, not complete answers, exits or generalization. '
                          'Observation rows participated in earlier training. Full-prefix grad-enabled vs '
                          'cached no-grad, both BF16 autocast. No claim that longer rollout improves.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
