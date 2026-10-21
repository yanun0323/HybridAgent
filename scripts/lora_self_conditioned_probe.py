"""One fixed training trajectory, official two-pass objective, backward only.

LoRA A/B + FP32 HybridLink receive gradients. Original weights and anchor deltas
remain frozen. No optimizer, model export, evaluation data or automatic retries.
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
from link_three_step_probe import select_record, gradient_report

SOURCE_LINK_SHA256 = '63e175ff8c49a09c3ffe0c5dd2e35b50b89cddac9e1acd9033afb77d43f935df'
SOURCE_LORA_SHA256 = 'bf67b117cef645cb14af97924ac72c19ea355440d7b5b0170529a5ed14dae48a'


def validate_paths(source, pilot, output):
    for protected in (source, pilot):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('output must be separate from source and data')
    if output.exists():
        raise FileExistsError('output exists; preserve it')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    source, pilot, output = (Path(v).resolve() for v in (args.model, args.pilot, args.output_dir))
    validate_paths(source, pilot, output)
    data, _ = verify_data(pilot, 'train')
    row = select_record([json.loads(s) for s in data.read_text().splitlines() if s.strip()])
    if row['id'] != 'mul-sub-54-5-6':
        raise ValueError('unexpected first training record')
    source_args = json.loads((source / 'train_args.json').read_text())
    if source_args['stage'] != 'hybridlink_activation' or source_args.get('max_steps') != 64:
        raise ValueError('start from the original 64-step activation checkpoint')
    source_hashes = {str(f.relative_to(source)): sha256(f) for f in sorted(source.rglob('*')) if f.is_file()}
    if (source_hashes.get('adapter.pt') != SOURCE_LINK_SHA256 or
            source_hashes.get('lora/adapter_model.safetensors') != SOURCE_LORA_SHA256):
        raise ValueError('source weights differ from the original experiment')

    import torch
    import transformers
    import peft
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoTokenizer
    from hybrid_agent.compat import check_execution
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_agent.lora import is_lora_model
    from hybrid_sft.calibration import read_training_calibration
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.model import HybridSFTModel, SELF_CONDITIONED

    calibration = read_training_calibration(args.calibration, data)
    if json.loads((source / 'hybridlink_config.json').read_text())['output_scale'] != calibration['output_scale']:
        raise ValueError('source scale differs from training calibration')
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    if (accelerator.device.type != 'cuda' or accelerator.num_processes != 1
            or str(accelerator.distributed_type).endswith('DEEPSPEED')):
        raise ValueError('requires one CUDA process without DeepSpeed')
    set_seed(6064)
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    add_anchor_tokens(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    sample = tokenize_trajectory(tokenizer, row['system'], row['question'], row['rounds'], 1024)
    if sample is None or not any(sample['latent_mask'][:-1]):
        raise ValueError('training record rejected or has no latent span')
    # Single unpadded trajectory; retain text/exit CE and the original latent mask.
    batch = collate_batch(tokenizer, [sample], pad_to_multiple=1)
    output.mkdir(parents=True)
    write_json(output / 'run.json', {
        'source': str(source), 'source_hashes': source_hashes,
        'data_sha256': sha256(data), 'sample_id': row['id'], 'question': row['question'],
        'sample_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        'tokenization_sha256': hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest(),
        'calibration': calibration, 'seed': 6064, 'optimizer_steps': 0,
        'objective': 'official_two_pass_text_ce_plus_latent_and_exit_regression',
        'trainable': 'LoRA A/B and HybridLink; anchor deltas frozen',
        'gradient_checkpointing': 'non-reentrant decoder layers and CE',
        'n_tokens': sample['n_tokens'], 'latent_sources': sum(sample['latent_mask']),
        'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
        'gpu': torch.cuda.get_device_name(accelerator.device),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    })
    print('Fixed training record:', row['id'], flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model):
        raise ValueError('expected saved LoRA checkpoint')
    check_execution(model.backbone, batch_size=1)
    if [tokenizer.convert_tokens_to_ids(t) for t in ('<latent>', '</latent>')] != model.lora_anchor_ids:
        raise ValueError('anchor IDs differ')
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = accelerator.prepare(model)
    model.train()
    raw = accelerator.unwrap_model(model)
    trainable = {n: p for n, p in raw.named_parameters() if p.requires_grad}
    if not trainable or any(not (n.startswith('hybridlink.') or '.lora_A.' in n or '.lora_B.' in n)
                            for n in trainable):
        raise RuntimeError('unexpected trainable parameters')
    if any(p.dtype != torch.float32 for p in trainable.values()):
        raise RuntimeError('LoRA and Link must be FP32')
    versions = {n: p._version for n, p in raw.named_parameters()}
    small_before = {n: p.detach().cpu().clone() for n, p in raw.named_parameters()
                    if n in trainable or 'trainable_tokens_delta' in n}
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
    raw.zero_grad(set_to_none=True)
    out = model(**batch, stage=SELF_CONDITIONED)
    forward = {k: v.detach().float().item() for k, v in out.items()}
    if not all(torch.isfinite(v).all() for v in out.values()):
        raise RuntimeError('nonfinite self-conditioned loss')
    write_json(output / 'forward.json', forward)
    print(json.dumps(forward), flush=True)
    accelerator.backward(out['loss'])
    # Zero individual gradients are allowed (e.g. freshly initialized LoRA A),
    # but each group must contain a finite nonzero gradient.
    lora_grads = gradient_report((n, p) for n, p in trainable.items() if n.startswith('backbone.'))
    link_grads = gradient_report(raw.hybridlink.named_parameters())
    for n, p in raw.named_parameters():
        if p._version != versions[n]:
            raise RuntimeError(f'parameter version changed: {n}')
        if n not in trainable and (p.requires_grad or p.grad is not None):
            raise RuntimeError(f'frozen parameter received gradients: {n}')
        if n in small_before and not torch.equal(p.detach().cpu(), small_before[n]):
            raise RuntimeError(f'LoRA/Link/anchor value changed: {n}')
    after_hashes = {str(f.relative_to(source)): sha256(f) for f in sorted(source.rglob('*')) if f.is_file()}
    if after_hashes != source_hashes or sha256(data) != json.loads((output / 'run.json').read_text())['data_sha256']:
        raise RuntimeError('source checkpoint or training file changed')
    torch.cuda.synchronize()
    summary = {
        **forward, 'sample_id': row['id'], 'lora_gradient_norms': lora_grads,
        'link_gradient_norms': link_grads,
        'trainable_parameters': sum(p.numel() for p in trainable.values()),
        'frozen_parameters_no_grad': True, 'all_parameter_versions_unchanged': True,
        'lora_link_anchors_values_unchanged': True, 'source_hashes_unchanged': True,
        'optimizer_steps': 0, 'weights_saved': False,
        'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'peak_reserved_gpu_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'elapsed_seconds': time.monotonic() - started,
        'warning': 'Backward feasibility only. Two-pass teacher-conditioned predictions, not free-running '
                   'rollout. No measured learning, cache equivalence, reasoning or natural exit success. '
                   'Frozen backbone uses version/no-gradient checks, not a full bytewise comparison. '
                   'No optimizer state allocated; a later update may require more memory.',
    }
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
