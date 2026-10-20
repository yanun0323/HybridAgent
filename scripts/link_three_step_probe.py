"""One training record, three recursive inputs, backward only. Never update/save weights."""
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


def select_record(rows):
    if len(rows) != 64 or len({r['id'] for r in rows}) != 64:
        raise ValueError('expected 64 unique training records')
    return rows[0]  # fixed file order, never chosen by CE


def gradient_report(named_tensors):
    import torch
    report = {}
    for name, tensor in named_tensors:
        grad = tensor.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f'missing or nonfinite gradient: {name}')
        norm = grad.detach().float().norm().item()
        if not torch.isfinite(torch.tensor(norm)):
            raise RuntimeError(f'nonfinite gradient norm: {name}')
        report[name] = norm
    if not report or not any(v > 0 for v in report.values()):
        raise RuntimeError('all gradients are zero')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    source, output = Path(args.model).resolve(), Path(args.output_dir).resolve()
    for protected in (source, Path(args.pilot).resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('output must be separate from source and data')
    if output.exists():
        raise FileExistsError('output exists; preserve it')
    data, _ = verify_data(args.pilot, 'train')
    row = select_record([json.loads(s) for s in data.read_text().splitlines() if s.strip()])
    source_args = json.loads((source / 'train_args.json').read_text())
    if source_args['stage'] != 'hybridlink_activation' or source_args.get('max_steps') != 64:
        raise ValueError('start from the original 64-step activation checkpoint')

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoTokenizer
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_agent.lora import is_lora_model
    from hybrid_sft.calibration import read_training_calibration
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.model import HybridSFTModel, LINK_NEXT_TOKEN

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
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    sample = tokenize_trajectory(tokenizer, row['system'], row['question'], row['rounds'], 1024)
    if sample is None:
        raise ValueError('training record rejected')
    positions = [i for i, value in enumerate(sample['latent_mask']) if value]
    if len(positions) < 4 or positions[:4] != list(range(positions[0], positions[0] + 4)):
        raise ValueError('requires entry and three contiguous content inputs')
    batch = collate_batch(tokenizer, [sample])
    output.mkdir(parents=True)
    hashes = {'source_link_sha256': sha256(source / 'adapter.pt'),
              'source_lora_sha256': sha256(source / 'lora/adapter_model.safetensors')}
    write_json(output / 'run.json', {
        'source': str(source), **hashes, 'data_sha256': sha256(data),
        'sample_id': row['id'], 'question': row['question'],
        'sample_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        'input_ids_sha256': hashlib.sha256(json.dumps(sample['input_ids']).encode()).hexdigest(),
        'calibration': calibration, 'seed': 6064, 'link_steps': 3, 'optimizer_steps': 0,
        'torch': torch.__version__,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    })
    print('Fixed training record:', row['id'], flush=True)
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model):
        raise ValueError('expected saved LoRA checkpoint')
    if [tokenizer.convert_tokens_to_ids(t) for t in ('<latent>', '</latent>')] != model.lora_anchor_ids:
        raise ValueError('anchor IDs differ')
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(LINK_NEXT_TOKEN)
    model.output_scale = calibration['output_scale']
    model = accelerator.prepare(model)
    model.train()
    raw = accelerator.unwrap_model(model)
    link_before = {k: v.detach().cpu().clone() for k, v in raw.hybridlink.state_dict().items()}
    # Checking tensor versions avoids a second 27B copy; there is no optimizer.
    backbone_versions = {n: p._version for n, p in raw.backbone.named_parameters()}
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
    raw.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    out = model(**batch, stage=LINK_NEXT_TOKEN, link_steps=3)
    if not torch.isfinite(out['step_losses']).all():
        raise RuntimeError('nonfinite three-step loss')
    forward = {'sample_id': row['id'], 'mean_ce': out['loss'].detach().item(),
               'step_ce': out['step_losses'].detach().tolist(),
               'target_ids': out['target_ids'].tolist()[0],
               'target_texts': [tokenizer.decode([t]) for t in out['target_ids'].tolist()[0]],
               'substitution_positions': out['substitution_positions'],
               'target_positions': out['target_positions']}
    write_json(output / 'forward.json', forward)
    print(json.dumps(forward, ensure_ascii=False), flush=True)
    accelerator.backward(out['loss'])
    link_grads = gradient_report(raw.hybridlink.named_parameters())
    vector_grads = gradient_report((str(i + 1), v) for i, v in enumerate(out['rollout_vectors']))
    if not all(v > 0 for v in vector_grads.values()):
        raise RuntimeError('a recursive input received zero gradient')
    if any(p.requires_grad or p.grad is not None or p._version != backbone_versions[n]
           for n, p in raw.backbone.named_parameters()):
        raise RuntimeError('backbone/LoRA/anchors not frozen or were modified')
    if any(not torch.equal(v.detach().cpu(), link_before[k]) for k, v in raw.hybridlink.state_dict().items()):
        raise RuntimeError('HybridLink weights changed during backward-only probe')
    if (sha256(source / 'adapter.pt') != hashes['source_link_sha256'] or
            sha256(source / 'lora/adapter_model.safetensors') != hashes['source_lora_sha256']):
        raise RuntimeError('source checkpoint files changed')
    summary = {**forward, 'link_gradient_norms': link_grads, 'input_gradient_norms': vector_grads,
               'backbone_frozen_no_grad_versions_unchanged': True, 'link_weights_unchanged': True,
               'source_hashes_unchanged': True, 'optimizer_steps': 0, 'weights_saved': False,
               'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
               'elapsed_seconds': time.monotonic() - started,
               'warning': 'Gradient feasibility only, one training record. Full-prefix no-cache recomputation, '
                          'not the sequential cached diagnostic. No measured learning or reasoning improvement.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
