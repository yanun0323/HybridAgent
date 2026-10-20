"""Compare three-input execution paths on eight fixed training records. No backward."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_experiment import ROOT, verify_data, sha256, write_json
from link_three_step_probe import select_record
from latent_diagnostic import span_bounds


def cached_rollout(model, sample, entry_id, exit_id):
    """Same segmentation as latent_diagnostic: prompt, assistant tokens, vectors.

    Caller supplies no_grad and BF16 autocast. The head/decoder are the same ones
    used by the full-prefix objective. Never feed known latent content.
    """
    import torch
    import torch.nn.functional as F
    from hybrid_agent.compat import text_decoder
    start, end = span_bounds(sample, entry_id, exit_id)
    if end - start - 1 < 3:
        raise ValueError('requires at least three reference content tokens')
    device = next(model.backbone.parameters()).device
    ids = sample['input_ids']
    prompt_len = sample['prompt_len']
    decoder = text_decoder(model.backbone)
    embed = model.backbone.get_input_embeddings()
    head = model.backbone.get_output_embeddings()
    out = decoder(input_ids=torch.tensor([ids[:prompt_len]], device=device),
                  use_cache=True, return_dict=True)
    position = prompt_len

    def feed(vector):
        nonlocal out, position
        out = decoder(inputs_embeds=vector, past_key_values=out.past_key_values,
                      attention_mask=torch.ones(1, position + 1, dtype=torch.long, device=device),
                      position_ids=torch.tensor([[position]], device=device),
                      use_cache=True, return_dict=True)
        position += 1
        if out.past_key_values.get_seq_length() != position:
            raise RuntimeError('cache position mismatch')

    for token in ids[prompt_len:start + 1]:
        feed(embed(torch.tensor([[token]], device=device)))
    vectors, losses, predicted = [], [], []
    dtype = embed.weight.dtype
    for step in range(3):
        with torch.autocast(device_type=device.type, enabled=False):
            vector = model.run_hybridlink(out.last_hidden_state[:, -1:], output_dtype=torch.float32)
        vectors.append(vector)
        feed(vector.to(dtype))
        logits = head(out.last_hidden_state[:, -1:]).float()[:, -1]
        losses.append(F.cross_entropy(logits, torch.tensor([ids[start + step + 2]], device=device)))
        predicted.append(logits.argmax(-1))
    losses = torch.stack(losses)
    return {'loss': losses.mean(), 'step_losses': losses, 'rollout_vectors': tuple(vectors),
            'predicted_ids': torch.stack(predicted, dim=1),
            'target_ids': torch.tensor([ids[start + 2:start + 5]], device=device),
            'substitution_positions': list(range(start + 1, start + 4)),
            'target_positions': list(range(start + 2, start + 5))}


def measurements(out, tokenizer):
    import torch
    if not torch.isfinite(out['step_losses']).all():
        raise RuntimeError('nonfinite CE')
    vectors = [v.detach().float().cpu().clone() for v in out['rollout_vectors']]
    if any(not torch.isfinite(v).all() for v in vectors):
        raise RuntimeError('nonfinite input vector')
    target = out['target_ids'].tolist()[0]
    predicted = out['predicted_ids'].tolist()[0]
    ce = out['step_losses'].detach().tolist()
    return {'mean_ce': sum(ce) / 3, 'step_ce': ce,
            'target_probabilities': [math.exp(-v) for v in ce],
            'target_ids': target, 'predicted_ids': predicted,
            'target_texts': [tokenizer.decode([v]) for v in target],
            'predicted_texts': [tokenizer.decode([v]) for v in predicted],
            'substitution_positions': out['substitution_positions'],
            'target_positions': out['target_positions']}, vectors


def compare_paths(a, b, av, bv):
    import torch.nn.functional as F
    for key in ('target_ids', 'target_positions', 'substitution_positions'):
        if a[key] != b[key]:
            raise ValueError('execution paths target different positions')
    return {'delta_ce_b_minus_a': [y - x for x, y in zip(a['step_ce'], b['step_ce'])],
            'argmax_matches': [x == y for x, y in zip(a['predicted_ids'], b['predicted_ids'])],
            'vector_cosines': [F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item() for x, y in zip(av, bv)],
            'vector_max_abs_differences': [(x - y).abs().max().item() for x, y in zip(av, bv)]}


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
            raise ValueError('use a separate output directory')
    if output.exists():
        raise FileExistsError('output exists; preserve it')
    data, _ = verify_data(args.pilot, 'train')
    rows = [json.loads(s) for s in data.read_text().splitlines() if s.strip()]
    select_record(rows)  # validates all 64 unique IDs
    rows = rows[:8]
    settings = json.loads((source / 'train_args.json').read_text())
    if settings['stage'] != 'hybridlink_activation' or settings.get('max_steps') != 64:
        raise ValueError('requires original 64-step activation checkpoint')

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
        raise ValueError('source scale differs')
    accelerator = Accelerator(mixed_precision='bf16')
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
    samples = []
    for row in rows:
        sample = tokenize_trajectory(tokenizer, row['system'], row['question'], row['rounds'], 1024)
        if sample is None:
            raise ValueError(f'rejected record {row["id"]}')
        start, end = span_bounds(sample, entry, exit)
        if end - start - 1 < 3:
            raise ValueError('requires three content tokens')
        samples.append(sample)
    hashes = {name: sha256(source / name) for name in ('adapter.pt', 'lora/adapter_model.safetensors')}
    output.mkdir(parents=True)
    write_json(output / 'run.json', {
        'source': str(source), 'source_hashes': hashes, 'data_sha256': sha256(data),
        'sample_ids': [r['id'] for r in rows],
        'input_ids_sha256': [hashlib.sha256(json.dumps(s['input_ids']).encode()).hexdigest() for s in samples],
        'calibration': calibration, 'seed': 6064, 'torch': torch.__version__,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'paths': ['full_grad', 'full_no_grad', 'cached_no_grad'],
        'precision': 'BF16 autocast in all paths; FP32 HybridLink, cast to embedding dtype',
    })
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model) or model.lora_anchor_ids != [entry, exit]:
        raise ValueError('expected matching LoRA anchors')
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(LINK_NEXT_TOKEN)
    model.output_scale = calibration['output_scale']
    model = accelerator.prepare(model)
    model.train()
    raw = accelerator.unwrap_model(model)
    versions = {n: p._version for n, p in raw.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    results = []
    with (output / 'measurements.jsonl').open('x') as stream:
        for row, sample in zip(rows, samples):
            batch = {k: v.to(accelerator.device) for k, v in collate_batch(tokenizer, [sample]).items()}
            out = model(**batch, stage=LINK_NEXT_TOKEN, link_steps=3)
            full, fv = measurements(out, tokenizer)
            del out  # discard graph BEFORE the next path, never backward
            with torch.no_grad():
                out = model(**batch, stage=LINK_NEXT_TOKEN, link_steps=3)
                plain, pv = measurements(out, tokenizer)
                del out
                with accelerator.autocast():
                    out = cached_rollout(raw, sample, entry, exit)
                cached, cv = measurements(out, tokenizer)
                del out  # fresh cache for the next record
            item = {'id': row['id'], 'question': row['question'],
                    'full_grad': full, 'full_no_grad': plain, 'cached_no_grad': cached,
                    'grad_mode_comparison': compare_paths(full, plain, fv, pv),
                    'cache_segmentation_comparison': compare_paths(plain, cached, pv, cv)}
            results.append(item)
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            print(f"{row['id']}: full_grad={full['step_ce']} full_no_grad={plain['step_ce']} cached={cached['step_ce']}", flush=True)
    if any(p.grad is not None or p._version != versions[n] for n, p in raw.named_parameters()):
        raise RuntimeError('parameters changed or accumulated gradients')
    if any(sha256(source / name) != value for name, value in hashes.items()):
        raise RuntimeError('source checkpoint changed')
    summary = {'count': len(results), 'results': results,
               'mean_ce_by_path': {name: sum(r[name]['mean_ce'] for r in results) / len(results)
                                   for name in ('full_grad', 'full_no_grad', 'cached_no_grad')},
               'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
               'optimizer_steps': 0, 'weights_saved': False, 'versions_and_source_hashes_unchanged': True,
               'warning': 'Training-set execution comparison only. Each path generates its own vectors. '
                          'Cached mode uses the diagnostic segmentation but common BF16 autocast, '
                          'not an exact replay of the older inference-mode diagnostic. Differences do not alone identify a kernel bug.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
