"""Replay A's six fixed vectors through full-prefix and cached execution; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_conditioned_swap import (
    PAIRS, VERSIONS, read_json, read_rows, verify_control, install_pair,
    SOURCE_LINK_SHA256, SOURCE_LORA_SHA256, SAVED_LINK_SHA256, SAVED_LORA_SHA256,
    CONTROL_LINK_SHA256, check_reference_config, verify_calibration,
)
from format_experiment import ROOT, verify_data, sha256, write_json
from lora_self_conditioned_probe import validate_paths
from lora_self_conditioned_joint import (
    TRAIN_SHA256, file_hashes, training_plan, verify_joint_reference, frozen_evaluation,
)
from lora_self_conditioned_evaluate import require_replay
from link_execution_probe import cached_rollout, measurements, compare_paths
from latent_diagnostic import span_bounds

SWAP_COMMIT = 'b48900422e8ac8f8fda9b11ab4b6dd2048cc9607'


def verify_swap(path, protected, plan, ids, reference, control):
    run, summary = read_json(path / 'run.json'), read_json(path / 'summary.json')
    pairs = {k: list(v) for k, v in PAIRS.items()}
    require_replay(run['code_commit'], SWAP_COMMIT, 'swap commit')
    require_replay(run['sequence'], ['source', 'B', 'C', 'A', 'A_reloaded', 'D', 'D_reloaded'], 'swap sequence')
    require_replay(run['lora_sha256'], {'source': SOURCE_LORA_SHA256, 'joint': SAVED_LORA_SHA256,
                                     'control': SOURCE_LORA_SHA256}, 'swap LoRA hashes')
    require_replay(run['link_sha256'], {'source': SOURCE_LINK_SHA256, 'joint': SAVED_LINK_SHA256,
                                     'control': CONTROL_LINK_SHA256}, 'swap Link hashes')
    require_replay(run['pairs'], pairs, 'swap pairs')
    require_replay(summary['pairs'], pairs, 'swap summary pairs')
    require_replay(run['original_plan'], plan, 'swap original plan')
    require_replay(run['protected_hashes'], {str(p): h for p, h in protected.items()}, 'swap sources')
    require_replay(run['tokenization_sha256'], plan['tokenization_sha256'], 'swap tokens')
    require_replay(run['calibration'], plan['calibration'], 'swap calibration')
    require_replay(run['data_sha256'], TRAIN_SHA256, 'swap data')
    for k in VERSIONS:
        require_replay(run[k], plan[k], f'swap {k}')
    for document in (run, summary):
        if document.get('optimizer_steps') != 0 or document.get('weights_saved') is not False:
            raise ValueError('swap must not train or export weights')
    for k in ('original_and_B_C_controls_exact', 'A_D_reload_exact', 'anchors_values_unchanged',
              'original_backbone_versions_no_grad_unchanged', 'protected_files_unchanged',
              'text_only_same_lora_exact'):
        if summary.get(k) is not True:
            raise ValueError(f'swap check failed: {k}')
    rows = {k: read_rows(path / f'{k}.jsonl') for k in
            ('source', 'B', 'C', 'A', 'A_reloaded', 'D', 'D_reloaded')}
    for k, value in rows.items():
        require_replay([r['id'] for r in value], ids, f'{k} IDs')
    for k, expected in (('source', reference['before']), ('B', control), ('C', reference['after']),
                        ('A_reloaded', rows['A']), ('D_reloaded', rows['D'])):
        require_replay(rows[k], expected, f'swap {k}')
    return rows['A']


def fixed_rollout(model, sample, entry_id, exit_id, vectors, path):
    """Feed six supplied FP32 vectors, cast once to embedding dtype. Never call Link.

    Reproduce the respective recursive path's prefix segmentation and positions.
    Capture entry and post-input hidden states/logits; labels never become inputs.
    """
    import torch
    import torch.nn.functional as F
    from hybrid_agent.compat import text_decoder
    if path not in ('full', 'cached'):
        raise ValueError('path must be full or cached')
    if torch.is_grad_enabled() or any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
        raise ValueError('requires frozen eval mode and no_grad')
    start, end = span_bounds(sample, entry_id, exit_id)
    ids, prompt_len = sample['input_ids'], sample['prompt_len']
    if end - start - 1 < 6 or not 0 < prompt_len <= start:
        raise ValueError('requires prompt, entry and six content tokens')
    embed = model.backbone.get_input_embeddings()
    device, dtype = embed.weight.device, embed.weight.dtype
    if len(vectors) != 6 or any(v.shape != (1, 1, embed.weight.shape[1]) or
            v.dtype != torch.float32 or not torch.isfinite(v).all() for v in vectors):
        raise ValueError('requires six finite FP32 vectors of matching shape')
    supplied = tuple(v.detach().to(device=device, dtype=dtype).clone() for v in vectors)
    decoder, head = text_decoder(model.backbone), model.backbone.get_output_embeddings()
    prefix = embed(torch.tensor([ids[:start + 1]], device=device)) if path == 'full' else None
    if path == 'full':
        out = decoder(inputs_embeds=prefix, attention_mask=torch.ones(1, start + 1,
                      dtype=torch.long, device=device), use_cache=False, return_dict=True)
    else:
        out = decoder(input_ids=torch.tensor([ids[:prompt_len]], device=device),
                      use_cache=True, return_dict=True)
        position = prompt_len
        if out.past_key_values.get_seq_length() != position:
            raise RuntimeError('prompt cache position mismatch')

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
    hidden, logits = [], []

    def capture():
        h = out.last_hidden_state[:, -1:]
        z = head(h).float()[:, -1]
        if not torch.isfinite(h).all() or not torch.isfinite(z).all():
            raise RuntimeError('nonfinite fixed-path output')
        hidden.append(h.detach().cpu().clone())
        logits.append(z.detach().cpu().clone())
        return z

    capture()  # entry output, before feeding any predicted vector
    losses, predicted = [], []
    for step, vector in enumerate(supplied):
        if path == 'full':
            modified = torch.cat((prefix, *supplied[:step + 1]), dim=1)
            out = decoder(inputs_embeds=modified, attention_mask=torch.ones(
                1, start + step + 2, dtype=torch.long, device=device),
                use_cache=False, return_dict=True)
        else:
            feed(vector)
        z = capture()
        losses.append(F.cross_entropy(z, torch.tensor([ids[start + step + 2]], device=device)))
        predicted.append(z.argmax(-1))
    losses = torch.stack(losses)
    return {'loss': losses.mean(), 'step_losses': losses,
            'rollout_vectors': tuple(v.detach().clone() for v in vectors),
            'predicted_ids': torch.stack(predicted, dim=1),
            'target_ids': torch.tensor([ids[start + 2:start + 8]], device=device),
            'substitution_positions': list(range(start + 1, start + 7)),
            'target_positions': list(range(start + 2, start + 8)),
            'hidden': hidden, 'logits': logits, 'fed_vectors': supplied}


def tensor_difference(a, b):
    import torch
    import torch.nn.functional as F
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('tensor shape or dtype differs')
    x, y = a.float().flatten(), b.float().flatten()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise RuntimeError('nonfinite comparison')
    d = y - x
    return {'exact': torch.equal(a, b), 'max_abs': d.abs().max().item(),
            'rms_difference': d.square().mean().sqrt().item(),
            'cosine': F.cosine_similarity(x, y, dim=0).item()}


def compare_fixed(full, cached, fm, cm):
    import torch
    for out in (full, cached):
        if len(out['fed_vectors']) != 6 or len(out['hidden']) != 7 or len(out['logits']) != 7:
            raise ValueError('requires six fixed inputs and seven captured states')
    for a, b in zip(full['fed_vectors'], cached['fed_vectors']):
        if a.dtype != b.dtype or not torch.equal(a, b):
            raise RuntimeError('fixed inputs differ after casting')
    report = compare_paths(fm, cm, full['rollout_vectors'], cached['rollout_vectors'])
    states = []
    for step in range(7):
        states.append({'step': step, 'hidden': tensor_difference(full['hidden'][step], cached['hidden'][step]),
                       'logits': tensor_difference(full['logits'][step], cached['logits'][step]),
                       'full_argmax': full['logits'][step].argmax(-1).item(),
                       'cached_argmax': cached['logits'][step].argmax(-1).item()})
    return {**report, 'same_cast_inputs_exact': True, 'states': states,
            'first_hidden_difference': next((s['step'] for s in states if not s['hidden']['exact']), None),
            'first_logit_difference': next((s['step'] for s in states if not s['logits']['exact']), None),
            'first_argmax_difference': next((s['step'] for s in states
                                            if s['full_argmax'] != s['cached_argmax']), None)}


def require_fixed_repeat(first, second, tokenizer):
    import torch
    require_replay(measurements(first, tokenizer)[0], measurements(second, tokenizer)[0], 'fixed repeat metrics')
    for key in ('hidden', 'logits', 'fed_vectors'):
        if len(first[key]) != len(second[key]) or any(a.dtype != b.dtype or not torch.equal(a, b)
                for a, b in zip(first[key], second[key])):
            raise RuntimeError(f'exact fixed replay mismatch: {key}')


def collect_controls(rows, expected, measure, record):
    """All 32 historical path controls must pass before any crossed execution."""
    if len(rows) != 16 or len(expected) != 16:
        raise ValueError('requires all sixteen controls')
    results = []
    for row, control in zip(rows, expected):
        require_replay(row['id'], control['id'], 'control ID')
        metrics, vectors = measure(row)
        for path in ('full', 'cached'):
            require_replay(metrics[path], control[path], f'A/{row["id"]}/{path}')
        record({'id': row['id'], **metrics})
        results.append((metrics, vectors))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'model', 'calibration', 'joint-run', 'joint-evaluation',
                 'control-run', 'swap-run', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source, pilot, joint, evaluation, control, swap, output = [Path(v).resolve() for v in
        (args.model, args.pilot, args.joint_run, args.joint_evaluation, args.control_run,
         args.swap_run, args.output_dir)]
    for p in (pilot, joint, evaluation, control, swap, Path(args.calibration).resolve()):
        validate_paths(source, p, output)
    data, _ = verify_data(pilot, 'train')
    if sha256(data) != TRAIN_SHA256:
        raise ValueError('training data differs')
    protected = {p: file_hashes(p) for p in (source, joint, evaluation, control)}
    update, observe, _ = training_plan(read_rows(data))
    rows = update + observe
    reference = verify_joint_reference(joint, evaluation, protected[source], update, observe)
    control_rows = verify_control(control, reference, protected[source], update, observe, joint, evaluation)
    plan = reference['run']
    expected = verify_swap(swap, protected, plan, [r['id'] for r in rows], reference, control_rows)
    protected[swap] = file_hashes(swap)
    checkpoints = {'source': source, 'joint': joint / 'checkpoint', 'control': control / 'checkpoint'}
    for label, lora_hash, link_hash in (
            ('source', SOURCE_LORA_SHA256, SOURCE_LINK_SHA256),
            ('joint', SAVED_LORA_SHA256, SAVED_LINK_SHA256),
            ('control', SOURCE_LORA_SHA256, CONTROL_LINK_SHA256)):
        p = checkpoints[label]
        if sha256(p / 'adapter.pt') != link_hash or sha256(p / 'lora/adapter_model.safetensors') != lora_hash:
            raise ValueError(f'weight hash differs: {label}')
        check_reference_config(p, source)
        require_replay(read_json(p / 'hybridlink_config.json'), read_json(source / 'hybridlink_config.json'), 'Link config')

    import torch
    from importlib.metadata import version
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoTokenizer
    from hybrid_agent.compat import check_execution
    from hybrid_agent.lora import is_lora_model
    from hybrid_sft.model import HybridSFTModel, SELF_CONDITIONED, LINK_NEXT_TOKEN
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.calibration import read_training_calibration

    versions = {k: version(k) for k in VERSIONS}
    require_replay(versions, {k: plan[k] for k in VERSIONS}, 'library versions')
    calibration = read_training_calibration(args.calibration, data)
    scale = read_json(source / 'hybridlink_config.json')['output_scale']
    verify_calibration(joint / 'checkpoint/link_calibration.json', plan['calibration'], calibration, scale)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    if accelerator.device.type != 'cuda' or accelerator.num_processes != 1 or str(accelerator.distributed_type).endswith('DEEPSPEED'):
        raise ValueError('requires one CUDA process without DeepSpeed')
    set_seed(plan['seed'])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    entry, exit = [tokenizer.convert_tokens_to_ids(t) for t in ('<latent>', '</latent>')]
    samples = [tokenize_trajectory(tokenizer, r['system'], r['question'], r['rounds'], 1024) for r in rows]
    if any(s is None for s in samples):
        raise ValueError('tokenization rejected')
    hashes = [hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for s in samples]
    require_replay(hashes, plan['tokenization_sha256'], 'tokenization')
    batches = [{k: v.to(accelerator.device) for k, v in collate_batch(tokenizer, [s], pad_to_multiple=1).items()}
               for s in samples]
    output.mkdir(parents=True)
    write_json(output / 'run.json', {
        'pair': 'A', 'sample_ids': [r['id'] for r in rows], 'groups': {'update': 8, 'observation': 8},
        'protected_hashes': {str(p): h for p, h in protected.items()},
        'calibration': calibration, 'tokenization_sha256': hashes, 'seed': plan['seed'],
        'optimizer_steps': 0, 'weights_saved': False, 'vectors_saved': True,
        'precision': 'eval/no_grad; BF16 autocast, FP32 Link; fixed inputs cast to embedding dtype',
        'donors': ['full', 'cached'], 'paths': ['full', 'cached'],
        'step_numbering': '0 = entry output; 1..6 = outputs after fixed vectors',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'script_sha256': sha256(Path(__file__)), **versions,
    })
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model) or model.lora_anchor_ids != [entry, exit]:
        raise ValueError('expected LoRA checkpoint and matching anchors')
    check_execution(model.backbone, batch_size=1)
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = accelerator.prepare(model)
    raw = accelerator.unwrap_model(model)
    anchors = {n: p.detach().cpu().clone() for n, p in raw.named_parameters() if 'trainable_tokens_delta' in n}
    base_versions = {n: p._version for n, p in raw.named_parameters() if not p.requires_grad and n not in anchors}
    install_pair(raw, source / 'lora/adapter_model.safetensors', joint / 'checkpoint/adapter.pt',
                 SOURCE_LORA_SHA256, SAVED_LINK_SHA256)
    if len(anchors) != 2 or any(p._version != base_versions[n] for n, p in raw.named_parameters() if n in base_versions):
        raise RuntimeError('original backbone changed during install')
    versions_before = {n: p._version for n, p in raw.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    indexes = {r['id']: i for i, r in enumerate(rows)}
    results = []
    with frozen_evaluation(raw), torch.no_grad(), accelerator.autocast():
        def measure(row):
            i = indexes[row['id']]
            metrics, vectors = {}, {}
            out = raw(**batches[i], stage=LINK_NEXT_TOKEN, link_steps=6)
            metrics['full'], vectors['full'] = measurements(out, tokenizer)
            del out
            out = cached_rollout(raw, samples[i], entry, exit, link_steps=6)
            metrics['cached'], vectors['cached'] = measurements(out, tokenizer)
            return metrics, vectors
        with (output / 'controls.jsonl').open('x') as stream:
            def record(item):
                stream.write(json.dumps(item) + '\n')
                stream.flush()
                print('control exact:', item['id'], flush=True)
            controls = collect_controls(rows, expected, measure, record)
        with (output / 'measurements.jsonl').open('x') as stream:
            for i, (row, (metrics, vectors)) in enumerate(zip(rows, controls)):
                item = {'id': row['id'], 'question': row['question'],
                        'group': 'update' if i < 8 else 'observation',
                        'recursive': metrics, 'recursive_comparison': compare_paths(
                            metrics['full'], metrics['cached'], vectors['full'], vectors['cached']), 'fixed': {}}
                vector_path = output / f'vectors-{row["id"]}.pt'
                with vector_path.open('xb') as f:
                    torch.save(vectors, f)
                saved = torch.load(vector_path, map_location='cpu', weights_only=True)
                for donor in ('full', 'cached'):
                    if len(saved[donor]) != 6 or any(a.dtype != b.dtype or not torch.equal(a, b)
                            for a, b in zip(vectors[donor], saved[donor])):
                        raise RuntimeError('saved vectors differ')
                    outputs, measures = {}, {}
                    for path in ('full', 'cached'):
                        out = fixed_rollout(raw, samples[i], entry, exit, saved[donor], path)
                        repeated = fixed_rollout(raw, samples[i], entry, exit, saved[donor], path)
                        require_fixed_repeat(out, repeated, tokenizer)
                        del repeated
                        measures[path], _ = measurements(out, tokenizer)
                        if path == donor:
                            require_replay(measures[path], metrics[path], f'{row["id"]}/{donor} self replay')
                        outputs[path] = out
                    item['fixed'][donor] = {'metrics': measures, 'comparison': compare_fixed(
                        outputs['full'], outputs['cached'], measures['full'], measures['cached'])}
                    del outputs, out
                item['vectors_sha256'] = sha256(vector_path)
                results.append(item)
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                stream.flush()
                print(row['id'], json.dumps({k: v['comparison'] for k, v in item['fixed'].items()}), flush=True)
    if any(p.grad is not None or p._version != versions_before[n] for n, p in raw.named_parameters()):
        raise RuntimeError('parameters changed or gradients accumulated')
    if any(not torch.equal(p.detach().cpu(), anchors[n]) for n, p in raw.named_parameters() if n in anchors):
        raise RuntimeError('anchors changed')
    if any(file_hashes(p) != h for p, h in protected.items()) or sha256(data) != TRAIN_SHA256 or sha256(args.calibration) != calibration['source_calibration_sha256']:
        raise RuntimeError('protected files changed')
    torch.cuda.synchronize()
    summary = {'count': 16, 'pair': 'A', 'optimizer_steps': 0, 'weights_saved': False,
               'historical_32_path_controls_exact': True, 'fixed_self_replay_exact': True,
               'fixed_repeats_hidden_logits_exact': True, 'same_cast_inputs_exact': True,
               'parameters_versions_no_grad_unchanged': True, 'anchors_values_unchanged': True,
               'protected_files_unchanged': True, 'vectors_saved': True,
               'results': results, 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
               'elapsed_seconds': time.monotonic() - started,
               'warning': 'Training records only, pair A only. No generation or exit evaluation repeated. '
                          'Fixed-path differences include prefix segmentation and numerical execution; '
                          'they do not alone identify a cache bug or prove latent reasoning.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
