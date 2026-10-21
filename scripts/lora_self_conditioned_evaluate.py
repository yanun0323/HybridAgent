"""Evaluate the saved interrupted one-update experiment. Never train or export.

Replay all 16 original controls before installing saved LoRA + Link tensors into
one backbone. Write only to a fresh evaluation directory; preserve the failed run.
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
from lora_self_conditioned_probe import validate_paths, SOURCE_LINK_SHA256, SOURCE_LORA_SHA256
from lora_self_conditioned_joint import (
    TRAIN_SHA256, LR, LOSS_KEYS, HYBRID_LIMITS, file_hashes, training_plan,
    measure_record, compare_records, reload_adapters,
)
from latent_experiment import check_reference_config
from latent_diagnostic import span_bounds

SAVED_LINK_SHA256 = '517d36c46910bf08bc5d49ebef67fc2ea1e45884129ab1143ecab9b07c73b861'
SAVED_LORA_SHA256 = '14d741abc90c8ef2634ab8b2c1ef7b2ab778004b2917ecf84bde2f4204c6315b'
FAILED_COMMIT = '40fef113adf910ddfea5c0a5037451fa526ffb1a'


def verify_plan(run, update_report, before, update, observe):
    """Require the recorded single update and its ordered pre-update evidence."""
    expected = {
        'code_commit': FAILED_COMMIT, 'data_sha256': TRAIN_SHA256,
        'planned_optimizer_steps': 1, 'gradient_average_count': 8,
        'learning_rate_lora': LR, 'learning_rate_link': LR, 'betas': [.9, .95],
        'weight_decay': 0., 'max_grad_norm': 1., 'shuffle': False, 'scheduler': None,
        'anchors_frozen': True, 'hybrid_limits': HYBRID_LIMITS,
        'text_budgets': {'plain': 32, 'format': 256}, 'recursive_steps': 6, 'seed': 6064,
        'update_ids': [r['id'] for r in update], 'observation_ids': [r['id'] for r in observe],
    }
    if any(run.get(k) != v for k, v in expected.items()):
        raise ValueError('recorded experiment plan differs')
    if len(before) != 16 or [r['id'] for r in before] != expected['update_ids'] + expected['observation_ids']:
        raise ValueError('requires 16 ordered pre-update records')
    if update_report.get('optimizer_steps') != 1:
        raise ValueError('requires exactly one completed update')
    if update_report['pre_update_losses'] != [r['two_pass'] for r in before[:8]]:
        raise ValueError('update losses differ from pre-update controls')


def verify_calibration(saved_path, recorded, original, scale):
    """Compare the saved record verbatim; do not reassign its source-file hash."""
    saved = json.loads(Path(saved_path).read_text())
    if saved != recorded or original != recorded or original['output_scale'] != scale:
        raise ValueError('calibration differs')
    return original


def require_replay(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f'exact replay mismatch: {label}; stop, preserve evidence')


def install_saved(raw, checkpoint):
    """Install only a compact PEFT adapter and Link, preserving frozen anchors."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    saved = load_file(str(checkpoint / 'lora/adapter_model.safetensors'))
    live = get_peft_model_state_dict(raw.backbone, save_embedding_layers=False)
    link = torch.load(checkpoint / 'adapter.pt', map_location='cpu', weights_only=True)
    live_link = raw.hybridlink.state_dict()
    for a, b in ((saved, live), (link, live_link)):
        if a.keys() != b.keys() or any(a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
                                      or not torch.isfinite(a[k]).all() for k in a):
            raise ValueError('saved adapter keys, shapes, dtypes or values invalid')
    if any(not ('lora_' in k or 'trainable_tokens_delta' in k) for k in saved):
        raise ValueError('checkpoint contains non-adapter backbone weights')
    anchors = [k for k in saved if 'trainable_tokens_delta' in k]
    if len(anchors) != 2 or any(not torch.equal(saved[k], live[k].detach().cpu()) for k in anchors):
        raise ValueError('saved anchor values differ from source')
    changed_lora = [k for k in saved if 'lora_' in k and not torch.equal(saved[k], live[k].detach().cpu())]
    changed_link = [k for k in link if not torch.equal(link[k], live_link[k].detach().cpu())]
    if not changed_lora or not changed_link:
        raise ValueError('expected changed LoRA and Link')
    loaded = set_peft_model_state_dict(raw.backbone, saved, adapter_name='default')
    if loaded.unexpected_keys:
        raise RuntimeError(f'unexpected PEFT keys: {loaded.unexpected_keys}')
    raw.hybridlink.load_state_dict(link, strict=True)
    for a, b in ((saved, get_peft_model_state_dict(raw.backbone, save_embedding_layers=False)),
                 (link, raw.hybridlink.state_dict())):
        if a.keys() != b.keys() or any(a[k].dtype != b[k].dtype or
                                     not torch.equal(a[k], b[k].detach().cpu()) for k in a):
            raise RuntimeError('installed adapter values differ from disk')
    return {'lora': changed_lora, 'link': changed_link}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'model', 'joint-run', 'calibration', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    pilot, source, failed, output = [Path(v).resolve() for v in
                                    (args.pilot, args.model, args.joint_run, args.output_dir)]
    validate_paths(source, pilot, output)
    validate_paths(source, failed, output)
    checkpoint = failed / 'checkpoint'
    if any((failed / n).exists() for n in ('after.jsonl', 'reloaded.jsonl', 'summary.json')):
        raise ValueError('expected the preserved run interrupted before after-measurements')
    data, _ = verify_data(pilot, 'train')
    if sha256(data) != TRAIN_SHA256:
        raise ValueError('training file differs')
    source_hashes, failed_hashes = file_hashes(source), file_hashes(failed)
    if (source_hashes.get('adapter.pt') != SOURCE_LINK_SHA256 or
            source_hashes.get('lora/adapter_model.safetensors') != SOURCE_LORA_SHA256 or
            failed_hashes.get('checkpoint/adapter.pt') != SAVED_LINK_SHA256 or
            failed_hashes.get('checkpoint/lora/adapter_model.safetensors') != SAVED_LORA_SHA256):
        raise ValueError('weights differ from the recorded source/saved update')
    run = json.loads((failed / 'run.json').read_text())
    report = json.loads((failed / 'update.json').read_text())
    before = [json.loads(s) for s in (failed / 'before.jsonl').read_text().splitlines() if s.strip()]
    update, observe, _ = training_plan([json.loads(s) for s in data.read_text().splitlines() if s.strip()])
    rows = update + observe
    verify_plan(run, report, before, update, observe)
    if run['source_hashes'] != source_hashes:
        raise ValueError('source files differ from original run')
    settings = json.loads((checkpoint / 'train_args.json').read_text())
    for key, value in {'stage': 'self_conditioned', 'lora_self_conditioned': True, 'max_steps': 1,
                       'grad_accum': 8, 'batch_size': 1, 'lr': LR, 'hybridlink_lr': LR}.items():
        if settings.get(key) != value:
            raise ValueError(f'checkpoint training settings differ: {key}')
    check_reference_config(checkpoint, source)
    config = json.loads((source / 'hybridlink_config.json').read_text())
    if config != json.loads((checkpoint / 'hybridlink_config.json').read_text()):
        raise ValueError('saved link configuration differs')

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

    versions = {k: version(k) for k in ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers')}
    if any(v != run[k] for k, v in versions.items()):
        raise ValueError('library versions differ from interrupted run')
    calibration = verify_calibration(
        checkpoint / 'link_calibration.json', run['calibration'],
        read_training_calibration(args.calibration, data), config['output_scale'])
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    if (accelerator.device.type != 'cuda' or accelerator.num_processes != 1
            or str(accelerator.distributed_type).endswith('DEEPSPEED')):
        raise ValueError('requires one CUDA process without DeepSpeed')
    set_seed(run['seed'])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    saved_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    for tok in (tokenizer, saved_tokenizer):
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
    entry, exit = [tokenizer.convert_tokens_to_ids(t) for t in ('<latent>', '</latent>')]
    samples, batches = [], []
    for row in rows:
        sample = tokenize_trajectory(tokenizer, row['system'], row['question'], row['rounds'], 1024)
        other = tokenize_trajectory(saved_tokenizer, row['system'], row['question'], row['rounds'], 1024)
        if sample is None or sample != other:
            raise ValueError('saved/source tokenization differs or record rejected')
        start, end = span_bounds(sample, entry, exit)
        if end - start - 1 < 6:
            raise ValueError('requires six content tokens')
        samples.append(sample)
        batches.append(collate_batch(tokenizer, [sample], pad_to_multiple=1))
    token_hashes = [hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for s in samples]
    if token_hashes != run['tokenization_sha256']:
        raise ValueError('tokenization differs from interrupted run')
    output.mkdir(parents=True)
    write_json(output / 'run.json', {
        'source': str(source), 'joint_run': str(failed), 'source_hashes': source_hashes,
        'interrupted_run_hashes': failed_hashes, 'data_sha256': TRAIN_SHA256,
        'tokenization_sha256': token_hashes, 'original_plan': run,
        'optimizer_steps_this_run': 0, 'weights_saved': False,
        'ignored_tokenizer_loading_flags': ['is_local', 'local_files_only'],
        **versions, 'gpu': torch.cuda.get_device_name(accelerator.device),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    })
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model) or model.lora_anchor_ids != [entry, exit]:
        raise ValueError('expected matching LoRA checkpoint')
    check_execution(model.backbone, batch_size=1)
    model.resize_to_tokenizer(tokenizer)
    # Grad-enabled two-pass forward is kept solely to replay the exact original
    # measurement path. There is no backward or optimizer anywhere in this script.
    model.set_trainable(SELF_CONDITIONED, lora_self_conditioned=True)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = accelerator.prepare(model)
    raw = accelerator.unwrap_model(model)
    raw.zero_grad(set_to_none=True)
    batches = [{k: v.to(accelerator.device) for k, v in b.items()} for b in batches]
    anchors = {n: p.detach().cpu().clone() for n, p in raw.named_parameters() if 'trainable_tokens_delta' in n}
    if len(anchors) != 2:
        raise RuntimeError('expected two frozen anchor tensors')
    original_versions = {n: p._version for n, p in raw.named_parameters() if not p.requires_grad and n not in anchors}
    def check_original():
        for n, p in raw.named_parameters():
            if p.grad is not None:
                raise RuntimeError(f'unexpected gradient: {n}')
            if n in original_versions and (p.requires_grad or p._version != original_versions[n]):
                raise RuntimeError(f'original backbone changed: {n}')
            if n in anchors and (p.requires_grad or not torch.equal(p.detach().cpu(), anchors[n])):
                raise RuntimeError(f'anchor values changed: {n}')
    def measure_all(phase, controls=None):
        results = []
        versions_before = {n: p._version for n, p in raw.named_parameters()}
        with (output / f'{phase}.jsonl').open('x') as stream:
            for i, (row, sample, batch) in enumerate(zip(rows, samples, batches)):
                item = measure_record(model, raw, accelerator, tokenizer, row, sample, batch, entry, exit)
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                stream.flush()
                if controls is not None:
                    require_replay(item, controls[i], f'{phase}/{row["id"]}')
                check_original()
                if any(p._version != versions_before[n] for n, p in raw.named_parameters()):
                    raise RuntimeError('parameters changed during measurement')
                results.append(item)
                print(f'{phase} {row["id"]}: loss={item["two_pass"]["loss"]:.8g} '
                      f'cached6={item["cached"]["mean_ce"]:.8g} '
                      f'natural_exit={item["hybrid"]["natural_exit"]}', flush=True)
        return results
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    measure_all('control', before)  # all sixteen must pass BEFORE changing adapters
    changed = install_saved(raw, checkpoint)
    check_original()
    after = measure_all('after')
    reload_adapters(raw, checkpoint)  # clear trainable tensors, reload from disk, exact tensor check
    check_original()
    measure_all('reloaded', after)
    if (file_hashes(source) != source_hashes or file_hashes(failed) != failed_hashes
            or sha256(data) != TRAIN_SHA256
            or sha256(args.calibration) != calibration['source_calibration_sha256']):
        raise RuntimeError('protected source, interrupted run, data or calibration changed')
    torch.cuda.synchronize()
    summary = {
        'optimizer_steps_this_run': 0, 'original_optimizer_steps': 1, 'weights_saved': False,
        'checkpoint': str(checkpoint), 'saved_link_sha256': SAVED_LINK_SHA256,
        'saved_lora_sha256': SAVED_LORA_SHA256, 'changed_tensors_vs_source': changed,
        'original_16_controls_exact': True, 'saved_adapter_replay_exact': True,
        'anchors_values_unchanged': True, 'original_backbone_versions_no_grad_unchanged': True,
        'protected_files_unchanged': True, 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'peak_reserved_gpu_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'elapsed_seconds': time.monotonic() - started,
        'update': compare_records(update, before[:8], after[:8]),
        'observation': compare_records(observe, before[8:], after[8:]),
        'warning': 'Evaluation of the already saved update, not resumed training. Original after-metrics '
                   'were never measured; saved-adapter replay only proves reproducible loading now. '
                   'Memory/time are for this evaluation, not the interrupted training step. '
                   'No held-out generalization or causal proof of latent reasoning. '
                   'Original backbone checked by versions/no gradients, not a full bytewise comparison.',
    }
    write_json(output / 'summary.json', summary)
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
    print(json.dumps({k: v for k, v in summary.items() if k not in ('update', 'observation', 'changed_tensors_vs_source')},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
