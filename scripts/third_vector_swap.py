"""Fixed 72*2-4 history/third-vector swap helpers; CLI lives in fixed_vector_execution."""
from fixed_vector_execution import fixed_rollout, compare_fixed, require_fixed_repeat
from self_conditioned_swap import read_json, read_rows, VERSIONS
from lora_self_conditioned_evaluate import require_replay
from format_experiment import sha256
from link_execution_probe import measurements

CASE_ID = 'mul-sub-72-2-4'
FIXED_COMMIT = 'abf88ad1bc541ca77d60f69778deb0caf853f895'
FIXED_SCRIPT_SHA256 = 'dc5568ac873cc0ea17be24b497ed32a83d6261eb2f573981aac116b017c98651'
COMBINATIONS = (('full', 'full'), ('cached', 'cached'), ('full', 'cached'), ('cached', 'full'))


def verify_fixed_run(path, protected, plan, expected):
    """Bind saved vectors to the completed sixteen-record A run, before model loading."""
    run, summary = read_json(path / 'run.json'), read_json(path / 'summary.json')
    for k, value in {'code_commit': FIXED_COMMIT, 'script_sha256': FIXED_SCRIPT_SHA256,
                     'pair': 'A', 'optimizer_steps': 0, 'weights_saved': False,
                     'sample_ids': [r['id'] for r in expected],
                     'protected_hashes': {str(p): h for p, h in protected.items()},
                     'calibration': plan['calibration'], 'seed': plan['seed'],
                     'tokenization_sha256': plan['tokenization_sha256'],
                     **{k: plan[k] for k in VERSIONS}}.items():
        require_replay(run[k], value, f'fixed run {k}')
    for k, value in {'count': 16, 'pair': 'A', 'optimizer_steps': 0, 'weights_saved': False,
                     **dict.fromkeys(('historical_32_path_controls_exact', 'fixed_self_replay_exact',
                        'fixed_repeats_hidden_logits_exact', 'same_cast_inputs_exact',
                        'parameters_versions_no_grad_unchanged', 'anchors_values_unchanged',
                        'protected_files_unchanged', 'vectors_saved'), True)}.items():
        require_replay(summary[k], value, f'fixed summary {k}')
    rows = read_rows(path / 'measurements.jsonl')
    controls = read_rows(path / 'controls.jsonl')
    require_replay(rows, summary['results'], 'fixed results')
    require_replay([r['id'] for r in rows], run['sample_ids'], 'fixed ordered IDs')
    require_replay(controls, [{'id': r['id'], 'full': r['full'], 'cached': r['cached']}
                              for r in expected], 'fixed historical controls')
    for row, control in zip(rows, controls):
        require_replay(row['recursive'], {k: control[k] for k in ('full', 'cached')}, 'fixed recursive')
        if sha256(path / f'vectors-{row["id"]}.pt') != row['vectors_sha256']:
            raise ValueError('saved vector hash differs')
    row = next(r for r in rows if r['id'] == CASE_ID)
    return {'row': row, 'vectors_path': path / f'vectors-{CASE_ID}.pt'}


def mix_vectors(saved, history, third):
    if set(saved) != {'full', 'cached'} or any(len(v) != 6 for v in saved.values()):
        raise ValueError('requires both six-vector donors')
    if history not in saved or third not in saved:
        raise ValueError('unknown donor')
    return tuple(v.detach().clone() for v in (*saved[history][:2], saved[third][2]))


def prefix_metrics(metrics):
    result = {k: v[:3] for k, v in metrics.items() if k != 'mean_ce'}
    result['mean_ce'] = sum(result['step_ce']) / 3
    return result


def require_history(first, second):
    """Changing only the third input must not change states 0..2."""
    import torch
    for key in ('hidden', 'logits'):
        if len(first[key]) < 3 or len(second[key]) < 3:
            raise ValueError('missing pre-third history states')
        if any(a.dtype != b.dtype or not torch.equal(a, b)
               for a, b in zip(first[key][:3], second[key][:3])):
            raise RuntimeError('history changed before third input')


def third_effects(results):
    pairs = {
        'third_with_full_history': ('full/full', 'full/cached'),
        'third_with_cached_history': ('cached/full', 'cached/cached'),
        'history_with_full_third': ('full/full', 'cached/full'),
        'history_with_cached_third': ('full/cached', 'cached/cached'),
    }
    effects = {}
    for name, (a, b) in pairs.items():
        effects[name] = {'before': a, 'after': b, 'paths': {}}
        for path in ('full', 'cached'):
            x, y = [results[k]['metrics'][path] for k in (a, b)]
            require_replay(x['target_ids'], y['target_ids'], 'effect targets')
            effects[name]['paths'][path] = {
                'delta_third_ce': y['step_ce'][2] - x['step_ce'][2],
                'before_probability': x['target_probabilities'][2],
                'after_probability': y['target_probabilities'][2],
                'before_prediction': x['predicted_texts'][2],
                'after_prediction': y['predicted_texts'][2],
                'before_correct': x['predicted_ids'][2] == x['target_ids'][2],
                'after_correct': y['predicted_ids'][2] == y['target_ids'][2]}
    interaction = {path: effects['third_with_cached_history']['paths'][path]['delta_third_ce'] -
                         effects['third_with_full_history']['paths'][path]['delta_third_ce']
                   for path in ('full', 'cached')}
    return {'effects': effects, 'third_ce_interaction_cached_minus_full_history': interaction}


def execute_combinations(evaluate, expected, emit):
    """Both historical donors on both paths gate all mixed runs."""
    results, baselines = {}, {}
    for donor in ('full', 'cached'):
        outputs, metrics = evaluate(donor, donor, 6)
        for path in ('full', 'cached'):
            require_replay(metrics[path], expected['fixed'][donor]['metrics'][path], f'{donor}/{path} historical')
        comparison = compare_fixed(outputs['full'], outputs['cached'], metrics['full'], metrics['cached'])
        require_replay(comparison, expected['fixed'][donor]['comparison'], f'{donor} historical comparison')
        emit({'phase': 'historical', 'donor': donor, 'metrics': metrics, 'comparison': comparison})
        baselines[donor] = outputs
    histories = {}
    for history, third in COMBINATIONS:
        outputs, metrics = evaluate(history, third, 3)
        for path in ('full', 'cached'):
            if history == third:
                require_replay(metrics[path], prefix_metrics(expected['fixed'][history]['metrics'][path]),
                               f'{history}/{path} truncated metrics')
                require_history(baselines[history][path], outputs[path])
                # Include the third output when checking truncation of the six-step control.
                import torch
                for key in ('hidden', 'logits'):
                    a, b = baselines[history][path][key][3], outputs[path][key][3]
                    if a.dtype != b.dtype or not torch.equal(a, b):
                        raise RuntimeError('third state changed on truncation')
            else:
                require_history(histories[history][path], outputs[path])
        if history == third:
            histories[history] = outputs
        label = f'{history}/{third}'
        results[label] = {'history_donor': history, 'third_donor': third, 'metrics': metrics,
                         'comparison': compare_fixed(outputs['full'], outputs['cached'], metrics['full'], metrics['cached'])}
        emit({'phase': 'three_inputs', 'combination': label, **results[label]})
    return {'combinations': results, **third_effects(results)}


def run_third_swap(raw, sample, entry, exit, tokenizer, reference, output):
    import json
    import torch
    path = reference['vectors_path']
    if sha256(path) != reference['row']['vectors_sha256']:
        raise ValueError('vector hash changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    # Validate the entire saved pack before any decoder work, including unused suffixes.
    hidden = raw.backbone.get_input_embeddings().weight.shape[1]
    if set(saved) != {'full', 'cached'} or any(len(vs) != 6 for vs in saved.values()) or any(
            v.shape != (1, 1, hidden) or v.dtype != torch.float32 or not torch.isfinite(v).all()
            for vs in saved.values() for v in vs):
        raise ValueError('invalid saved vector pack')
    before = {k: [v.clone() for v in vs] for k, vs in saved.items()}
    def evaluate(history, third, steps):
        vectors = saved[history] if steps == 6 else mix_vectors(saved, history, third)
        outputs, metrics = {}, {}
        for path in ('full', 'cached'):
            out = fixed_rollout(raw, sample, entry, exit, vectors, path, steps=steps)
            repeat = fixed_rollout(raw, sample, entry, exit, vectors, path, steps=steps)
            require_fixed_repeat(out, repeat, tokenizer)
            outputs[path] = out
            metrics[path], _ = measurements(out, tokenizer)
        return outputs, metrics
    with (output / 'third-vector.jsonl').open('x') as stream:
        def emit(item):
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            print(item['phase'], item.get('combination', item.get('donor')), flush=True)
        results = execute_combinations(evaluate, reference['row'], emit)
    if any(not torch.equal(v, b) for k in saved for v, b in zip(saved[k], before[k])):
        raise RuntimeError('saved vectors mutated')
    return {'id': CASE_ID, 'vectors_sha256': sha256(reference['vectors_path']), **results}
