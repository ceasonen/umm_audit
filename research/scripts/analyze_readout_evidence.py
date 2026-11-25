#!/usr/bin/env python3
"""Reconstruct paired U/G contrasts from raw predictions and official item scores."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from skillbridge.evaluation import answers_match
from skillbridge.readout_followup import crossed_interval


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def contrast_label(result):
    name = result['contrast']
    if name == 'full-matched-random-mean' and not result.get(
        'all_random_controls_pass_target_activation_validation', False
    ):
        return name + '（匹配未通过；仅描述性结果）'
    if name in ('full-sign-random-mean', 'full-mirror') and not result.get(
        'activation_control_integrity_passed', False
    ):
        return name + '（激活控制未通过；仅描述性结果）'
    return name


def endpoint(run, task):
    predictions = lines(run / 'predictions.jsonl')
    if task == 'understanding':
        keys = [(r['image'], r['question'], r['reference']) for r in predictions]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate understanding evaluation units')
        scores = [float(answers_match(r['capability'], r['reference'], r['prediction'])) for r in predictions]
        saved = json.loads((run / 'metrics.json').read_text())
        if (not scores or not np.isfinite(saved['exact_match'])
                or abs(np.mean(scores) - saved['exact_match']) > 1e-8):
            raise ValueError('raw understanding scores do not reproduce aggregate')
        strata = [r['reference'] for r in predictions]
        return keys, np.asarray(scores), strata
    stage = run / 'official-stage'
    summary = json.loads((stage / 't2i_compbench/official-summary.json').read_text())
    index = lines(stage / 'index.jsonl')
    categories = {}
    for c, info in summary['categories'].items():
        # Resolve relative to this copy, not the original absolute remote path.
        relative = info['result'].split('/t2i_compbench/', 1)[1]
        items = lines(stage / 't2i_compbench' / relative) if relative.endswith('.jsonl') else json.loads((stage / 't2i_compbench' / relative).read_text())
        values = {int(r['question_id']): float(r['answer']) for r in items}
        if len(values) != len(items) or len(items) != info['rows']:
            raise ValueError('duplicate or missing official question IDs')
        categories[c] = values
    pred = {r['split_key']: r for r in predictions}
    index_ids = [r['split_key'] for r in index]
    benchmark_ids = [(r['image'].split('/')[1], int(r['benchmark_id'])) for r in index]
    if (len(pred) != len(predictions) or len(set(index_ids)) != len(index_ids)
            or len(set(benchmark_ids)) != len(benchmark_ids)
            or set(index_ids) != set(pred)):
        raise ValueError('duplicate or incomplete official/prediction pairing')
    scores, keys, strata = [], [], []
    for row in index:
        c = row['image'].split('/')[1]
        r = pred[row['split_key']]
        keys.append((c, int(row['benchmark_id']), r['prompt'], r['seed']))
        scores.append(categories[c][int(row['benchmark_id'])])
        strata.append(c)
    if len(keys) != len(predictions) or len(keys) != summary['rows']:
        raise ValueError('official/prediction row count mismatch')
    scores = np.asarray(scores)
    if (not scores.size or not np.isfinite(scores).all()
            or not np.isfinite(summary['micro_average'])
            or abs(scores.mean() - summary['micro_average']) > 1e-8):
        raise ValueError('raw official scores do not reproduce reported aggregate')
    return keys, scores, strata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--followup', type=Path)
    p.add_argument('--selection-root', type=Path, help='reuse completed source-only selected/complement/group controls')
    p.add_argument('--activation-root', type=Path, help='separate native-output sign-control campaign')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    seeds = [17, 29, 43, 71]
    conditions = ['zero', 'full', 'top64', 'complement64', 'random']
    new = ['source_selected', 'source_complement', 'random_groups', 'random_matched_0', 'random_matched_1', 'random_matched_2']
    signs = ['mirror', 'sign_random_0', 'sign_random_1', 'sign_random_2'] if args.activation_root else []
    contrasts = {'full-zero': {'full': 1, 'zero': -1},
                 'full-random': {'full': 1, 'random': -1},
                 'top64-zero': {'top64': 1, 'zero': -1},
                 'complement64-zero': {'complement64': 1, 'zero': -1},
                 'interaction': {'full': 1, 'top64': -1, 'complement64': -1, 'zero': 1}}
    if args.followup:
        contrasts.update({'selected-zero': {'source_selected': 1, 'zero': -1},
            'selected-random-groups': {'source_selected': 1, 'random_groups': -1},
            'selected-complement': {'source_selected': 1, 'source_complement': -1},
            'full-matched-random-mean': {'full': 1, **{f'random_matched_{i}': -1/3 for i in range(3)}}})
    if args.activation_root:
        contrasts.update({'full-mirror': {'full': 1, 'mirror': -1},
            'full-sign-random-mean': {'full': 1, **{f'sign_random_{i}': -1/3 for i in range(3)}}})
    results, endpoint_rows, missing = [], [], []
    validation_status = {}
    if args.followup:
        for cap, seed in itertools.product(('counting', 'spatial'), seeds):
            path = args.followup / cap / f'seed-{seed}' / 'validation-status.json'
            validation_status[f'{cap}/seed-{seed}'] = json.loads(path.read_text()) if path.exists() else None
    for capability, task in itertools.product(('counting', 'spatial'), ('understanding', 'generation')):
        arrays, reference_keys, strata = {}, None, None
        for c in conditions + (new if args.followup else []) + signs:
            matrices = []
            for seed in seeds:
                root = args.root if c in conditions else args.followup
                if c in signs:
                    root = args.activation_root
                if args.selection_root and c in ('source_selected', 'source_complement', 'random_groups'):
                    root = args.selection_root
                run = root / capability / f'seed-{seed}' / task / c
                try:
                    keys, scores, labels = endpoint(run, task)
                except FileNotFoundError:
                    missing.append(str(run))
                    continue
                if reference_keys is None:
                    reference_keys, strata = keys, labels
                if keys != reference_keys or labels != strata:
                    raise ValueError(f'pairing mismatch: {run}')
                matrices.append(scores)
                endpoint_rows.append({
                    'capability': capability,
                    'task': task,
                    'condition': c,
                    'source_seed': seed,
                    'n': len(scores),
                    'metric': ('exact_match_accuracy' if task == 'understanding'
                               else 't2i_compbench_official_score'),
                    'unit': ('accuracy_percent' if task == 'understanding' else 'score_x100'),
                    'value': scores.mean() * 100,
                })
            if len(matrices) == len(seeds):
                arrays[c] = np.stack(matrices)
        for name, weights in contrasts.items():
            if not all(c in arrays for c in weights):
                continue
            differences = sum(arrays[c] * weight for c, weight in weights.items())
            subgroups = ['all'] + (sorted(set(strata)) if task == 'generation' and capability == 'spatial' else [])
            for subgroup in subgroups:
                mask = np.ones(len(strata), bool) if subgroup == 'all' else np.asarray(strata) == subgroup
                result = crossed_interval(differences[:, mask], strata=np.asarray(strata)[mask])
                means = differences[:, mask].mean(axis=1)
                permuted = [abs(np.mean(means * signs)) for signs in itertools.product((-1, 1), repeat=len(seeds))]
                result['seed_signflip_p_exploratory'] = float(np.mean(np.asarray(permuted) >= abs(means.mean()) - 1e-12))
                result.update(capability=capability, task=task, contrast=name, subgroup=subgroup)
                if name == 'full-matched-random-mean':
                    gates = [validation_status.get(f'{capability}/seed-{s}') for s in seeds]
                    result['all_random_controls_pass_target_activation_validation'] = all(
                        bool(g) and all(g[f'random_matched_{i}']['all_sites_within_20_percent']
                            and g[f'random_matched_{i}']['calibration_all_sites_within_2_percent']
                            for i in range(3)) for g in gates)
                if name in ('full-mirror', 'full-sign-random-mean'):
                    passed = True
                    for seed in seeds:
                        run = args.activation_root / capability / f'seed-{seed}'
                        validation_file = run / 'validation-complete.json'
                        validation = json.loads(validation_file.read_text()) if validation_file.exists() else {}
                        for condition in weights:
                            if condition not in signs:
                                continue
                            done = run / task / condition / 'condition-complete.json'
                            endpoint_gate = json.loads(done.read_text()).get('gate', {}) if done.exists() else {}
                            passed = passed and validation.get('conditions', {}).get(condition, {}).get('passed') is True
                            passed = passed and endpoint_gate.get('passed') is True
                    result['activation_control_integrity_passed'] = bool(passed)
                    result['control_level'] = 'native output signs, not random parameter cores'
                results.append(result)
    # Holm correction over every reported contrast/subgroup; no significance shopping.
    ordered = sorted(range(len(results)), key=lambda i: results[i]['seed_signflip_p_exploratory'])
    previous = 0.
    for rank, i in enumerate(ordered):
        previous = max(previous, min(1., results[i]['seed_signflip_p_exploratory'] * (len(results)-rank)))
        results[i]['seed_signflip_p_holm'] = previous
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'results': results, 'missing': missing, 'target_activation_validation': validation_status,
              'notes': ['Already observed v1 contrasts are exploratory, not newly preregistered.',
                        'Four training seeds are not 4000 independent training replicates.',
                        'Fixed inference seed per prompt: inference-seed variability is not separately identified.',
                        'Random directions are averaged within source seed, never counted as independent training seeds.',
                        'Matched random comparisons require the separately reported per-site validation gates.',
                        'Output-sign controls preserve local magnitudes but retain amplitude/support information; not semantic swaps.',
                        'Subspace interaction is endpoint nonadditivity, not proof of a unique semantic mechanism.']}
    (args.output / 'paired-statistics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (args.output / 'endpoints.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    md = ['# 配对机制证据', '', '理解端点效应单位为 accuracy percentage points；生成端点效应单位为 official continuous score ×100。区间为配对的 seed × 样本 bootstrap 95% 区间。已有数据分析属于探索性分析；四个训练 seed 不足以支持强总体推断。', '',
          '| 能力 | 端点 | 对比 | 效应 | 95% 区间 | 正/负 seed |', '|---|---|---|---:|---|---|']
    for r in results:
        lo, hi = r['ci95_pp']
        md.append(f"| {r['capability']} / {r['subgroup']} | {r['task']} | {contrast_label(r)} | {r['mean_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['positive_seeds']}/{r['negative_seeds']} |")
    md.extend(['', f'未完成条件数：{len(missing)}。缺失数据未填零。', '', '解释限制：激活匹配不等于语义匹配；层组选择不等于识别独立语义方向。full / top / complement 的非加性也可能来自有界调制和下游网络非线性。'])
    (args.output / 'analysis.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'contrasts': len(results), 'missing_conditions': len(missing)}))
    if missing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
