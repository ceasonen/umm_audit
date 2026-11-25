#!/usr/bin/env python3
"""Score spatial counterfactual images under recipient and donor prompts."""
from __future__ import annotations
import argparse, json, subprocess, os
from pathlib import Path


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def validate_checkpoint_metadata(metadata, capability, seed):
    """Lock the semantic campaign to the focused BLIP3-o rank-64 core."""
    if (metadata.get('backbone'), metadata.get('capability'), int(metadata.get('seed', -1))) != (
            'blip3o', capability, seed):
        raise ValueError('expected matching BLIP3-o capability checkpoint')
    if int(metadata.get('rank', -1)) != 64 or metadata.get('method') != 'tied_core':
        raise ValueError('expected rank-64 tied_core checkpoint')


def validate_condition_rows(pairs, rows, condition_root, *, condition_name=None,
                            require_passed_gates=False):
    """Validate recipient identity, image provenance, and per-unit gates."""
    if len(rows) != len(pairs):
        raise ValueError('invalid spatial prediction family length')
    by_key = {}
    for row in rows:
        key = row.get('split_key')
        if not key or key in by_key:
            raise ValueError('duplicate or missing spatial split_key')
        by_key[key] = row
    root = condition_root.resolve()
    for i, pair in enumerate(pairs):
        expected = pair['recipient']
        row = by_key.get(expected['split_key'])
        if row is None:
            raise ValueError('missing recipient split_key')
        if row.get('prompt') != expected['user_prompt']:
            raise ValueError('recipient prompt does not match design')
        if row.get('seed') != 8087 + i:
            raise ValueError('recipient decoder seed does not match design')
        if condition_name is not None and row.get('condition') != condition_name:
            raise ValueError('condition metadata does not match scoring arm')
        image = (root / row.get('image', '')).resolve()
        if root not in image.parents or not image.is_file():
            raise ValueError('spatial image path is missing or escapes condition root')
        if require_passed_gates:
            audit_path = root.parent / 'units' / f'{i:05d}' / 'activation-audit.json'
            if not audit_path.is_file():
                raise ValueError('missing activation audit for spatial unit')
            audit = json.loads(audit_path.read_text())
            if not all(audit.get(name, {}).get('passed')
                       for name in ('semantic_donor', 'nuisance_donor')):
                raise ValueError('donor interface-energy gate failed')
    return by_key


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--interface', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    design = json.loads((args.interface/'design.json').read_text())
    complete = json.loads((args.interface/'complete.json').read_text())
    pairs = design['pairs']
    conditions = tuple(complete['conditions'])
    if complete['rows'] != len(pairs) or set(conditions) != {'zero','full','conditional_only','unconditional_only','interface_replay','semantic_donor','nuisance_donor','random_sign'}:
        raise ValueError('incomplete spatial interface family')
    source_records = {r['split_key']: r for r in read_jsonl(args.interface/'records.jsonl')}
    for condition in conditions:
        rows = read_jsonl(args.interface/condition/'predictions.jsonl')
        by_key = validate_condition_rows(pairs, rows, args.interface/condition,
                                         condition_name=condition,
                                         require_passed_gates=True)
        for criterion in ('recipient', 'donor'):
            records, predictions = [], []
            for i, pair in enumerate(pairs):
                target = pair['semantic_donor'] if criterion == 'donor' else pair['recipient']
                src = by_key[pair['recipient']['split_key']]
                key = f'{pair["recipient"]["split_key"]}::{criterion}'
                rec = dict(target)
                rec['split_key'] = key
                rec['benchmark_id'] = str(i)
                pred = dict(src)
                pred['split_key'] = key
                pred['prompt'] = target['user_prompt']
                records.append(rec); predictions.append(pred)
            out = args.output/condition/criterion
            out.mkdir(parents=True, exist_ok=True)
            image_dir = out/'images'
            image_dir.mkdir(exist_ok=True)
            for src in (args.interface/condition/'images').glob('*.png'):
                dst = image_dir/src.name
                if not dst.exists():
                    os.symlink(src.resolve(), dst)
            (out/'records.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))
            (out/'predictions.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in predictions))
            stage = out/'official-stage'
            if not stage.exists():
                subprocess.run([str(args.root/'environments/dimoo/.venv/bin/python'), '-P', str(args.root/'scripts/stage_oral_predictions.py'), '--records', str(out/'records.jsonl'), '--predictions', str(out/'predictions.jsonl'), '--output', str(stage)], check=True)
            summary = stage/'t2i_compbench/official-summary.json'
            if not summary.exists():
                py = args.root/'environments/t2i_compbench/.venv/bin/python'
                subprocess.run([str(py), '-P', str(args.root/'scripts/run_t2i_compbench_official.py'), '--repository', str(args.root/'external/T2I-CompBench'), '--lock', str(args.root/'configs/evaluators.lock.json'), '--staged-root', str(stage/'t2i_compbench'), '--output', str(summary), '--python', str(py), '--no-revision-verification'], check=True)
    print(json.dumps({'conditions': conditions, 'pairs': len(pairs), 'scoring': 'complete'}))


if __name__ == '__main__':
    main()
