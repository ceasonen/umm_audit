#!/usr/bin/env python3
"""Count-specific donor deltas and conditional/unconditional path interventions."""
import argparse
import json
import os
import random
from pathlib import Path

import torch

from scripts.evaluate_cross_backbone_generation import checkpoint_route_spec, load_blip3o
from scripts.score_spatial_counterfactual import validate_checkpoint_metadata
from scripts.run_blip3o_readout_probe import write_json
from skillbridge.dimoo_data import load_route_records
from skillbridge.mechanism_sequence import (
    capture_condition, condition_interventions, counting_pairs, lock_json, replace_condition, validate_gpu,
)
from skillbridge.readout_probe import set_core_state


def spatial_pairs(records):
    """Build literal same-subject/object relation counterfactuals."""
    import re
    pattern = re.compile(r"^(.+?) (on the top of|on the bottom of|on the left of|on the right of|next to|near|on side of) (.+)$")
    parsed = []
    for row in records:
        match = pattern.fullmatch(row['user_prompt'].strip())
        if match:
            parsed.append((row, match.group(1), match.group(2), match.group(3)))
    pairs = []
    for row, subject, relation, obj in parsed:
        semantic = [r for r, s, rel, o in parsed if s == subject and o == obj and rel != relation]
        nuisance = [r for r, s, rel, o in parsed if rel == relation and (s != subject or o != obj)]
        if semantic and nuisance:
            pairs.append(dict(recipient=row,
                semantic_donor=sorted(semantic, key=lambda r: (r['user_prompt'], r['split_key']))[0],
                nuisance_donor=sorted(nuisance, key=lambda r: (r['user_prompt'], r['split_key']))[0]))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--capability', choices=('counting', 'spatial'), default='counting')
    p.add_argument('--max-pairs', type=int, default=None)
    args = p.parse_args()
    validate_gpu(int(os.environ.get('CUDA_VISIBLE_DEVICES', '-1')))
    records = load_route_records(args.root / f'data/oral_benchmarks_confirmatory_v1/{args.capability}/tiers/confirmatory/records.jsonl')
    pairs = counting_pairs(records) if args.capability == 'counting' else spatial_pairs(records)
    if args.max_pairs is not None:
        if args.max_pairs < 1 or len(pairs) < args.max_pairs:
            raise ValueError('max-pairs must be positive and no larger than eligible pairs')
        pairs = pairs[:args.max_pairs]
    if not pairs:
        raise ValueError(f'no literal same-input {args.capability} counterfactuals exist in the benchmark')
    lock_json(args.output / 'design.json', dict(pairs=pairs, source_seed=args.seed,
        core=str(args.core), capability=args.capability,
        selection='all eligible input-only literal counterfactual pairs',
        scope='target-input diagnostic at AR-to-diffusion interface, not parameter portability',
        decoder_rng='reset at interface; separate protocol from earlier generation runs',
        donor_selection_uses_scores=False, donor_energy_tolerance=.2,
        inference_unit='training seed crossed with recipient/donor object families'))
    target_rows = [p['recipient'] for p in pairs]
    records_path = args.output / 'records.jsonl'
    text = ''.join(json.dumps(r) + '\n' for r in target_rows)
    if records_path.exists() and records_path.read_text() != text:
        raise ValueError('counterfactual recipient records changed')
    records_path.write_text(text)
    ck = torch.load(args.core, map_location='cpu', weights_only=True)
    md = ck['metadata']
    validate_checkpoint_metadata(md, args.capability, args.seed)
    full = ck['state_dict']
    zero = {k: torch.zeros_like(v) for k, v in full.items()}
    gain, suffixes = checkpoint_route_spec(args.core)
    adapter, route, kwargs = load_blip3o(args.root, rank=int(md['rank']), timestep_buckets=1,
        method=md['method'], gain_limit=gain, target_suffixes=suffixes, task='generation')
    condition_rows = {}
    for i, pair in enumerate(pairs):
        unit = args.output / 'units' / f'{i:05d}'
        marker = unit / 'complete.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            for name, row in saved['predictions'].items():
                if not (args.output / name / row['image']).is_file():
                    raise ValueError('completed interface unit has missing image')
                condition_rows.setdefault(name, []).append(row)
            continue
        embeddings = {}
        for role in ('recipient', 'semantic_donor', 'nuisance_donor'):
            prompt = pair[role]['user_prompt']
            for condition, state in (('zero', zero), ('full', full)):
                set_core_state(route, state)
                embeddings[f'{role}_{condition}'] = capture_condition(adapter, prompt, seed=8087+i, kwargs=kwargs).to(
                    dtype=adapter.diffusion_pipeline.dtype(adapter.diffusion_pipeline.unet))
        states, audit = condition_interventions(
            embeddings['recipient_zero'], embeddings['recipient_full'],
            embeddings['semantic_donor_zero'], embeddings['semantic_donor_full'],
            embeddings['nuisance_donor_zero'], embeddings['nuisance_donor_full'], seed=1000+args.seed)
        unit.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, unit / 'interfaces.pt')
        write_json(unit / 'activation-audit.json', audit)
        # Compare replay with a real full-core encoder execution, not merely a
        # second call to the same replacement. All arms use the same decoder RNG.
        set_core_state(route, full)
        adapter.reset_condition_cache()
        with replace_condition(adapter.diffusion_pipeline, states['full'], seed=8087+i, verify_native=True):
            native_image = adapter.generate(pair['recipient']['user_prompt'], seed=8087+i, **kwargs)
        native_image.save(unit / 'native-full.png')
        order = list(states)
        random.Random(args.seed + i).shuffle(order)
        output_rows = {}
        for name in order:
            state = states[name]
            set_core_state(route, zero)
            adapter.reset_condition_cache()
            with replace_condition(adapter.diffusion_pipeline, state, seed=8087+i):
                image = adapter.generate(pair['recipient']['user_prompt'], seed=8087+i, **kwargs)
            out = args.output / name / 'images' / f'{i:05d}.png'
            out.parent.mkdir(parents=True, exist_ok=True)
            image.save(out)
            row = dict(split_key=pair['recipient']['split_key'], prompt=pair['recipient']['user_prompt'],
                       seed=8087+i, image=f'images/{i:05d}.png', condition=name,
                       semantic_donor=pair['semantic_donor']['user_prompt'],
                       nuisance_donor=pair['nuisance_donor']['user_prompt'])
            output_rows[name] = row
            condition_rows.setdefault(name, []).append(row)
        # This replay checks deterministic interface restoration only.
        from PIL import Image
        with Image.open(args.output / 'full' / output_rows['full']['image']) as a, Image.open(
            args.output / 'interface_replay' / output_rows['interface_replay']['image']) as b:
            if a.size != b.size or a.tobytes() != b.tobytes() or a.tobytes() != native_image.tobytes():
                raise ValueError('interface replay image differs from full-condition image')
        write_json(marker, {'predictions': output_rows, 'interface_replay_exact': True,
            'donor_norm_matching_passed': all(audit[c]['passed'] for c in ('semantic_donor', 'nuisance_donor'))})
        print(f'interface pair {i+1}/{len(pairs)}', flush=True)
    for name, rows in condition_rows.items():
        path = args.output / name / 'predictions.jsonl'
        temporary = path.with_suffix('.partial')
        temporary.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        temporary.replace(path)
    write_json(args.output / 'complete.json', dict(rows=len(pairs), conditions=list(condition_rows),
        status='generated; official scoring and semantic tests still required'))


if __name__ == '__main__':
    main()
