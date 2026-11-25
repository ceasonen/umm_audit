#!/usr/bin/env python3
"""Locked source localization and independently validated activation controls."""
import argparse
import itertools
import json
import random
from pathlib import Path

import torch

from scripts.evaluate_cross_backbone_generation import checkpoint_route_spec, load_blip3o
from scripts.run_blip3o_readout_probe import FreshGeneration, write_json
from skillbridge.cross_backbone_evaluation import generate_fixed_predictions, generate_understanding_predictions
from skillbridge.dimoo_data import load_route_records
from skillbridge.interventions import rank_norm_matched_core_delta
from skillbridge.readout_probe import CommonInputProbe, set_core_state
from skillbridge.readout_followup import InputBank, depth_groups, select_groups, split_prompts, native_output_change


def save_state(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def targets(root, capability):
    name = 'records.jsonl' if capability == 'counting' else 'records-balanced-1000.jsonl'
    return root / f'data/oral_benchmarks_confirmatory_v1/{capability}/tiers/confirmatory' / name


def prepare_records(root, output, capability):
    source = root / f'data/route_transfer_oral_v2/capabilities/{capability}/understanding'
    dev = load_route_records(source / 'development/records.jsonl')
    test = load_route_records(source / 'test/records.jsonl')
    test_images = {Path(r['user_image']).name for r in test}
    # Selection must not use any image appearing in the source test.
    clean = [r for r in dev if Path(r['user_image']).name not in test_images]
    if len(clean) < 128:
        raise ValueError('fewer than 128 source development rows after image isolation')
    bank = list(itertools.chain.from_iterable(load_route_records(root /
        f'data/oral_benchmarks_confirmatory_v1/{c}/dsg1k/records.jsonl') for c in ('counting', 'spatial')))
    excluded = list(itertools.chain.from_iterable(load_route_records(targets(root, c)) for c in ('counting', 'spatial')))
    fit, validation = split_prompts(bank, excluded)
    write_json(output / 'data-design.json', {
        'source_dev_original': len(dev), 'source_dev_after_image_isolation': len(clean),
        'source_test': len(test), 'target_test': str(targets(root, capability)),
        'calibration_prompts': fit, 'validation_prompts': validation,
        'protocol': 'target-input-calibrated diagnostic; not strict no-target-input source-only claim',
        'selection': 'source development exact match only; target scores never used',
        'target_holdout_scope': 'literal normalized prompt disjointness; no claim of unseen dataset provenance',
    })
    return clean, test, fit, validation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capability', choices=('counting', 'spatial'), required=True)
    p.add_argument('--source-seed', type=int, required=True)
    p.add_argument('--phase', choices=('select', 'calibrate', 'understanding', 'generation'), required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    core = args.root / f'runs/oral_scale_calibrated_v2/tied_core/blip3o/{args.capability}/seed-{args.source_seed}/core.pt'
    checkpoint = torch.load(core, map_location='cpu', weights_only=True)
    full = checkpoint['state_dict']
    zero = {k: torch.zeros_like(v) for k, v in full.items()}
    dev, test, fit_records, validation_records = prepare_records(args.root, args.output, args.capability)
    gain, suffixes = checkpoint_route_spec(core)
    task = 'understanding' if args.phase in ('select', 'understanding') else 'generation'
    adapter, route, kwargs = load_blip3o(args.root, rank=64, timestep_buckets=1,
        method='tied_core', gain_limit=gain, target_suffixes=suffixes, task=task)
    fresh = FreshGeneration(adapter)
    if args.phase == 'select':
        groups = depth_groups(full)
        conditions = {'full': full}
        for i, group in enumerate(groups):
            conditions[f'without_group_{i}'] = {k: torch.zeros_like(v) if k in group else v for k, v in full.items()}
        order = list(conditions)
        random.Random(args.source_seed).shuffle(order)
        scores = {}
        for c in order:
            print(f'source selection {c}, rows={len(dev)}', flush=True)
            set_core_state(route, conditions[c])
            out = args.output / 'source-selection' / c
            generate_understanding_predictions(adapter, dev, out, metadata={'condition': c}, resume=True)
            scores[c] = json.loads((out / 'metrics.json').read_text())['exact_match']
        damages = [scores['full'] - scores[f'without_group_{i}'] for i in range(6)]
        selected, complement = select_groups(full, groups, damages)
        chosen = sorted(range(6), key=lambda i: (-damages[i], i))[:2]
        controls = [pair for pair in itertools.combinations(range(6), 2) if set(pair) != set(chosen)]
        random_pair = random.Random(9000 + args.source_seed).choice(controls)
        random_sites = {k for i in random_pair for k in groups[i]}
        states = {'source_selected': selected, 'source_complement': complement,
                  'random_groups': {k: v.clone() if k in random_sites else torch.zeros_like(v) for k, v in full.items()}}
        save_state(args.output / 'selected-states.pt', states)
        write_json(args.output / 'selection.json', {
            'scores': scores, 'damages': damages, 'groups': groups, 'selected': chosen,
            'random_groups': random_pair, 'budget_groups': 2, 'total_groups': 6,
            'selection_used_target': False, 'level': 'depth-group support, not individual semantic directions',
            'source_seed': args.source_seed, 'core': str(core)})
    elif args.phase == 'calibrate':
        selected = torch.load(args.output / 'selected-states.pt', map_location='cpu', weights_only=True)
        set_core_state(route, zero)
        candidates = {f'random_matched_{i}': rank_norm_matched_core_delta(full, seed=1000 + args.source_seed + 10000*i) for i in range(3)}
        matched_path = args.output / 'matched-states.pt'
        if not matched_path.exists():
            bank = InputBank()
            with bank.observe(route):
                for i, row in enumerate(fit_records):
                    bank.next_prompt()
                    fresh.generate(row['user_prompt'], seed=30000+i, **kwargs)
                    print(f'calibration {i+1}/{len(fit_records)}', flush=True)
            matched, calibration = bank.fit(route, full, candidates)
            write_json(args.output / 'calibration.json', calibration)
            save_state(matched_path, matched)
            del bank
        matched = torch.load(matched_path, map_location='cpu', weights_only=True)
        states = {'full': full, **selected, **matched, 'raw_random': candidates['random_matched_0']}
        probe = CommonInputProbe(states, change_function=native_output_change)
        with probe.observe(route):
            for i, row in enumerate(validation_records):
                fresh.generate(row['user_prompt'], seed=40000+i, **kwargs)
                print(f'validation {i+1}/{len(validation_records)}', flush=True)
        report = probe.report()
        validation = {}
        calibration = json.loads((args.output / 'calibration.json').read_text())
        for c in matched:
            ratios = [v['conditions'][c]['relative_energy_to_full'] for v in report['layers'].values()]
            finite = [r for r in ratios if r is not None]
            ok = len(finite) == len(ratios) and all(.8 <= r <= 1.2 for r in finite)
            validation[c] = {'all_sites_within_20_percent': ok,
                             'calibration_all_sites_within_2_percent': all(v['matched'] for v in calibration[c].values()),
                             'sites_outside_tolerance': sum(r is None or not .8 <= r <= 1.2 for r in ratios),
                             'total_sites': len(ratios),
                             'policy': 'retain all results, never call failed matches energy-matched evidence'}
        write_json(args.output / 'validation-activation.json', report)
        write_json(args.output / 'validation-status.json', validation)
    else:
        states = {**torch.load(args.output / 'selected-states.pt', map_location='cpu', weights_only=True),
                  **torch.load(args.output / 'matched-states.pt', map_location='cpu', weights_only=True)}
        # Baselines reuse the completed identical-seed, identical-cache-policy v1 run.
        order = list(states)
        random.Random(args.source_seed + 100).shuffle(order)
        write_json(args.output / task / 'design.json', {'order': order, 'inference_seed': 8087,
            'source_seed': args.source_seed, 'baseline_campaign': 'blip3o_readout_probe_v1',
            'validation': json.loads((args.output / 'validation-status.json').read_text())})
        if task == 'understanding':
            set_core_state(route, zero)
            probe = CommonInputProbe({'full': full, **states}, change_function=native_output_change)
            with probe.observe(route):
                for row in dev[:32]:
                    adapter.understand(Path(row['user_image']), row['user_prompt'])
            write_json(args.output / task / 'same-input-activation.json', probe.report())
        records = test if task == 'understanding' else load_route_records(targets(args.root, args.capability))
        for c in order:
            set_core_state(route, states[c])
            print(f'evaluate {task} {c} rows={len(records)}', flush=True)
            out = args.output / task / c
            if task == 'understanding':
                generate_understanding_predictions(adapter, records, out, metadata={'condition': c}, resume=True)
            else:
                generate_fixed_predictions(fresh, records, out, seed=8087, generation_kwargs=kwargs,
                                           metadata={'condition': c}, resume=True)
        write_json(args.output / task / 'complete.json', {'conditions': order, 'rows': len(records)})
    write_json(args.output / f'{args.phase}-complete.json', {'source_seed': args.source_seed})


if __name__ == '__main__':
    main()
