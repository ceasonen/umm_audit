#!/usr/bin/env python3
"""Source-only selection, shortcut checks, and matched U/G endpoint execution."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch

from scripts.evaluate_cross_backbone_generation import checkpoint_route_spec, load_blip3o, load_janusflow
from scripts.run_blip3o_readout_probe import FreshGeneration, write_json
from skillbridge.cross_backbone_evaluation import (
    generate_fixed_predictions, generate_understanding_predictions, generate_route_interventions,
)
from skillbridge.dimoo_data import load_route_records
from skillbridge.mechanism_sequence import image_permutation, lock_json, singular_bands, selected_band_states, validate_gpu
from skillbridge.readout_followup import native_output_change
from skillbridge.readout_probe import CommonInputProbe, set_core_state


class FreshAdapter(FreshGeneration):
    def generate(self, prompt, *, seed, **kwargs):
        reset = getattr(self.adapter, "reset_condition_cache", None)
        if reset is not None:
            reset()
        return self.adapter.generate(prompt, seed=seed, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capability', choices=('counting', 'spatial'), required=True)
    p.add_argument('--backbone', choices=('blip3o', 'janusflow'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--phase', choices=('shortcut', 'select', 'understanding', 'generation', 'replay'), required=True)
    p.add_argument('--family', choices=('base', 'directions'), default='base')
    args = p.parse_args()
    validate_gpu(int(os.environ.get('CUDA_VISIBLE_DEVICES', '-1')))
    ck = torch.load(args.core, map_location='cpu', weights_only=True)
    metadata = ck['metadata']
    if metadata['backbone'] != args.backbone or int(metadata['seed']) != args.seed:
        raise ValueError('checkpoint/backbone/seed mismatch')
    # Cross-capability target evaluations are allowed only as disclosed diagnostics.
    full = ck['state_dict']
    zero = {k: torch.zeros_like(v) for k, v in full.items()}
    gain, suffixes = checkpoint_route_spec(args.core)
    task = 'generation' if args.phase in ('generation', 'replay') else 'understanding'
    options = dict(rank=int(metadata['rank']), timestep_buckets=int(metadata['timestep_buckets']),
                   method=metadata['method'], gain_limit=gain, target_suffixes=suffixes)
    if args.backbone == 'blip3o':
        adapter, route, kwargs = load_blip3o(args.root, task=task, **options)
    else:
        adapter, route, kwargs = load_janusflow(args.root, **options)
    fresh = FreshAdapter(adapter)
    data = args.root / f'data/route_transfer_oral_v2/capabilities/{args.capability}/understanding'
    test = load_route_records(data / 'test/records.jsonl')
    dev = load_route_records(data / 'development/records.jsonl')
    train_inputs = args.core.parent / 'training-inputs.jsonl'
    if not train_inputs.exists():
        raise ValueError('new campaign requires explicit recorded training inputs')
    train = load_route_records(train_inputs)
    train_images = {Path(r['user_image']).name for r in train}
    test_images = {Path(r['user_image']).name for r in test}
    if train_images & test_images:
        raise ValueError('source train/test image overlap')
    dev = [r for r in dev if Path(r['user_image']).name not in train_images | test_images]
    if len(dev) < 128:
        raise ValueError('insufficient disjoint source development records')
    target_name = 'records.jsonl' if args.capability == 'counting' else 'records-balanced-1000.jsonl'
    target_path = args.root / f'data/oral_benchmarks_confirmatory_v1/{args.capability}/tiers/confirmatory' / target_name
    output = args.output / args.family
    output.mkdir(parents=True, exist_ok=True)
    lock_json(output / f'{args.phase}-design.json', dict(
        core=str(args.core), checkpoint_method=metadata['method'], checkpoint_capability=metadata['capability'],
        test_capability=args.capability, backbone=args.backbone, source_seed=args.seed,
        source_test_rows=len(test), source_development_rows=len(dev),
        phase=args.phase, family=args.family, generation_kwargs=kwargs, inference_seed=8087,
        source_selection_uses_target=False, cache_policy='reset per image',
        interpretation='instrument-specific; cross-backbone difference is not architectural causality'))
    if args.phase == 'select':
        if args.family != 'directions' or metadata['capability'] != args.capability:
            raise ValueError('direction selection requires matching source capability')
        bands = singular_bands(full, groups=8)
        states = {'full': full, **{f'without_band_{i}': {k: v - b[k] for k, v in full.items()}
                                 for i, b in enumerate(bands)}}
        order = list(states)
        random.Random(args.seed).shuffle(order)
        scores = {}
        for name in order:
            set_core_state(route, states[name])
            out = output / 'source-selection' / name
            generate_understanding_predictions(adapter, dev, out, metadata={'condition': name}, resume=True)
            scores[name] = json.loads((out / 'metrics.json').read_text())['exact_match']
        damages = [scores['full'] - scores[f'without_band_{i}'] for i in range(8)]
        states, design = selected_band_states(full, bands, damages, seed=args.seed)
        torch.save(states, output / 'selected-states.pt')
        lock_json(output / 'selection.json', {**design, 'scores': scores, 'development_images': [r['user_image'] for r in dev]})
    elif args.phase == 'replay':
        set_core_state(route, full)
        records = load_route_records(target_path)
        generate_route_interventions(fresh, route, records, output / 'replay', seed=8087,
                                     generation_kwargs=kwargs, resume=True)
    else:
        states = {'zero': zero, 'full': full}
        if args.family == 'directions':
            selection = json.loads((output / 'selection.json').read_text())
            if selection['selection_used_target'] is not False:
                raise ValueError('target-informed direction selection is forbidden')
            states.update(torch.load(output / 'selected-states.pt', map_location='cpu', weights_only=True))
        if args.phase == 'shortcut':
            diagnostics = {'original': test, 'image_permuted': image_permutation(test, seed=args.seed + 20000),
                           'instruction_reworded': [{**r, 'user_prompt': r['user_prompt'] +
                               '\nAnswer using the image, with a short answer only.'} for r in test]}
            for mode, records in diagnostics.items():
                for name, state in states.items():
                    set_core_state(route, state)
                    generate_understanding_predictions(adapter, records, output / 'shortcut' / mode / name,
                        metadata={'condition': name, 'diagnostic': mode}, resume=True)
        else:
            set_core_state(route, zero)
            probe = CommonInputProbe(states, change_function=native_output_change)
            probe_path = output / task / 'same-input-activation.json'
            if not probe_path.exists():
                with probe.observe(route):
                    if task == 'understanding':
                        for row in dev[:32]:
                            adapter.understand(Path(row['user_image']), row['user_prompt'])
                    else:
                        # Fixed diagnostic input bank excludes endpoint prompts; not used for selection.
                        from scripts.run_blip3o_readout_followup import prepare_records
                        _, _, fit, _ = prepare_records(args.root, output / 'diagnostic-inputs', args.capability)
                        for i, row in enumerate(fit):
                            fresh.generate(row['user_prompt'], seed=30000 + i, **kwargs)
                write_json(probe_path, probe.report())
            records = test if task == 'understanding' else load_route_records(target_path)
            order = list(states)
            random.Random(args.seed).shuffle(order)
            if args.family == 'directions':
                base = args.output / 'base'
                base_design = json.loads((base / f'{args.phase}-design.json').read_text())
                if (base_design['core'] != str(args.core) or base_design['source_seed'] != args.seed
                        or base_design['generation_kwargs'] != kwargs or not (base / f'{args.phase}-complete.json').exists()):
                    raise ValueError('baseline reuse requires an identical completed endpoint protocol')
                lock_json(output / task / 'baseline-reuse.json', {'zero': str(base / task / 'zero'),
                    'full': str(base / task / 'full'), 'reason': 'same checkpoint, inputs, noise and cache policy'})
                order = [name for name in order if name not in ('zero', 'full')]
            for name in order:
                set_core_state(route, states[name])
                out = output / task / name
                if task == 'understanding':
                    generate_understanding_predictions(adapter, records, out, metadata={'condition': name}, resume=True)
                else:
                    generate_fixed_predictions(fresh, records, out, seed=8087, generation_kwargs=kwargs,
                                               metadata={'condition': name}, resume=True)
    write_json(output / f'{args.phase}-complete.json', {'phase': args.phase, 'status': 'endpoint_outputs_only',
        'official_generation_scoring_complete': False, 'mechanism_proven': False})


if __name__ == '__main__':
    main()
