#!/usr/bin/env python3
"""Full held-out source answers; no changes to the original candidate-score gate."""
import argparse
import json
import os
from pathlib import Path
import random
import time


def make_plan(root, seed, backbone, capability):
    if seed not in (17, 29, 43, 71) or backbone not in ('blip3o', 'janusflow'):
        raise ValueError('unsupported training identity')
    if capability not in ('counting', 'spatial'):
        raise ValueError('unsupported capability')
    verbalizers = ('digits', 'words') if capability == 'counting' else ('yes_no', 'truth')
    cases = [dict(state=s, verbalizer=v) for s in ('full', 'zero', 'label_shuffle') for v in verbalizers]
    random.Random(20260914 + seed).shuffle(cases)
    return dict(seed=seed, backbone=backbone, capability=capability, cases=cases, max_new_tokens=32,
                output=str(root/f'runs/mechanism_controls/full_answer_v1/{backbone}/{capability}/seed-{seed}'),
                training_base=str(root/f'runs/mechanism_controls/mechanism_sequence_v1/{backbone}/{capability}/seed-{seed}'),
                scope='measurement repair on previously examined panels, not independent confirmation',
                semantic_rule='case-insensitive answer-only number/boolean aliases; all unrecognized outputs retained',
                image_control='preselected same-question/different-answer natural image pairs; not isolated attribute counterfactuals',
                original_candidate_gate_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--backbone', choices=('blip3o', 'janusflow'), required=True)
    parser.add_argument('--capability', choices=('counting', 'spatial'), required=True)
    parser.add_argument('--memory-fraction', type=float, default=0.45)
    parser.add_argument('--method', choices=('tied_core', 'svd_core', 'additive_tied_core', 'shared_lora'), default='tied_core')
    parser.add_argument('--training-base', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if bool(args.training_base) != bool(args.output_root) or (args.method != 'tied_core' and not args.training_base):
        raise ValueError('non-default references need isolated training and output paths together')
    import torch
    from scripts.evaluate_cross_backbone_generation import checkpoint_route_spec, load_blip3o, load_janusflow
    from skillbridge.closed_answer_audit import answer_design, natural_pairs, validate_training, validate_image_disjointness
    from skillbridge.dimoo_data import load_route_records
    from skillbridge.full_answer_audit import classify_answer, summarize, validate_resume, RULE_VERSION
    from skillbridge.full_answer_runtime import generate_answer
    from skillbridge.mechanism_sequence import lock_json
    from skillbridge.readout_probe import set_core_state
    from skillbridge.source_measurement import disjoint_records
    from skillbridge.training_overlap import training_lock

    gpu = int(os.environ.get('CUDA_VISIBLE_DEVICES', '-1'))
    if gpu not in range(8) or not 0 < args.memory_fraction <= 0.9:
        raise ValueError('single physical GPU and bounded memory fraction required')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
    root = args.root.resolve()
    plan = make_plan(root, args.seed, args.backbone, args.capability)
    custom = args.training_base is not None
    if custom:
        plan.update(method=args.method, training_base=str(args.training_base.resolve()), output=str(args.output_root.resolve()))
        if 'full_answer_v1' in args.output_root.parts:
            raise ValueError('keep alternate reference results outside original full_answer_v1')
    output, base = Path(plan['output']), Path(plan['training_base'])
    records = load_route_records(root/f'data/route_transfer_oral_v2/capabilities/{args.capability}/understanding/test/records.jsonl')
    arms = (dict(full='reference', label_shuffle='label_shuffle') if args.backbone == 'blip3o'
            else dict(full='replication', label_shuffle='replication_label_shuffle'))
    states, checkpoints = {}, {}
    for state, arm in arms.items():
        folder = base/arm
        if custom:
            from skillbridge.learning_reference import validate_reference
            config, checkpoint = validate_reference(folder, shuffled=state == 'label_shuffle',
                backbone=args.backbone, capability=args.capability, seed=args.seed, method=args.method)
        else:
            config = validate_training(folder, shuffled=state == 'label_shuffle')
            checkpoint = torch.load(folder/'core.pt', map_location='cpu', weights_only=True)
        md = checkpoint['metadata']
        if (md['backbone'], md['capability'], md['seed'], md['method']) != (args.backbone, args.capability, args.seed, args.method):
            raise ValueError('wrong source checkpoint identity')
        training = load_route_records(folder/'training-inputs.jsonl')
        disjoint_records(training, records)
        validate_image_disjointness(training, records)
        checkpoints[state] = dict(path=str(folder/('adapter.pt' if custom else 'core.pt')), metadata=md, training=config)
        states[state] = checkpoint['state_dict']
    if set(states['full']) != set(states['label_shuffle']):
        raise ValueError('checkpoint sites differ')
    states['zero'] = states['full'] if custom else {k: torch.zeros_like(v) for k, v in states['full'].items()}
    pairs = natural_pairs(records)
    if not pairs:
        raise ValueError('no natural image contrasts; require a separately locked image-control design')
    with training_lock(output, blocking=False):
        lock_json(output/'design.json', dict(**plan, records=records, pairs=pairs,
                                           checkpoints=checkpoints, rule_version=RULE_VERSION))
        gain, suffixes = checkpoint_route_spec(Path(checkpoints['full']['path']))
        loader = load_blip3o if args.backbone == 'blip3o' else load_janusflow
        rank = checkpoints['full']['metadata']['rank']
        adapter, route, _ = loader(root, rank=rank, timestep_buckets=1,
                                   method=('tied_core' if args.method == 'svd_core' else args.method),
                                   gain_limit=gain, target_suffixes=suffixes,
                                   **({'task': 'understanding'} if args.backbone == 'blip3o' else {}))
        adapter.model.eval()
        for case in plan['cases']:
            state, verbalizer = case['state'], case['verbalizer']
            if custom:
                from skillbridge.learning_reference import apply_state
                apply_state(route, states[state], method=args.method, zero=state == 'zero')
            else:
                set_core_state(route, states[state])
            path = output/verbalizer/state/'predictions.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            predictions = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            count = validate_resume(records, predictions, args.capability, state, verbalizer)
            started = time.monotonic()
            with path.open('a') as handle:
                for i, row in enumerate(records[count:], start=count):
                    measured = generate_answer(adapter, args.backbone, row['user_image'],
                                               row['user_prompt']+answer_design(args.capability, verbalizer)['suffix'],
                                               max_new_tokens=plan['max_new_tokens'])
                    scored = classify_answer(measured['raw_answer'], args.capability, verbalizer,
                                             row['answer_text'], truncated=measured['truncated'])
                    result = dict(split_key=row['split_key'], image=row['user_image'], question=row['user_prompt'], **case)
                    result.update(measured)
                    result.update(scored)
                    handle.write(json.dumps(result, allow_nan=False)+'\n')
                    handle.flush()
                    predictions.append(result)
                    if (i+1) % 32 == 0 or i+1 == len(records):
                        value = dict(seed=args.seed, backbone=args.backbone, capability=args.capability, **case,
                                     rows=i+1, total=len(records), elapsed=time.monotonic()-started,
                                     peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, updated=time.time())
                        temporary = output/'progress.partial'
                        temporary.write_text(json.dumps(value, indent=2)+'\n')
                        temporary.replace(output/'progress.json')
                        print(json.dumps(value), flush=True)
            validate_resume(records, predictions, args.capability, state, verbalizer)
            if len(predictions) != len(records):
                raise ValueError('incomplete panel')
            lock_json(path.parent/'summary.json', summarize(predictions, pairs))
        lock_json(output/'complete.json', dict(seed=args.seed, backbone=args.backbone, capability=args.capability,
                                              cases=6, rows_per_case=len(records), rule_version=RULE_VERSION,
                                              semantic_mechanism_proven=False, scope=plan['scope']))


if __name__ == '__main__':
    main()
