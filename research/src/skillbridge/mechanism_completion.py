"""Fixed completion controls; no endpoint score enters scale fitting or selection."""
from __future__ import annotations

import math
from pathlib import Path
import re

TASKS = ('understanding', 'generation')
SEEDS = (17, 29, 43, 71)
COUNTS = dict(zip('one two three four five six'.split(), range(1, 7)))


def rms_check(values, targets, *, tolerance):
    if set(values) != set(TASKS) or set(targets) != set(TASKS) or not 0 < tolerance < 1:
        raise ValueError('both endpoints and a fixed relative tolerance required')
    ratios, errors, zeros = {}, [], []
    for task in TASKS:
        value, target = values[task], targets[task]
        if not all(math.isfinite(x) and x >= 0 for x in (value, target)):
            raise ValueError('finite nonnegative native output energies required')
        if target == 0:
            ratios[task] = None
            zeros.append(value == 0)
        else:
            ratios[task] = math.sqrt(value / target)
            errors.append(abs(ratios[task] - 1))
    error = max(errors, default=0.)
    return dict(passed=all(zeros) and error <= tolerance, rms_ratios=ratios,
                max_relative_rms_error=error if all(zeros) else None,
                zero_reference_exact=all(zeros), tolerance=tolerance,
                reference_energies=targets, control_energies=values)


def fit_joint_scale(energy, targets):
    # Shared scalar at each site for U and G. A finite search is not a proof
    # that no feasible scalar exists outside the tested interval.
    grid = [0.] + [2. ** (-5 + i * .25) for i in range(41)]
    evaluated = {}
    def evaluate(scale):
        if scale not in evaluated:
            check = rms_check(energy(scale), targets, tolerance=.02)
            evaluated[scale] = check
        check = evaluated[scale]
        value = check['max_relative_rms_error']
        return float('inf') if value is None else value
    for scale in grid:
        evaluate(scale)
    best = min(grid, key=lambda s: (evaluate(s), s))
    for _ in range(2):
        ordered = sorted(evaluated)
        i = ordered.index(best)
        lo, hi = ordered[max(0, i-1)], ordered[min(len(ordered)-1, i+1)]
        for j in range(1, 9):
            evaluate(lo + (hi-lo)*j/9)
        best = min(evaluated, key=lambda s: (evaluate(s), s))
    return dict(scale=best, **evaluated[best], evaluations=len(evaluated),
                search='fixed [0,32] grid and two local refinements; minimax U/G RMS error')


def joint_gate(calibration, validation):
    if not calibration or set(calibration) != set(validation):
        raise ValueError('calibration/validation condition coverage differs')
    conditions = {}
    for name, sites in calibration.items():
        if not sites or set(sites) != set(validation[name]):
            raise ValueError('calibration/validation layer coverage differs')
        failures = [site for site in sites if sites[site].get('passed') is not True
                    or validation[name][site].get('passed') is not True]
        conditions[name] = dict(passed=not failures, sites=len(sites), failed_sites=failures)
    return dict(passed=all(v['passed'] for v in conditions.values()), conditions=conditions,
                policy='every fixed control and site must pass on both endpoints; no successful subset selection')


def literal_count(prompt):
    match = re.fullmatch(r'(one|two|three|four|five|six) ([a-z]+)', prompt.strip())
    if match is None:
        raise ValueError('single literal count/object criterion required')
    return COUNTS[match[1]], match[2]


def attribute_outcome(recipient_score, donor_score):
    if any(x not in (0., .5, 1.) for x in (recipient_score, donor_score)):
        raise ValueError('single-object official numeracy must score 0, 0.5 or 1')
    if (recipient_score > 0) != (donor_score > 0) or recipient_score == donor_score == 1:
        raise ValueError('inconsistent same-object detector results or distinct-count criteria')
    return dict(object_present=int(recipient_score > 0),
                recipient_count_hit=int(recipient_score == 1), donor_count_hit=int(donor_score == 1),
                scope='automatic detector proxy; not a human count or distance-to-donor measure')


def attribute_scoring_view(pairs, predictions, source_root, *, criterion):
    if criterion not in ('recipient', 'donor'):
        raise ValueError('unknown attribute scoring criterion')
    indexed = {r['split_key']: r for r in predictions}
    if (not pairs or len(indexed) != len(predictions)
            or len({p['recipient']['split_key'] for p in pairs}) != len(pairs)
            or set(indexed) != {p['recipient']['split_key'] for p in pairs}):
        raise ValueError('duplicate or incomplete recipient coverage')
    records, outputs = [], []
    for pair in pairs:
        recipient, donor, nuisance = (pair[x] for x in ('recipient', 'semantic_donor', 'nuisance_donor'))
        n, obj = literal_count(recipient['user_prompt'])
        nd, od = literal_count(donor['user_prompt'])
        nn, on = literal_count(nuisance['user_prompt'])
        if n == nd or obj != od or n != nn or obj == on:
            raise ValueError('invalid semantic/nuisance counterfactual')
        original = indexed[recipient['split_key']]
        if original['prompt'] != recipient['user_prompt']:
            raise ValueError('generation prompt does not match recipient')
        image = (source_root / original['image']).resolve()
        if not image.is_relative_to(source_root.resolve()) or not image.is_file():
            raise ValueError('missing or out-of-scope generated image')
        prompt = donor['user_prompt'] if criterion == 'donor' else recipient['user_prompt']
        records.append({**recipient, 'user_prompt': prompt})
        outputs.append({**original, 'prompt': prompt, 'image': str(image),
                        'generation_prompt': original['prompt'], 'scoring_criterion': criterion})
    return records, outputs


def admit_extension(gpu, seed, status, *, active_commands, free_mib):
    if gpu not in range(4) or seed != SEEDS[gpu]:
        raise ValueError('only fixed seed assignment on GPU 0 through 3 permitted')
    # A free-memory snapshot did not bound JanusFlow's later attention-mask peak.
    # Do not overlap with a main queue, including the gap between its child jobs.
    return (not active_commands and free_mib >= 20000
            and status.get('status') in ('scheduled_computational_jobs_finished', 'failed'))


def _bank_computations(model, banks, reference, candidates, operators, *, fit):
    import torch
    from skillbridge.readout_followup import native_output_change
    from skillbridge.sidecar import RoutedLowRankLinear

    layers = {(f'{name}.capability_delta' if name else 'capability_delta'): layer
              for name, layer in model.named_modules() if isinstance(layer, RoutedLowRankLinear)}
    if (not layers or set(layers) != set(reference) or set(banks) != set(TASKS)
            or set(candidates) != set(operators) or not candidates
            or any(set(state) != set(layers) for state in candidates.values())
            or any(set(bank.inputs) != set(layers) for bank in banks.values())
            or any(op not in ('multiplicative', 'additive') for op in operators.values())):
        raise ValueError('complete site, task, state and operator coverage required')
    states = {name: {} for name in candidates}
    reports = {name: {} for name in candidates}
    with torch.no_grad():
        for site, layer in layers.items():
            previous = layer.active_task, layer.active_timestep_bucket, layer.shared_enabled, layer.shared_update
            inputs = {task: torch.cat(bank.inputs[site]).to(layer.capability_delta) for task, bank in banks.items()}
            def change(task, delta, operator):
                actual_task, bucket = banks[task].routes[site]
                if actual_task != task:
                    raise ValueError('recorded input task mismatch')
                layer.set_route(task=task, timestep_bucket=bucket)
                layer.shared_update = operator
                return native_output_change(layer, inputs[task], delta).float()
            try:
                refs = {task: change(task, reference[site], 'multiplicative') for task in TASKS}
                targets = {task: float(value.square().mean()) for task, value in refs.items()}
                for name, state in candidates.items():
                    candidate = state[site].to(layer.capability_delta)
                    def energy(scale):
                        return {task: float(change(task, candidate*scale, operators[name]).square().mean())
                                for task in TASKS}
                    report = fit_joint_scale(energy, targets) if fit else rms_check(energy(1.), targets, tolerance=.2)
                    scaled = candidate * report['scale'] if fit else candidate
                    states[name][site] = scaled.detach().cpu()
                    report['tokens'] = {task: len(x) for task, x in inputs.items()}
                    report['cosine_to_reference'] = {}
                    for task in TASKS:
                        delta = change(task, scaled, operators[name])
                        denominator = float(delta.norm()*refs[task].norm())
                        report['cosine_to_reference'][task] = float((delta*refs[task]).sum())/denominator if denominator else None
                    reports[name][site] = report
            finally:
                layer.active_task, layer.active_timestep_bucket, layer.shared_enabled, layer.shared_update = previous
            print(f'joint {"fit" if fit else "validation"}: {site}', flush=True)
    return states, reports


def fit_banks(model, banks, reference, candidates, operators):
    return _bank_computations(model, banks, reference, candidates, operators, fit=True)


def validate_banks(model, banks, reference, candidates, operators):
    return _bank_computations(model, banks, reference, candidates, operators, fit=False)[1]
