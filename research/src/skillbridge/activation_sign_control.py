"""Activation-level sign control; not a random low-rank parameter state."""

import torch
from contextlib import contextmanager
import math

from skillbridge.readout_followup import native_route_outputs
from skillbridge.sidecar import RoutedLowRankLinear


@torch.no_grad()
def signed_native_control(baseline, adapted, signs):
    if (baseline.shape != adapted.shape or baseline.dtype != adapted.dtype
            or baseline.device != adapted.device or baseline.ndim == 0 or baseline.numel() == 0
            or not baseline.is_floating_point()):
        raise ValueError('equal nonempty native output tensors required')
    signs = torch.as_tensor(signs, device=baseline.device)
    if signs.shape != (baseline.shape[-1],) or not ((signs == 1) | (signs == -1)).all():
        raise ValueError('one fixed +1/-1 sign per output channel required')
    if not torch.isfinite(baseline).all() or not torch.isfinite(adapted).all():
        raise ValueError('nonfinite native output')
    work_dtype = torch.float64 if baseline.dtype == torch.float64 else torch.float32
    base = baseline.to(work_dtype)
    delta = adapted.to(work_dtype) - base
    if not torch.isfinite(delta).all():
        raise ValueError('nonfinite output difference')
    proposed = (base + delta * signs).to(baseline.dtype)
    realized = proposed.to(work_dtype) - base
    exact = torch.isfinite(proposed) & (realized == delta * signs)
    # At exponent boundaries a reflected update may not be representable.
    # Retain the original coordinate and explicitly report the reduced intervention.
    output = torch.where(exact, proposed, adapted)
    changed = (output != adapted) & (delta != 0)
    energy = float(delta.double().square().sum())
    control_energy = float((output.to(work_dtype) - base).double().square().sum())
    changed_energy = float(torch.where(changed, delta, 0).double().square().sum())
    return output, {
        'reference_energy_sum': energy, 'control_energy_sum': control_energy,
        'energy_ratio': control_energy / energy if energy else None,
        'changed_energy_fraction': changed_energy / energy if energy else None,
        'fallback_elements': int(((signs == -1) & ~exact & (delta != 0)).sum()),
        'nonzero_elements': int((delta != 0).sum()), 'changed_elements': int(changed.sum()),
        'elements': delta.numel(),
        'scope': 'same-input output-sign intervention; retains magnitudes, not a semantic shuffle',
    }


@contextmanager
def native_sign_intervention(model, *, condition, source_seed):
    """Apply fixed channel signs on the visited trajectory, with per-call audits."""
    if condition not in ('mirror', 'sign_random_0', 'sign_random_1', 'sign_random_2'):
        raise ValueError('unknown activation sign condition')
    layers = {f'{name}.capability_delta' if name else 'capability_delta': layer
              for name, layer in model.named_modules() if isinstance(layer, RoutedLowRankLinear)}
    if not layers:
        raise ValueError('no routed layers')
    index = int(condition[-1]) if condition.startswith('sign_random_') else 0
    generator = torch.Generator(device='cpu').manual_seed(1000 + source_seed + 10000*index)
    handles, audit = [], {}
    try:
        for site in sorted(layers):
            layer = layers[site]
            width = layer.base.out_features
            signs = (-torch.ones(width, dtype=torch.int8) if condition == 'mirror'
                     else torch.randint(0, 2, (width,), generator=generator)*2-1)
            def hook(module, inputs, output, *, key=site, channel_signs=signs):
                if module.active_task is None or not module.shared_enabled:
                    return output
                baseline, expected = native_route_outputs(module, inputs[0], module.capability_delta)
                if not torch.equal(output, expected):
                    raise ValueError('native diagnostic does not reproduce live layer output')
                controlled, stats = signed_native_control(baseline, output, channel_signs)
                if stats['energy_ratio'] not in (None, 1.):
                    raise ValueError('live control changed local perturbation energy')
                row = audit.setdefault(key, {'calls': 0, 'reference_energy_sum': 0.,
                    'control_energy_sum': 0., 'changed_energy_sum': 0.,
                    'fallback_elements': 0, 'nonzero_elements': 0, 'changed_elements': 0})
                row['calls'] += 1
                for name in ('reference_energy_sum', 'control_energy_sum', 'fallback_elements',
                             'nonzero_elements', 'changed_elements'):
                    row[name] += stats[name]
                row['changed_energy_sum'] += stats['reference_energy_sum'] * (stats['changed_energy_fraction'] or 0.)
                return controlled
            handles.append(layer.register_forward_hook(hook))
        yield audit
    finally:
        for handle in handles:
            handle.remove()


def live_control_gate(audit, expected_sites, minimum_calls=1):
    complete = bool(expected_sites) and set(audit) == set(expected_sites)
    valid = complete and all(
        r['calls'] >= minimum_calls
        and math.isfinite(r['reference_energy_sum']) and r['reference_energy_sum'] > 0
        and r['control_energy_sum'] == r['reference_energy_sum']
        and math.isfinite(r['changed_energy_sum']) and 0 < r['changed_energy_sum'] <= r['reference_energy_sum']
        for r in audit.values())
    return {'passed': bool(valid), 'complete_sites': complete,
            'expected_sites': len(expected_sites), 'observed_sites': len(audit),
            'policy': 'all sites visited, exact local magnitude preservation, nonzero direction intervention',
            'scope': 'implementation validity only; no semantic claim'}
