"""Source-only localization, bounded-core energy calibration, paired inference."""
from contextlib import contextmanager
import math

import numpy as np
import torch

from skillbridge.sidecar import RoutedLowRankLinear


@torch.no_grad()
def native_route_outputs(layer, x, delta):
    """Return zero-write and adapted native outputs without mutating parameters."""
    task = layer.active_task
    if task is None or not layer.shared_enabled or layer._capability_scales or layer._shared_modulation_transform is not None:
        raise ValueError('requires unmodified active shared route')
    base = layer.base(x)
    private = layer.private_output[task](layer.private_input[task](x)) * layer.scaling
    z = layer.shared_input[task](x)
    gate = torch.sigmoid(layer.understanding_route_logit if task == 'understanding'
                         else layer.route_logits[layer.active_timestep_bucket])
    def output(core):
        value = layer.shared_output[task](torch.nn.functional.linear(z, core)) * gate * layer.scaling
        modulation = layer.gain_limit * torch.tanh(value) if layer.shared_update == 'multiplicative' else value
        return base + private + (base * modulation if layer.shared_update == 'multiplicative' else modulation)
    return output(layer.shared_core.weight), output(layer.shared_core.weight + delta.to(layer.capability_delta))


@torch.no_grad()
def native_output_change(layer, x, delta):
    """Two actual native-precision layer outputs, including final residual rounding."""
    baseline, adapted = native_route_outputs(layer, x, delta)
    return adapted - baseline


def fit_energy(energy, target, tolerance=.02):
    if not math.isfinite(target) or target <= 0:
        return {'scale': 1., 'relative_energy': None, 'matched': False, 'reason': 'zero reference energy'}
    grid = [0., .125, .25, .5, 1., 2., 4., 8., 16., 32.]
    values = [float(energy(a)) for a in grid]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('invalid activation energy')
    # Tanh energy need not be globally monotone. Refine every observed crossing.
    candidates = list(zip(grid, values))
    for lo, hi, elo, ehi in zip(grid, grid[1:], values, values[1:]):
        if (elo - target) * (ehi - target) > 0:
            continue
        for _ in range(14):
            mid = (lo + hi) / 2
            emid = float(energy(mid))
            if not math.isfinite(emid) or emid < 0:
                raise ValueError('invalid activation energy during refinement')
            candidates.append((mid, emid))
            if (elo - target) * (emid - target) <= 0:
                hi = mid
            else:
                lo, elo = mid, emid
    scale, value = min(candidates, key=lambda av: (abs(av[1] / target - 1), av[0]))
    return {'scale': scale, 'relative_energy': value / target,
            'matched': abs(value / target - 1) <= tolerance,
            'reason': 'bounded-core scalar search; grid max 32'}


def depth_groups(state, count=6):
    depths = sorted({int(k.split('.')[0]) for k in state})
    if len(depths) % count:
        raise ValueError('depth groups must have equal size')
    groups = [[k for k in sorted(state) if int(k.split('.')[0]) in chunk]
              for chunk in np.array_split(depths, count)]
    if len({len(g) for g in groups}) != 1:
        raise ValueError('unequal site budgets')
    return groups


def select_groups(state, groups, damages, budget=2):
    if len(groups) != len(damages) or not all(math.isfinite(d) for d in damages):
        raise ValueError('invalid source damages')
    chosen = sorted(range(len(groups)), key=lambda i: (-damages[i], i))[:budget]
    sites = {k for i in chosen for k in groups[i]}
    selected = {k: v.clone() if k in sites else torch.zeros_like(v) for k, v in state.items()}
    return selected, {k: v - selected[k] for k, v in state.items()}


def split_prompts(bank, excluded, per_split=32):
    normalize = lambda s: ' '.join(str(s).casefold().split())
    seen = {normalize(r['user_prompt']) for r in excluded}
    eligible = []
    for record in bank:
        key = normalize(record['user_prompt'])
        if key and key not in seen:
            seen.add(key)
            # Deliberately omit answers, quality scores, and evaluator metadata.
            eligible.append({'user_prompt': record['user_prompt']})
    rng = np.random.default_rng(20260909)
    rng.shuffle(eligible)
    if len(eligible) < per_split * 2:
        raise ValueError('insufficient disjoint calibration prompts')
    return eligible[:per_split], eligible[per_split:per_split * 2]


def crossed_interval(differences, draws=10000, seed=20260909, strata=None):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 2 or not d.size or not np.isfinite(d).all():
        raise ValueError('expected finite seed-by-prompt paired differences')
    rng = np.random.default_rng(seed)
    labels = np.zeros(d.shape[1], int) if strata is None else np.asarray(strata)
    if len(labels) != d.shape[1]:
        raise ValueError('stratum count mismatch')
    groups = [np.flatnonzero(labels == k) for k in np.unique(labels)]
    values = np.empty(draws)
    for i in range(draws):
        rows = rng.integers(d.shape[0], size=d.shape[0])
        cols = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values[i] = d[np.ix_(rows, cols)].mean()
    means = d.mean(axis=1)
    return {'mean_pp': float(d.mean() * 100),
            'ci95_pp': (np.quantile(values, [.025, .975]) * 100).tolist(),
            'seed_effects_pp': (means * 100).tolist(),
            'positive_seeds': int((means > 0).sum()), 'negative_seeds': int((means < 0).sum()),
            'training_seeds': d.shape[0], 'paired_prompts': d.shape[1],
            'interpretation': 'exploratory crossed seed/prompt percentile bootstrap; four-seed uncertainty is limited'}


def validation_gate(report, calibration):
    if not report.get('layers') or not calibration:
        raise ValueError('empty validation or calibration')
    conditions = {}
    for name, fits in calibration.items():
        if set(fits) != set(report['layers']):
            raise ValueError('calibration and validation sites differ')
        ratios = [v['conditions'][name]['relative_energy_to_full'] for v in report['layers'].values()]
        outside = sum(r is None or not math.isfinite(r) or not .8 <= r <= 1.2 for r in ratios)
        cal_ok = all(v['matched'] for v in fits.values())
        conditions[name] = {'calibration_passed': cal_ok, 'validation_failed_sites': outside,
                            'total_sites': len(ratios), 'passed': cal_ok and outside == 0}
    return {'passed': all(v['passed'] for v in conditions.values()), 'conditions': conditions,
            'policy': 'all directions must pass before expensive target evaluation; do not select successful seeds'}


class InputBank:
    """Balanced per-prompt zero-trajectory inputs, on CPU, for local energy fitting."""
    def __init__(self, tokens_per_call=2, per_prompt=64):
        self.tokens_per_call, self.per_prompt = tokens_per_call, per_prompt
        self.inputs, self.routes, self.counts = {}, {}, {}

    def next_prompt(self):
        self.counts = {}

    @contextmanager
    def observe(self, model):
        handles = []
        for name, layer in model.named_modules():
            if not isinstance(layer, RoutedLowRankLinear):
                continue
            site = f'{name}.capability_delta' if name else 'capability_delta'
            def hook(module, args, output, key=site):
                if module.active_task is None or not module.shared_enabled:
                    return
                remaining = self.per_prompt - self.counts.get(key, 0)
                x = args[0].detach().reshape(-1, args[0].shape[-1])
                n = min(remaining, self.tokens_per_call, x.shape[0])
                if n <= 0:
                    return
                index = torch.linspace(0, x.shape[0]-1, n, device=x.device).long()
                self.inputs.setdefault(key, []).append(x[index].cpu())
                self.routes[key] = (module.active_task, module.active_timestep_bucket)
                self.counts[key] = self.counts.get(key, 0) + n
            handles.append(layer.register_forward_hook(hook))
        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def fit(self, model, full, candidates):
        result = {c: {} for c in candidates}
        report = {c: {} for c in candidates}
        for name, layer in model.named_modules():
            if not isinstance(layer, RoutedLowRankLinear):
                continue
            site = f'{name}.capability_delta' if name else 'capability_delta'
            if site not in self.inputs:
                raise ValueError(f'no calibration inputs at {site}')
            x = torch.cat(self.inputs[site]).to(layer.capability_delta)
            task, bucket = self.routes[site]
            previous = (layer.active_task, layer.active_timestep_bucket, layer.shared_enabled)
            layer.set_route(task=task, timestep_bucket=bucket)
            try:
                target = float(native_output_change(layer, x, full[site]).float().square().mean())
                for c, state in candidates.items():
                    delta = state[site].to(layer.capability_delta)
                    fit = fit_energy(lambda scale: float(native_output_change(layer, x, delta * scale).float().square().mean()), target)
                    result[c][site] = (delta * fit['scale']).cpu()
                    report[c][site] = {**fit, 'tokens': len(x), 'reference_energy': target}
            finally:
                layer.active_task, layer.active_timestep_bucket, layer.shared_enabled = previous
            print(f'calibrated {site}', flush=True)
        return result, report

    @torch.no_grad()
    def report(self, model, states):
        if 'full' not in states:
            raise ValueError('full reference required')
        report = {}
        for name, layer in model.named_modules():
            if not isinstance(layer, RoutedLowRankLinear):
                continue
            site = f'{name}.capability_delta' if name else 'capability_delta'
            x = torch.cat(self.inputs[site]).to(layer.capability_delta)
            previous = layer.active_task, layer.active_timestep_bucket, layer.shared_enabled
            task, bucket = self.routes[site]
            layer.set_route(task=task, timestep_bucket=bucket)
            try:
                ref = native_output_change(layer, x, states['full'][site]).float()
                ref_energy = float(ref.square().mean())
                values = {}
                for c, state in states.items():
                    delta = native_output_change(layer, x, state[site]).float()
                    energy = float(delta.square().mean())
                    dot = float((delta * ref).mean())
                    if not all(math.isfinite(v) for v in (ref_energy, energy, dot)):
                        raise ValueError('nonfinite activation report')
                    values[c] = {'mean_squared_output_change': energy,
                                 'relative_energy_to_full': energy / ref_energy if ref_energy > 0 else None,
                                 'cosine_to_full': dot / math.sqrt(energy * ref_energy) if energy * ref_energy > 0 else None}
                report[site] = {'sampled_elements': ref.numel(), 'tokens': len(x), 'conditions': values}
            finally:
                layer.active_task, layer.active_timestep_bucket, layer.shared_enabled = previous
        if not report:
            raise ValueError('no active layers observed')
        return {'reference_trajectory': 'zero_core', 'change_function': 'native_output_change',
                'tokens_per_call': self.tokens_per_call, 'per_prompt_limit': self.per_prompt,
                'aggregation': 'same concatenated input computation as calibration fit', 'layers': report}
