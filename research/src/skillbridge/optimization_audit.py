"""Opt-in source-only telemetry and fixed-basis coordinate diagnostics."""
import json
import math

import torch

from skillbridge.sidecar import RoutedLowRankLinear


class OptimizationAudit:
    def __init__(self, path, *, every=100, total_steps):
        if every < 1 or total_steps < 1:
            raise ValueError('positive telemetry interval and training budget required')
        if path.exists():
            raise FileExistsError('do not append telemetry from a restarted optimizer run')
        self.path, self.every, self.total_steps = path, every, total_steps
        self.handles, self.transforms, self.before, self.sites = [], [], {}, {}

    def __call__(self, event, step, route):
        if step not in (1, self.total_steps) and step % self.every:
            return
        if event == 'before_step':
            self.before = {n: p.detach().float().cpu().clone() for n, p in route.named_parameters() if p.requires_grad}
            self.sites = {}
            for name, layer in route.named_modules():
                if not hasattr(layer, 'base') or not (isinstance(layer, RoutedLowRankLinear)
                                                      or getattr(layer, '_skillbridge_route_compatible', False)):
                    continue
                self.sites[name] = dict(elements=0, delta_sum_squares=0., base_sum_squares=0., saturated=0, modulation_elements=0)
                def hook(module, inputs, output, site=name):
                    with torch.no_grad():
                        frozen = module.base(inputs[0]).detach().float()
                        delta = output.detach().float()-frozen
                        item = self.sites[site]
                        item['elements'] += delta.numel()
                        item['delta_sum_squares'] += float(delta.square().sum())
                        item['base_sum_squares'] += float(frozen.square().sum())
                self.handles.append(layer.register_forward_hook(hook))
                if isinstance(layer, RoutedLowRankLinear) and layer.shared_update == 'multiplicative':
                    if layer._shared_modulation_transform is not None:
                        raise ValueError('training telemetry must not replace an intervention')
                    def observe(value, site=name, limit=layer.gain_limit):
                        detached = value.detach()
                        self.sites[site]['saturated'] += int((detached.abs() >= .95*limit).sum())
                        self.sites[site]['modulation_elements'] += detached.numel()
                        return value
                    previous = layer.replace_modulation_transform(observe)
                    self.transforms.append((layer, previous))
        elif event == 'after_forward':
            # No repeated counting of activation-checkpoint backward recomputation.
            for handle in self.handles:
                handle.remove()
            for layer, previous in self.transforms:
                layer.replace_modulation_transform(previous)
            self.handles, self.transforms = [], []
        elif event == 'after_step':
            gradients, changes, parameter_norms, missing = [], [], [], []
            for name, parameter in route.named_parameters():
                if not parameter.requires_grad:
                    continue
                current = parameter.detach().float().cpu()
                if not torch.isfinite(current).all():
                    raise ValueError('nonfinite trained parameter')
                changes.append(float((current-self.before[name]).square().sum()))
                parameter_norms.append(float(current.square().sum()))
                if parameter.grad is None:
                    missing.append(name)
                else:
                    gradients.append(float(parameter.grad.detach().float().square().sum()))
            if not all(math.isfinite(x) for x in gradients+changes+parameter_norms):
                raise ValueError('nonfinite gradient or parameter-update telemetry')
            sites = {}
            for name, item in self.sites.items():
                n = item['elements']
                sites[name] = dict(elements=n,
                    realized_output_delta_rms=math.sqrt(item['delta_sum_squares']/n) if n else None,
                    frozen_output_rms=math.sqrt(item['base_sum_squares']/n) if n else None,
                    saturation_fraction=item['saturated']/item['modulation_elements'] if item['modulation_elements'] else None)
            row = dict(step=step, trainable_parameters=sum(v.numel() for v in self.before.values()),
                       gradient_l2=math.sqrt(sum(gradients)), update_l2=math.sqrt(sum(changes)),
                       parameter_l2=math.sqrt(sum(parameter_norms)), missing_gradient_names=missing, sites=sites,
                       update_scope='one actual optimizer step in stored dtype; not a parameter budget proxy',
                       activation_scope='source training forward only; not dual-task matched validation')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
            self.before = {}
        else:
            raise ValueError('unknown optimization event')


def coordinate_check(a, b, q, left, right, *, keep):
    """Gauge invariance of BQA versus entrywise top-k in Q; CPU algebra only."""
    a, b, q, left, right = [value.detach().double().cpu() for value in (a, b, q, left, right)]
    rank = q.shape[0]
    if q.shape != (rank, rank) or not 0 < keep < q.numel():
        raise ValueError('square core and nontrivial coordinate budget required')
    for rotation in (left, right):
        if not torch.allclose(rotation.T@rotation, torch.eye(rank, dtype=torch.float64), atol=1e-6):
            raise ValueError('orthogonal coordinate changes required')
    ar, br, qr = right.T@a, b@left, left.T@q@right
    def sparse(value):
        result = torch.zeros_like(value).flatten()
        indices = torch.argsort(value.flatten().abs(), descending=True, stable=True)[:keep]
        result[indices] = value.flatten()[indices]
        return result.reshape_as(value)
    original = b@q@a
    norm = original.norm().clamp_min(1e-30)
    return dict(operator_relative_error=float((original-br@qr@ar).norm()/norm),
                coordinate_topk_relative_difference=float((b@sparse(q)@a-br@sparse(qr)@ar).norm()/norm),
                keep_entries=keep, core_entries=q.numel(),
                scope='linear pre-gain algebra; native-precision forward and task effects require separate tests')
