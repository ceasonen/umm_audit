"""Fixed core interventions and same-input activation diagnostics."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator, Mapping
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from skillbridge.interventions import rank_norm_matched_core_delta
from skillbridge.sidecar import RoutedLowRankLinear


def core_conditions(state: Mapping[str, Tensor], *, budget: int, seed: int) -> dict[str, dict[str, Tensor]]:
    if not state or any(v.ndim != 2 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("expected nonempty finite core matrices")
    full = {k: v.detach().cpu().clone() for k, v in state.items()}
    spectra = {k: torch.linalg.svd(v.float(), full_matrices=False) for k, v in full.items()}
    order = sorted(((float(s[i]), k, i) for k, (_, s, _) in spectra.items() for i in range(len(s))),
                   key=lambda x: (-x[0], x[1], x[2]))
    if not 0 <= budget <= len(order):
        raise ValueError("global component budget out of range")
    top = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in full.items()}
    for value, k, i in order[:budget]:
        u, _, vh = spectra[k]
        top[k].add_(torch.outer(u[:, i], vh[i]), alpha=value)
    top = {k: v.to(full[k]) for k, v in top.items()}
    return {
        "zero": {k: torch.zeros_like(v) for k, v in full.items()},
        "full": full,
        f"top{budget}": top,
        f"complement{budget}": {k: v - top[k] for k, v in full.items()},
        "random": rank_norm_matched_core_delta(full, seed=seed),
    }


def set_core_state(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    layers = {f"{k}.capability_delta" if k else "capability_delta": v
              for k, v in model.named_modules() if isinstance(v, RoutedLowRankLinear)}
    if layers.keys() != state.keys():
        raise ValueError("core state sites do not match model")
    if any(state[k].shape != v.capability_delta.shape or not torch.isfinite(state[k]).all()
           for k, v in layers.items()):
        raise ValueError("invalid core state tensor")
    with torch.no_grad():
        for k, v in layers.items():
            v.capability_delta.copy_(state[k].to(v.capability_delta))


@torch.no_grad()
def same_input_change(layer: RoutedLowRankLinear, x: Tensor, delta: Tensor) -> Tensor:
    """Actual local output difference on identical inputs, including tanh and base output."""
    task = layer.active_task
    if task is None or not layer.shared_enabled:
        raise ValueError("probe requires an active shared route")
    z = layer.shared_input[task](x)
    gate = torch.sigmoid(layer.understanding_route_logit if task == "understanding"
                         else layer.route_logits[layer.active_timestep_bucket])
    def realized(core: Tensor) -> Tensor:
        value = layer.shared_output[task](F.linear(z, core)) * gate * layer.scaling
        return layer.gain_limit * torch.tanh(value) if layer.shared_update == "multiplicative" else value
    change = realized(layer.shared_core.weight + delta.to(layer.capability_delta)) - realized(layer.shared_core.weight)
    return layer.base(x) * change if layer.shared_update == "multiplicative" else change


class CommonInputProbe:
    """Observe zero-core trajectories; evaluate every candidate on exactly the same inputs.

    Tokens are evenly subsampled per invocation; per-layer sample counts are reported.
    This is a local activation diagnostic, not downstream semantic accuracy.
    """
    def __init__(self, states: dict[str, dict[str, Tensor]], *, tokens: int = 8, change_function=same_input_change):
        if tokens < 1 or "full" not in states:
            raise ValueError("probe requires positive token budget and full state")
        self.states = states
        self.tokens = tokens
        self.change_function = change_function
        self.sums: dict[str, dict[str, Tensor]] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def observe(self, model: nn.Module) -> Iterator[None]:
        handles = []
        for name, layer in model.named_modules():
            if not isinstance(layer, RoutedLowRankLinear):
                continue
            key = f"{name}.capability_delta" if name else "capability_delta"
            candidates = {c: s[key].to(layer.capability_delta) for c, s in self.states.items()}
            def hook(module, args, output, *, site=key, values=candidates):
                if module.active_task is None or not module.shared_enabled:
                    return
                with torch.no_grad():
                    flat = args[0].detach().reshape(-1, args[0].shape[-1])
                    if flat.shape[0] == 0:
                        return
                    indices = torch.linspace(0, flat.shape[0] - 1, min(self.tokens, flat.shape[0]), device=flat.device).long()
                    x = flat[indices]
                    changes = {c: self.change_function(module, x, d).float() for c, d in values.items()}
                    reference = changes["full"]
                    self.counts[site] = self.counts.get(site, 0) + reference.numel()
                    bucket = self.sums.setdefault(site, {})
                    for c, change in changes.items():
                        stats = torch.stack((change.square().sum(), reference.square().sum(), (change * reference).sum())).double()
                        bucket[c] = bucket[c] + stats if c in bucket else stats
            handles.append(layer.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def report(self) -> dict:
        layers = {}
        for site, conditions in self.sums.items():
            rows = {}
            for condition, stats in conditions.items():
                energy, reference, dot = stats.cpu().tolist()
                if not all(math.isfinite(v) for v in (energy, reference, dot)):
                    raise ValueError("non-finite probe statistic")
                rows[condition] = {
                    "mean_squared_output_change": energy / self.counts[site],
                    "relative_energy_to_full": energy / reference if reference > 0 else None,
                    "cosine_to_full": dot / math.sqrt(energy * reference) if energy * reference > 0 else None,
                }
            layers[site] = {"sampled_elements": self.counts[site], "conditions": rows}
        if not layers:
            raise ValueError("no active route was observed")
        return {"reference_trajectory": "zero_core", "tokens_per_call": self.tokens,
                "change_function": self.change_function.__name__,
                "interpretation": "same-input local output perturbation; not semantic accuracy",
                "layers": layers}
