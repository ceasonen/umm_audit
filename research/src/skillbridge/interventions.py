from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import torch
from torch import Tensor, nn

from skillbridge.sidecar import RouteComponentBank, SidecarSite, SidecarTask, named_sidecars

PatchBank = dict[SidecarSite, list[Tensor]]
CoreDeltaBank = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class CandidateState:
    calibration_frame_sha256: str
    delta: CoreDeltaBank

    def __post_init__(self) -> None:
        if len(self.calibration_frame_sha256) != 64:
            msg = "candidate state requires a full calibration frame SHA-256"
            raise ValueError(msg)
        if not self.delta:
            msg = "candidate state requires a non-empty core delta"
            raise ValueError(msg)


def capture_core_delta(adapted: nn.Module, calibrated: nn.Module) -> CoreDeltaBank:
    """Return adapted-minus-calibrated shared cores in one calibration frame."""
    adapted_layers = dict(named_sidecars(adapted))
    calibrated_layers = dict(named_sidecars(calibrated))
    if adapted_layers.keys() != calibrated_layers.keys():
        msg = "adapted and calibrated checkpoints must contain the same sidecar layers"
        raise ValueError(msg)
    if not adapted_layers:
        msg = "core delta capture requires at least one sidecar layer"
        raise ValueError(msg)

    delta: CoreDeltaBank = {}
    for name in sorted(adapted_layers):
        adapted_core = adapted_layers[name].capability_delta
        calibrated_core = calibrated_layers[name].capability_delta
        if adapted_core.shape != calibrated_core.shape:
            msg = f"shared-core shape mismatch at {name}"
            raise ValueError(msg)
        delta[name] = adapted_core.detach().clone() - calibrated_core.detach().clone()
    return delta


def select_core_delta(delta: CoreDeltaBank, layer_names: Collection[str]) -> CoreDeltaBank:
    """Select a locked layer subset from a core-delta bank."""
    if not layer_names:
        msg = "at least one core layer is required"
        raise ValueError(msg)
    missing = set(layer_names) - set(delta)
    if missing:
        msg = f"core delta does not contain layers: {sorted(missing)}"
        raise ValueError(msg)
    return {name: delta[name].detach().clone() for name in sorted(layer_names)}


@contextmanager
def temporary_core_delta(
    module: nn.Module,
    delta: CoreDeltaBank,
    *,
    scale: float = 1.0,
) -> Iterator[None]:
    """Temporarily add a locked core delta and restore the checkpoint afterward."""
    if not delta:
        msg = "temporary core intervention requires a non-empty delta"
        raise ValueError(msg)
    if not math.isfinite(scale):
        msg = "core intervention scale must be finite"
        raise ValueError(msg)
    layers = dict(named_sidecars(module))
    missing = set(delta) - set(layers)
    if missing:
        msg = f"checkpoint does not contain sidecar layers: {sorted(missing)}"
        raise ValueError(msg)

    targets: dict[str, Tensor] = {}
    originals: dict[str, Tensor] = {}
    for name, value in delta.items():
        core = layers[name].capability_delta
        if core.shape != value.shape:
            msg = f"shared-core intervention shape mismatch at {name}"
            raise ValueError(msg)
        if not torch.isfinite(value).all():
            msg = f"shared-core intervention contains non-finite values at {name}"
            raise ValueError(msg)
        targets[name] = core
        originals[name] = core.detach().clone()
    try:
        with torch.no_grad():
            for name, value in delta.items():
                core = targets[name]
                core.add_(value.to(device=core.device, dtype=core.dtype), alpha=scale)
        yield
    finally:
        with torch.no_grad():
            for name, value in originals.items():
                layers[name].capability_delta.copy_(value)


def rank_norm_matched_core_delta(delta: CoreDeltaBank, *, seed: int) -> CoreDeltaBank:
    """Sample deterministic controls with matched shape, rank, and singular values."""
    if not delta:
        msg = "matched core controls require a non-empty delta"
        raise ValueError(msg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    controls: CoreDeltaBank = {}
    for name in sorted(delta):
        reference = delta[name]
        if reference.ndim != 2:
            msg = f"shared-core delta must be a matrix at {name}"
            raise ValueError(msg)
        source = reference.detach().to(device="cpu", dtype=torch.float64)
        rank = int(torch.linalg.matrix_rank(source).item())
        if rank == 0:
            controls[name] = torch.zeros_like(reference)
            continue
        singular_values = torch.linalg.svdvals(source)[:rank]
        left_random = torch.randn(source.shape[0], rank, generator=generator, dtype=source.dtype)
        right_random = torch.randn(source.shape[1], rank, generator=generator, dtype=source.dtype)
        left = torch.linalg.qr(left_random, mode="reduced").Q
        right = torch.linalg.qr(right_random, mode="reduced").Q
        control = left @ torch.diag(singular_values) @ right.T
        controls[name] = control.to(device=reference.device, dtype=reference.dtype)
    return controls


def singular_value_concentration(
    delta: CoreDeltaBank,
) -> list[tuple[str, int, float, float]]:
    """Rank singular directions globally and report cumulative squared energy."""
    if not delta:
        msg = "singular concentration requires a non-empty candidate state"
        raise ValueError(msg)
    components: list[tuple[str, int, float]] = []
    for name in sorted(delta):
        value = delta[name]
        if value.ndim != 2 or not torch.isfinite(value).all():
            msg = f"candidate state must contain finite matrices: {name}"
            raise ValueError(msg)
        for index, singular_value in enumerate(torch.linalg.svdvals(value.float()).tolist()):
            components.append((name, index, float(singular_value)))
    components.sort(key=lambda item: (-item[2], item[0], item[1]))
    total = sum(value * value for _, _, value in components)
    if total == 0.0:
        return [(name, index, value, 0.0) for name, index, value in components]
    cumulative = 0.0
    output: list[tuple[str, int, float, float]] = []
    for name, index, value in components:
        cumulative += value * value
        output.append((name, index, value, cumulative / total))
    return output


def singular_component_core_delta(
    delta: CoreDeltaBank,
    *,
    budget: int,
    ordered_components: tuple[tuple[str, int], ...] | None = None,
) -> CoreDeltaBank:
    """Keep a prefix of singular directions under a fixed locked ordering."""
    ranked = singular_value_concentration(delta)
    if not 1 <= budget <= len(ranked):
        msg = "singular-component budget lies outside the candidate-state rank"
        raise ValueError(msg)
    available = {(name, index) for name, index, _, _ in ranked}
    ordering = ordered_components or tuple((name, index) for name, index, _, _ in ranked)
    if len(ordering) != len(set(ordering)) or set(ordering) != available:
        msg = "singular-component ordering must be a permutation of all directions"
        raise ValueError(msg)
    selected = set(ordering[:budget])
    output: CoreDeltaBank = {}
    for name in sorted(delta):
        source = delta[name].detach().to(device="cpu", dtype=torch.float64)
        left, singular_values, right = torch.linalg.svd(source, full_matrices=False)
        keep = torch.zeros_like(singular_values)
        for index, singular_value in enumerate(singular_values):
            if (name, index) in selected:
                keep[index] = singular_value
        reconstructed = left @ torch.diag(keep) @ right
        output[name] = reconstructed.to(device=delta[name].device, dtype=delta[name].dtype)
    return output


def rank_singular_components_by_development(
    delta: CoreDeltaBank,
    *,
    evaluate: Callable[[CoreDeltaBank], float],
) -> tuple[tuple[str, int], ...]:
    """Rank individual singular directions by absolute development-set effect."""
    ranked = singular_value_concentration(delta)
    all_components = tuple((name, index) for name, index, _, _ in ranked)
    scored: list[tuple[float, str, int]] = []
    for component in all_components:
        ordering = (component, *(item for item in all_components if item != component))
        effect = evaluate(
            singular_component_core_delta(
                delta,
                budget=1,
                ordered_components=ordering,
            )
        )
        if not math.isfinite(effect):
            msg = f"development effect is non-finite for singular component {component}"
            raise ValueError(msg)
        scored.append((abs(effect), component[0], component[1]))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple((name, index) for _, name, index in scored)


def minimum_recovery_budget(recovery: Mapping[int, float], *, threshold: float) -> int:
    """Return the smallest tested budget that reaches a recovery fraction."""
    if not recovery or not 0.0 < threshold <= 1.0:
        msg = "recovery curve and threshold must be valid"
        raise ValueError(msg)
    invalid = [
        budget for budget, value in recovery.items() if budget < 1 or not math.isfinite(value)
    ]
    if invalid:
        msg = f"invalid recovery-curve budgets: {sorted(invalid)}"
        raise ValueError(msg)
    matches = [budget for budget, value in recovery.items() if value >= threshold]
    if not matches:
        msg = f"no tested budget reaches recovery threshold {threshold}"
        raise ValueError(msg)
    return min(matches)


def functional_energy_matched_core_delta(
    reference: CoreDeltaBank,
    control: CoreDeltaBank,
    *,
    model: nn.Module,
    task: SidecarTask,
    layer_inputs: Mapping[str, Tensor],
) -> CoreDeltaBank:
    """Scale each control matrix to match functional B-Q-A-x energy."""
    layers = dict(named_sidecars(model))
    if reference.keys() != control.keys() or reference.keys() != layer_inputs.keys():
        msg = "functional matching requires identical layer keys"
        raise ValueError(msg)
    if not reference.keys() <= layers.keys():
        msg = "functional matching references unknown sidecar layers"
        raise ValueError(msg)
    matched: CoreDeltaBank = {}
    for name in sorted(reference):
        layer = layers[name]
        inputs = layer_inputs[name].to(device=reference[name].device, dtype=reference[name].dtype)
        read = layer.shared_input[task](inputs)
        reference_routed = layer.shared_output[task](read @ reference[name].T)
        control_routed = layer.shared_output[task](read @ control[name].T)
        reference_energy = reference_routed.float().square().mean()
        control_energy = control_routed.float().square().mean()
        if reference_energy == 0:
            matched[name] = torch.zeros_like(control[name])
            continue
        if control_energy <= 0:
            msg = f"control has zero functional energy at {name}"
            raise ValueError(msg)
        scale = torch.sqrt(reference_energy / control_energy)
        matched[name] = control[name] * scale.to(control[name])
    return matched


def local_gain_bound_metrics(  # noqa: PLR0915
    model: nn.Module,
    *,
    task: SidecarTask,
    layer_inputs: Mapping[str, Tensor],
    timestep_bucket: int = 0,
) -> dict[str, float]:
    """Measure every factor in the same-input multiplicative-gain bound."""
    layers = dict(named_sidecars(model))
    if not layer_inputs or not layer_inputs.keys() <= layers.keys():
        msg = "local-bound inputs must identify a non-empty sidecar subset"
        raise ValueError(msg)

    access_values: list[Tensor] = []
    activation_infinity: list[Tensor] = []
    routed_input_norms: list[Tensor] = []
    output_norms: list[Tensor] = []
    delta_norms: list[Tensor] = []
    empirical_effects: list[Tensor] = []
    theoretical_bounds: list[Tensor] = []
    gain_limits: list[float] = []
    for name in sorted(layer_inputs):
        layer = layers[name]
        if layer.shared_update != "multiplicative":
            msg = f"local gain bound requires a multiplicative sidecar: {name}"
            raise ValueError(msg)
        if not 0 <= timestep_bucket < layer.timestep_buckets:
            msg = f"invalid timestep bucket for local-bound audit: {(name, timestep_bucket)}"
            raise ValueError(msg)
        inputs = layer_inputs[name].to(
            device=layer.base.weight.device, dtype=layer.base.weight.dtype
        )
        if inputs.shape[-1] != layer.base.in_features:
            msg = f"local-bound input width mismatch at {name}"
            raise ValueError(msg)
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        if flat_inputs.shape[0] == 0 or not torch.isfinite(flat_inputs).all():
            msg = f"local-bound inputs must be finite and non-empty at {name}"
            raise ValueError(msg)

        base_output = layer.base(flat_inputs)
        routed_input = layer.shared_input[task](flat_inputs)
        gate_logit = (
            layer.understanding_route_logit
            if task == "understanding"
            else layer.route_logits[timestep_bucket]
        )
        gate = torch.sigmoid(gate_logit).detach().float()
        output_projection = cast("nn.Linear", layer.shared_output[task])
        output_weight = output_projection.weight
        calibrated = output_projection(
            torch.nn.functional.linear(routed_input, layer.shared_core.weight)
        )
        adapted = output_projection(
            torch.nn.functional.linear(
                routed_input,
                layer.shared_core.weight + layer.capability_delta,
            )
        )
        calibrated_gain = layer.gain_limit * torch.tanh(calibrated * gate * layer.scaling)
        adapted_gain = layer.gain_limit * torch.tanh(adapted * gate * layer.scaling)
        empirical = torch.linalg.vector_norm((adapted_gain - calibrated_gain) * base_output, dim=-1)

        output_norm = torch.linalg.matrix_norm(output_weight.detach().float(), ord=2)
        delta_norm = torch.linalg.matrix_norm(layer.capability_delta.detach().float(), ord=2)
        hidden_infinity = base_output.detach().float().abs().amax(dim=-1)
        routed_norm = torch.linalg.vector_norm(routed_input.detach().float(), dim=-1)
        bound = (
            layer.scaling
            * layer.gain_limit
            * gate
            * hidden_infinity
            * output_norm
            * delta_norm
            * routed_norm
        )

        access_values.append(gate.reshape(1))
        activation_infinity.append(hidden_infinity)
        routed_input_norms.append(routed_norm)
        output_norms.append(output_norm.reshape(1))
        delta_norms.append(delta_norm.reshape(1))
        empirical_effects.append(empirical.detach().float())
        theoretical_bounds.append(bound.detach().float())
        gain_limits.append(layer.gain_limit)

    empirical_max = torch.cat(empirical_effects).max()
    bound_max = torch.cat(theoretical_bounds).max()
    if empirical_max <= 0:
        msg = "local-bound ratio is undefined for a zero realized capability effect"
        raise ValueError(msg)
    return {
        "gain_limit": max(gain_limits),
        "access_q95": torch.quantile(torch.cat(access_values), 0.95).item(),
        "activation_infinity_q95": torch.quantile(torch.cat(activation_infinity), 0.95).item(),
        "max_output_spectral_norm": torch.cat(output_norms).max().item(),
        "max_delta_spectral_norm": torch.cat(delta_norms).max().item(),
        "routed_input_norm_max": torch.cat(routed_input_norms).max().item(),
        "empirical_effect_max": empirical_max.item(),
        "theoretical_bound_max": bound_max.item(),
        "bound_to_empirical_ratio": (bound_max / empirical_max).item(),
    }


def linear_combine_core_deltas(
    deltas: tuple[CoreDeltaBank, ...],
    *,
    weights: tuple[float, ...],
) -> CoreDeltaBank:
    """Combine candidate states that share one locked calibration frame."""
    if not deltas or len(deltas) != len(weights):
        msg = "state composition requires equally many states and weights"
        raise ValueError(msg)
    if any(not math.isfinite(weight) for weight in weights):
        msg = "state composition weights must be finite"
        raise ValueError(msg)
    keys = deltas[0].keys()
    if any(delta.keys() != keys for delta in deltas[1:]):
        msg = "composed states must contain identical layer keys"
        raise ValueError(msg)
    output: CoreDeltaBank = {}
    for name in sorted(keys):
        reference = deltas[0][name]
        if any(delta[name].shape != reference.shape for delta in deltas[1:]):
            msg = f"composed state shape mismatch at {name}"
            raise ValueError(msg)
        value = torch.zeros_like(reference)
        for delta, weight in zip(deltas, weights, strict=True):
            value = value + delta[name].to(value) * weight
        output[name] = value
    return output


def linear_combine_candidate_states(
    states: tuple[CandidateState, ...],
    *,
    weights: tuple[float, ...],
) -> CandidateState:
    """Compose only states derived from the identical calibration artifact."""
    if not states:
        msg = "candidate-state composition requires at least one state"
        raise ValueError(msg)
    frames = {state.calibration_frame_sha256 for state in states}
    if len(frames) != 1:
        msg = "candidate states do not share one calibration frame"
        raise ValueError(msg)
    return CandidateState(
        states[0].calibration_frame_sha256,
        linear_combine_core_deltas(
            tuple(state.delta for state in states),
            weights=weights,
        ),
    )


def functional_core_similarity(
    left: CoreDeltaBank,
    right: CoreDeltaBank,
    *,
    model: nn.Module,
    task: SidecarTask,
    layer_inputs: Mapping[str, Tensor],
) -> float:
    """Compare candidate states after the calibrated task-specific maps."""
    layers = dict(named_sidecars(model))
    if left.keys() != right.keys() or left.keys() != layer_inputs.keys():
        msg = "functional similarity requires identical layer keys"
        raise ValueError(msg)
    left_values: list[Tensor] = []
    right_values: list[Tensor] = []
    for name in sorted(left):
        if name not in layers or left[name].shape != right[name].shape:
            msg = f"functional similarity has an invalid layer: {name}"
            raise ValueError(msg)
        layer = layers[name]
        inputs = layer_inputs[name].to(device=left[name].device, dtype=left[name].dtype)
        read = layer.shared_input[task](inputs)
        left_values.append(layer.shared_output[task](read @ left[name].T).float().reshape(-1))
        right_values.append(layer.shared_output[task](read @ right[name].T).float().reshape(-1))
    left_flat = torch.cat(left_values)
    right_flat = torch.cat(right_values)
    denominator = torch.linalg.norm(left_flat) * torch.linalg.norm(right_flat)
    if denominator <= 0:
        msg = "functional similarity is undefined for a zero-energy state"
        raise ValueError(msg)
    return float((torch.dot(left_flat, right_flat) / denominator).item())


def functional_candidate_similarity(
    left: CandidateState,
    right: CandidateState,
    *,
    model: nn.Module,
    task: SidecarTask,
    layer_inputs: Mapping[str, Tensor],
) -> float:
    """Compare two states only when their calibration coordinates are identical."""
    if left.calibration_frame_sha256 != right.calibration_frame_sha256:
        msg = "candidate states do not share one calibration frame"
        raise ValueError(msg)
    return functional_core_similarity(
        left.delta,
        right.delta,
        model=model,
        task=task,
        layer_inputs=layer_inputs,
    )


def route_cancellation_metrics(bank: RouteComponentBank) -> dict[str, float]:
    """Measure energy and cosine cancellation between private and shared paths."""
    if not bank or any(not calls for calls in bank.values()):
        msg = "route cancellation requires captured private and shared components"
        raise ValueError(msg)
    private_energy: list[Tensor] = []
    shared_energy: list[Tensor] = []
    cosines: list[Tensor] = []
    for calls in bank.values():
        for private, shared in calls:
            if private.shape != shared.shape:
                msg = "private and shared component shapes must match"
                raise ValueError(msg)
            private_flat = private.detach().float().reshape(-1)
            shared_flat = shared.detach().float().reshape(-1)
            private_energy.append(private_flat.square().mean())
            shared_energy.append(shared_flat.square().mean())
            denominator = torch.linalg.norm(private_flat) * torch.linalg.norm(shared_flat)
            cosine = (
                torch.dot(private_flat, shared_flat) / denominator
                if denominator > 0
                else torch.tensor(0.0)
            )
            cosines.append(cosine)
    cosine_values = torch.stack(cosines)
    return {
        "private_energy": torch.stack(private_energy).mean().item(),
        "shared_energy": torch.stack(shared_energy).mean().item(),
        "mean_cosine": cosine_values.mean().item(),
        "negative_cosine_fraction": (cosine_values < 0).float().mean().item(),
    }


def route_stability_metrics(bank: PatchBank) -> dict[str, float]:
    """Measure realized modulation magnitude and variation across time buckets."""
    if not bank or any(not calls for calls in bank.values()):
        msg = "route stability metrics require a non-empty patch bank"
        raise ValueError(msg)

    energies: list[Tensor] = []
    maxima: list[Tensor] = []
    site_means: dict[SidecarSite, Tensor] = {}
    for site, calls in bank.items():
        reference_shape = calls[0].shape
        if any(value.shape != reference_shape for value in calls):
            msg = f"captured modulation shapes differ at site {site}"
            raise ValueError(msg)
        values = [value.detach().to(dtype=torch.float32) for value in calls]
        energies.extend(value.square().mean() for value in values)
        maxima.extend(value.abs().max() for value in values)
        site_means[site] = torch.stack(values).mean(dim=0)

    adjacent_variations: list[Tensor] = []
    layer_names = {name for name, _ in site_means}
    for name in layer_names:
        ordered = sorted(
            ((bucket, value) for (layer, bucket), value in site_means.items() if layer == name),
            key=lambda item: item[0],
        )
        for (left_bucket, left), (right_bucket, right) in pairwise(ordered):
            if right_bucket != left_bucket + 1:
                continue
            if left.shape != right.shape:
                msg = f"adjacent modulation shapes differ in layer {name}"
                raise ValueError(msg)
            adjacent_variations.append((right - left).square().mean())

    variation = torch.stack(adjacent_variations).mean().item() if adjacent_variations else 0.0
    return {
        "mean_modulation_energy": torch.stack(energies).mean().item(),
        "max_absolute_modulation": torch.stack(maxima).max().item(),
        "mean_adjacent_bucket_variation": variation,
    }


def select_patch_bank(bank: PatchBank, sites: Collection[SidecarSite]) -> PatchBank:
    missing = set(sites) - set(bank)
    if missing:
        msg = f"patch bank does not contain sites: {sorted(missing)}"
        raise ValueError(msg)
    if not sites:
        msg = "at least one patch site is required"
        raise ValueError(msg)
    return {site: [value.clone() for value in bank[site]] for site in sorted(sites)}


def swap_patch_bank(bank: PatchBank, *, batch_permutation: Tensor) -> PatchBank:
    if batch_permutation.ndim != 1:
        msg = "batch_permutation must be one-dimensional"
        raise ValueError(msg)
    swapped: PatchBank = {}
    for site, calls in bank.items():
        swapped[site] = []
        for value in calls:
            if value.shape[0] != batch_permutation.numel():
                msg = f"batch permutation does not match patch at {site}"
                raise ValueError(msg)
            swapped[site].append(value.index_select(0, batch_permutation.to(value.device)))
    return swapped


def noise_patch_bank(bank: PatchBank, *, seed: int, covariance_matched: bool = True) -> PatchBank:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output: PatchBank = {}
    for site, calls in bank.items():
        output[site] = []
        for value in calls:
            source = value.detach().to(device="cpu", dtype=torch.float32)
            noise = torch.randn(source.shape, generator=generator)
            if covariance_matched and source.ndim >= 2 and source.shape[0] > 1:
                flat = source.reshape(-1, source.shape[-1])
                centered = flat - flat.mean(dim=0, keepdim=True)
                covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                root = eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt())
                noise = noise.reshape(-1, source.shape[-1]) @ root.T
                noise = noise.reshape(source.shape)
            else:
                noise = noise * source.std(unbiased=False).clamp_min(1e-8) + source.mean()
            output[site].append(noise.to(device=value.device, dtype=value.dtype))
    return output
