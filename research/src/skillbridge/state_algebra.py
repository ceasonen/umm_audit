from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn

from skillbridge.adapter_checkpoint import AdapterCheckpoint
from skillbridge.sidecar import SidecarTask, named_sidecars

Normalization = Literal["none", "max_input_norm"]

_FRAME_KEYS = (
    "backbone",
    "method",
    "seed",
    "rank",
    "timestep_buckets",
    "image_size",
    "gain_limit",
    "target_suffixes",
)


def _load(path: Path) -> AdapterCheckpoint:
    checkpoint = cast(
        "AdapterCheckpoint",
        torch.load(path, map_location="cpu", weights_only=True),
    )
    if checkpoint.get("format_version") != 1:
        message = f"unsupported capability checkpoint: {path}"
        raise ValueError(message)
    metadata = checkpoint.get("metadata")
    state = checkpoint.get("state_dict")
    if not isinstance(metadata, dict) or not isinstance(state, dict) or not state:
        message = f"invalid capability checkpoint payload: {path}"
        raise ValueError(message)
    if metadata.get("method") != "tied_core":
        message = "state algebra requires tied_core checkpoints"
        raise ValueError(message)
    if any(
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        for value in state.values()
    ):
        message = f"capability checkpoint contains invalid tensors: {path}"
        raise ValueError(message)
    return checkpoint


def _state_norm(state: dict[str, Tensor]) -> float:
    squared = sum(
        float(torch.sum(value.detach().to(dtype=torch.float64).square()).item())
        for value in state.values()
    )
    return math.sqrt(squared)


def functional_operator_energy(
    state: Mapping[str, Tensor],
    *,
    model: nn.Module,
    task: SidecarTask,
) -> float:
    """Return the squared Frobenius energy of the realized linear maps $BQA$.

    The trace identity avoids materializing the full output-by-input operator,
    which would be prohibitively large for transformer projections.
    """
    layers = dict(named_sidecars(model))
    keyed_layers = {
        ("capability_delta" if name == "<root>" else f"{name}.capability_delta"): layer
        for name, layer in layers.items()
    }
    if not keyed_layers or set(state) != set(keyed_layers):
        missing = sorted(set(keyed_layers) - set(state))
        unexpected = sorted(set(state) - set(keyed_layers))
        message = (
            "operator-energy state mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
        raise ValueError(message)

    total = 0.0
    for name in sorted(state):
        layer = keyed_layers[name]
        core = state[name]
        if core.ndim != 2 or tuple(core.shape) != (layer.rank, layer.rank):
            message = f"invalid operator-energy core shape at {name}"
            raise ValueError(message)
        if not core.is_floating_point() or not torch.isfinite(core).all():
            message = f"invalid operator-energy core values at {name}"
            raise ValueError(message)
        device = layer.shared_input[task].weight.device
        read = layer.shared_input[task].weight.detach().to(device=device, dtype=torch.float32)
        write = layer.shared_output[task].weight.detach().to(
            device=device, dtype=torch.float32
        )
        routed_core = core.detach().to(device=device, dtype=torch.float32)
        read_gram = read @ read.T
        write_gram = write.T @ write
        energy = torch.trace(
            routed_core.T @ write_gram @ routed_core @ read_gram
        ).double()
        total += max(float(energy.item()), 0.0)
    return total


def match_composed_operator_energy(
    composed: AdapterCheckpoint,
    sources: Sequence[Path],
    *,
    model: nn.Module,
    task: SidecarTask,
) -> AdapterCheckpoint:
    """Scale a composition to the largest component $BQA$ operator energy."""
    checkpoints = tuple(_load(Path(path).resolve()) for path in sources)
    _validate_frames(checkpoints)
    reference_frame = tuple(checkpoints[0]["metadata"].get(key) for key in _FRAME_KEYS)
    composed_frame = tuple(composed["metadata"].get(key) for key in _FRAME_KEYS)
    if reference_frame != composed_frame:
        message = "composed state does not share the source coordinate frame"
        raise ValueError(message)

    component_energies = tuple(
        functional_operator_energy(checkpoint["state_dict"], model=model, task=task)
        for checkpoint in checkpoints
    )
    if any(value <= 0.0 or not math.isfinite(value) for value in component_energies):
        message = "operator-energy matching requires nonzero finite source states"
        raise ValueError(message)
    pre_energy = functional_operator_energy(
        composed["state_dict"], model=model, task=task
    )
    if pre_energy <= 0.0 or not math.isfinite(pre_energy):
        message = "operator-energy matching requires a nonzero finite composition"
        raise ValueError(message)
    target_energy = max(component_energies)
    scale = math.sqrt(target_energy / pre_energy)
    state = {name: value.mul(scale) for name, value in composed["state_dict"].items()}
    post_energy = functional_operator_energy(state, model=model, task=task)
    matching_error = abs(post_energy - target_energy) / target_energy
    if matching_error > 0.01:
        message = (
            "operator-energy matching exceeds one-percent tolerance: "
            f"{matching_error:.6g}"
        )
        raise ValueError(message)
    metadata = dict(composed["metadata"])
    metadata.update(
        {
            "normalization": "max_component_operator_energy",
            "operator_energy_task": task,
            "component_operator_energies": ",".join(
                format(value, ".8g") for value in component_energies
            ),
            "pre_normalization_operator_energy": pre_energy,
            "post_normalization_operator_energy": post_energy,
            "operator_energy_target": target_energy,
            "operator_energy_matching_relative_error": matching_error,
            "post_normalization_norm": _state_norm(state),
        }
    )
    return {"format_version": 1, "metadata": metadata, "state_dict": state}


def _validate_request(
    sources: Sequence[Path], coefficients: Sequence[float], normalization: Normalization
) -> tuple[float, ...]:
    if len(sources) < 2 or len(sources) != len(coefficients):
        message = "state algebra requires equally sized source and coefficient lists"
        raise ValueError(message)
    if normalization not in {"none", "max_input_norm"}:
        message = f"unknown state-algebra normalization: {normalization}"
        raise ValueError(message)
    weights = tuple(float(value) for value in coefficients)
    if any(not math.isfinite(value) for value in weights) or not any(weights):
        message = "state-algebra coefficients must be finite and nonzero"
        raise ValueError(message)
    return weights


def _validate_frames(
    checkpoints: Sequence[AdapterCheckpoint],
) -> tuple[tuple[str, ...], dict[str, Tensor]]:
    reference_metadata = checkpoints[0]["metadata"]
    reference_frame = tuple(reference_metadata.get(key) for key in _FRAME_KEYS)
    for checkpoint in checkpoints[1:]:
        frame = tuple(checkpoint["metadata"].get(key) for key in _FRAME_KEYS)
        if frame != reference_frame:
            message = "capability checkpoints do not share one coordinate frame"
            raise ValueError(message)
    capabilities = tuple(str(item["metadata"].get("capability", "")) for item in checkpoints)
    if any(not capability for capability in capabilities) or len(set(capabilities)) != len(
        capabilities
    ):
        message = "state algebra requires distinct source capabilities"
        raise ValueError(message)
    reference_state = checkpoints[0]["state_dict"]
    keys = set(reference_state)
    for checkpoint in checkpoints[1:]:
        state = checkpoint["state_dict"]
        if set(state) != keys or any(
            state[name].shape != reference_state[name].shape for name in keys
        ):
            message = "capability checkpoints use incompatible state coordinates"
            raise ValueError(message)
    return capabilities, reference_state


def compose_capability_states(
    sources: Sequence[Path],
    *,
    coefficients: Sequence[float],
    normalization: Normalization = "none",
) -> AdapterCheckpoint:
    """Compose TMC states only inside an identical deterministic coordinate frame."""
    weights = _validate_request(sources, coefficients, normalization)
    checkpoints = tuple(_load(Path(path).resolve()) for path in sources)
    reference_metadata = checkpoints[0]["metadata"]
    capabilities, reference_state = _validate_frames(checkpoints)

    composed: dict[str, Tensor] = {}
    for name, reference in reference_state.items():
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for weight, checkpoint in zip(weights, checkpoints, strict=True):
            accumulator.add_(checkpoint["state_dict"][name].to(dtype=torch.float64), alpha=weight)
        composed[name] = accumulator.to(dtype=reference.dtype)

    input_norms = tuple(_state_norm(checkpoint["state_dict"]) for checkpoint in checkpoints)
    pre_normalization_norm = _state_norm(composed)
    if pre_normalization_norm == 0:
        message = "state composition cancels to the zero state"
        raise ValueError(message)
    if normalization == "max_input_norm":
        scale = max(input_norms) / pre_normalization_norm
        composed = {name: value.mul(scale) for name, value in composed.items()}

    metadata = dict(reference_metadata)
    metadata.update(
        {
            "capability": "composition",
            "source_capabilities": ",".join(capabilities),
            "coefficients": ",".join(format(value, ".8g") for value in weights),
            "normalization": normalization,
            "component_norms": ",".join(format(value, ".8g") for value in input_norms),
            "pre_normalization_norm": pre_normalization_norm,
            "post_normalization_norm": _state_norm(composed),
        }
    )
    return {
        "format_version": 1,
        "metadata": metadata,
        "state_dict": composed,
    }
