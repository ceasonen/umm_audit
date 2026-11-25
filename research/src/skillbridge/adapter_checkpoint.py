from __future__ import annotations

from skillbridge.artifact_policy import forbidden_hashing as hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor, nn

from skillbridge.sidecar import RoutedLowRankLinear

MetadataValue = str | int | float | bool
AdapterState = dict[str, Tensor]


class AdapterCheckpoint(TypedDict):
    format_version: int
    metadata: dict[str, MetadataValue]
    state_dict: dict[str, Tensor]


def _adapter_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    parameters: dict[str, nn.Parameter] = {}
    for module_name, child in module.named_modules():
        if not (
            isinstance(child, RoutedLowRankLinear)
            or bool(getattr(child, "_skillbridge_route_compatible", False))
        ):
            continue
        for name, parameter in child.named_parameters(recurse=True):
            if name.startswith("base."):
                continue
            key = f"{module_name}.{name}" if module_name else name
            parameters[key] = parameter
    return parameters


def capture_adapter_state(module: nn.Module) -> AdapterState:
    """Snapshot adapter tensors without retaining references to later updates."""
    parameters = _adapter_parameters(module)
    if not parameters:
        msg = "module contains no adapter parameters"
        raise ValueError(msg)
    return {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}


def _capability_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in _adapter_parameters(module).items()
        if name == "capability_delta" or name.endswith(".capability_delta")
    }


def capture_capability_state(module: nn.Module) -> AdapterState:
    """Snapshot only the learned shared core increments.

    Tied access bases are reconstructed deterministically from the frozen
    backbone, so they are not part of the persistent capability state.
    """
    parameters = _capability_parameters(module)
    if not parameters:
        msg = "module contains no capability deltas"
        raise ValueError(msg)
    return {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}


def adapter_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash tensor content independently of torch serialization and run metadata."""
    if not state:
        msg = "adapter state must be non-empty"
        raise ValueError(msg)
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_checkpoint_state_sha256(path: Path) -> str:
    checkpoint: AdapterCheckpoint = torch.load(path, map_location="cpu", weights_only=True)
    return adapter_state_sha256(checkpoint["state_dict"])


def save_adapter_state(
    state: Mapping[str, Tensor],
    path: Path,
    *,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> AdapterCheckpoint:
    """Persist an immutable adapter snapshot captured at a stage boundary."""
    if path.exists():
        msg = f"refusing to overwrite adapter checkpoint: {path}"
        raise FileExistsError(msg)
    if not state or any(
        not name or not torch.isfinite(value).all() for name, value in state.items()
    ):
        msg = "adapter state must contain named finite tensors"
        raise ValueError(msg)
    checkpoint: AdapterCheckpoint = {
        "format_version": 1,
        "metadata": dict(metadata or {}),
        "state_dict": {name: value.detach().cpu().clone() for name, value in state.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def save_adapter_checkpoint(
    module: nn.Module,
    path: Path,
    *,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> AdapterCheckpoint:
    return save_adapter_state(capture_adapter_state(module), path, metadata=metadata)


def save_capability_checkpoint(
    module: nn.Module,
    path: Path,
    *,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> AdapterCheckpoint:
    """Persist the minimal learned state after deterministic access setup."""
    return save_adapter_state(capture_capability_state(module), path, metadata=metadata)


def load_adapter_checkpoint(module: nn.Module, path: Path) -> AdapterCheckpoint:
    checkpoint: AdapterCheckpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        msg = f"unsupported adapter checkpoint version: {checkpoint['format_version']}"
        raise ValueError(msg)
    parameters = _adapter_parameters(module)
    state = checkpoint["state_dict"]
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        msg = f"adapter checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        raise ValueError(msg)
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                msg = f"adapter checkpoint shape mismatch for {name}"
                raise ValueError(msg)
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return checkpoint


def restore_svd_adapter(module: nn.Module, core_path: Path) -> AdapterCheckpoint:
    """Restore persisted access maps; SVD sign/subspace choices must not drift."""
    core = torch.load(core_path, map_location="cpu", weights_only=True)
    full_path = core_path.parent / "adapter.pt"
    full = torch.load(full_path, map_location="cpu", weights_only=True)
    if core["metadata"].get("method") != "svd_core" or full["metadata"] != core["metadata"]:
        raise ValueError("SVD core/adapter metadata differ")
    expected = {k: v for k, v in full["state_dict"].items()
                if k == "capability_delta" or k.endswith(".capability_delta")}
    if set(expected) != set(core["state_dict"]) or any(
        not torch.equal(v, core["state_dict"][k]) for k, v in expected.items()
    ):
        raise ValueError("SVD core differs from its saved adapter")
    if any(not torch.isfinite(v).all() for v in full["state_dict"].values()):
        raise ValueError("nonfinite recorded SVD adapter")
    return load_adapter_checkpoint(module, full_path)


def load_capability_checkpoint(module: nn.Module, path: Path) -> AdapterCheckpoint:
    """Restore only capability deltas into an initialized tied-access route."""
    checkpoint: AdapterCheckpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        msg = f"unsupported adapter checkpoint version: {checkpoint['format_version']}"
        raise ValueError(msg)
    parameters = _capability_parameters(module)
    state = checkpoint["state_dict"]
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        msg = f"capability checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        raise ValueError(msg)
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                msg = f"capability checkpoint shape mismatch for {name}"
                raise ValueError(msg)
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return checkpoint
