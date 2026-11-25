from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor, nn

from skillbridge.sidecar import inject_sidecars, named_sidecars
from skillbridge.sidecar_config import SidecarExperimentConfig

MetadataValue = str | int | float | bool


class SidecarCheckpoint(TypedDict):
    format_version: int
    target_names: list[str]
    metadata: dict[str, MetadataValue]
    state_dict: dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class SidecarAttachmentReport:
    target_names: tuple[str, ...]
    sidecar_parameter_count: int


def _sidecar_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    parameters: dict[str, nn.Parameter] = {}
    for layer_name, layer in named_sidecars(module):
        for parameter_name, parameter in layer.named_parameters():
            if not parameter_name.startswith("base."):
                parameters[f"{layer_name}.{parameter_name}"] = parameter
    return parameters


def attach_configured_sidecars(
    backbone: nn.Module,
    config: SidecarExperimentConfig,
) -> SidecarAttachmentReport:
    targets = inject_sidecars(
        backbone,
        rank=config.rank,
        alpha=config.alpha,
        timestep_buckets=config.timestep_buckets,
        gain_limit=config.gain_limit,
        shared_update=config.shared_update,
        target_suffixes=config.target_suffixes,
        exclude_names=config.excluded_target_names,
    )
    if not targets:
        suffixes = ", ".join(config.target_suffixes)
        msg = f"backbone contains no configured linear targets: {suffixes}"
        raise ValueError(msg)
    count = sum(parameter.numel() for parameter in _sidecar_parameters(backbone).values())
    return SidecarAttachmentReport(tuple(targets), count)


def resolve_dimoo_transformer(model: nn.Module) -> nn.Module:
    inner_model = getattr(model, "model", None)
    transformer = getattr(inner_model, "transformer", None)
    if not isinstance(transformer, nn.Module):
        msg = "expected the official unified model path model.model.transformer"
        raise ValueError(msg)
    return transformer


def attach_dimoo_sidecars(
    model: nn.Module,
    config: SidecarExperimentConfig,
) -> SidecarAttachmentReport:
    return attach_configured_sidecars(resolve_dimoo_transformer(model), config)


def resolve_janusflow_transformer(model: nn.Module) -> nn.Module:
    language_model = getattr(model, "language_model", None)
    transformer = getattr(language_model, "model", None)
    if not isinstance(transformer, nn.Module):
        msg = "expected the official JanusFlow path model.language_model.model"
        raise ValueError(msg)
    return transformer


def attach_janusflow_sidecars(
    model: nn.Module,
    config: SidecarExperimentConfig,
) -> SidecarAttachmentReport:
    return attach_configured_sidecars(resolve_janusflow_transformer(model), config)


def resolve_uniworld_v1_transformer(model: nn.Module) -> nn.Module:
    transformer = getattr(model, "model", None)
    if not isinstance(transformer, nn.Module):
        msg = "expected the official UniWorld-V1 Qwen conditioner at model.model"
        raise ValueError(msg)
    return transformer


def attach_uniworld_v1_sidecars(
    model: nn.Module,
    config: SidecarExperimentConfig,
) -> SidecarAttachmentReport:
    return attach_configured_sidecars(resolve_uniworld_v1_transformer(model), config)


def resolve_module_path(model: nn.Module, path: str) -> nn.Module:
    current: object = model
    for component in path.split("."):
        if component.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
            current = current[int(component)]
        else:
            current = getattr(current, component, None)
        if current is None:
            msg = f"model does not expose module path: {path}"
            raise ValueError(msg)
    if not isinstance(current, nn.Module):
        msg = f"resolved object is not a module: {path}"
        raise ValueError(msg)
    return current


def attach_backbone_sidecars(
    model: nn.Module,
    config: SidecarExperimentConfig,
    *,
    transformer_path: str,
) -> SidecarAttachmentReport:
    return attach_configured_sidecars(resolve_module_path(model, transformer_path), config)


def save_sidecar_checkpoint(
    module: nn.Module,
    path: Path,
    *,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> SidecarCheckpoint:
    parameters = _sidecar_parameters(module)
    if not parameters:
        msg = "module contains no routed low-rank sidecars"
        raise ValueError(msg)
    checkpoint: SidecarCheckpoint = {
        "format_version": 1,
        "target_names": [name for name, _ in named_sidecars(module)],
        "metadata": dict(metadata or {}),
        "state_dict": {
            name: parameter.detach().cpu().clone() for name, parameter in parameters.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def load_sidecar_checkpoint(module: nn.Module, path: Path) -> SidecarCheckpoint:
    checkpoint: SidecarCheckpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        msg = f"unsupported sidecar checkpoint version: {checkpoint['format_version']}"
        raise ValueError(msg)
    parameters = _sidecar_parameters(module)
    state = checkpoint["state_dict"]
    if set(state) != set(parameters):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        msg = f"sidecar checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        raise ValueError(msg)
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                msg = f"sidecar checkpoint shape mismatch for {name}"
                raise ValueError(msg)
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return checkpoint
