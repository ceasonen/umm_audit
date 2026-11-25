from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from skillbridge.sidecar import RoutedLowRankLinear
from skillbridge.state_algebra import functional_operator_energy


@dataclass(frozen=True, slots=True)
class CoreLayerSpectrum:
    left: Tensor
    singular_values: Tensor
    right: Tensor


CoreSpectrum = dict[str, CoreLayerSpectrum]
CoreLayerBand = Literal["early", "middle", "late"]


def capture_core_spectrum(module: nn.Module) -> CoreSpectrum:
    """Capture the SVD of every trained TMC capability core."""
    spectrum: CoreSpectrum = {}
    for name, child in module.named_modules():
        if not isinstance(child, RoutedLowRankLinear):
            continue
        left, values, right = torch.linalg.svd(
            child.capability_delta.detach().float().cpu(), full_matrices=False
        )
        spectrum[name] = CoreLayerSpectrum(left, values, right)
    if not spectrum:
        msg = "module contains no tied multiplicative cores"
        raise ValueError(msg)
    return spectrum


def core_singular_concentration(
    spectrum: CoreSpectrum,
) -> list[dict[str, float | int | str]]:
    """Globally order capability-core directions by singular-value energy."""
    if not spectrum:
        msg = "core spectrum must be non-empty"
        raise ValueError(msg)
    components = sorted(
        (
            (name, index, float(value))
            for name, layer in spectrum.items()
            for index, value in enumerate(layer.singular_values)
        ),
        key=lambda item: (-item[2], item[0], item[1]),
    )
    total_energy = sum(value * value for _, _, value in components)
    if total_energy <= 0.0:
        msg = "tied capability cores have zero energy"
        raise ValueError(msg)
    cumulative = 0.0
    rows: list[dict[str, float | int | str]] = []
    for name, index, value in components:
        cumulative += value * value
        rows.append(
            {
                "layer": name,
                "component": index,
                "singular_value": value,
                "cumulative_energy": cumulative / total_energy,
            }
        )
    return rows


def _depth_index(name: str) -> int:
    matches = re.findall(r"(?:^|\.)(\d+)(?=\.|$)", name)
    if not matches:
        msg = f"TMC site has no transformer depth index: {name}"
        raise ValueError(msg)
    return int(matches[-1])


def core_layer_bands(spectrum: CoreSpectrum) -> dict[CoreLayerBand, tuple[str, ...]]:
    """Partition transformer depths into fixed early, middle, and late bands."""
    if not spectrum:
        msg = "core spectrum must be non-empty"
        raise ValueError(msg)
    depths = sorted({_depth_index(name) for name in spectrum})
    if len(depths) < 3:
        msg = "layer-band localization requires at least three transformer depths"
        raise ValueError(msg)
    width, remainder = divmod(len(depths), 3)
    sizes = tuple(width + (index < remainder) for index in range(3))
    partitions: list[set[int]] = []
    offset = 0
    for size in sizes:
        partitions.append(set(depths[offset : offset + size]))
        offset += size
    labels: tuple[CoreLayerBand, ...] = ("early", "middle", "late")
    return {
        label: tuple(
            name for name in sorted(spectrum) if _depth_index(name) in partitions[index]
        )
        for index, label in enumerate(labels)
    }


def _spectrum_state(spectrum: CoreSpectrum) -> dict[str, Tensor]:
    return {
        ("capability_delta" if name == "<root>" else f"{name}.capability_delta"): (
            item.left @ torch.diag(item.singular_values) @ item.right
        )
        for name, item in spectrum.items()
    }


def _site_from_state_key(name: str) -> str:
    return "<root>" if name == "capability_delta" else name.removesuffix(".capability_delta")


def apply_core_layer_band(
    module: nn.Module,
    spectrum: CoreSpectrum,
    *,
    band: CoreLayerBand,
    task: Literal["understanding", "generation"],
    match_full_operator_energy: bool,
) -> dict[str, float | str]:
    """Keep one fixed depth band and optionally match its $BQA$ energy to full state."""
    bands = core_layer_bands(spectrum)
    if band not in bands:
        msg = f"unknown core layer band: {band}"
        raise ValueError(msg)
    selected = set(bands[band])
    full_state = _spectrum_state(spectrum)
    band_state = {
        name: value.clone()
        if _site_from_state_key(name) in selected
        else torch.zeros_like(value)
        for name, value in full_state.items()
    }
    full_energy = functional_operator_energy(full_state, model=module, task=task)
    band_energy = functional_operator_energy(band_state, model=module, task=task)
    if full_energy <= 0.0 or band_energy <= 0.0:
        msg = "layer-band intervention requires nonzero full and band operator energy"
        raise ValueError(msg)
    scale = math.sqrt(full_energy / band_energy) if match_full_operator_energy else 1.0
    realized_state = {name: value.mul(scale) for name, value in band_state.items()}
    realized_energy = functional_operator_energy(realized_state, model=module, task=task)
    matching_error = (
        abs(realized_energy - full_energy) / full_energy
        if match_full_operator_energy
        else math.nan
    )
    layers = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, RoutedLowRankLinear)
    }
    expected = {
        "capability_delta" if name == "<root>" else f"{name}.capability_delta"
        for name in layers
    }
    if set(realized_state) != expected:
        msg = "core spectrum does not match the attached TMC layers"
        raise ValueError(msg)
    with torch.no_grad():
        for name, layer in layers.items():
            key = "capability_delta" if name == "<root>" else f"{name}.capability_delta"
            layer.capability_delta.copy_(realized_state[key].to(layer.capability_delta))
    return {
        "band": band,
        "scale": scale,
        "full_operator_energy": full_energy,
        "band_operator_energy_before_matching": band_energy,
        "realized_operator_energy": realized_energy,
        "operator_energy_matching_relative_error": matching_error,
    }


def apply_core_component_budget(
    module: nn.Module,
    spectrum: CoreSpectrum,
    *,
    budget: int,
) -> None:
    """Replace every capability delta by a global nested spectral subset."""
    layers = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, RoutedLowRankLinear)
    }
    if layers.keys() != spectrum.keys():
        msg = "core spectrum does not match the attached TMC layers"
        raise ValueError(msg)
    ordering = [
        (str(row["layer"]), int(row["component"]))
        for row in core_singular_concentration(spectrum)
    ]
    if not 0 <= budget <= len(ordering):
        msg = "core component budget lies outside the available rank"
        raise ValueError(msg)
    selected = set(ordering[:budget])
    with torch.no_grad():
        for name, layer in layers.items():
            item = spectrum[name]
            reconstructed = torch.zeros_like(item.left @ item.right)
            for index, singular_value in enumerate(item.singular_values):
                if (name, index) not in selected:
                    continue
                reconstructed.add_(
                    torch.outer(item.left[:, index], item.right[index, :]),
                    alpha=float(singular_value),
                )
            layer.capability_delta.copy_(reconstructed.to(layer.capability_delta))
