from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from skillbridge.baseline import PrivateSharedLowRankLinear


@dataclass(frozen=True, slots=True)
class WitnessLayerSpectrum:
    left: Tensor
    singular_values: Tensor
    right: Tensor
    scaling: float


WitnessSpectrum = dict[str, WitnessLayerSpectrum]


def _factorized_svd(output: Tensor, input_: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the SVD of `output @ input_` through only rank-sized cores."""
    left_basis, left_core = torch.linalg.qr(output.float(), mode="reduced")
    right_basis, right_core = torch.linalg.qr(input_.float().T, mode="reduced")
    middle = left_core @ right_core.T
    middle_left, singular_values, middle_right = torch.linalg.svd(
        middle, full_matrices=False
    )
    left = left_basis @ middle_left
    right = middle_right @ right_basis.T
    return left, singular_values, right


def capture_witness_spectrum(module: nn.Module) -> WitnessSpectrum:
    """Capture a compact SVD of every trained shared-LoRA witness update."""
    output: WitnessSpectrum = {}
    for name, child in module.named_modules():
        if not isinstance(child, PrivateSharedLowRankLinear):
            continue
        left, values, right = _factorized_svd(
            child.shared_output.weight.detach().cpu(),
            child.shared_input.weight.detach().cpu(),
        )
        output[name] = WitnessLayerSpectrum(left, values, right, child.scaling)
    if not output:
        message = "module contains no shared-LoRA transfer witness"
        raise ValueError(message)
    return output


def witness_singular_concentration(
    spectrum: WitnessSpectrum,
) -> list[dict[str, float | int | str]]:
    if not spectrum:
        message = "witness spectrum must be non-empty"
        raise ValueError(message)
    components = sorted(
        (
            (name, index, float(value) * layer.scaling)
            for name, layer in spectrum.items()
            for index, value in enumerate(layer.singular_values)
        ),
        key=lambda item: (-abs(item[2]), item[0], item[1]),
    )
    total_energy = sum(value * value for _, _, value in components)
    if total_energy <= 0:
        message = "transfer witness has zero shared-update energy"
        raise ValueError(message)
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


def _attached_witness_layers(module: nn.Module) -> dict[str, PrivateSharedLowRankLinear]:
    layers = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, PrivateSharedLowRankLinear)
    }
    if not layers:
        message = "module contains no shared-LoRA transfer witness"
        raise ValueError(message)
    return layers


def _witness_component_ordering(spectrum: WitnessSpectrum) -> list[tuple[str, int]]:
    return [
        (str(row["layer"]), int(row["component"]))
        for row in witness_singular_concentration(spectrum)
    ]


def _validate_witness_budget(spectrum: WitnessSpectrum, budget: int) -> list[tuple[str, int]]:
    ordering = _witness_component_ordering(spectrum)
    if not 0 <= budget <= len(ordering):
        message = "witness component budget lies outside the available rank"
        raise ValueError(message)
    return ordering


def _apply_witness_components(
    module: nn.Module,
    spectrum: WitnessSpectrum,
    *,
    selected: set[tuple[str, int]],
) -> None:
    layers = _attached_witness_layers(module)
    if layers.keys() != spectrum.keys():
        message = "witness spectrum does not match the attached shared-LoRA layers"
        raise ValueError(message)
    with torch.no_grad():
        for name, layer in layers.items():
            item = spectrum[name]
            output = torch.zeros_like(layer.shared_output.weight)
            input_ = torch.zeros_like(layer.shared_input.weight)
            slot = 0
            for index, singular_value in enumerate(item.singular_values):
                if (name, index) not in selected:
                    continue
                root = torch.sqrt(singular_value.clamp_min(0.0))
                output[:, slot].copy_((item.left[:, index] * root).to(output))
                input_[slot, :].copy_((item.right[index, :] * root).to(input_))
                slot += 1
            layer.shared_output.weight.copy_(output)
            layer.shared_input.weight.copy_(input_)


def apply_witness_component_budget(
    module: nn.Module,
    spectrum: WitnessSpectrum,
    *,
    budget: int,
) -> None:
    """Replace a witness by its globally largest singular components."""
    ordering = _validate_witness_budget(spectrum, budget)
    _apply_witness_components(module, spectrum, selected=set(ordering[:budget]))


def apply_witness_component_complement(
    module: nn.Module,
    spectrum: WitnessSpectrum,
    *,
    budget: int,
) -> None:
    """Remove the globally leading witness components and keep their complement."""
    ordering = _validate_witness_budget(spectrum, budget)
    _apply_witness_components(module, spectrum, selected=set(ordering[budget:]))


def apply_witness_random_control(
    module: nn.Module,
    spectrum: WitnessSpectrum,
    *,
    budget: int,
    seed: int,
) -> None:
    """Randomize selected singular directions while preserving their singular values."""
    ordering = _validate_witness_budget(spectrum, budget)
    selected = set(ordering[:budget])
    layers = _attached_witness_layers(module)
    if layers.keys() != spectrum.keys():
        message = "witness spectrum does not match the attached shared-LoRA layers"
        raise ValueError(message)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, layer in layers.items():
            item = spectrum[name]
            output = torch.zeros_like(layer.shared_output.weight)
            input_ = torch.zeros_like(layer.shared_input.weight)
            indices = [
                index
                for index in range(item.singular_values.numel())
                if (name, index) in selected
            ]
            if indices:
                count = len(indices)
                left = torch.linalg.qr(
                    torch.randn(output.shape[0], count, generator=generator),
                    mode="reduced",
                ).Q
                right = torch.linalg.qr(
                    torch.randn(input_.shape[1], count, generator=generator),
                    mode="reduced",
                ).Q.transpose(0, 1)
            else:
                left = None
                right = None
            for slot, index in enumerate(indices):
                if left is None or right is None:
                    message = "selected witness control must have random directions"
                    raise RuntimeError(message)
                singular_value = item.singular_values[index]
                root = torch.sqrt(singular_value.clamp_min(0.0))
                output[:, slot].copy_((left[:, slot] * root).to(output))
                input_[slot, :].copy_((right[slot, :] * root).to(input_))
            layer.shared_output.weight.copy_(output)
            layer.shared_input.weight.copy_(input_)
