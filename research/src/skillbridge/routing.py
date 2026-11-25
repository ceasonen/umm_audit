from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from skillbridge.sidecar import RoutedLowRankLinear, SidecarTask

if TYPE_CHECKING:
    from torch import nn


def diffusion_step_to_bucket(step: int, *, total_steps: int, buckets: int) -> int:
    if total_steps < 1 or buckets < 1:
        msg = "total_steps and buckets must be positive"
        raise ValueError(msg)
    if not 0 <= step < total_steps:
        msg = f"step must lie in [0, {total_steps})"
        raise ValueError(msg)
    return min(buckets - 1, step * buckets // total_steps)


def mask_ratio_to_bucket(mask_ratio: float, *, buckets: int) -> int:
    if buckets < 1:
        msg = "buckets must be positive"
        raise ValueError(msg)
    if not 0.0 <= mask_ratio <= 1.0:
        msg = "mask_ratio must lie in [0, 1]"
        raise ValueError(msg)
    return min(buckets - 1, int(mask_ratio * buckets))


@contextmanager
def forward_step_routing(
    model: nn.Module,
    route_module: nn.Module,
    *,
    task: SidecarTask,
    total_steps: int,
    calls_per_step: int = 1,
    shared_enabled: bool = True,
) -> Iterator[None]:
    if calls_per_step < 1:
        msg = "calls_per_step must be positive"
        raise ValueError(msg)
    sidecars = [
        cast("RoutedLowRankLinear", child)
        for child in route_module.modules()
        if isinstance(child, RoutedLowRankLinear)
        or bool(getattr(child, "_skillbridge_route_compatible", False))
    ]
    if not sidecars:
        msg = "route_module contains no routed layers"
        raise ValueError(msg)
    previous = [
        (layer.active_task, layer.active_timestep_bucket, layer.shared_enabled)
        for layer in sidecars
    ]
    call_index = 0

    def select_route(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        nonlocal call_index
        step = min(total_steps - 1, call_index // calls_per_step)
        bucket = diffusion_step_to_bucket(
            step,
            total_steps=total_steps,
            buckets=sidecars[0].timestep_buckets,
        )
        for layer in sidecars:
            layer.set_route(
                task=task,
                timestep_bucket=min(bucket, layer.timestep_buckets - 1),
                shared_enabled=shared_enabled,
            )
        call_index += 1

    handle = model.register_forward_pre_hook(select_route)
    try:
        yield
    finally:
        handle.remove()
        for layer, state in zip(sidecars, previous, strict=True):
            layer.set_route(
                task=state[0],
                timestep_bucket=state[1],
                shared_enabled=state[2],
            )
