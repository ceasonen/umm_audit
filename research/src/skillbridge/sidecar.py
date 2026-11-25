from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SidecarTask = Literal["understanding", "generation"]
SharedUpdate = Literal["multiplicative", "additive"]
SidecarSite = tuple[str, int]
RouteComponentBank = dict[SidecarSite, list[tuple[Tensor, Tensor]]]
TrainingPhase = Literal[
    "private_calibration",
    "route_calibration",
    "joint_calibration",
    "transfer",
    "private",
]

_TASKS: tuple[SidecarTask, ...] = ("understanding", "generation")


class RoutedLowRankLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        timestep_buckets: int = 1,
        gain_limit: float = 0.5,
        shared_update: SharedUpdate = "multiplicative",
    ) -> None:
        super().__init__()
        if rank < 1:
            msg = "rank must be positive"
            raise ValueError(msg)
        if timestep_buckets < 1:
            msg = "timestep_buckets must be positive"
            raise ValueError(msg)
        if not 0.0 < gain_limit < 1.0:
            msg = "gain_limit must lie strictly between zero and one"
            raise ValueError(msg)
        if shared_update not in ("multiplicative", "additive"):
            msg = f"unknown shared_update: {shared_update}"
            raise ValueError(msg)

        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.timestep_buckets = timestep_buckets
        self.gain_limit = gain_limit
        self.shared_update = shared_update
        self.base.requires_grad_(requires_grad=False)
        adapter_dtype = (
            base.weight.dtype if base.weight.dtype.is_floating_point else torch.bfloat16
        )
        adapter_kwargs = {"device": base.weight.device, "dtype": adapter_dtype}

        self.shared_input = nn.ModuleDict(
            {
                task: nn.Linear(base.in_features, rank, bias=False, **adapter_kwargs)
                for task in _TASKS
            }
        )
        self.shared_core = nn.Linear(rank, rank, bias=False, **adapter_kwargs)
        self.capability_delta = nn.Parameter(torch.zeros(rank, rank, **adapter_kwargs))
        self.shared_output = nn.ModuleDict(
            {
                task: nn.Linear(rank, base.out_features, bias=False, **adapter_kwargs)
                for task in _TASKS
            }
        )
        self.private_input = nn.ModuleDict(
            {
                task: nn.Linear(base.in_features, rank, bias=False, **adapter_kwargs)
                for task in _TASKS
            }
        )
        self.private_output = nn.ModuleDict(
            {
                task: nn.Linear(rank, base.out_features, bias=False, **adapter_kwargs)
                for task in _TASKS
            }
        )
        self.route_logits = nn.Parameter(
            torch.zeros(
                timestep_buckets,
                device=base.weight.device,
                dtype=adapter_dtype,
            )
        )
        self.understanding_route_logit = nn.Parameter(
            torch.zeros((), device=base.weight.device, dtype=adapter_dtype)
        )
        self.active_task: SidecarTask | None = None
        self.active_timestep_bucket = 0
        self.shared_enabled = True
        self._shared_modulation_transform: Callable[[Tensor], Tensor] | None = None
        self._capability_scales: dict[int, float] = {}
        self._route_component_observer: Callable[[Tensor, Tensor], None] | None = None
        self.reset_parameters()

    def replace_modulation_transform(
        self,
        transform: Callable[[Tensor], Tensor] | None,
    ) -> Callable[[Tensor], Tensor] | None:
        previous = self._shared_modulation_transform
        self._shared_modulation_transform = transform
        return previous

    def replace_residual_transform(
        self,
        transform: Callable[[Tensor], Tensor] | None,
    ) -> Callable[[Tensor], Tensor] | None:
        """Backward-compatible alias for modulation interventions."""
        return self.replace_modulation_transform(transform)

    def replace_capability_scales(self, values: dict[int, float]) -> dict[int, float]:
        previous = self._capability_scales
        self._capability_scales = dict(values)
        return previous

    def set_capability_scale(self, bucket: int, scale: float) -> None:
        self._capability_scales[bucket] = scale

    def capability_scale(self, bucket: int) -> float:
        return self._capability_scales.get(bucket, 1.0)

    def replace_route_component_observer(
        self,
        observer: Callable[[Tensor, Tensor], None] | None,
    ) -> Callable[[Tensor, Tensor], None] | None:
        previous = self._route_component_observer
        self._route_component_observer = observer
        return previous

    def reset_parameters(self) -> None:
        for projection in (*self.shared_input.values(), *self.private_input.values()):
            linear = cast("nn.Linear", projection)
            nn.init.kaiming_uniform_(linear.weight, a=5**0.5)
        nn.init.eye_(self.shared_core.weight)
        nn.init.zeros_(self.capability_delta)
        for projection in (*self.shared_output.values(), *self.private_output.values()):
            linear = cast("nn.Linear", projection)
            nn.init.zeros_(linear.weight)
        nn.init.zeros_(self.route_logits)
        nn.init.zeros_(self.understanding_route_logit)

    def set_route(
        self,
        *,
        task: SidecarTask | None,
        timestep_bucket: int = 0,
        shared_enabled: bool = True,
    ) -> None:
        if task is not None and task not in _TASKS:
            msg = f"unknown sidecar task: {task}"
            raise ValueError(msg)
        if not 0 <= timestep_bucket < self.timestep_buckets:
            msg = f"timestep bucket must lie in [0, {self.timestep_buckets})"
            raise ValueError(msg)
        self.active_task = task
        self.active_timestep_bucket = timestep_bucket
        self.shared_enabled = shared_enabled

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        task = self.active_task
        if task is None:
            return output

        private = self.private_output[task](self.private_input[task](inputs))
        if not self.shared_enabled:
            return output + private * self.scaling

        low_rank_inputs = self.shared_input[task](inputs)
        gate_logit = (
            self.understanding_route_logit
            if task == "understanding"
            else self.route_logits[self.active_timestep_bucket]
        )
        gate = torch.sigmoid(gate_logit)

        def realize(core: Tensor) -> Tensor:
            shared = self.shared_output[task](F.linear(low_rank_inputs, core))
            routed = shared * gate * self.scaling
            return (
                self.gain_limit * torch.tanh(routed)
                if self.shared_update == "multiplicative"
                else routed
            )

        capability_scale = self._capability_scales.get(self.active_timestep_bucket)
        if capability_scale is None:
            modulation = realize(self.shared_core.weight + self.capability_delta)
        else:
            calibrated_modulation = realize(self.shared_core.weight)
            adapted_modulation = realize(self.shared_core.weight + self.capability_delta)
            modulation = calibrated_modulation + capability_scale * (
                adapted_modulation - calibrated_modulation
            )
        if self._shared_modulation_transform is not None:
            modulation = self._shared_modulation_transform(modulation)
        private_contribution = private * self.scaling
        if self.shared_update == "multiplicative":
            shared_contribution = output * modulation
        else:
            shared_contribution = modulation
        if self._route_component_observer is not None:
            self._route_component_observer(private_contribution, shared_contribution)
        return output + private_contribution + shared_contribution


def _parent_and_attribute(module: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_sidecars(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    timestep_buckets: int = 1,
    gain_limit: float = 0.5,
    shared_update: SharedUpdate = "multiplicative",
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    module.requires_grad_(requires_grad=False)
    targets = [
        name
        for name, child in module.named_modules()
        if name
        and isinstance(child, nn.Linear)
        and name.endswith(target_suffixes)
        and name not in exclude_names
    ]
    for name in targets:
        parent, attribute = _parent_and_attribute(module, name)
        base = getattr(parent, attribute)
        setattr(
            parent,
            attribute,
            RoutedLowRankLinear(
                base,
                rank=rank,
                alpha=alpha,
                timestep_buckets=timestep_buckets,
                gain_limit=gain_limit,
                shared_update=shared_update,
            ),
        )
    return targets


def svd_access(weight: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Unit-norm truncated SVD bases; no singular-value gain or base subtraction.

    This is an access-geometry control, not the original additive LoRA-XS.
    Persist the resulting maps in adapter.pt for exact inference restoration.
    """
    if weight.ndim != 2 or not 0 < rank <= min(weight.shape) or not torch.isfinite(weight).all():
        raise ValueError("finite matrix and valid positive SVD rank required")
    left, _, right = torch.linalg.svd(weight.float(), full_matrices=False)
    write, read = left[:, :rank].clone(), right[:rank].clone()
    pivots = write.abs().argmax(dim=0)
    signs = torch.sign(write[pivots, torch.arange(rank, device=weight.device)])
    write.mul_(signs.unsqueeze(0))
    read.mul_(signs.unsqueeze(1))
    return read, write


def initialize_svd_backbone_access(module: nn.Module) -> None:
    """Use SVD axes while preserving the existing bounded multiplicative operator."""
    initialize_tied_backbone_access(module, basis="svd")


def initialize_tied_backbone_access(module: nn.Module, *, basis: str = "energy") -> None:
    """Derive fixed task-shared read/write bases from high-energy backbone slices.

    The shared core and capability write remain zero, so insertion stays an
    exact no-op.  A source-task gradient can nevertheless reach the capability
    write through the nonzero frozen bases.
    """
    if basis not in ("energy", "svd"):
        raise ValueError("unknown fixed access basis")
    layers = list(iter_sidecars(module))
    if not layers:
        msg = "module contains no routed low-rank sidecars"
        raise ValueError(msg)
    with torch.no_grad():
        for layer in layers:
            stored_weight = layer.base.weight.detach()
            expected_shape = (layer.base.out_features, layer.base.in_features)
            if tuple(stored_weight.shape) == expected_shape:
                weight = stored_weight.float()
            else:
                quant_state = getattr(layer.base.weight, "quant_state", None)
                if quant_state is None:
                    msg = (
                        "packed backbone weight lacks quantization state: "
                        f"stored={tuple(stored_weight.shape)}, expected={expected_shape}"
                    )
                    raise ValueError(msg)
                from bitsandbytes.functional import dequantize_4bit

                weight = dequantize_4bit(
                    stored_weight, quant_state=quant_state
                ).float()
                if tuple(weight.shape) != expected_shape:
                    msg = (
                        "dequantized backbone weight shape mismatch: "
                        f"observed={tuple(weight.shape)}, expected={expected_shape}"
                    )
                    raise ValueError(msg)
            if layer.rank > min(weight.shape):
                msg = "sidecar rank exceeds backbone dimensions"
                raise ValueError(msg)
            if basis == "svd":
                read, write = svd_access(weight, layer.rank)
            else:
                row_indices = torch.topk(weight.square().sum(dim=1), layer.rank).indices
                column_indices = torch.topk(weight.square().sum(dim=0), layer.rank).indices
                read = F.normalize(weight[row_indices, :], dim=1)
                write = F.normalize(weight[:, column_indices], dim=0)
            for task in _TASKS:
                layer.shared_input[task].weight.copy_(
                    read.to(layer.shared_input[task].weight)
                )
                layer.shared_output[task].weight.copy_(
                    write.to(layer.shared_output[task].weight)
                )
            layer.shared_core.weight.zero_()
            layer.capability_delta.zero_()


def initialize_random_shared_access(module: nn.Module, *, seed: int) -> None:
    """Initialize task-shared orthonormal random bases for a matched control."""
    layers = list(iter_sidecars(module))
    if not layers:
        msg = "module contains no routed low-rank sidecars"
        raise ValueError(msg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for layer in layers:
            read_basis, _ = torch.linalg.qr(
                torch.randn(
                    layer.base.in_features,
                    layer.rank,
                    generator=generator,
                    dtype=torch.float32,
                ),
                mode="reduced",
            )
            write_basis, _ = torch.linalg.qr(
                torch.randn(
                    layer.base.out_features,
                    layer.rank,
                    generator=generator,
                    dtype=torch.float32,
                ),
                mode="reduced",
            )
            for task in _TASKS:
                layer.shared_input[task].weight.copy_(
                    read_basis.T.to(layer.shared_input[task].weight)
                )
                layer.shared_output[task].weight.copy_(
                    write_basis.to(layer.shared_output[task].weight)
                )
            layer.shared_core.weight.zero_()
            layer.capability_delta.zero_()


def iter_sidecars(module: nn.Module) -> Iterator[RoutedLowRankLinear]:
    if isinstance(module, RoutedLowRankLinear):
        yield module
        return
    for child in module.modules():
        if isinstance(child, RoutedLowRankLinear):
            yield child


def named_sidecars(module: nn.Module) -> Iterator[tuple[str, RoutedLowRankLinear]]:
    if isinstance(module, RoutedLowRankLinear):
        yield "<root>", module
        return
    for name, child in module.named_modules():
        if isinstance(child, RoutedLowRankLinear):
            yield name, child


@contextmanager
def sidecar_route(
    module: nn.Module,
    *,
    task: SidecarTask,
    timestep_bucket: int = 0,
    shared_enabled: bool = True,
) -> Iterator[None]:
    sidecars = [
        cast("RoutedLowRankLinear", child)
        for child in module.modules()
        if isinstance(child, RoutedLowRankLinear)
        or bool(getattr(child, "_skillbridge_route_compatible", False))
    ]
    previous = [
        (layer.active_task, layer.active_timestep_bucket, layer.shared_enabled)
        for layer in sidecars
    ]
    for layer in sidecars:
        layer.set_route(
            task=task,
            timestep_bucket=timestep_bucket,
            shared_enabled=shared_enabled,
        )
    try:
        yield
    finally:
        for layer, state in zip(sidecars, previous, strict=True):
            layer.set_route(
                task=state[0],
                timestep_bucket=state[1],
                shared_enabled=state[2],
            )


@contextmanager
def sidecars_disabled(module: nn.Module) -> Iterator[None]:
    sidecars = [
        cast("RoutedLowRankLinear", child)
        for child in module.modules()
        if isinstance(child, RoutedLowRankLinear)
        or bool(getattr(child, "_skillbridge_route_compatible", False))
    ]
    previous = [
        (layer.active_task, layer.active_timestep_bucket, layer.shared_enabled)
        for layer in sidecars
    ]
    for layer in sidecars:
        layer.set_route(task=None, timestep_bucket=0, shared_enabled=False)
    try:
        yield
    finally:
        for layer, state in zip(sidecars, previous, strict=True):
            layer.set_route(
                task=state[0],
                timestep_bucket=state[1],
                shared_enabled=state[2],
            )


@contextmanager
def functional_capability_scale(
    module: nn.Module,
    *,
    scale: float,
    sites: tuple[SidecarSite, ...] | None = None,
) -> Iterator[None]:
    """Scale the candidate write while each layer keeps its current input."""
    if not torch.isfinite(torch.tensor(scale)):
        msg = "capability scale must be finite"
        raise ValueError(msg)
    layers = dict(named_sidecars(module))
    if not layers:
        msg = "functional intervention requires at least one sidecar layer"
        raise ValueError(msg)
    selected = (
        tuple(
            (name, bucket)
            for name, layer in layers.items()
            for bucket in range(layer.timestep_buckets)
        )
        if sites is None
        else sites
    )
    if not selected:
        msg = "functional intervention requires at least one site"
        raise ValueError(msg)
    for name, bucket in selected:
        if name not in layers:
            msg = f"unknown sidecar layer in functional intervention: {name}"
            raise ValueError(msg)
        if not 0 <= bucket < layers[name].timestep_buckets:
            msg = f"invalid timestep bucket in functional intervention: {(name, bucket)}"
            raise ValueError(msg)
    previous = {name: layer.replace_capability_scales({}) for name, layer in layers.items()}
    for name, bucket in selected:
        layers[name].set_capability_scale(bucket, scale)
    try:
        yield
    finally:
        for name, values in previous.items():
            layers[name].replace_capability_scales(values)


@contextmanager
def functional_capability_site_mask(
    module: nn.Module,
    *,
    active_sites: tuple[SidecarSite, ...],
) -> Iterator[None]:
    """Realize the candidate delta only at a locked subset of current-input sites."""
    layers = dict(named_sidecars(module))
    all_sites = tuple(
        (name, bucket) for name, layer in layers.items() for bucket in range(layer.timestep_buckets)
    )
    if not active_sites or not set(active_sites) <= set(all_sites):
        msg = "functional site mask requires a non-empty subset of valid sites"
        raise ValueError(msg)
    previous = {name: layer.replace_capability_scales({}) for name, layer in layers.items()}
    for name, bucket in all_sites:
        layers[name].set_capability_scale(bucket, 1.0 if (name, bucket) in active_sites else 0.0)
    try:
        yield
    finally:
        for name, values in previous.items():
            layers[name].replace_capability_scales(values)


@contextmanager
def capture_route_components(module: nn.Module) -> Iterator[RouteComponentBank]:
    """Capture the private and shared additive effects for cancellation audits."""
    layers = list(named_sidecars(module))
    if not layers:
        msg = "component capture requires at least one sidecar layer"
        raise ValueError(msg)
    captured: RouteComponentBank = {}
    previous = [layer.replace_route_component_observer(None) for _, layer in layers]

    def observer(name: str, layer: RoutedLowRankLinear) -> Callable[[Tensor, Tensor], None]:
        def record(private: Tensor, shared: Tensor) -> None:
            site = (name, layer.active_timestep_bucket)
            captured.setdefault(site, []).append(
                (private.detach().clone(), shared.detach().clone())
            )

        return record

    for name, layer in layers:
        layer.replace_route_component_observer(observer(name, layer))
    try:
        yield captured
    finally:
        for (_, layer), value in zip(layers, previous, strict=True):
            layer.replace_route_component_observer(value)


@contextmanager
def capture_shared_residuals(
    module: nn.Module,
) -> Iterator[dict[str, list[Tensor]]]:
    layers = list(named_sidecars(module))
    captured: dict[str, list[Tensor]] = {name: [] for name, _ in layers}
    previous = [layer.replace_residual_transform(None) for _, layer in layers]

    def recorder(name: str) -> Callable[[Tensor], Tensor]:
        def record(residual: Tensor) -> Tensor:
            captured[name].append(residual.detach().clone())
            return residual

        return record

    for name, layer in layers:
        layer.replace_residual_transform(recorder(name))
    try:
        yield captured
    finally:
        for (_, layer), transform in zip(layers, previous, strict=True):
            layer.replace_residual_transform(transform)


@contextmanager
def patch_shared_residuals(
    module: nn.Module,
    patches: dict[str, list[Tensor]],
    *,
    strict: bool = False,
) -> Iterator[None]:
    layers = list(named_sidecars(module))
    available = {name for name, _ in layers}
    if not patches or not set(patches).issubset(available):
        msg = "patch names must identify a non-empty subset of sidecar layers"
        raise ValueError(msg)
    selected = [(name, layer) for name, layer in layers if name in patches]
    previous = [layer.replace_residual_transform(None) for _, layer in selected]
    call_indices = dict.fromkeys(patches, 0)

    def patcher(name: str) -> Callable[[Tensor], Tensor]:
        def patch(residual: Tensor) -> Tensor:
            index = call_indices[name]
            if index >= len(patches[name]):
                msg = f"missing residual patch for {name} call {index}"
                raise ValueError(msg)
            call_indices[name] += 1
            replacement = patches[name][index]
            if replacement.shape != residual.shape:
                msg = f"residual patch shape mismatch for {name}"
                raise ValueError(msg)
            return replacement.to(device=residual.device, dtype=residual.dtype)

        return patch

    for name, layer in selected:
        layer.replace_residual_transform(patcher(name))
    body_completed = False
    try:
        yield
        body_completed = True
    finally:
        for (_, layer), transform in zip(selected, previous, strict=True):
            layer.replace_residual_transform(transform)
        unused = {
            name: len(patches[name]) - call_indices[name]
            for name in patches
            if call_indices[name] != len(patches[name])
        }
        if body_completed and strict and unused:
            msg = f"unused residual patches after forward pass: {unused}"
            raise ValueError(msg)


@contextmanager
def capture_shared_residual_sites(
    module: nn.Module,
    *,
    capture_device: torch.device | str | None = None,
) -> Iterator[dict[SidecarSite, list[Tensor]]]:
    layers = list(named_sidecars(module))
    captured: dict[SidecarSite, list[Tensor]] = {}
    previous = [layer.replace_residual_transform(None) for _, layer in layers]

    def recorder(name: str, layer: RoutedLowRankLinear) -> Callable[[Tensor], Tensor]:
        def record(residual: Tensor) -> Tensor:
            site = (name, layer.active_timestep_bucket)
            value = residual.detach()
            if capture_device is not None:
                value = value.to(capture_device)
            captured.setdefault(site, []).append(value.clone())
            return residual

        return record

    for name, layer in layers:
        layer.replace_residual_transform(recorder(name, layer))
    try:
        yield captured
    finally:
        for (_, layer), transform in zip(layers, previous, strict=True):
            layer.replace_residual_transform(transform)


@contextmanager
def patch_shared_residual_sites(
    module: nn.Module,
    patches: dict[SidecarSite, list[Tensor]],
    *,
    strict: bool = False,
) -> Iterator[None]:
    layers_by_name = dict(named_sidecars(module))
    _validate_site_patches(layers_by_name, patches)
    selected_names = {name for name, _ in patches}
    selected = [(name, layers_by_name[name]) for name in selected_names]
    previous = [layer.replace_residual_transform(None) for _, layer in selected]
    call_indices = dict.fromkeys(patches, 0)

    def patcher(name: str, layer: RoutedLowRankLinear) -> Callable[[Tensor], Tensor]:
        def patch(residual: Tensor) -> Tensor:
            site = (name, layer.active_timestep_bucket)
            if site not in patches:
                return residual
            index = call_indices[site]
            if index >= len(patches[site]):
                msg = f"missing residual patch for site {site} call {index}"
                raise ValueError(msg)
            call_indices[site] += 1
            replacement = patches[site][index]
            if replacement.shape != residual.shape:
                msg = f"residual patch shape mismatch for site {site}"
                raise ValueError(msg)
            return replacement.to(device=residual.device, dtype=residual.dtype)

        return patch

    for name, layer in selected:
        layer.replace_residual_transform(patcher(name, layer))
    body_completed = False
    try:
        yield
        body_completed = True
    finally:
        for (_, layer), transform in zip(selected, previous, strict=True):
            layer.replace_residual_transform(transform)
        unused = {
            site: len(patches[site]) - call_indices[site]
            for site in patches
            if call_indices[site] != len(patches[site])
        }
        if body_completed and strict and unused:
            msg = f"unused residual patches after forward pass: {unused}"
            raise ValueError(msg)


def _validate_site_patches(
    layers_by_name: dict[str, RoutedLowRankLinear],
    patches: dict[SidecarSite, list[Tensor]],
) -> None:
    if not patches:
        msg = "site patches must be non-empty"
        raise ValueError(msg)
    for name, bucket in patches:
        if name not in layers_by_name:
            msg = f"unknown sidecar layer in site patch: {name}"
            raise ValueError(msg)
        if not 0 <= bucket < layers_by_name[name].timestep_buckets:
            msg = f"invalid timestep bucket for site patch: {(name, bucket)}"
            raise ValueError(msg)


# Canonical names for multiplicative interventions. The residual names remain
# import-compatible with checkpoints and scripts produced before the route
# changed from an additive residual to a bounded gain.
capture_shared_modulations = capture_shared_residuals
patch_shared_modulations = patch_shared_residuals
capture_shared_modulation_sites = capture_shared_residual_sites
patch_shared_modulation_sites = patch_shared_residual_sites


def configure_sidecar_training(  # noqa: C901
    module: nn.Module,
    *,
    phase: TrainingPhase,
    task: SidecarTask | None = None,
) -> None:
    module.requires_grad_(requires_grad=False)
    sidecars = list(iter_sidecars(module))
    if not sidecars:
        msg = "module contains no routed low-rank sidecars"
        raise ValueError(msg)
    if phase == "private" and task is None:
        msg = "private training requires a task"
        raise ValueError(msg)

    for layer in sidecars:
        if phase == "private_calibration":
            for branch in (layer.private_input, layer.private_output):
                branch.requires_grad_(requires_grad=True)
        elif phase == "route_calibration":
            for branch in (layer.shared_input, layer.shared_output, layer.shared_core):
                branch.requires_grad_(requires_grad=True)
            layer.route_logits.requires_grad = True
            layer.understanding_route_logit.requires_grad = True
        elif phase == "joint_calibration":
            for branch in (
                layer.private_input,
                layer.private_output,
                layer.shared_input,
                layer.shared_output,
                layer.shared_core,
            ):
                branch.requires_grad_(requires_grad=True)
            layer.route_logits.requires_grad = True
            layer.understanding_route_logit.requires_grad = True
        elif phase == "transfer":
            layer.capability_delta.requires_grad = True
        elif phase == "private" and task is not None:
            layer.private_input[task].requires_grad_(requires_grad=True)
            layer.private_output[task].requires_grad_(requires_grad=True)
        else:
            msg = f"unknown sidecar training phase: {phase}"
            raise ValueError(msg)
