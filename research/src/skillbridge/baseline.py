from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal, cast

import torch
from torch import Tensor, nn

from skillbridge.sidecar import SidecarTask

BaselinePhase = Literal["private_calibration", "shared_calibration", "transfer"]
_TASKS: tuple[SidecarTask, ...] = ("understanding", "generation")


class _SharedBaselineRoute(nn.Module):
    _skillbridge_route_compatible = True

    def __init__(self, *, timestep_buckets: int) -> None:
        super().__init__()
        if timestep_buckets < 1:
            msg = "timestep_buckets must be positive"
            raise ValueError(msg)
        self.timestep_buckets = timestep_buckets
        self.active_task: SidecarTask | None = None
        self.active_timestep_bucket = 0
        self.shared_enabled = True
        self._scale = 1.0

    def set_route(
        self,
        *,
        task: SidecarTask | None,
        timestep_bucket: int = 0,
        shared_enabled: bool = True,
    ) -> None:
        if task is not None and task not in _TASKS:
            msg = f"unknown baseline task: {task}"
            raise ValueError(msg)
        if not 0 <= timestep_bucket < self.timestep_buckets:
            msg = f"timestep bucket must lie in [0, {self.timestep_buckets})"
            raise ValueError(msg)
        self.active_task = task
        self.active_timestep_bucket = timestep_bucket
        self.shared_enabled = shared_enabled

    def replace_scale(self, scale: float) -> float:
        if not torch.isfinite(torch.tensor(scale)):
            msg = "baseline scale must be finite"
            raise ValueError(msg)
        previous = self._scale
        self._scale = float(scale)
        return previous

    def route_active(self) -> bool:
        return self.active_task is not None and self.shared_enabled and self._scale != 0.0


class SharedBiasLinear(_SharedBaselineRoute):
    """A removable BitFit-style output-bias write on a frozen projection."""

    def __init__(self, base: nn.Linear, *, timestep_buckets: int = 1) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        self.base = base
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        self.bias_delta = nn.Parameter(
            torch.zeros(base.out_features, device=base.weight.device, dtype=dtype)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        return frozen + self._scale * self.bias_delta


class SharedDiffFitLinear(_SharedBaselineRoute):
    """A removable DiffFit-style output scale-and-bias write."""

    def __init__(self, base: nn.Linear, *, timestep_buckets: int = 1) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        self.base = base
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        kwargs = {"device": base.weight.device, "dtype": dtype}
        self.gain_delta = nn.Parameter(torch.zeros(base.out_features, **kwargs))
        self.bias_delta = nn.Parameter(torch.zeros(base.out_features, **kwargs))

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        update = frozen * self.gain_delta + self.bias_delta
        return frozen + self._scale * update


class SharedIA3InputLinear(_SharedBaselineRoute):
    """A removable IA3 input-channel scaling on a frozen projection."""

    def __init__(self, base: nn.Linear, *, timestep_buckets: int = 1) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        self.base = base
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        self.gain_delta = nn.Parameter(
            torch.zeros(base.in_features, device=base.weight.device, dtype=dtype)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        update = torch.nn.functional.linear(
            inputs * self.gain_delta,
            self.base.weight,
            bias=None,
        )
        return frozen + self._scale * update


class SharedFullRankDeltaLinear(_SharedBaselineRoute):
    """A removable full-rank update on the same shared projections as TMC."""

    def __init__(self, base: nn.Linear, *, timestep_buckets: int = 1) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        self.base = base
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        self.delta_weight = nn.Parameter(
            torch.zeros(
                base.out_features,
                base.in_features,
                device=base.weight.device,
                dtype=dtype,
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        delta = torch.nn.functional.linear(inputs, self.delta_weight)
        return frozen + self._scale * delta


class SharedDoRALinear(_SharedBaselineRoute):
    """A shared DoRA baseline with a removable source-written update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        if rank < 1:
            msg = "rank must be positive"
            raise ValueError(msg)
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        kwargs = {"device": base.weight.device, "dtype": dtype}
        self.lora_input = nn.Linear(base.in_features, rank, bias=False, **kwargs)
        self.lora_output = nn.Linear(rank, base.out_features, bias=False, **kwargs)
        initial_magnitude = torch.linalg.vector_norm(
            base.weight.detach().to(dtype=dtype), dim=1
        )
        self.magnitude = nn.Parameter(initial_magnitude)
        nn.init.kaiming_uniform_(self.lora_input.weight, a=5**0.5)
        nn.init.zeros_(self.lora_output.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        base_weight = self.base.weight.to(dtype=self.lora_input.weight.dtype)
        output_factor = self.lora_output.weight * self.scaling
        base_norm_sq = base_weight.square().sum(dim=1)
        base_by_input = base_weight @ self.lora_input.weight.transpose(0, 1)
        cross = 2.0 * (output_factor * base_by_input).sum(dim=1)
        input_gram = self.lora_input.weight @ self.lora_input.weight.transpose(0, 1)
        delta_norm_sq = (output_factor @ input_gram * output_factor).sum(dim=1)
        direction_norm = (base_norm_sq + cross + delta_norm_sq).clamp_min(1e-12).sqrt()
        ratio = self.magnitude / direction_norm
        shape = (1,) * (frozen.ndim - 1) + (ratio.numel(),)
        directed = frozen + self.lora_output(self.lora_input(inputs)) * self.scaling
        adapted = directed * ratio.view(shape)
        return frozen + self._scale * (adapted - frozen)


class SharedVeRALinear(_SharedBaselineRoute):
    """A shared VeRA baseline with deterministic frozen bases."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        seed: int,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        if rank < 1:
            msg = "rank must be positive"
            raise ValueError(msg)
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        generator = torch.Generator(device="cpu").manual_seed(seed)
        fixed_input = torch.randn(rank, base.in_features, generator=generator)
        fixed_output = torch.randn(base.out_features, rank, generator=generator)
        fixed_input = fixed_input.to(device=base.weight.device, dtype=dtype) / max(
            base.in_features, 1
        ) ** 0.5
        fixed_output = fixed_output.to(device=base.weight.device, dtype=dtype) / max(rank, 1) ** 0.5
        self.register_buffer("fixed_input", fixed_input, persistent=False)
        self.register_buffer("fixed_output", fixed_output, persistent=False)
        self.rank_scale = nn.Parameter(torch.ones(rank, device=base.weight.device, dtype=dtype))
        self.output_scale = nn.Parameter(
            torch.zeros(base.out_features, device=base.weight.device, dtype=dtype)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        compressed = torch.nn.functional.linear(inputs, self.fixed_input)
        expanded = torch.nn.functional.linear(compressed * self.rank_scale, self.fixed_output)
        return frozen + self._scale * self.scaling * expanded * self.output_scale


class SharedLoRAFALinear(_SharedBaselineRoute):
    """LoRA-FA with a frozen random down projection and trainable up projection."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        seed: int,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        if rank < 1:
            msg = "rank must be positive"
            raise ValueError(msg)
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        generator = torch.Generator(device="cpu").manual_seed(seed)
        fixed_input = torch.empty(rank, base.in_features)
        fixed_input.uniform_(-1.0 / max(base.in_features, 1) ** 0.5,
                             1.0 / max(base.in_features, 1) ** 0.5,
                             generator=generator)
        self.register_buffer(
            "fixed_input",
            fixed_input.to(device=base.weight.device, dtype=dtype),
            persistent=False,
        )
        self.up = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=dtype,
        )
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        compressed = torch.nn.functional.linear(inputs, self.fixed_input)
        return frozen + self._scale * self.scaling * self.up(compressed)


class SharedLoRAXSLinear(_SharedBaselineRoute):
    """LoRA-XS with frozen SVD bases and one trainable square core."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        seed: int,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        maximum_rank = min(base.in_features, base.out_features)
        if not 1 <= rank <= maximum_rank:
            message = f"rank must lie in [1, {maximum_rank}]"
            raise ValueError(message)
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        weight = base.weight.detach().to(device="cpu", dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            left, singular, right = torch.svd_lowrank(
                weight,
                q=rank,
                niter=10,
            )
        root = singular[:rank].clamp_min(0.0).sqrt()
        fixed_output = left[:, :rank] * root.unsqueeze(0)
        fixed_input = root.unsqueeze(1) * right[:, :rank].transpose(0, 1)
        self.register_buffer(
            "fixed_input",
            fixed_input.to(device=base.weight.device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "fixed_output",
            fixed_output.to(device=base.weight.device, dtype=dtype),
            persistent=False,
        )
        self.core = nn.Parameter(
            torch.zeros(rank, rank, device=base.weight.device, dtype=dtype)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        compressed = torch.nn.functional.linear(inputs, self.fixed_input)
        mixed = torch.nn.functional.linear(compressed, self.core)
        update = torch.nn.functional.linear(mixed, self.fixed_output)
        return frozen + self._scale * self.scaling * update


class SharedPiSSALinear(_SharedBaselineRoute):
    """A removable PiSSA-style update initialized from principal singular vectors."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        seed: int,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        maximum_rank = min(base.in_features, base.out_features)
        if not 1 <= rank <= maximum_rank:
            message = f"rank must lie in [1, {maximum_rank}]"
            raise ValueError(message)
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        weight = base.weight.detach().to(device="cpu", dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            left, singular, right = torch.svd_lowrank(weight, q=rank, niter=4)
        root = singular[:rank].clamp_min(0.0).sqrt()
        output_factor = left[:, :rank] * root.unsqueeze(0)
        input_factor = root.unsqueeze(1) * right[:, :rank].transpose(0, 1)
        self.output_factor = nn.Parameter(
            output_factor.to(device=base.weight.device, dtype=dtype)
        )
        self.input_factor = nn.Parameter(
            input_factor.to(device=base.weight.device, dtype=dtype)
        )
        self.register_buffer(
            "initial_delta",
            (self.output_factor.detach() @ self.input_factor.detach()).clone(),
            persistent=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        delta = self.output_factor @ self.input_factor - self.initial_delta
        update = torch.nn.functional.linear(inputs, delta)
        return frozen + self._scale * self.scaling * update


class SharedMRTLinear(_SharedBaselineRoute):
    """A removable projection-matched MRT-style editor.

    This adapts MRT's representation-editor equation to the same selected
    projection outputs used by the other controls; it does not reproduce MRT's
    original site-selection procedure. The frozen projection is edited only
    inside a learned orthonormal output subspace. Initializing the target
    coordinates to the access coordinates makes the editor an exact no-op
    before source-only training.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        seed: int,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        if not 1 <= rank <= base.out_features:
            msg = f"MRT-style rank must lie in [1, {base.out_features}]"
            raise ValueError(msg)
        self.base = base
        self.rank = rank
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.randn(base.out_features, rank, generator=generator)
        initial = torch.linalg.qr(initial, mode="reduced").Q.transpose(0, 1)
        initial = initial.to(device=base.weight.device, dtype=dtype)
        self.u_raw = nn.Parameter(initial.clone())
        self.target = nn.Parameter(initial.clone())
        self.bias = nn.Parameter(
            torch.zeros(rank, device=base.weight.device, dtype=dtype)
        )

    def orthonormal_rows(self) -> Tensor:
        compute = (
            self.u_raw.float()
            if self.u_raw.dtype in {torch.float16, torch.bfloat16}
            else self.u_raw
        )
        basis = torch.linalg.qr(compute.transpose(0, 1), mode="reduced").Q.transpose(0, 1)
        return basis.to(dtype=self.u_raw.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        basis = self.orthonormal_rows()
        coordinates = torch.nn.functional.linear(frozen, basis)
        target = torch.nn.functional.linear(frozen, self.target) + self.bias
        update = torch.nn.functional.linear(target - coordinates, basis.transpose(0, 1))
        return frozen + self._scale * update


class SharedLoKrLinear(_SharedBaselineRoute):
    """A removable LoKr-style update computed without a dense delta matrix."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        factor: int,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        if rank < 1 or factor < 1:
            msg = "LoKr rank and factor must be positive"
            raise ValueError(msg)
        if base.in_features % factor or base.out_features % factor:
            msg = "LoKr factor must divide both projection dimensions"
            raise ValueError(msg)
        self.base = base
        self.rank = rank
        self.factor = factor
        self.input_block = base.in_features // factor
        self.output_block = base.out_features // factor
        if rank > min(self.input_block, self.output_block):
            msg = "LoKr rank exceeds the factored projection dimensions"
            raise ValueError(msg)
        self.scaling = alpha / rank
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        kwargs = {"device": base.weight.device, "dtype": dtype}
        self.left_factor = nn.Parameter(torch.empty(factor, factor, **kwargs))
        self.down = nn.Linear(self.input_block, rank, bias=False, **kwargs)
        self.up = nn.Linear(rank, self.output_block, bias=False, **kwargs)
        nn.init.kaiming_uniform_(self.left_factor, a=5**0.5)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        blocks = inputs.reshape(*inputs.shape[:-1], self.factor, self.input_block)
        compressed = torch.einsum("...ki,ri->...kr", blocks, self.down.weight)
        mixed = torch.einsum("...kr,ok->...or", compressed, self.left_factor)
        expanded = torch.einsum("...or,jr->...oj", mixed, self.up.weight)
        update = expanded.reshape(*inputs.shape[:-1], self.base.out_features)
        return frozen + self._scale * self.scaling * update


class SharedLoHaLinear(_SharedBaselineRoute):
    """A removable LoHa update with a Hadamard product of two low-rank maps."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        if rank < 1:
            msg = "LoHa rank must be positive"
            raise ValueError(msg)
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(2 * rank, 1)
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        kwargs = {"device": base.weight.device, "dtype": dtype}
        self.down_one = nn.Linear(base.in_features, rank, bias=False, **kwargs)
        self.up_one = nn.Linear(rank, base.out_features, bias=False, **kwargs)
        self.down_two = nn.Linear(base.in_features, rank, bias=False, **kwargs)
        self.up_two = nn.Linear(rank, base.out_features, bias=False, **kwargs)
        nn.init.kaiming_uniform_(self.down_one.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.up_one.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_two.weight, a=5**0.5)
        nn.init.zeros_(self.up_two.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        first = self.up_one.weight @ self.down_one.weight
        second = self.up_two.weight @ self.down_two.weight
        update = torch.nn.functional.linear(inputs, first * second)
        return frozen + self._scale * self.scaling * update


class SharedFourierFTLinear(_SharedBaselineRoute):
    """A removable sparse-frequency full-matrix update following FourierFT."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        coefficients: int,
        alpha: float,
        seed: int,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        frequency_columns = base.in_features // 2 + 1
        available = base.out_features * frequency_columns
        if not 1 <= coefficients <= available:
            msg = f"FourierFT coefficients must lie in [1, {available}]"
            raise ValueError(msg)
        self.base = base
        self.base.requires_grad_(requires_grad=False)
        self.output_features = base.out_features
        self.input_features = base.in_features
        self.frequency_columns = frequency_columns
        self.scaling = alpha / coefficients**0.5
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(available, generator=generator)[:coefficients]
        self.register_buffer(
            "frequency_indices",
            indices.to(device=base.weight.device),
            persistent=False,
        )
        self.coefficients = nn.Parameter(
            torch.zeros(coefficients, 2, device=base.weight.device, dtype=torch.float32)
        )

    def delta_weight(self) -> Tensor:
        values = torch.complex(self.coefficients[:, 0], self.coefficients[:, 1])
        flat = torch.zeros(
            self.output_features * self.frequency_columns,
            device=values.device,
            dtype=values.dtype,
        ).index_copy(0, self.frequency_indices, values)
        spectrum = flat.reshape(self.output_features, self.frequency_columns)
        return torch.fft.irfft2(
            spectrum,
            s=(self.output_features, self.input_features),
            norm="ortho",
        )

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        delta = self.delta_weight().to(device=inputs.device, dtype=inputs.dtype)
        update = torch.nn.functional.linear(inputs, delta)
        return frozen + self._scale * self.scaling * update


class SharedOFTLinear(_SharedBaselineRoute):
    """A removable block-diagonal orthogonal transform of frozen features."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        block_size: int,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        if block_size < 2 or base.out_features % block_size:
            msg = "OFT block size must divide the output dimension"
            raise ValueError(msg)
        self.base = base
        self.block_size = block_size
        self.block_count = base.out_features // block_size
        self.base.requires_grad_(requires_grad=False)
        self.skew_raw = nn.Parameter(
            torch.zeros(
                self.block_count,
                block_size,
                block_size,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )

    def orthogonal_blocks(self) -> Tensor:
        skew = self.skew_raw - self.skew_raw.transpose(-1, -2)
        identity = torch.eye(
            self.block_size,
            device=skew.device,
            dtype=skew.dtype,
        ).expand(self.block_count, -1, -1)
        return torch.linalg.solve(identity + skew, identity - skew)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        blocks = frozen.float().reshape(
            *frozen.shape[:-1], self.block_count, self.block_size
        )
        rotated = torch.einsum("...ni,nji->...nj", blocks, self.orthogonal_blocks())
        adapted = rotated.reshape_as(frozen).to(dtype=frozen.dtype)
        return frozen + self._scale * (adapted - frozen)


class SharedBottleneckAdapterLinear(_SharedBaselineRoute):
    """A shared nonlinear bottleneck-adapter baseline."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__(timestep_buckets=timestep_buckets)
        if rank < 1:
            msg = "rank must be positive"
            raise ValueError(msg)
        self.base = base
        self.scaling = alpha / rank
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        kwargs = {"device": base.weight.device, "dtype": dtype}
        self.down = nn.Linear(base.in_features, rank, bias=False, **kwargs)
        self.up = nn.Linear(rank, base.out_features, bias=False, **kwargs)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        if not self.route_active():
            return frozen
        update = self.up(torch.nn.functional.gelu(self.down(inputs)))
        return frozen + self._scale * self.scaling * update


class StaticMultiplicativeLinear(nn.Module):
    """An (IA)^3/SSF-style, input-independent shared gain baseline."""

    _skillbridge_route_compatible = True

    def __init__(
        self,
        base: nn.Linear,
        *,
        gain_limit: float,
        timestep_buckets: int = 1,
    ) -> None:
        super().__init__()
        if not 0.0 < gain_limit < 1.0:
            message = "gain_limit must lie strictly between zero and one"
            raise ValueError(message)
        if timestep_buckets < 1:
            message = "timestep_buckets must be positive"
            raise ValueError(message)
        self.base = base
        self.gain_limit = gain_limit
        self.timestep_buckets = timestep_buckets
        self.base.requires_grad_(requires_grad=False)
        dtype = base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        self.gain_delta = nn.Parameter(
            torch.zeros(base.out_features, device=base.weight.device, dtype=dtype)
        )
        self.active_task: SidecarTask | None = None
        self.active_timestep_bucket = 0
        self.shared_enabled = True
        self._scale = 1.0

    def set_route(
        self,
        *,
        task: SidecarTask | None,
        timestep_bucket: int = 0,
        shared_enabled: bool = True,
    ) -> None:
        if task is not None and task not in _TASKS:
            message = f"unknown baseline task: {task}"
            raise ValueError(message)
        if not 0 <= timestep_bucket < self.timestep_buckets:
            message = f"timestep bucket must lie in [0, {self.timestep_buckets})"
            raise ValueError(message)
        self.active_task = task
        self.active_timestep_bucket = timestep_bucket
        self.shared_enabled = shared_enabled

    def replace_scale(self, scale: float) -> float:
        if not torch.isfinite(torch.tensor(scale)):
            message = "static gain scale must be finite"
            raise ValueError(message)
        previous = self._scale
        self._scale = float(scale)
        return previous

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        if self.active_task is None or not self.shared_enabled:
            return output
        gain = self._scale * self.gain_limit * torch.tanh(self.gain_delta)
        return output + output * gain


class PrivateSharedLowRankLinear(nn.Module):
    _skillbridge_route_compatible = True

    def __init__(
        self, base: nn.Linear, *, rank: int, alpha: float, timestep_buckets: int = 1
    ) -> None:
        super().__init__()
        if rank < 1:
            msg = "rank must be positive"
            raise ValueError(msg)
        if timestep_buckets < 1:
            msg = "timestep_buckets must be positive"
            raise ValueError(msg)
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.timestep_buckets = timestep_buckets
        self.base.requires_grad_(requires_grad=False)
        adapter_dtype = (
            base.weight.dtype if base.weight.is_floating_point() else torch.bfloat16
        )
        adapter_device = base.weight.device
        self.private_input = nn.ModuleDict(
            {
                task: nn.Linear(base.in_features, rank, bias=False).to(
                    device=adapter_device, dtype=adapter_dtype
                )
                for task in _TASKS
            }
        )
        self.private_output = nn.ModuleDict(
            {
                task: nn.Linear(rank, base.out_features, bias=False).to(
                    device=adapter_device, dtype=adapter_dtype
                )
                for task in _TASKS
            }
        )
        self.shared_input = nn.Linear(base.in_features, rank, bias=False).to(
            device=adapter_device, dtype=adapter_dtype
        )
        self.shared_output = nn.Linear(rank, base.out_features, bias=False).to(
            device=adapter_device, dtype=adapter_dtype
        )
        self.active_task: SidecarTask | None = None
        self.active_timestep_bucket = 0
        self.shared_enabled = True
        self._scale = 1.0
        self._residual_transform: Callable[[Tensor], Tensor] | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.shared_input.weight, a=5**0.5)
        nn.init.zeros_(self.shared_output.weight)
        for projection in self.private_input.values():
            nn.init.kaiming_uniform_(cast("nn.Linear", projection).weight, a=5**0.5)
        for projection in self.private_output.values():
            nn.init.zeros_(cast("nn.Linear", projection).weight)

    def set_route(
        self,
        *,
        task: SidecarTask | None,
        timestep_bucket: int = 0,
        shared_enabled: bool = True,
    ) -> None:
        if task is not None and task not in _TASKS:
            msg = f"unknown baseline task: {task}"
            raise ValueError(msg)
        if not 0 <= timestep_bucket < self.timestep_buckets:
            msg = f"timestep bucket must lie in [0, {self.timestep_buckets})"
            raise ValueError(msg)
        self.active_task = task
        self.active_timestep_bucket = timestep_bucket
        self.shared_enabled = shared_enabled

    def replace_scale(self, scale: float) -> float:
        if not torch.isfinite(torch.tensor(scale)):
            msg = "LoRA scale must be finite"
            raise ValueError(msg)
        previous = self._scale
        self._scale = float(scale)
        return previous

    def replace_residual_transform(
        self, transform: Callable[[Tensor], Tensor] | None
    ) -> Callable[[Tensor], Tensor] | None:
        previous = self._residual_transform
        self._residual_transform = transform
        return previous

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        if self.active_task is None:
            return output
        task = self.active_task
        private = self.private_output[task](self.private_input[task](inputs))
        shared = self.shared_output(self.shared_input(inputs)) if self.shared_enabled else 0.0
        residual = self._scale * (private + shared) * self.scaling
        if self._residual_transform is not None:
            residual = self._residual_transform(residual)
        return output + residual


def _parent_and_attribute(module: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _inject_shared_baseline(
    module: nn.Module,
    *,
    factory: Callable[[nn.Linear], nn.Module],
    target_suffixes: tuple[str, ...],
    exclude_names: tuple[str, ...],
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
        setattr(parent, attribute, factory(getattr(parent, attribute)))
    return targets


def inject_shared_dora(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    return _inject_shared_baseline(
        module,
        factory=lambda base: SharedDoRALinear(
            base, rank=rank, alpha=alpha, timestep_buckets=timestep_buckets
        ),
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_bias(
    module: nn.Module,
    *,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    return _inject_shared_baseline(
        module,
        factory=lambda base: SharedBiasLinear(
            base, timestep_buckets=timestep_buckets
        ),
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_difffit(
    module: nn.Module,
    *,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    return _inject_shared_baseline(
        module,
        factory=lambda base: SharedDiffFitLinear(
            base, timestep_buckets=timestep_buckets
        ),
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_ia3_input(
    module: nn.Module,
    *,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    return _inject_shared_baseline(
        module,
        factory=lambda base: SharedIA3InputLinear(
            base, timestep_buckets=timestep_buckets
        ),
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_full_rank_delta(
    module: nn.Module,
    *,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    return _inject_shared_baseline(
        module,
        factory=lambda base: SharedFullRankDeltaLinear(
            base, timestep_buckets=timestep_buckets
        ),
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_vera(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    seed: int = 0,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    site_index = 0

    def factory(base: nn.Linear) -> nn.Module:
        nonlocal site_index
        layer = SharedVeRALinear(
            base,
            rank=rank,
            alpha=alpha,
            seed=seed + site_index,
            timestep_buckets=timestep_buckets,
        )
        site_index += 1
        return layer

    return _inject_shared_baseline(
        module,
        factory=factory,
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_lora_fa(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    seed: int = 0,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    site_index = 0

    def factory(base: nn.Linear) -> nn.Module:
        nonlocal site_index
        layer = SharedLoRAFALinear(
            base,
            rank=rank,
            alpha=alpha,
            seed=seed + site_index,
            timestep_buckets=timestep_buckets,
        )
        site_index += 1
        return layer

    return _inject_shared_baseline(
        module,
        factory=factory,
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_lora_xs(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    seed: int = 0,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    site_index = 0

    def factory(base: nn.Linear) -> nn.Module:
        nonlocal site_index
        layer = SharedLoRAXSLinear(
            base,
            rank=rank,
            alpha=alpha,
            seed=seed + site_index,
            timestep_buckets=timestep_buckets,
        )
        site_index += 1
        return layer

    return _inject_shared_baseline(
        module,
        factory=factory,
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_pissa(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    seed: int = 0,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    site_index = 0

    def factory(base: nn.Linear) -> nn.Module:
        nonlocal site_index
        layer = SharedPiSSALinear(
            base,
            rank=rank,
            alpha=alpha,
            seed=seed + site_index,
            timestep_buckets=timestep_buckets,
        )
        site_index += 1
        return layer

    return _inject_shared_baseline(
        module,
        factory=factory,
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_mrt(
    module: nn.Module,
    *,
    rank: int,
    alpha: float | None = None,
    seed: int = 0,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    del alpha
    site_index = 0

    def factory(base: nn.Linear) -> nn.Module:
        nonlocal site_index
        layer = SharedMRTLinear(
            base,
            rank=min(rank, base.out_features),
            seed=seed + site_index,
            timestep_buckets=timestep_buckets,
        )
        site_index += 1
        return layer

    return _inject_shared_baseline(
        module,
        factory=factory,
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def _largest_common_factor(base: nn.Linear, *, maximum: int) -> int:
    for factor in range(min(maximum, base.in_features, base.out_features), 1, -1):
        if base.in_features % factor == 0 and base.out_features % factor == 0:
            return factor
    return 1


def inject_shared_lokr(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    factor: int | None = None,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    return _inject_shared_baseline(
        module,
        factory=lambda base: SharedLoKrLinear(
            base,
            rank=min(
                rank,
                base.in_features // (factor or _largest_common_factor(base, maximum=8)),
                base.out_features // (factor or _largest_common_factor(base, maximum=8)),
            ),
            alpha=alpha,
            factor=factor or _largest_common_factor(base, maximum=8),
            timestep_buckets=timestep_buckets,
        ),
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_loha(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    factor_rank = max(1, rank // 2)
    return _inject_shared_baseline(
        module,
        factory=lambda base: SharedLoHaLinear(
            base,
            rank=min(factor_rank, base.in_features, base.out_features),
            alpha=alpha,
            timestep_buckets=timestep_buckets,
        ),
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_fourierft(
    module: nn.Module,
    *,
    coefficients: int,
    alpha: float | None = None,
    seed: int = 0,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    site_index = 0

    def factory(base: nn.Linear) -> nn.Module:
        nonlocal site_index
        layer = SharedFourierFTLinear(
            base,
            coefficients=min(
                coefficients,
                base.out_features * (base.in_features // 2 + 1),
            ),
            alpha=float(coefficients if alpha is None else alpha),
            seed=seed + site_index,
            timestep_buckets=timestep_buckets,
        )
        site_index += 1
        return layer

    return _inject_shared_baseline(
        module,
        factory=factory,
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_shared_oft(
    module: nn.Module,
    *,
    block_size: int | None = None,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    def factory(base: nn.Linear) -> nn.Module:
        resolved = block_size or _largest_common_factor(base, maximum=4)
        if resolved < 2:
            msg = "OFT requires an output dimension with a factor of at least two"
            raise ValueError(msg)
        return SharedOFTLinear(
            base,
            block_size=resolved,
            timestep_buckets=timestep_buckets,
        )

    return _inject_shared_baseline(
        module,
        factory=factory,
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_bottleneck_adapter(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    timestep_buckets: int = 1,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_names: tuple[str, ...] = (),
) -> list[str]:
    return _inject_shared_baseline(
        module,
        factory=lambda base: SharedBottleneckAdapterLinear(
            base, rank=rank, alpha=alpha, timestep_buckets=timestep_buckets
        ),
        target_suffixes=target_suffixes,
        exclude_names=exclude_names,
    )


def inject_private_shared_lora(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    timestep_buckets: int = 1,
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
            PrivateSharedLowRankLinear(
                base,
                rank=rank,
                alpha=alpha,
                timestep_buckets=timestep_buckets,
            ),
        )
    return targets


def inject_static_multiplicative_gain(
    module: nn.Module,
    *,
    gain_limit: float,
    timestep_buckets: int = 1,
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
            StaticMultiplicativeLinear(
                base,
                gain_limit=gain_limit,
                timestep_buckets=timestep_buckets,
            ),
        )
    return targets


def iter_private_shared_lora(module: nn.Module) -> Iterator[PrivateSharedLowRankLinear]:
    if isinstance(module, PrivateSharedLowRankLinear):
        yield module
        return
    for child in module.modules():
        if isinstance(child, PrivateSharedLowRankLinear):
            yield child


def iter_static_multiplicative_gain(
    module: nn.Module,
) -> Iterator[StaticMultiplicativeLinear]:
    if isinstance(module, StaticMultiplicativeLinear):
        yield module
        return
    for child in module.modules():
        if isinstance(child, StaticMultiplicativeLinear):
            yield child


def iter_shared_peft_baselines(module: nn.Module) -> Iterator[_SharedBaselineRoute]:
    route_types = (
        SharedBiasLinear,
        SharedDiffFitLinear,
        SharedIA3InputLinear,
        SharedFullRankDeltaLinear,
        SharedDoRALinear,
        SharedVeRALinear,
        SharedLoRAFALinear,
        SharedLoRAXSLinear,
        SharedPiSSALinear,
        SharedMRTLinear,
        SharedLoKrLinear,
        SharedLoHaLinear,
        SharedFourierFTLinear,
        SharedOFTLinear,
        SharedBottleneckAdapterLinear,
    )
    if isinstance(module, route_types):
        yield module
        return
    for child in module.modules():
        if isinstance(child, route_types):
            yield child


@contextmanager
def private_shared_lora_scale(module: nn.Module, *, scale: float) -> Iterator[None]:
    layers = list(iter_private_shared_lora(module))
    if not layers:
        msg = "module contains no private/shared LoRA layers"
        raise ValueError(msg)
    previous = [layer.replace_scale(scale) for layer in layers]
    try:
        yield
    finally:
        for layer, value in zip(layers, previous, strict=True):
            layer.replace_scale(value)


@contextmanager
def capture_private_shared_lora_sites(
    module: nn.Module,
    *,
    capture_device: torch.device | str | None = None,
) -> Iterator[dict[tuple[str, int], list[Tensor]]]:
    layers = [
        (name, layer)
        for name, layer in module.named_modules()
        if isinstance(layer, PrivateSharedLowRankLinear)
    ]
    if not layers:
        raise ValueError("module contains no private/shared LoRA layers")
    captured: dict[tuple[str, int], list[Tensor]] = {}
    previous = [layer.replace_residual_transform(None) for _, layer in layers]

    def recorder(
        name: str, layer: PrivateSharedLowRankLinear
    ) -> Callable[[Tensor], Tensor]:
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
def patch_private_shared_lora_sites(
    module: nn.Module,
    patches: dict[tuple[str, int], list[Tensor]],
    *,
    strict: bool = False,
) -> Iterator[None]:
    layers_by_name = {
        name: layer
        for name, layer in module.named_modules()
        if isinstance(layer, PrivateSharedLowRankLinear)
    }
    if not patches:
        raise ValueError("LoRA residual patches must be non-empty")
    for name, bucket in patches:
        if name not in layers_by_name:
            raise ValueError(f"unknown LoRA layer in residual patch: {name}")
        if not 0 <= bucket < layers_by_name[name].timestep_buckets:
            raise ValueError(f"invalid LoRA timestep bucket: {(name, bucket)}")
    selected_names = {name for name, _ in patches}
    selected = [(name, layers_by_name[name]) for name in selected_names]
    previous = [layer.replace_residual_transform(None) for _, layer in selected]
    call_indices = dict.fromkeys(patches, 0)

    def patcher(
        name: str, layer: PrivateSharedLowRankLinear
    ) -> Callable[[Tensor], Tensor]:
        def patch(residual: Tensor) -> Tensor:
            site = (name, layer.active_timestep_bucket)
            if site not in patches:
                return residual
            index = call_indices[site]
            if index >= len(patches[site]):
                raise ValueError(f"missing LoRA residual patch for site {site} call {index}")
            call_indices[site] += 1
            replacement = patches[site][index]
            if replacement.shape != residual.shape:
                raise ValueError(f"LoRA residual patch shape mismatch for site {site}")
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
            raise ValueError(f"unused LoRA residual patches after forward pass: {unused}")


@contextmanager
def static_gain_scale(module: nn.Module, *, scale: float) -> Iterator[None]:
    layers = list(iter_static_multiplicative_gain(module))
    if not layers:
        message = "module contains no static multiplicative gain layers"
        raise ValueError(message)
    previous = [layer.replace_scale(scale) for layer in layers]
    try:
        yield
    finally:
        for layer, value in zip(layers, previous, strict=True):
            layer.replace_scale(value)


@contextmanager
def baseline_scale(module: nn.Module, *, scale: float) -> Iterator[None]:
    layers = list(iter_shared_peft_baselines(module))
    if not layers:
        msg = "module contains no shared PEFT baseline layers"
        raise ValueError(msg)
    previous = [layer.replace_scale(scale) for layer in layers]
    try:
        yield
    finally:
        for layer, value in zip(layers, previous, strict=True):
            layer.replace_scale(value)


def configure_baseline_training(  # noqa: C901, PLR0912
    module: nn.Module, *, phase: BaselinePhase
) -> None:
    module.requires_grad_(requires_grad=False)
    layers = list(iter_private_shared_lora(module))
    static_layers = list(iter_static_multiplicative_gain(module))
    shared_peft_layers = list(iter_shared_peft_baselines(module))
    if not layers and not static_layers and not shared_peft_layers:
        msg = "module contains no supported baseline layers"
        raise ValueError(msg)
    for layer in layers:
        if phase == "private_calibration":
            for branch in (layer.private_input, layer.private_output):
                branch.requires_grad_(requires_grad=True)
        elif phase in {"shared_calibration", "transfer"}:
            layer.shared_input.requires_grad_(requires_grad=True)
            layer.shared_output.requires_grad_(requires_grad=True)
        else:
            msg = f"unknown baseline training phase: {phase}"
            raise ValueError(msg)
    if static_layers:
        if phase not in {"shared_calibration", "transfer"}:
            message = "static multiplicative gain has no private calibration phase"
            raise ValueError(message)
        for layer in static_layers:
            layer.gain_delta.requires_grad_(requires_grad=True)
    if shared_peft_layers:
        if phase not in {"shared_calibration", "transfer"}:
            message = "shared PEFT baselines have no private calibration phase"
            raise ValueError(message)
        for layer in shared_peft_layers:
            for name, parameter in layer.named_parameters():
                if not name.startswith("base."):
                    parameter.requires_grad_(requires_grad=True)
