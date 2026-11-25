from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SharedUpdate = Literal["multiplicative", "additive"]
BackboneKind = Literal["dimoo", "janusflow", "uniworld_v1"]

DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_out",
    "ff_proj",
    "up_proj",
    "ff_out",
)


@dataclass(frozen=True, slots=True)
class SidecarExperimentConfig:
    model_id: str
    model_revision: str | None = None
    backbone: BackboneKind = "dimoo"
    external_repo: Path = Path("external/Lumina-DiMOO")
    output_dir: Path = Path("runs/sidecar")
    rank: int = 8
    alpha: float = 16.0
    gain_limit: float = 0.5
    shared_update: SharedUpdate = "multiplicative"
    timestep_buckets: int = 8
    private_calibration_steps: int = 500
    route_calibration_steps: int = 1_000
    transfer_steps: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 1
    preservation_weight: float = 0.1
    adaptive_preservation_power: float = 0.0
    adaptive_preservation_epsilon: float = 1e-4
    max_length: int = 1024
    image_size: int = 384
    vae_id: str = "stabilityai/sdxl-vae"
    vae_revision: str | None = None
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES
    excluded_target_names: tuple[str, ...] = ("ff_out",)
    capabilities: tuple[str, ...] = ("counting", "spatial", "text")
    seeds: tuple[int, ...] = (17, 29, 43)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        if not self.model_id.strip():
            msg = "model_id must be non-empty"
            raise ValueError(msg)
        if self.backbone not in ("dimoo", "janusflow", "uniworld_v1"):
            msg = f"unknown backbone: {self.backbone}"
            raise ValueError(msg)
        if self.backbone == "janusflow":
            for name, revision in (
                ("model_revision", self.model_revision),
                ("vae_revision", self.vae_revision),
            ):
                if revision is None or len(revision) != 40:
                    msg = f"{name} must be a full 40-character commit for JanusFlow"
                    raise ValueError(msg)
        if self.rank < 1:
            msg = "rank must be positive"
            raise ValueError(msg)
        if self.alpha <= 0:
            msg = "alpha must be positive"
            raise ValueError(msg)
        if not 0.0 < self.gain_limit < 1.0:
            msg = "gain_limit must lie strictly between zero and one"
            raise ValueError(msg)
        if self.shared_update not in ("multiplicative", "additive"):
            msg = f"unknown shared_update: {self.shared_update}"
            raise ValueError(msg)
        if self.backbone == "uniworld_v1" and self.timestep_buckets != 1:
            msg = "UniWorld-V1 uses a cached generation conditioner and requires one bucket"
            raise ValueError(msg)
        if self.backbone != "uniworld_v1" and self.timestep_buckets < 2:
            msg = "timestep_buckets must be at least two"
            raise ValueError(msg)
        if (
            self.private_calibration_steps < 1
            or self.route_calibration_steps < 1
            or self.transfer_steps < 1
        ):
            msg = "training steps must be positive"
            raise ValueError(msg)
        if self.learning_rate <= 0 or self.weight_decay < 0:
            msg = "learning rate must be positive and weight decay must be non-negative"
            raise ValueError(msg)
        if self.batch_size < 1 or self.max_length < 1 or self.image_size < 8:
            msg = "batch size, max length, and image size must be positive"
            raise ValueError(msg)
        if not self.vae_id.strip():
            msg = "vae_id must be non-empty"
            raise ValueError(msg)
        if self.preservation_weight < 0:
            msg = "preservation weight must be non-negative"
            raise ValueError(msg)
        if self.adaptive_preservation_power < 0:
            msg = "adaptive preservation power must be non-negative"
            raise ValueError(msg)
        if self.adaptive_preservation_epsilon <= 0:
            msg = "adaptive preservation epsilon must be positive"
            raise ValueError(msg)
        if not self.target_suffixes:
            msg = "target_suffixes must be non-empty"
            raise ValueError(msg)
        if not self.capabilities or not self.seeds:
            msg = "capabilities and seeds must be non-empty"
            raise ValueError(msg)


def load_sidecar_config(path: Path) -> SidecarExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "configuration must be a JSON object"
        raise ValueError(msg)
    values = dict(payload)
    for key in ("external_repo", "output_dir"):
        if key in values:
            value = Path(values[key])
            values[key] = value if value.is_absolute() else path.parent / value
    for key in ("target_suffixes", "excluded_target_names", "capabilities", "seeds"):
        if key in values:
            values[key] = tuple(values[key])
    return SidecarExperimentConfig(**values)
