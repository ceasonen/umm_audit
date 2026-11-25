from __future__ import annotations

from skillbridge.artifact_policy import forbidden_hashing as hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from skillbridge.baseline import BaselinePhase, configure_baseline_training
from skillbridge.sidecar import (
    SidecarTask,
    TrainingPhase,
    configure_sidecar_training,
    sidecar_route,
)

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
TrainingStage = TrainingPhase | BaselinePhase
LossFunction = Callable[[nn.Module, "SidecarTrainBatch"], Tensor]
CheckpointCallback = Callable[[int, nn.Module], None]
OptimizerName = Literal["adamw", "sgd"]


@dataclass(frozen=True, slots=True)
class SidecarTrainBatch:
    task: SidecarTask
    timestep_bucket: int
    model_inputs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TrainingStep:
    step: int
    phase: TrainingStage
    task: SidecarTask
    timestep_bucket: int
    loss: float


def _extract_loss(output: object) -> Tensor:
    if isinstance(output, Tensor):
        loss = output
    elif isinstance(output, Mapping):
        candidate = output.get("loss")
        loss = candidate if isinstance(candidate, Tensor) else None
    else:
        loss = getattr(output, "loss", None)
    if not isinstance(loss, Tensor) or loss.numel() != 1:
        msg = "model output must contain one scalar loss tensor"
        raise ValueError(msg)
    return loss


def train_sidecar_stage(
    model: nn.Module,
    route_module: nn.Module,
    batches: Sequence[SidecarTrainBatch],
    *,
    phase: TrainingPhase,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    loss_function: LossFunction | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    optimizer_name: OptimizerName = "adamw",
    optimization_observer: Callable[[str, int, nn.Module], None] | None = None,
) -> list[TrainingStep]:
    training_task = batches[0].task if phase == "private" and batches else None
    configure_sidecar_training(route_module, phase=phase, task=training_task)
    return _train_configured_stage(
        model,
        route_module,
        batches,
        phase=phase,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_function=loss_function,
        checkpoint_callback=checkpoint_callback,
        optimizer_name=optimizer_name,
        optimization_observer=optimization_observer,
    )


def train_baseline_stage(
    model: nn.Module,
    route_module: nn.Module,
    batches: Sequence[SidecarTrainBatch],
    *,
    phase: BaselinePhase,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    loss_function: LossFunction | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    optimizer_name: OptimizerName = "adamw",
    optimization_observer: Callable[[str, int, nn.Module], None] | None = None,
) -> list[TrainingStep]:
    configure_baseline_training(route_module, phase=phase)
    return _train_configured_stage(
        model,
        route_module,
        batches,
        phase=phase,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_function=loss_function,
        checkpoint_callback=checkpoint_callback,
        optimizer_name=optimizer_name,
        optimization_observer=optimization_observer,
    )


def _train_configured_stage(
    model: nn.Module,
    route_module: nn.Module,
    batches: Sequence[SidecarTrainBatch],
    *,
    phase: TrainingStage,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    loss_function: LossFunction | None,
    checkpoint_callback: CheckpointCallback | None,
    optimizer_name: OptimizerName,
    optimization_observer: Callable[[str, int, nn.Module], None] | None = None,
) -> list[TrainingStep]:
    if not batches:
        msg = "training batches must be non-empty"
        raise ValueError(msg)
    if steps < 1 or learning_rate <= 0 or weight_decay < 0:
        msg = "steps and learning rate must be positive; weight decay must be non-negative"
        raise ValueError(msg)
    parameters = [parameter for parameter in route_module.parameters() if parameter.requires_grad]
    if not parameters:
        msg = f"training phase {phase} selected no parameters"
        raise ValueError(msg)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.0,
            weight_decay=weight_decay,
        )
    else:
        message = f"unknown optimizer: {optimizer_name}"
        raise ValueError(message)
    history: list[TrainingStep] = []
    # The pretrained backbone is part of the fixed experimental state. Evaluation
    # mode prevents dropout and mutable normalization buffers from changing even
    # though gradients still flow into the selected sidecar parameters.
    model.eval()
    for step in range(steps):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        if optimization_observer is not None:
            optimization_observer("before_step", step + 1, route_module)
        shared_enabled = phase != "private_calibration"
        with sidecar_route(
            route_module,
            task=batch.task,
            timestep_bucket=batch.timestep_bucket,
            shared_enabled=shared_enabled,
        ):
            loss = (
                loss_function(model, batch)
                if loss_function is not None
                else _extract_loss(model(**dict(batch.model_inputs)))
            )
            if not bool(torch.isfinite(loss)):
                msg = f"non-finite loss at step {step}"
                raise FloatingPointError(msg)
            if optimization_observer is not None:
                optimization_observer("after_forward", step + 1, route_module)
            # Non-reentrant activation checkpointing replays forward functions
            # during backward. Keep the task and timestep route active until
            # recomputation has finished, otherwise it sees a different graph.
            loss.backward()
        optimizer.step()
        if optimization_observer is not None:
            optimization_observer("after_step", step + 1, route_module)
        if checkpoint_callback is not None:
            checkpoint_callback(step + 1, route_module)
        history.append(
            TrainingStep(
                step=step,
                phase=phase,
                task=batch.task,
                timestep_bucket=batch.timestep_bucket,
                loss=float(loss.detach().cpu()),
            )
        )
    return history


def _canonical_json(payload: Mapping[str, JsonValue]) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_artifacts(
    output_dir: Path,
    *,
    run_id: str,
    config: Mapping[str, JsonValue],
    history: Sequence[TrainingStep],
    metrics: Mapping[str, float],
    extra_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, JsonValue]:
    if not run_id.strip():
        msg = "run_id must be non-empty"
        raise ValueError(msg)
    if any(not math.isfinite(value) for value in metrics.values()):
        msg = "metrics must be finite"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        msg = f"refusing to overwrite completed run: {output_dir}"
        raise FileExistsError(msg)
    config_path = output_dir / "config.json"
    history_path = output_dir / "history.jsonl"
    metrics_path = output_dir / "metrics.json"
    config_text = json.dumps(config, allow_nan=False, indent=2, sort_keys=True) + "\n"
    config_path.write_text(config_text, encoding="utf-8")
    history_path.write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in history),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_sha256 = hashlib.sha256(_canonical_json(config).encode()).hexdigest()
    artifacts: dict[str, JsonValue] = {
        "config": {"path": "config.json", "sha256": _sha256(config_path)},
        "history": {"path": "history.jsonl", "sha256": _sha256(history_path)},
        "metrics": {"path": "metrics.json", "sha256": _sha256(metrics_path)},
    }
    for name, path in (extra_artifacts or {}).items():
        if not path.is_file():
            msg = f"extra artifact is not a file: {path}"
            raise FileNotFoundError(msg)
        try:
            relative = path.resolve().relative_to(output_dir.resolve())
        except ValueError as error:
            msg = f"extra artifact must live inside the run directory: {path}"
            raise ValueError(msg) from error
        artifacts[name] = {"path": str(relative), "sha256": _sha256(path)}
    manifest: dict[str, JsonValue] = {
        "format_version": 1,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
