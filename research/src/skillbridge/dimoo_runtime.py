from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from skillbridge.objectives import calibration_loss
from skillbridge.route_data import RouteRecord
from skillbridge.routing import mask_ratio_to_bucket
from skillbridge.sidecar import SidecarTask, sidecars_disabled
from skillbridge.training import SidecarTrainBatch

MASK_TOKEN_ID = 126336


class DimooItemProcessorProtocol(Protocol):
    def process_item(
        self,
        record: dict[str, object],
        training_mode: bool = False,
    ) -> tuple[list[int], list[int]]: ...


@dataclass(frozen=True, slots=True)
class DimooComponents:
    model: nn.Module
    tokenizer: object
    processor: DimooItemProcessorProtocol


def infer_supervised_mask_ratio(
    input_ids: Tensor,
    labels: Tensor,
    *,
    mask_token_id: int = MASK_TOKEN_ID,
) -> float:
    supervised = labels.ne(-100)
    count = int(supervised.sum())
    if count == 0:
        msg = "batch contains no supervised tokens"
        raise ValueError(msg)
    masked = input_ids.eq(mask_token_id) & supervised
    return float(masked.sum()) / count


def _padded_tensor(rows: Sequence[Sequence[int]], *, padding: int) -> Tensor:
    if not rows:
        msg = "cannot pad an empty batch"
        raise ValueError(msg)
    width = max(map(len, rows))
    output = torch.full((len(rows), width), padding, dtype=torch.long)
    for index, row in enumerate(rows):
        output[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return output


def build_dimoo_batches(
    records: Sequence[RouteRecord],
    *,
    processor: DimooItemProcessorProtocol,
    task: SidecarTask,
    batch_size: int,
    timestep_buckets: int,
) -> list[SidecarTrainBatch]:
    if batch_size < 1:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    batches: list[SidecarTrainBatch] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        processed = [processor.process_item(dict(record), training_mode=True) for record in chunk]
        input_ids = _padded_tensor([item[0] for item in processed], padding=0)
        labels = _padded_tensor([item[1] for item in processed], padding=-100)
        ratio = infer_supervised_mask_ratio(input_ids, labels)
        batches.append(
            SidecarTrainBatch(
                task=task,
                timestep_bucket=mask_ratio_to_bucket(ratio, buckets=timestep_buckets),
                model_inputs={"input_ids": input_ids, "labels": labels},
            )
        )
    return batches


def load_dimoo_components(
    external_repo: Path,
    *,
    model_id: str,
    max_length: int = 1024,
    torch_dtype: str = "bfloat16",
    device_map: str | Mapping[str, str | int | torch.device] = "auto",
) -> DimooComponents:
    repo = external_repo.resolve()
    if not (repo / "model" / "__init__.py").is_file():
        msg = f"official Lumina-DiMOO repository is missing at {repo}"
        raise FileNotFoundError(msg)
    sys.path.insert(0, str(repo))
    transformers = importlib.import_module("transformers")
    model_module = importlib.import_module("model")
    train_module = importlib.import_module("train.train")
    dtype = getattr(torch, torch_dtype, None)
    if not isinstance(dtype, torch.dtype):
        msg = f"unknown torch dtype: {torch_dtype}"
        raise ValueError(msg)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model_class = model_module.LLaDAForMultiModalGeneration
    model = model_class.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
    processor_class = train_module.ItemProcessor
    processor = processor_class(tokenizer, max_length)
    return DimooComponents(model=model, tokenizer=tokenizer, processor=processor)


def _extract_logits(output: object) -> Tensor:
    if isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        msg = "model output does not contain logits"
        raise ValueError(msg)
    return logits


class DimooMaskedTokenObjective:
    def __init__(
        self,
        route_module: nn.Module,
        *,
        preservation_weight: float = 0.0,
        temperature: float = 1.0,
        adaptive_preservation_power: float = 0.0,
        adaptive_preservation_epsilon: float = 1e-4,
    ) -> None:
        if preservation_weight < 0:
            msg = "preservation_weight must be non-negative"
            raise ValueError(msg)
        self.route_module = route_module
        self.preservation_weight = preservation_weight
        self.temperature = temperature
        self.adaptive_preservation_power = adaptive_preservation_power
        self.adaptive_preservation_epsilon = adaptive_preservation_epsilon

    def __call__(self, model: nn.Module, batch: SidecarTrainBatch) -> Tensor:
        input_ids = cast("Tensor", batch.model_inputs["input_ids"])
        labels = cast("Tensor", batch.model_inputs["labels"])
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        frozen_logits: Tensor | None = None
        if self.preservation_weight:
            with sidecars_disabled(self.route_module), torch.no_grad():
                frozen_logits = _extract_logits(model(input_ids=input_ids, infer=True))
        adapted_logits = _extract_logits(model(input_ids=input_ids, infer=True))
        task_loss = torch.nn.functional.cross_entropy(
            adapted_logits.reshape(-1, adapted_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        if frozen_logits is None:
            return task_loss
        return calibration_loss(
            task_loss,
            adapted_logits=adapted_logits,
            frozen_logits=frozen_logits,
            keep_weight=self.preservation_weight,
            valid_mask=labels.ne(-100),
            temperature=self.temperature,
            adaptive_keep_power=self.adaptive_preservation_power,
            adaptive_keep_epsilon=self.adaptive_preservation_epsilon,
        )
