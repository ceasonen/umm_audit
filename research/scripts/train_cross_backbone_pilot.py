#!/usr/bin/env python3
"""Train one source-only shared LoRA update on a complete frozen UMM backbone."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from skillbridge.training_overlap import completed_training, mark_training_complete, training_lock, make_progress_reporter

from skillbridge.adapter_checkpoint import (
    save_adapter_checkpoint,
    save_capability_checkpoint,
)
from skillbridge.baseline import (
    BaselinePhase,
    inject_bottleneck_adapter,
    inject_private_shared_lora,
    inject_shared_bias,
    inject_shared_difffit,
    inject_shared_dora,
    inject_shared_fourierft,
    inject_shared_full_rank_delta,
    inject_shared_ia3_input,
    inject_shared_loha,
    inject_shared_lokr,
    inject_shared_lora_fa,
    inject_shared_lora_xs,
    inject_shared_mrt,
    inject_shared_oft,
    inject_shared_pissa,
    inject_shared_vera,
    inject_static_multiplicative_gain,
)
from skillbridge.dimoo_data import (
    build_dimoo_encoder,
    load_route_records,
    pretokenize_route_records,
)
from skillbridge.dimoo_runtime import (
    DimooMaskedTokenObjective,
    build_dimoo_batches,
    load_dimoo_components,
)
from skillbridge.janusflow_runtime import (
    JanusFlowObjective,
    build_janusflow_batches,
    load_janusflow_components,
)
from skillbridge.sidecar import (
    initialize_random_shared_access,
    initialize_tied_backbone_access,
    inject_sidecars,
)
from skillbridge.sidecar_backbone import (
    resolve_dimoo_transformer,
    resolve_janusflow_transformer,
    resolve_uniworld_v1_transformer,
)
from skillbridge.training import SidecarTrainBatch, train_baseline_stage, train_sidecar_stage
from skillbridge.witness import capture_witness_spectrum, witness_singular_concentration

BACKBONES = ("janusflow", "dimoo", "blip3o", "uniworld_v1")
TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
SHARED_LORA_METHODS = ("shared_lora", "shared_lora_witness")


def training_order_id(records: list[dict[str, object]]) -> str:
    """Allocate a fresh cache namespace; ordered inputs are recorded explicitly.

    This is a random run label, not a content identity or a verification digest.
    Fresh namespaces deliberately prevent stale token-cache reuse across labels.
    """
    if not records:
        raise ValueError("training records must not be empty")
    return str(uuid.uuid4())


def balance_numeric_answers(
    records: list[dict[str, object]],
    *,
    examples: int,
    values: tuple[int, ...],
    seed: int,
) -> list[dict[str, object]]:
    """Draw a deterministic equal-size source curriculum over numeric answers."""
    if examples < 1 or not values or len(set(values)) != len(values):
        message = "balanced sampling requires examples and unique answer values"
        raise ValueError(message)
    if examples < len(values):
        message = "examples must cover every requested answer stratum"
        raise ValueError(message)
    groups = {
        value: [
            row
            for row in records
            if str(row.get("answer_text", "")).strip() == str(value)
        ]
        for value in values
    }
    missing = [value for value, rows in groups.items() if not rows]
    if missing:
        message = f"missing requested answer strata: {missing}"
        raise ValueError(message)
    base, remainder = divmod(examples, len(values))
    selected: list[dict[str, object]] = []
    for offset, value in enumerate(values):
        required = base + int(offset < remainder)
        if required > len(groups[value]):
            message = (
                "cannot balance without repeating examples: "
                f"answer {value} has {len(groups[value])}, requires {required}"
            )
            raise ValueError(message)
        generator = random.Random(seed + 1009 * (offset + 1))
        candidates = list(groups[value])
        generator.shuffle(candidates)
        selected.extend(candidates[:required])
    random.Random(seed).shuffle(selected)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument(
        "--method",
        choices=(
            "tied_core",
            "svd_core",
            "additive_tied_core",
            "random_basis_tmc",
            "shared_lora",
            "shared_lora_witness",
            "source_private_lora",
            "static_multiplicative_gain",
            "shared_dora",
            "shared_vera",
            "bottleneck_adapter",
            "shared_full_rank_delta",
            "shared_lora_xs",
            "shared_pissa",
            "shared_lora_fa",
            "shared_mrt",
            "shared_bias",
            "shared_difffit",
            "shared_ia3_input",
            "shared_oft",
            "shared_lokr",
            "shared_loha",
            "shared_fourierft",
        ),
        default="tied_core",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-records", type=Path)
    parser.add_argument(
        "--capability",
        choices=("counting", "spatial", "text", "world_knowledge"),
        default="counting",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--timestep-buckets", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gain-limit", type=float, default=0.25)
    parser.add_argument("--target-suffixes", nargs="+", default=TARGET_SUFFIXES)
    parser.add_argument("--balanced-numeric-answers", type=int, nargs="+")
    parser.add_argument("--shuffle-source-labels", action="store_true")
    parser.add_argument("--preserve-source-order", action="store_true")
    parser.add_argument("--optimization-audit", action="store_true",
                        help="record source-only gradient/update/activation diagnostics every 100 steps")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def native_loss(model: nn.Module, batch: SidecarTrainBatch) -> Tensor:
    inputs = {
        name: value.to("cuda") if isinstance(value, Tensor) else value
        for name, value in batch.model_inputs.items()
    }
    output = model(**inputs)
    loss = output[0] if isinstance(output, tuple) and output else getattr(output, "loss", None)
    if not isinstance(loss, Tensor):
        raise ValueError("native model output lacks a scalar loss")
    return loss


def attach_tied_route(
    route_module: nn.Module,
    *,
    rank: int,
    timestep_buckets: int,
    method: str,
    gain_limit: float = 0.25,
    target_suffixes: tuple[str, ...] = TARGET_SUFFIXES,
) -> list[str]:
    targets = inject_sidecars(
        route_module,
        rank=rank,
        alpha=float(rank),
        timestep_buckets=timestep_buckets,
        gain_limit=gain_limit,
        shared_update=("additive" if method == "additive_tied_core" else "multiplicative"),
        target_suffixes=target_suffixes,
    )
    if not targets:
        raise RuntimeError("no compatible transformer projections found")
    if method == "random_basis_tmc":
        initialize_random_shared_access(route_module, seed=0)
    else:
        initialize_tied_backbone_access(route_module, basis="svd" if method == "svd_core" else "energy")
    return targets


def attach_route(
    route_module: nn.Module,
    *,
    rank: int,
    timestep_buckets: int,
    method: str,
    gain_limit: float = 0.25,
    target_suffixes: tuple[str, ...] = TARGET_SUFFIXES,
) -> list[str]:
    if method in {"tied_core", "svd_core", "additive_tied_core", "random_basis_tmc"}:
        return attach_tied_route(
            route_module,
            rank=rank,
            timestep_buckets=timestep_buckets,
            method=method,
            gain_limit=gain_limit,
            target_suffixes=target_suffixes,
        )
    if method in {*SHARED_LORA_METHODS, "source_private_lora"}:
        targets = inject_private_shared_lora(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
        if not targets:
            raise RuntimeError("no compatible transformer projections found")
        return targets
    if method == "static_multiplicative_gain":
        targets = inject_static_multiplicative_gain(
            route_module,
            gain_limit=gain_limit,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
        if not targets:
            raise RuntimeError("no compatible transformer projections found")
        return targets
    public_peft = {
        "shared_dora": lambda: inject_shared_dora(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_vera": lambda: inject_shared_vera(
            route_module,
            rank=rank,
            alpha=float(rank),
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "bottleneck_adapter": lambda: inject_bottleneck_adapter(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_full_rank_delta": lambda: inject_shared_full_rank_delta(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_lora_xs": lambda: inject_shared_lora_xs(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_pissa": lambda: inject_shared_pissa(
            route_module,
            rank=rank,
            alpha=float(rank),
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_lora_fa": lambda: inject_shared_lora_fa(
            route_module,
            rank=rank,
            alpha=float(rank),
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_mrt": lambda: inject_shared_mrt(
            route_module,
            rank=rank,
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_bias": lambda: inject_shared_bias(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_difffit": lambda: inject_shared_difffit(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_ia3_input": lambda: inject_shared_ia3_input(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_oft": lambda: inject_shared_oft(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_lokr": lambda: inject_shared_lokr(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_loha": lambda: inject_shared_loha(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
        "shared_fourierft": lambda: inject_shared_fourierft(
            route_module,
            coefficients=rank * rank,
            alpha=float(rank),
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        ),
    }
    if method in public_peft:
        targets = public_peft[method]()
        if not targets:
            raise RuntimeError("no compatible transformer projections found")
        return targets
    raise ValueError(f"unknown transfer method: {method}")


def baseline_training_phase(method: str) -> BaselinePhase:
    if method == "source_private_lora":
        return "private_calibration"
    if method in {
        "shared_lora",
        "shared_lora_witness",
        "static_multiplicative_gain",
        "shared_dora",
        "shared_vera",
        "bottleneck_adapter",
        "shared_full_rank_delta",
        "shared_lora_xs",
        "shared_pissa",
        "shared_lora_fa",
        "shared_mrt",
        "shared_bias",
        "shared_difffit",
        "shared_ia3_input",
        "shared_oft",
        "shared_lokr",
        "shared_loha",
        "shared_fourierft",
    }:
        return "transfer"
    raise ValueError(f"method is not a baseline: {method}")


def load_janusflow(
    root: Path,
    records: list[dict[str, object]],
    rank: int,
    seed: int,
    capability: str,
    timestep_buckets: int,
    method: str,
    image_size: int,
    gain_limit: float = 0.25,
    target_suffixes: tuple[str, ...] = TARGET_SUFFIXES,
) -> tuple[nn.Module, nn.Module, Sequence[SidecarTrainBatch], object]:
    del capability
    components = load_janusflow_components(
        root / "external/Janus",
        model_id=str(root / "models/janusflow/JanusFlow-1.3B"),
        model_revision=None,
        vae_id=str(root / "models/janusflow/sdxl-vae"),
        vae_revision=None,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    route_module = resolve_janusflow_transformer(components.model)
    attach_route(
        route_module,
        rank=rank,
        timestep_buckets=timestep_buckets,
        method=method,
        gain_limit=gain_limit,
        target_suffixes=target_suffixes,
    )
    batches = build_janusflow_batches(
        cast("list", records),
        processor=components.processor,
        vae=components.vae,
        task="understanding",
        batch_size=1,
        timestep_buckets=timestep_buckets,
        image_size=image_size,
        stochastic_seed=seed,
    )
    objective = JanusFlowObjective(
        route_module,
        timestep_buckets=timestep_buckets,
        stochastic_seed=seed + 1_000,
    )
    return components.model, route_module, batches, objective


def load_dimoo(
    root: Path,
    records: list[dict[str, object]],
    rank: int,
    seed: int,
    capability: str,
    timestep_buckets: int,
    method: str,
    image_size: int,
    gain_limit: float = 0.25,
    target_suffixes: tuple[str, ...] = TARGET_SUFFIXES,
) -> tuple[nn.Module, nn.Module, list[SidecarTrainBatch], object]:
    model_path = root / "models/dimoo/Lumina-DiMOO"
    order_id = training_order_id(records)
    token_root = (
        root
        / f"data/cross_backbone_tokens/{capability}/understanding/{image_size}"
        / f"n-{len(records)}/seed-{seed}-order-{order_id}"
    )
    records_path = token_root / "records.jsonl"
    if not records_path.is_file():
        encoder = build_dimoo_encoder(
            root / "external/Lumina-DiMOO",
            model_id=str(model_path),
            target_size=image_size,
            device="cuda",
        )
        pretokenize_route_records(
            cast("list", records), token_root, task="understanding", encoder=encoder
        )
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    components = load_dimoo_components(
        root / "external/Lumina-DiMOO",
        model_id=str(model_path),
        max_length=1024,
        torch_dtype="bfloat16",
        device_map={"": "cuda:0"},
    )
    checkpointable = getattr(components.model, "model", None)
    if checkpointable is None or not hasattr(checkpointable, "set_activation_checkpointing"):
        raise RuntimeError("official DiMOO activation checkpointing API is unavailable")
    checkpointable.set_activation_checkpointing("whole_layer")
    route_module = resolve_dimoo_transformer(components.model)
    attach_route(
        route_module,
        rank=rank,
        timestep_buckets=timestep_buckets,
        method=method,
        gain_limit=gain_limit,
        target_suffixes=target_suffixes,
    )
    batches = build_dimoo_batches(
        load_route_records(records_path),
        processor=components.processor,
        task="understanding",
        batch_size=1,
        timestep_buckets=timestep_buckets,
    )
    return (
        components.model,
        route_module,
        batches,
        DimooMaskedTokenObjective(route_module),
    )


def load_blip3o(
    root: Path,
    records: list[dict[str, object]],
    rank: int,
    seed: int,
    capability: str,
    timestep_buckets: int,
    method: str,
    image_size: int,
    gain_limit: float = 0.25,
    target_suffixes: tuple[str, ...] = TARGET_SUFFIXES,
) -> tuple[nn.Module, nn.Module, list[SidecarTrainBatch], object]:
    del seed, capability, image_size
    from skillbridge.blip3o_runtime import (
        build_understanding_inputs,
        load_original_model,
    )

    model, processor, _generation_tokenizer = load_original_model(root)
    route_module = model.model.layers
    attach_route(
        route_module,
        rank=rank,
        timestep_buckets=timestep_buckets,
        method=method,
        gain_limit=gain_limit,
        target_suffixes=target_suffixes,
    )
    batches: list[SidecarTrainBatch] = []
    for record in records:
        prompt = str(record["user_prompt"])
        answer = str(record["answer_text"])
        batches.append(
            SidecarTrainBatch(
                task="understanding",
                timestep_bucket=0,
                model_inputs=build_understanding_inputs(
                    processor,
                    Path(str(record["user_image"])),
                    prompt,
                    answer,
                ),
            )
        )
    return model, route_module, batches, native_loss


def load_uniworld_v1(
    root: Path,
    records: list[dict[str, object]],
    rank: int,
    seed: int,
    capability: str,
    timestep_buckets: int,
    method: str,
    image_size: int,
    gain_limit: float = 0.25,
    target_suffixes: tuple[str, ...] = TARGET_SUFFIXES,
) -> tuple[nn.Module, nn.Module, list[SidecarTrainBatch], object]:
    del seed, capability, image_size
    repo = root / "external/UniWorld-V1/UniWorld-V1"
    sys.path.insert(0, str(repo))
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import (
        UnivaQwen2p5VLForConditionalGeneration,
    )

    model_path = root / "models/uniworld_v1/UniWorld-V1-NF4"
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    route_module = resolve_uniworld_v1_transformer(model)
    attach_route(
        route_module,
        rank=rank,
        timestep_buckets=timestep_buckets,
        method=method,
        gain_limit=gain_limit,
        target_suffixes=target_suffixes,
    )
    batches: list[SidecarTrainBatch] = []
    from skillbridge.uniworld_training import encode_answer_training
    for record in records:
        encoded = encode_answer_training(processor, process_vision_info, record)
        batches.append(
            SidecarTrainBatch(
                task="understanding", timestep_bucket=0, model_inputs=dict(encoded)
            )
        )
    return model, route_module, batches, native_loss


LOADERS = {
    "janusflow": load_janusflow,
    "dimoo": load_dimoo,
    "blip3o": load_blip3o,
    "uniworld_v1": load_uniworld_v1,
}


def run_training(args) -> None:
    if min(args.steps, args.examples, args.rank, args.timestep_buckets, args.image_size) < 1:
        raise ValueError(
            "steps, examples, rank, timestep buckets, and image size must be positive"
        )
    if not 0.0 < args.gain_limit < 1.0 or not args.target_suffixes:
        raise ValueError("gain limit must lie in (0, 1) and projection suffixes cannot be empty")
    if args.backbone in {"blip3o", "uniworld_v1"} and args.timestep_buckets != 1:
        raise ValueError(f"{args.backbone} exposes one cached/static shared route")
    root = args.root.resolve()
    source_path = args.source_records
    if source_path is None:
        source_path = (
            root
            / "data/route_transfer/capabilities"
            / args.capability
            / "understanding/train/records.jsonl"
        )
    source_path = source_path.resolve()
    records = load_route_records(source_path)
    if args.preserve_source_order:
        if args.balanced_numeric_answers is not None:
            raise ValueError("preserved source order cannot be resampled")
        records = records[:args.examples]
    elif args.balanced_numeric_answers is None:
        random.Random(args.seed).shuffle(records)
        records = records[: args.examples]
    else:
        records = balance_numeric_answers(
            records,
            examples=args.examples,
            values=tuple(args.balanced_numeric_answers),
            seed=args.seed,
        )
    if args.shuffle_source_labels:
        shuffled_answers = [str(record["answer_text"]) for record in records]
        random.Random(args.seed + 10_000).shuffle(shuffled_answers)
        if len(shuffled_answers) > 1 and all(
            str(record["answer_text"]) == answer
            for record, answer in zip(records, shuffled_answers, strict=True)
        ):
            shuffled_answers = shuffled_answers[1:] + shuffled_answers[:1]
        records = [
            {**record, "answer_text": answer}
            for record, answer in zip(records, shuffled_answers, strict=True)
        ]
    order_id = training_order_id(cast("list[dict[str, object]]", records))
    if len(records) != args.examples:
        raise ValueError("source file contains fewer examples than requested")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if not configuration.get("optimization_audit", False):
        configuration.pop("optimization_audit", None)  # preserve legacy configuration identities
    configuration["weight_decay"] = 0.01
    if completed_training(output, configuration, inputs_text):
        print("Reusing completed identical training; no optimizer steps repeated.", flush=True)
        return
    if (output / "core.pt").exists() or (output / "adapter.pt").exists():
        raise FileExistsError("refusing to overwrite an existing trained checkpoint")
    inputs_path = output / "training-inputs.jsonl"
    if inputs_path.exists() and inputs_path.read_text() != inputs_text:
        raise ValueError("training inputs changed in an existing run directory")
    inputs_path.write_text(inputs_text)
    config_path = output / "training-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("training configuration changed in an existing run directory")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    torch.manual_seed(args.seed)
    model, route_module, batches, objective = LOADERS[args.backbone](
        root=root,
        records=records,
        rank=args.rank,
        seed=args.seed,
        capability=args.capability,
        timestep_buckets=args.timestep_buckets,
        method=args.method,
        image_size=args.image_size,
        gain_limit=args.gain_limit,
        target_suffixes=tuple(args.target_suffixes),
    )
    model.requires_grad_(False)
    training_kwargs = {
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "loss_function": objective,
        "optimizer_name": args.optimizer,
        "checkpoint_callback": make_progress_reporter(output / "training-progress.json", total_steps=args.steps),
    }
    if getattr(args, "optimization_audit", False):
        from skillbridge.optimization_audit import OptimizationAudit
        training_kwargs["optimization_observer"] = OptimizationAudit(
            output / "optimization.jsonl", every=100, total_steps=args.steps)
    history = (
        train_sidecar_stage(
            model,
            route_module,
            batches,
            phase="transfer",
            **training_kwargs,
        )
        if args.method in {"tied_core", "svd_core", "additive_tied_core", "random_basis_tmc"}
        else train_baseline_stage(
            model,
            route_module,
            batches,
            phase=baseline_training_phase(args.method),
            **training_kwargs,
        )
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_order.json").write_text(
        json.dumps(
            {
                "count": len(records),
                "unique_count": len({str(record["split_key"]) for record in records}),
                "seed": args.seed,
                "source_records": str(source_path),
                "training_order_id": order_id,
                "balanced_numeric_answers": args.balanced_numeric_answers or [],
                "split_keys": [str(record["split_key"]) for record in records],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in route_module.parameters()
        if parameter.requires_grad
    )
    checkpoint_metadata = {
        "backbone": args.backbone,
        "capability": args.capability,
        "method": args.method,
        "seed": args.seed,
        "rank": args.rank,
        "timestep_buckets": args.timestep_buckets,
        "image_size": args.image_size,
        "training_order_id": order_id,
        "shuffle_source_labels": args.shuffle_source_labels,
        "balanced_numeric_answers": ",".join(
            map(str, args.balanced_numeric_answers or [])
        ),
        "gain_limit": args.gain_limit,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "target_suffixes": ",".join(args.target_suffixes),
        "trainable_parameters": trainable_parameters,
    }
    if args.method in {"tied_core", "svd_core", "additive_tied_core", "random_basis_tmc"}:
        save_capability_checkpoint(
            route_module,
            output / "core.pt",
            metadata=checkpoint_metadata,
        )
    elif args.method in SHARED_LORA_METHODS:
        spectrum = capture_witness_spectrum(route_module)
        torch.save(
            {
                name: {
                    "left": item.left,
                    "singular_values": item.singular_values,
                    "right": item.right,
                    "scaling": item.scaling,
                }
                for name, item in spectrum.items()
            },
            output / "witness_spectrum.pt",
        )
        (output / "witness_concentration.json").write_text(
            json.dumps(witness_singular_concentration(spectrum), indent=2) + "\n",
            encoding="utf-8",
        )
    save_adapter_checkpoint(
        route_module,
        output / "adapter.pt",
        metadata=checkpoint_metadata,
    )
    (output / "history.jsonl").write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in history),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "capability": args.capability,
                "examples": len(records),
                "unique_examples": len(
                    {str(record["split_key"]) for record in records}
                ),
                "final_loss": history[-1].loss,
                "method": args.method,
                "rank": args.rank,
                "seed": args.seed,
                "steps": args.steps,
                "timestep_buckets": args.timestep_buckets,
                "image_size": args.image_size,
                "training_order_id": order_id,
                "shuffle_source_labels": args.shuffle_source_labels,
                "balanced_numeric_answers": args.balanced_numeric_answers or [],
                "gain_limit": args.gain_limit,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "target_suffixes": args.target_suffixes,
                "trainable_parameters": trainable_parameters,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    mark_training_complete(output, configuration)


def main() -> None:
    args = parse_args()
    with training_lock(args.output.resolve()):
        run_training(args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
