#!/usr/bin/env python3
"""Run paired target-generation trajectories through complete frozen UMMs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from skillbridge.adapter_checkpoint import load_adapter_checkpoint, load_capability_checkpoint
from skillbridge.baseline import (
    baseline_scale,
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
    private_shared_lora_scale,
    static_gain_scale,
)
from skillbridge.blip3o_inference import BLIP3oInferenceAdapter
from skillbridge.core_spectrum import (
    apply_core_component_budget,
    apply_core_layer_band,
    capture_core_spectrum,
    core_layer_bands,
    core_singular_concentration,
)
from skillbridge.cross_backbone_evaluation import (
    generate_fixed_predictions,
    generate_paired_trajectories,
    generate_route_interventions,
    generate_understanding_predictions,
    resolve_witness_budgets,
    summarize_understanding_budgets,
    summarize_understanding_doses,
)
from skillbridge.dimoo_data import load_route_records
from skillbridge.dimoo_inference import DimooInferenceAdapter, TokenizerProtocol
from skillbridge.dimoo_runtime import load_dimoo_components
from skillbridge.janusflow_inference import JanusFlowInferenceAdapter
from skillbridge.janusflow_runtime import load_janusflow_components
from skillbridge.sidecar import (
    functional_capability_scale,
    initialize_random_shared_access,
    initialize_tied_backbone_access,
    inject_sidecars,
)
from skillbridge.sidecar_backbone import (
    resolve_dimoo_transformer,
    resolve_janusflow_transformer,
    resolve_uniworld_v1_transformer,
)
from skillbridge.uniworld_v1_inference import UniWorldV1InferenceAdapter
from skillbridge.witness import (
    apply_witness_component_complement,
    apply_witness_component_budget,
    apply_witness_random_control,
    capture_witness_spectrum,
    witness_singular_concentration,
)

BACKBONES = ("janusflow", "dimoo", "blip3o", "uniworld_v1")
CORE_METHODS = ("tied_core", "svd_core", "additive_tied_core", "random_basis_tmc")
SHARED_LORA_METHODS = ("shared_lora", "shared_lora_witness")
STATIC_GAIN_METHOD = "static_multiplicative_gain"
SOURCE_PRIVATE_METHOD = "source_private_lora"
PUBLIC_PEFT_METHODS = (
    "shared_dora",
    "shared_vera",
    "bottleneck_adapter",
    "shared_full_rank_delta",
    "shared_lora_fa",
    "shared_lora_xs",
    "shared_pissa",
    "shared_ia3_input",
    "shared_mrt",
    "shared_bias",
    "shared_difffit",
    "shared_oft",
    "shared_lokr",
    "shared_loha",
    "shared_fourierft",
)
TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def uniworld_condition_metadata(
    condition_path: Literal["joint", "qwen_only", "t5_only"],
    *,
    reference_condition: Literal["inactive", "active", "ablated"] = "inactive",
) -> dict[str, str]:
    """Describe every text condition reaching FLUX for traceable path surgery."""
    sequence = {
        "joint": "t5_plus_qwen",
        "qwen_only": "qwen_only",
        "t5_only": "t5_only",
    }[condition_path]
    pooled = "null_prompt" if condition_path == "qwen_only" else "prompt"
    if reference_condition not in {"inactive", "active", "ablated"}:
        raise ValueError("unknown UniWorld reference condition")
    return {
        "uniworld_condition_path": condition_path,
        "sequence_condition": sequence,
        "pooled_clip_condition": pooled,
        "siglip_reference_condition": reference_condition,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument(
        "--method",
        choices=(
            *CORE_METHODS,
            *SHARED_LORA_METHODS,
            SOURCE_PRIVATE_METHOD,
            STATIC_GAIN_METHOD,
            *PUBLIC_PEFT_METHODS,
        ),
        default="tied_core",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("generation", "understanding", "editing"),
        default="generation",
    )
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--timestep-buckets", type=int, required=True)
    parser.add_argument("--witness-budget", type=int)
    parser.add_argument("--witness-budgets", type=int, nargs="+")
    parser.add_argument("--witness-budget-fractions", type=float, nargs="+")
    parser.add_argument(
        "--frozen-reference",
        type=Path,
        help="prediction manifest used as the frozen understanding baseline",
    )
    parser.add_argument(
        "--witness-intervention",
        choices=("prefix", "complement", "matched_random"),
        default="prefix",
    )
    parser.add_argument("--core-budget", type=int)
    parser.add_argument("--core-budgets", type=int, nargs="+")
    parser.add_argument("--core-budget-fractions", type=float, nargs="+")
    parser.add_argument("--core-layer-band-audit", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--understanding-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--route-interventions", action="store_true")
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument(
        "--uniworld-condition-path",
        choices=("joint", "qwen_only", "t5_only"),
        default="joint",
    )
    parser.add_argument(
        "--uniworld-reference-path",
        choices=("joint", "no_siglip"),
        default="joint",
    )
    parser.add_argument("--scales", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    return parser.parse_args()


def attach_core(
    route_module: nn.Module,
    *,
    rank: int,
    timestep_buckets: int,
    method: str,
    gain_limit: float = 0.25,
    target_suffixes: tuple[str, ...] = TARGET_SUFFIXES,
) -> None:
    if method in CORE_METHODS:
        targets = inject_sidecars(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            gain_limit=gain_limit,
            shared_update=("additive" if method == "additive_tied_core" else "multiplicative"),
            target_suffixes=target_suffixes,
        )
    elif method in {*SHARED_LORA_METHODS, SOURCE_PRIVATE_METHOD}:
        targets = inject_private_shared_lora(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == STATIC_GAIN_METHOD:
        targets = inject_static_multiplicative_gain(
            route_module,
            gain_limit=gain_limit,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_dora":
        targets = inject_shared_dora(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_vera":
        targets = inject_shared_vera(
            route_module,
            rank=rank,
            alpha=float(rank),
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "bottleneck_adapter":
        targets = inject_bottleneck_adapter(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_full_rank_delta":
        targets = inject_shared_full_rank_delta(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_lora_fa":
        targets = inject_shared_lora_fa(
            route_module,
            rank=rank,
            alpha=float(rank),
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_lora_xs":
        targets = inject_shared_lora_xs(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_pissa":
        targets = inject_shared_pissa(
            route_module,
            rank=rank,
            alpha=float(rank),
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_mrt":
        targets = inject_shared_mrt(
            route_module,
            rank=rank,
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_bias":
        targets = inject_shared_bias(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_difffit":
        targets = inject_shared_difffit(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_ia3_input":
        targets = inject_shared_ia3_input(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_oft":
        targets = inject_shared_oft(
            route_module,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_lokr":
        targets = inject_shared_lokr(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_loha":
        targets = inject_shared_loha(
            route_module,
            rank=rank,
            alpha=float(rank),
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    elif method == "shared_fourierft":
        targets = inject_shared_fourierft(
            route_module,
            coefficients=rank * rank,
            alpha=float(rank),
            seed=0,
            timestep_buckets=timestep_buckets,
            target_suffixes=target_suffixes,
        )
    else:
        raise ValueError(f"unknown transfer method: {method}")
    if not targets:
        raise RuntimeError("no compatible shared projections found")
    if method in CORE_METHODS:
        if method == "random_basis_tmc":
            initialize_random_shared_access(route_module, seed=0)
        else:
            initialize_tied_backbone_access(route_module, basis="svd" if method == "svd_core" else "energy")


def checkpoint_route_spec(path: Path) -> tuple[float, tuple[str, ...]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    gain_limit = float(metadata.get("gain_limit", 0.25))
    encoded = str(metadata.get("target_suffixes", ",".join(TARGET_SUFFIXES)))
    suffixes = tuple(value for value in encoded.split(",") if value)
    if not 0.0 < gain_limit < 1.0 or not suffixes:
        raise ValueError("checkpoint contains an invalid route specification")
    return gain_limit, suffixes


def load_janusflow(
    root: Path,
    *,
    rank: int,
    timestep_buckets: int,
    method: str,
    gain_limit: float,
    target_suffixes: tuple[str, ...],
):
    components = load_janusflow_components(
        root / "external/Janus",
        model_id=str(root / "models/janusflow/JanusFlow-1.3B"),
        model_revision=None,
        vae_id=str(root / "models/janusflow/sdxl-vae"),
        vae_revision=None,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    route = resolve_janusflow_transformer(components.model)
    attach_core(
        route,
        rank=rank,
        timestep_buckets=timestep_buckets,
        method=method,
        gain_limit=gain_limit,
        target_suffixes=target_suffixes,
    )
    adapter = JanusFlowInferenceAdapter(
        components.model,
        components.processor,
        components.vae,
        route,
        timestep_buckets,
    )
    return adapter, route, {"image_size": 384, "steps": 30, "cfg_scale": 5.0}


def load_dimoo(
    root: Path,
    *,
    rank: int,
    timestep_buckets: int,
    method: str,
    gain_limit: float,
    target_suffixes: tuple[str, ...],
):
    model_path = root / "models/dimoo/Lumina-DiMOO"
    components = load_dimoo_components(
        root / "external/Lumina-DiMOO",
        model_id=str(model_path),
        max_length=1024,
        torch_dtype="bfloat16",
        device_map={"": "cuda:0"},
    )
    route = resolve_dimoo_transformer(components.model)
    attach_core(
        route,
        rank=rank,
        timestep_buckets=timestep_buckets,
        method=method,
        gain_limit=gain_limit,
        target_suffixes=target_suffixes,
    )
    from diffusers import VQModel

    vqvae = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=torch.bfloat16)
    vqvae.to(next(components.model.parameters()).device).eval()
    adapter = DimooInferenceAdapter(
        components.model,
        cast("TokenizerProtocol", components.tokenizer),
        vqvae,
        route,
        root / "external/Lumina-DiMOO",
    )
    return (
        adapter,
        route,
        {
            "height": 512,
            "width": 512,
            "steps": 64,
            "cfg_scale": 4.0,
            "temperature": 1.0,
        },
    )


def load_blip3o(
    root: Path,
    *,
    rank: int,
    timestep_buckets: int,
    method: str,
    gain_limit: float,
    target_suffixes: tuple[str, ...],
    task: str = "generation",
):
    if timestep_buckets != 1:
        raise ValueError("BLIP3-o exposes one AR semantic-condition route")
    from skillbridge.blip3o_runtime import (
        load_original_diffusion_pipeline,
        load_original_model,
    )

    model, processor, generation_tokenizer = load_original_model(root)
    route = model.model.layers
    attach_core(
        route,
        rank=rank,
        timestep_buckets=1,
        method=method,
        gain_limit=gain_limit,
        target_suffixes=target_suffixes,
    )
    pipeline = (
        load_original_diffusion_pipeline(root, model, processor, generation_tokenizer)
        if task == "generation"
        else None
    )
    adapter = BLIP3oInferenceAdapter(model, processor, pipeline, route)
    return (
        adapter,
        route,
        {
            "height": 512,
            "width": 512,
            "diffusion_steps": 30,
            "guidance_scale": 3.0,
        },
    )


def load_uniworld_v1(
    root: Path,
    *,
    rank: int,
    timestep_buckets: int,
    method: str,
    gain_limit: float,
    target_suffixes: tuple[str, ...],
    task: str = "generation",
):
    if timestep_buckets != 1:
        raise ValueError("UniWorld-V1 exposes one cached conditioner route")
    repo = root / "external/UniWorld-V1/UniWorld-V1"
    sys.path.insert(0, str(repo))
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, SiglipImageProcessor, SiglipVisionModel
    from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import (
        UnivaQwen2p5VLForConditionalGeneration,
    )
    from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
    from univa.utils.flux_pipeline import FluxPipeline

    model_path = root / "models/uniworld_v1/UniWorld-V1-NF4"
    flux_path = root / "models/uniworld_v1/FLUX.1-dev-bnb-4bit"
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    route = resolve_uniworld_v1_transformer(model)
    attach_core(
        route,
        rank=rank,
        timestep_buckets=1,
        method=method,
        gain_limit=gain_limit,
        target_suffixes=target_suffixes,
    )
    pipe = FluxPipeline.from_pretrained(
        flux_path,
        transformer=model.denoise_tower.denoiser,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    siglip_processor = None
    siglip_model = None
    if task == "editing":
        siglip_path = root / "models/uniworld_v1/siglip2-so400m-patch16-512"
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
        siglip_model = SiglipVisionModel.from_pretrained(
            siglip_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()
    adapter = UniWorldV1InferenceAdapter(
        model=model,
        processor=processor,
        pipe=pipe,
        route_module=route,
        tokenizers=(pipe.tokenizer, pipe.tokenizer_2),
        text_encoders=(pipe.text_encoder, pipe.text_encoder_2),
        encode_prompt=encode_prompt,
        process_vision_info=process_vision_info,
        siglip_processor=siglip_processor,
        siglip_model=siglip_model,
    )
    return (
        adapter,
        route,
        {
            "height": 512,
            "width": 512,
            "diffusion_steps": 28,
            "guidance_scale": 3.5,
        },
    )


LOADERS = {
    "janusflow": load_janusflow,
    "dimoo": load_dimoo,
    "blip3o": load_blip3o,
    "uniworld_v1": load_uniworld_v1,
}


def main() -> None:
    args = parse_args()
    visual_task = args.task in {"generation", "editing"}
    if (
        args.rank < 1
        or args.timestep_buckets < 1
        or args.understanding_batch_size < 1
        or (args.limit is not None and args.limit < 1)
    ):
        message = "rank, timestep buckets, batch size, and optional limit must be positive"
        raise ValueError(message)
    if args.task == "editing" and args.backbone != "uniworld_v1":
        raise ValueError("editing currently requires --backbone uniworld_v1")
    if args.uniworld_condition_path != "joint" and (
        args.backbone != "uniworld_v1" or not visual_task
    ):
        raise ValueError(
            "--uniworld-condition-path requires a visual UniWorld-V1 target"
        )
    if args.uniworld_reference_path != "joint" and args.task != "editing":
        raise ValueError("--uniworld-reference-path requires UniWorld-V1 editing")
    if args.route_interventions and not visual_task:
        raise ValueError("route interventions require a visual target")
    if args.route_interventions and args.method not in (*CORE_METHODS, *SHARED_LORA_METHODS):
        raise ValueError("route interventions require tied core or shared LoRA")
    if args.active_only and (not visual_task or args.method not in SHARED_LORA_METHODS):
        raise ValueError("active-only generation requires standard shared LoRA")
    if args.core_layer_band_audit and (
        not visual_task or args.method not in CORE_METHODS
    ):
        raise ValueError("core layer-band audit requires tied-core target generation")
    budget_requested = any(
        value is not None
        for value in (
            args.core_budget,
            args.core_budgets,
            args.core_budget_fractions,
            args.witness_budget,
            args.witness_budgets,
            args.witness_budget_fractions,
        )
    )
    if args.route_interventions and budget_requested:
        raise ValueError("route interventions are mutually exclusive with budget schedules")
    if args.active_only and (budget_requested or args.route_interventions):
        raise ValueError("active-only generation is mutually exclusive with interventions and budgets")
    if args.core_layer_band_audit and budget_requested:
        raise ValueError("core layer-band audit excludes budget schedules")
    if args.route_interventions and args.core_layer_band_audit:
        raise ValueError("route and layer-band interventions are mutually exclusive")
    if args.witness_intervention != "prefix" and args.method not in SHARED_LORA_METHODS:
        raise ValueError("witness interventions require --method shared_lora")
    root = args.root.resolve()
    gain_limit, target_suffixes = checkpoint_route_spec(args.core.resolve())
    loader_kwargs = {
        "rank": args.rank,
        "timestep_buckets": args.timestep_buckets,
        "method": "tied_core" if args.method == "svd_core" else args.method,
        "gain_limit": gain_limit,
        "target_suffixes": target_suffixes,
    }
    if args.backbone in {"blip3o", "uniworld_v1"}:
        loader_kwargs["task"] = args.task
    adapter, route, generation_kwargs = LOADERS[args.backbone](root, **loader_kwargs)
    if args.method == "svd_core":
        from skillbridge.adapter_checkpoint import restore_svd_adapter
        restore_svd_adapter(route, args.core.resolve())
    if args.backbone == "uniworld_v1" and visual_task:
        generation_kwargs["condition_path"] = args.uniworld_condition_path
        if args.task == "editing":
            siglip_enabled = args.uniworld_reference_path == "joint"
            generation_kwargs["siglip_enabled"] = siglip_enabled
            reference_condition = "active" if siglip_enabled else "ablated"
        else:
            reference_condition = "inactive"
        condition_metadata = uniworld_condition_metadata(
            args.uniworld_condition_path,
            reference_condition=reference_condition,
        )
    else:
        condition_metadata = {}
    records = cast("list[dict[str, object]]", load_route_records(args.records.resolve()))
    if args.limit is not None:
        records = records[: args.limit]
    if args.method in CORE_METHODS:
        if any(
            value is not None
            for value in (
                args.witness_budget,
                args.witness_budgets,
                args.witness_budget_fractions,
            )
        ):
            raise ValueError("component budgets require --method shared_lora")
        load_capability_checkpoint(route, args.core.resolve())
        core_budget_requested = any(
            value is not None
            for value in (
                args.core_budget,
                args.core_budgets,
                args.core_budget_fractions,
            )
        )
        if args.route_interventions:
            generate_route_interventions(
                cast("Any", adapter),
                route,
                records,
                args.output.resolve(),
                seed=args.seed,
                generation_kwargs=generation_kwargs,
                resume=args.resume,
            )
        elif args.core_layer_band_audit:
            spectrum = capture_core_spectrum(route)
            output_root = args.output.resolve()
            bands = core_layer_bands(spectrum)
            total_components = len(core_singular_concentration(spectrum))
            plan: dict[str, object] = {
                "format_version": 1,
                "partition": {name: list(sites) for name, sites in bands.items()},
                "conditions": {},
            }
            conditions = cast("dict[str, object]", plan["conditions"])
            apply_core_component_budget(route, spectrum, budget=0)
            generate_fixed_predictions(
                cast("Any", adapter),
                records,
                output_root / "band-zero",
                seed=args.seed,
                generation_kwargs=generation_kwargs,
                metadata={"method": "tied_core_layer_band", "layer_band": "zero"},
                resume=args.resume,
            )
            conditions["zero"] = {"component_budget": 0}
            for band in ("early", "middle", "late"):
                report = apply_core_layer_band(
                    route,
                    spectrum,
                    band=band,
                    task="generation",
                    match_full_operator_energy=True,
                )
                generate_fixed_predictions(
                    cast("Any", adapter),
                    records,
                    output_root / f"band-{band}",
                    seed=args.seed,
                    generation_kwargs=generation_kwargs,
                    metadata={"method": "tied_core_layer_band", **report},
                    resume=args.resume,
                )
                conditions[band] = report
            apply_core_component_budget(route, spectrum, budget=total_components)
            generate_fixed_predictions(
                cast("Any", adapter),
                records,
                output_root / "band-full",
                seed=args.seed,
                generation_kwargs=generation_kwargs,
                metadata={
                    "method": "tied_core_layer_band",
                    "layer_band": "full",
                    "component_budget": total_components,
                },
                resume=args.resume,
            )
            conditions["full"] = {"component_budget": total_components}
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "layer-band-plan.json").write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif core_budget_requested:
            spectrum = capture_core_spectrum(route)
            total_components = len(core_singular_concentration(spectrum))
            budgets = resolve_witness_budgets(
                args.core_budget,
                args.core_budgets,
                total_components=total_components,
                fractions=args.core_budget_fractions,
            )
            use_subdirectories = (
                args.core_budgets is not None or args.core_budget_fractions is not None
            )
            for budget in budgets:
                apply_core_component_budget(route, spectrum, budget=budget)
                output = (
                    args.output.resolve() / f"budget-{budget}"
                    if use_subdirectories
                    else args.output.resolve()
                )
                metadata = {
                    "method": "tied_core_subset",
                    "component_budget": budget,
                    "total_components": total_components,
                    **condition_metadata,
                }
                if visual_task:
                    generate_fixed_predictions(
                        cast("Any", adapter),
                        records,
                        output,
                        seed=args.seed,
                        generation_kwargs=generation_kwargs,
                        metadata=metadata,
                        resume=args.resume,
                    )
                else:
                    generate_understanding_predictions(
                        cast("Any", adapter),
                        records,
                        output,
                        metadata=metadata,
                        resume=args.resume,
                        batch_size=args.understanding_batch_size,
                    )
            if args.task == "understanding":
                summary = summarize_understanding_budgets(
                    args.output.resolve(),
                    backbone=args.backbone,
                    seed=args.seed,
                    frozen_reference=args.frozen_reference,
                )
                (args.output.resolve() / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        elif visual_task:
            generate_paired_trajectories(
                cast("Any", adapter),
                route,
                records,
                args.output.resolve(),
                scales=tuple(args.scales),
                seed=args.seed,
                generation_kwargs=generation_kwargs,
                resume=args.resume,
            )
        else:
            output = args.output.resolve()
            for scale in args.scales:
                label = format(scale, ".6g").replace("-", "m").replace(".", "p")
                with functional_capability_scale(route, scale=scale):
                    generate_understanding_predictions(
                        cast("Any", adapter),
                        records,
                        output / f"scale-{label}",
                        metadata={"scale": scale},
                        resume=args.resume,
                        batch_size=args.understanding_batch_size,
                    )
            summary = summarize_understanding_doses(output, backbone=args.backbone, seed=args.seed)
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    elif args.method == SOURCE_PRIVATE_METHOD:
        if budget_requested:
            raise ValueError("source-private LoRA excludes component budgets")
        load_adapter_checkpoint(route, args.core.resolve())
        output_root = args.output.resolve()
        for scale in args.scales:
            label = format(scale, ".6g").replace("-", "m").replace(".", "p")
            output = output_root / f"scale-{label}"
            metadata = {
                "method": SOURCE_PRIVATE_METHOD,
                "scale": scale,
                **condition_metadata,
            }
            with private_shared_lora_scale(route, scale=scale):
                if visual_task:
                    generate_fixed_predictions(
                        cast("Any", adapter),
                        records,
                        output,
                        seed=args.seed,
                        generation_kwargs=generation_kwargs,
                        metadata=metadata,
                        resume=args.resume,
                    )
                else:
                    generate_understanding_predictions(
                        cast("Any", adapter),
                        records,
                        output,
                        metadata=metadata,
                        resume=args.resume,
                        batch_size=args.understanding_batch_size,
                    )
        if args.task == "understanding":
            summary = summarize_understanding_doses(
                output_root, backbone=args.backbone, seed=args.seed
            )
            (output_root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    elif args.method == STATIC_GAIN_METHOD:
        if budget_requested:
            raise ValueError("static gain baseline excludes component budgets")
        load_adapter_checkpoint(route, args.core.resolve())
        output_root = args.output.resolve()
        for scale in args.scales:
            label = format(scale, ".6g").replace("-", "m").replace(".", "p")
            output = output_root / f"scale-{label}"
            metadata = {
                "method": STATIC_GAIN_METHOD,
                "scale": scale,
                **condition_metadata,
            }
            with static_gain_scale(route, scale=scale):
                if visual_task:
                    generate_fixed_predictions(
                        cast("Any", adapter),
                        records,
                        output,
                        seed=args.seed,
                        generation_kwargs=generation_kwargs,
                        metadata=metadata,
                        resume=args.resume,
                    )
                else:
                    generate_understanding_predictions(
                        cast("Any", adapter),
                        records,
                        output,
                        metadata=metadata,
                        resume=args.resume,
                        batch_size=args.understanding_batch_size,
                    )
        if args.task == "understanding":
            summary = summarize_understanding_doses(
                output_root, backbone=args.backbone, seed=args.seed
            )
            (output_root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    elif args.method in PUBLIC_PEFT_METHODS:
        if budget_requested:
            raise ValueError("shared PEFT baselines exclude component budgets")
        load_adapter_checkpoint(route, args.core.resolve())
        output_root = args.output.resolve()
        for scale in args.scales:
            label = format(scale, ".6g").replace("-", "m").replace(".", "p")
            output = output_root / f"scale-{label}"
            metadata = {"method": args.method, "scale": scale, **condition_metadata}
            with baseline_scale(route, scale=scale):
                if visual_task:
                    generate_fixed_predictions(
                        cast("Any", adapter),
                        records,
                        output,
                        seed=args.seed,
                        generation_kwargs=generation_kwargs,
                        metadata=metadata,
                        resume=args.resume,
                    )
                else:
                    generate_understanding_predictions(
                        cast("Any", adapter),
                        records,
                        output,
                        metadata=metadata,
                        resume=args.resume,
                        batch_size=args.understanding_batch_size,
                    )
        if args.task == "understanding":
            summary = summarize_understanding_doses(
                output_root, backbone=args.backbone, seed=args.seed
            )
            (output_root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        if any(
            value is not None
            for value in (
                args.core_budget,
                args.core_budgets,
                args.core_budget_fractions,
            )
        ):
            raise ValueError("core budgets require --method tied_core")
        load_adapter_checkpoint(route, args.core.resolve())
        if args.active_only:
            generate_fixed_predictions(
                cast("Any", adapter),
                records,
                args.output.resolve(),
                seed=args.seed,
                generation_kwargs=generation_kwargs,
                metadata={"method": args.method, "condition": "active", **condition_metadata},
                resume=args.resume,
            )
            return
        if args.route_interventions:
            generate_route_interventions(
                cast("Any", adapter),
                route,
                records,
                args.output.resolve(),
                seed=args.seed,
                generation_kwargs=generation_kwargs,
                resume=args.resume,
                intervention_kind="shared_lora",
            )
            return
        spectrum = capture_witness_spectrum(route)
        total_components = len(witness_singular_concentration(spectrum))
        budgets = resolve_witness_budgets(
            args.witness_budget,
            args.witness_budgets,
            total_components=total_components,
            fractions=args.witness_budget_fractions,
        )
        use_subdirectories = (
            args.witness_budgets is not None or args.witness_budget_fractions is not None
        )
        for budget in budgets:
            if args.witness_intervention == "prefix":
                apply_witness_component_budget(route, spectrum, budget=budget)
            elif args.witness_intervention == "complement":
                apply_witness_component_complement(route, spectrum, budget=budget)
            else:
                apply_witness_random_control(
                    route,
                    spectrum,
                    budget=budget,
                    seed=args.seed,
                )
            output = (
                args.output.resolve() / f"budget-{budget}"
                if use_subdirectories
                else args.output.resolve()
            )
            metadata = {
                "method": args.method,
                "component_budget": budget,
                "total_components": total_components,
                "witness_intervention": args.witness_intervention,
                **condition_metadata,
            }
            if visual_task:
                generate_fixed_predictions(
                    cast("Any", adapter),
                    records,
                    output,
                    seed=args.seed,
                    generation_kwargs=generation_kwargs,
                    metadata=metadata,
                    resume=args.resume,
                )
            else:
                generate_understanding_predictions(
                    cast("Any", adapter),
                    records,
                    output,
                    metadata=metadata,
                    resume=args.resume,
                    batch_size=args.understanding_batch_size,
                )
        if args.task == "understanding":
            summary = summarize_understanding_budgets(
                args.output.resolve(),
                backbone=args.backbone,
                seed=args.seed,
                frozen_reference=args.frozen_reference,
            )
            (args.output.resolve() / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
