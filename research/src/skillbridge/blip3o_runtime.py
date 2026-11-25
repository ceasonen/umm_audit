from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from PIL import Image


@contextmanager
def disable_gradient_checkpointing_for_inference(
    model_class: type[object],
) -> Iterator[None]:
    """Bypass the released DiT's legacy Diffusers checkpointing hook."""
    with patch.object(
        model_class,
        "enable_gradient_checkpointing",
        return_value=None,
    ):
        yield


@contextmanager
def allow_offline_multimodal_encoder_override(
    pipeline_utils: object,
) -> Iterator[None]:
    """Skip only the stale Emu type check for the caller-supplied BLIP3-o model."""
    original = getattr(pipeline_utils, "maybe_raise_or_warn")

    def compatible_validator(
        library_name: object,
        library: object,
        class_name: object,
        importable_classes: object,
        passed_class_obj: object,
        name: object,
        is_pipeline_module: object,
    ) -> object:
        if name == "multimodal_encoder" and str(library_name).startswith(
            "transformers_modules.multimodal_encoder"
        ):
            return None
        return original(
            library_name,
            library,
            class_name,
            importable_classes,
            passed_class_obj,
            name,
            is_pipeline_module,
        )

    with patch.object(
        pipeline_utils,
        "maybe_raise_or_warn",
        side_effect=compatible_validator,
    ):
        yield


def _messages(
    image: Image.Image, question: str, answer: str | None = None
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def build_understanding_inputs(
    processor: Any,  # noqa: ANN401
    image_path: Path,
    question: str,
    answer: str,
) -> dict[str, torch.Tensor]:
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        prefix_text = processor.apply_chat_template(
            _messages(image, question),
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = processor.apply_chat_template(
            _messages(image, question, answer),
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix = processor(
            text=[prefix_text],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        full = processor(
            text=[full_text],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
    labels = full.input_ids.clone()
    labels[:, : int(prefix.input_ids.shape[1])] = -100
    inputs = {
        key: value
        for key, value in dict(full).items()
        if isinstance(value, torch.Tensor)
    }
    inputs["labels"] = labels
    return inputs


def load_original_model(root: Path) -> tuple[Any, Any, Any]:
    """Load the frozen original 4B release and its two official tokenizers."""
    repository = root / "external/BLIP3o-original"
    model_path = root / "models/blip3o/BLIP3o-Model-4B"
    processor_path = root / "models/blip3o/Qwen2.5-VL-7B-Instruct-processor"
    sys.path.insert(0, str(repository))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # The pinned official source downloads a 17 GB EVA tower at import time.
    # Text-to-image inference keeps that tower delayed and never reads it, so
    # guard this import with an explicit local placeholder instead of allowing
    # an unrelated hidden network dependency.
    eva_placeholder = root / "models/blip3o/eva-import-placeholder"
    eva_placeholder.mkdir(parents=True, exist_ok=True)
    with patch(
        "huggingface_hub.snapshot_download",
        return_value=str(eva_placeholder),
    ):
        from blip3o.model.language_model.blip3o_qwen_inference import (  # noqa: PLC0415
            blip3oQwenConfig,
            blip3oQwenForInferenceLM,
        )
        from blip3o.model.lumina_nextdit2d import (  # noqa: PLC0415
            LuminaNextDiT2DModel,
        )
        from blip3o.model.multimodal_encoder import (  # noqa: PLC0415
            builder as encoder_builder,
        )
    from transformers import (  # noqa: PLC0415
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    config = blip3oQwenConfig.from_pretrained(model_path, local_files_only=True)
    config.unfreeze_mm_vision_tower = False
    config.use_cache = False
    scheduler_loader = encoder_builder.FlowMatchEulerDiscreteScheduler.from_pretrained

    def local_scheduler(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
        return scheduler_loader(
            model_path / "diffusion-decoder",
            subfolder="scheduler",
            local_files_only=True,
        )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    with patch.object(
        encoder_builder.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        side_effect=local_scheduler,
    ), disable_gradient_checkpointing_for_inference(LuminaNextDiT2DModel):
        model = blip3oQwenForInferenceLM.from_pretrained(
            model_path,
            config=config,
            quantization_config=quantization,
            device_map={"": torch.cuda.current_device()},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            local_files_only=True,
        )
    object.__setattr__(model, "_skillbridge_flow_scheduler_loader", local_scheduler)
    processor = AutoProcessor.from_pretrained(
        processor_path, local_files_only=True
    )
    generation_tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    return model, processor, generation_tokenizer


def load_original_diffusion_pipeline(
    root: Path,
    model: Any,  # noqa: ANN401
    processor: Any,  # noqa: ANN401
    generation_tokenizer: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Load the released decoder while redirecting its hard-coded processor offline."""
    model_path = root / "models/blip3o/BLIP3o-Model-4B"
    pipeline_path = model_path / "diffusion-decoder/pipeline_llava_gen.py"
    from diffusers import DiffusionPipeline  # noqa: PLC0415
    from diffusers.pipelines import pipeline_utils  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415

    original_loader = AutoProcessor.from_pretrained

    def offline_processor(
        identifier: object, *args: object, **kwargs: object
    ) -> Any:  # noqa: ANN401
        if str(identifier).startswith("Qwen/Qwen2.5-VL-"):
            return processor
        kwargs["local_files_only"] = True
        return original_loader(identifier, *args, **kwargs)

    with patch.object(
        AutoProcessor, "from_pretrained", side_effect=offline_processor
    ), allow_offline_multimodal_encoder_override(pipeline_utils):
        pipeline = DiffusionPipeline.from_pretrained(
            model_path / "diffusion-decoder",
            custom_pipeline=str(pipeline_path),
            multimodal_encoder=model,
            tokenizer=generation_tokenizer,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="bf16",
            safety_checker=None,
            local_files_only=True,
        )
    pipeline.vae.to("cuda")
    pipeline.unet.to("cuda")
    return pipeline
