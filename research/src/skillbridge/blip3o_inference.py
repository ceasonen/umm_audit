from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from unittest.mock import patch

import torch
from PIL import Image
from torch import Tensor, nn

from skillbridge.sidecar import sidecar_route


@contextmanager
def flow_scheduler_override(model: object) -> Iterator[None]:
    """Redirect the released generation-time scheduler lookup when configured."""
    loader = getattr(model, "_skillbridge_flow_scheduler_loader", None)
    if loader is None:
        yield
        return
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: PLC0415

    with patch.object(
        FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        side_effect=loader,
    ):
        yield


class BLIPProcessorProtocol(Protocol):
    tokenizer: Any

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(self, **kwargs: object) -> Any: ...  # noqa: ANN401


@dataclass(slots=True)
class BLIP3oInferenceAdapter:
    """Original BLIP3-o Qwen-to-diffusion path with an explicit shared route."""

    model: nn.Module
    processor: BLIPProcessorProtocol
    diffusion_pipeline: Any
    route_module: nn.Module

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def reset_condition_cache(self) -> None:
        """Clear the released pipeline's unconditional condition between audit arms."""
        cache = getattr(self.diffusion_pipeline, "negative_prompt", None)
        if isinstance(cache, dict):
            cache.clear()

    @staticmethod
    def _understanding_messages(
        image: Image.Image, question: str
    ) -> list[dict[str, object]]:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

    @staticmethod
    def _generation_messages(prompt: str) -> list[dict[str, object]]:
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Please generate image based on the following caption: {prompt}",
            }],
        }]

    @torch.inference_mode()
    def understand(
        self,
        image_path: Path,
        question: str,
        *,
        max_new_tokens: int = 32,
    ) -> str:
        if max_new_tokens < 1:
            message = "max_new_tokens must be positive"
            raise ValueError(message)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            messages = self._understanding_messages(image, question)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
        prompt_length = int(inputs.input_ids.shape[1])
        with sidecar_route(self.route_module, task="understanding", timestep_bucket=0):
            output = cast("Any", self.model).generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        sequences = cast(
            "Tensor", output.sequences if hasattr(output, "sequences") else output
        )
        if sequences.ndim != 2 or sequences.shape[0] != 1:
            message = "BLIP3-o understanding returned invalid token IDs"
            raise ValueError(message)
        suffix = sequences[:, prompt_length:]
        return self.processor.tokenizer.batch_decode(
            suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @torch.inference_mode()
    def understand_batch(
        self,
        image_paths: list[Path],
        questions: list[str],
        *,
        max_new_tokens: int = 32,
    ) -> list[str]:
        if not image_paths or len(image_paths) != len(questions):
            message = "BLIP3-o understanding batch requires aligned non-empty inputs"
            raise ValueError(message)
        if max_new_tokens < 1:
            message = "max_new_tokens must be positive"
            raise ValueError(message)
        images: list[Image.Image] = []
        texts: list[str] = []
        for image_path, question in zip(image_paths, questions, strict=True):
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            images.append(image)
            texts.append(
                self.processor.apply_chat_template(
                    self._understanding_messages(image, question),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        tokenizer = self.processor.tokenizer
        original_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            inputs = self.processor(
                text=texts,
                images=images,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
        finally:
            tokenizer.padding_side = original_padding_side
        prompt_length = int(inputs.input_ids.shape[1])
        with sidecar_route(self.route_module, task="understanding", timestep_bucket=0):
            output = cast("Any", self.model).generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        sequences = cast(
            "Tensor", output.sequences if hasattr(output, "sequences") else output
        )
        if sequences.ndim != 2 or sequences.shape[0] != len(image_paths):
            message = "BLIP3-o understanding batch returned invalid token IDs"
            raise ValueError(message)
        decoded = self.processor.tokenizer.batch_decode(
            sequences[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [answer.strip() for answer in decoded]

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        image_token_length: int = 729,
        top_p: float = 0.95,
        top_k: int = 1200,
        guidance_scale: float = 3.0,
        diffusion_steps: int = 30,
        scale: int = 0,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
        if image_token_length < 1 or diffusion_steps < 1:
            message = "image token length and diffusion steps must be positive"
            raise ValueError(message)
        del image_token_length, top_p, top_k, scale
        text = self.processor.apply_chat_template(
            self._generation_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        cuda_devices = (
            [
                self._device.index
                if self._device.index is not None
                else torch.cuda.current_device()
            ]
            if self._device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed(seed)
            with flow_scheduler_override(self.model), sidecar_route(
                self.route_module, task="generation", timestep_bucket=0
            ):
                output = self.diffusion_pipeline(
                    [text],
                    guidance_scale=guidance_scale,
                    num_inference_steps=diffusion_steps,
                    height=height,
                    width=width,
                )
        image = getattr(output, "image", None)
        if not isinstance(image, Image.Image):
            message = "BLIP3-o diffusion decoder returned no image"
            raise ValueError(message)
        return image
