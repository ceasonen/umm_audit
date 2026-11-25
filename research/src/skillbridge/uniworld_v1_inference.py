from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from PIL import Image
from torch import Tensor, nn

from skillbridge.sidecar import sidecar_route


class UniWorldProcessorProtocol(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(self, **kwargs: object) -> Any: ...  # noqa: ANN401

    def batch_decode(self, tokens: Tensor, **kwargs: object) -> list[str]: ...


ProcessVisionInfo = Callable[[list[dict[str, object]]], tuple[object, object]]
EncodePrompt = Callable[..., tuple[Tensor, Tensor]]


@dataclass(slots=True)
class UniWorldV1InferenceAdapter:
    """Official Qwen-conditioner-to-FLUX path under a static shared route."""

    model: nn.Module
    processor: UniWorldProcessorProtocol
    pipe: Any
    route_module: nn.Module
    tokenizers: Sequence[object]
    text_encoders: Sequence[object]
    encode_prompt: EncodePrompt
    process_vision_info: ProcessVisionInfo
    siglip_processor: Any | None = None
    siglip_model: nn.Module | None = None

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _chat_text(processor: UniWorldProcessorProtocol, messages: list[dict[str, object]]) -> str:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        marker = "<|im_end|>\n"
        parts = text.split(marker)
        return marker.join(parts[1:]) if len(parts) > 1 else text

    def _inputs(self, messages: list[dict[str, object]]) -> Any:  # noqa: ANN401
        text = self._chat_text(self.processor, messages)
        images, videos = self.process_vision_info(messages)
        return self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

    @staticmethod
    def _optional(inputs: object, name: str) -> object | None:
        if isinstance(inputs, Mapping):
            return inputs.get(name)
        return getattr(inputs, name, None)

    @torch.inference_mode()
    def understand(
        self,
        image_path: Path,
        question: str,
        *,
        max_new_tokens: int = 128,
    ) -> str:
        if max_new_tokens < 1:
            message = "max_new_tokens must be positive"
            raise ValueError(message)
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = self._inputs(messages)
        with sidecar_route(self.route_module, task="understanding", timestep_bucket=0):
            generated = cast("Any", self.model).generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        suffix = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @torch.inference_mode()
    def generation_condition(self, prompt: str, *, return_input_ids: bool = False) -> Tensor | tuple[Tensor,list[int]]:
        """Capture the native Qwen generation carrier without a FLUX decoder call."""
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = self._inputs(messages)
        with sidecar_route(self.route_module, task="generation", timestep_bucket=0):
            lvmm_embeddings = cast("Any", self.model)(
                inputs.input_ids,
                pixel_values=self._optional(inputs, "pixel_values"),
                attention_mask=inputs.attention_mask,
                image_grid_thw=self._optional(inputs, "image_grid_thw"),
                siglip_hidden_states=None,
                output_type="denoise_embeds",
            )
        if return_input_ids:
            ids=inputs.input_ids[0].detach().cpu().tolist()
            if len(ids)!=lvmm_embeddings.shape[1]:
                raise ValueError('native carrier positions differ from tokenizer positions')
            return lvmm_embeddings,ids
        return lvmm_embeddings

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        height: int = 1024,
        width: int = 1024,
        diffusion_steps: int = 28,
        guidance_scale: float = 3.5,
        condition_path: Literal["joint", "qwen_only", "t5_only"] = "joint",
        condition_transform: Callable[[Tensor], Tensor] | None = None,
    ) -> Image.Image:
        if min(height, width, diffusion_steps) < 1:
            message = "image dimensions and diffusion steps must be positive"
            raise ValueError(message)
        if condition_path not in {"joint", "qwen_only", "t5_only"}:
            message = f"unknown UniWorld-V1 condition path: {condition_path}"
            raise ValueError(message)
        lvmm_embeddings = self.generation_condition(prompt)
        if condition_transform is not None:
            changed = condition_transform(lvmm_embeddings)
            if (not isinstance(changed, Tensor) or changed.shape != lvmm_embeddings.shape
                    or changed.dtype != lvmm_embeddings.dtype
                    or changed.device != lvmm_embeddings.device
                    or not torch.isfinite(changed).all()):
                raise ValueError("donor conditioning must preserve finite native tensor shape and dtype")
            lvmm_embeddings = changed
        t5_embeddings, pooled_embeddings = self.encode_prompt(
            self.text_encoders,
            self.tokenizers,
            prompt if condition_path != "qwen_only" else "",
            256,
            self._device,
            1,
        )
        if condition_path == "joint":
            prompt_embeddings = torch.cat((t5_embeddings, lvmm_embeddings), dim=1)
        elif condition_path == "qwen_only":
            prompt_embeddings = lvmm_embeddings
        else:
            prompt_embeddings = t5_embeddings
        generator = torch.Generator(device=self._device).manual_seed(seed)
        result = self.pipe(
            prompt_embeds=prompt_embeddings,
            pooled_prompt_embeds=pooled_embeddings,
            height=height,
            width=width,
            num_inference_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        images = getattr(result, "images", None)
        if not images or not isinstance(images[0], Image.Image):
            message = "official UniWorld-V1 generation returned no PIL image"
            raise ValueError(message)
        return images[0]

    @torch.inference_mode()
    def edit(
        self,
        image_path: Path,
        instruction: str,
        *,
        seed: int = 0,
        height: int = 1024,
        width: int = 1024,
        diffusion_steps: int = 28,
        guidance_scale: float = 3.5,
        condition_path: Literal["joint", "qwen_only", "t5_only"] = "joint",
        siglip_enabled: bool = True,
    ) -> Image.Image:
        if not image_path.is_file():
            message = f"editing reference image does not exist: {image_path}"
            raise FileNotFoundError(message)
        if not instruction.strip():
            message = "editing instruction must be non-empty"
            raise ValueError(message)
        if min(height, width, diffusion_steps) < 1:
            message = "image dimensions and diffusion steps must be positive"
            raise ValueError(message)
        if condition_path not in {"joint", "qwen_only", "t5_only"}:
            message = f"unknown UniWorld-V1 condition path: {condition_path}"
            raise ValueError(message)
        if siglip_enabled and (self.siglip_processor is None or self.siglip_model is None):
            message = "UniWorld-V1 editing requires the SigLIP reference encoder"
            raise RuntimeError(message)

        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": str(image_path)},
                ],
            }
        ]
        inputs = self._inputs(messages)
        siglip_hidden_states = None
        if siglip_enabled:
            assert self.siglip_processor is not None
            assert self.siglip_model is not None
            with Image.open(image_path) as source:
                pixels = self.siglip_processor.preprocess(
                    images=source.convert("RGB"),
                    do_resize=True,
                    return_tensors="pt",
                    do_convert_rgb=True,
                ).pixel_values
            try:
                reference_device = next(self.siglip_model.parameters()).device
            except StopIteration:
                reference_device = self._device
            siglip_hidden_states = self.siglip_model(
                pixels.to(reference_device)
            ).last_hidden_state

        with sidecar_route(self.route_module, task="generation", timestep_bucket=0):
            lvmm_embeddings = cast("Any", self.model)(
                inputs.input_ids,
                pixel_values=self._optional(inputs, "pixel_values"),
                attention_mask=inputs.attention_mask,
                image_grid_thw=self._optional(inputs, "image_grid_thw"),
                siglip_hidden_states=siglip_hidden_states,
                output_type="denoise_embeds",
            )
        t5_embeddings, pooled_embeddings = self.encode_prompt(
            self.text_encoders,
            self.tokenizers,
            instruction if condition_path != "qwen_only" else "",
            256,
            self._device,
            1,
        )
        if condition_path == "joint":
            prompt_embeddings = torch.cat((t5_embeddings, lvmm_embeddings), dim=1)
        elif condition_path == "qwen_only":
            prompt_embeddings = lvmm_embeddings
        else:
            prompt_embeddings = t5_embeddings
        generator = torch.Generator(device=self._device).manual_seed(seed)
        result = self.pipe(
            prompt_embeds=prompt_embeddings,
            pooled_prompt_embeds=pooled_embeddings,
            height=height,
            width=width,
            num_inference_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        images = getattr(result, "images", None)
        if not images or not isinstance(images[0], Image.Image):
            message = "official UniWorld-V1 editing returned no PIL image"
            raise ValueError(message)
        return images[0]
