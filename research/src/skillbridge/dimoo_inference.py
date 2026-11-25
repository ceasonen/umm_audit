from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import torch
from PIL import Image, ImageFile
from torch import Tensor, nn

from skillbridge.routing import forward_step_routing

MASK = 126336
NEWLINE = 126084
ANSWER_START = 126354
ANSWER_END = 126355
IMAGE_START = 126349
IMAGE_END = 126350


class TokenizerProtocol(Protocol):
    def __call__(self, text: str) -> dict[str, list[int]]: ...

    def batch_decode(self, tokens: Tensor, *, skip_special_tokens: bool) -> list[str]: ...


def _prompt(kind: str, content: str) -> str:
    if kind == "understanding":
        system = (
            "You are a multimodal model that can process both text and images. "
            "Answer the question based on the provided image."
        )
        return f"<system>{system}</system><user>{content}</user>"
    system = "Generate an image according to the text prompt."
    return f"<system>{system}</system><user>{content}</user>"


def build_understanding_tokens(
    tokenizer: TokenizerProtocol,
    *,
    question: str,
    image_tokens: list[int],
    generation_length: int,
) -> tuple[Tensor, int]:
    if generation_length < 1:
        msg = "generation_length must be positive"
        raise ValueError(msg)
    prompt_ids = tokenizer(_prompt("understanding", question))["input_ids"]
    prefix = prompt_ids[:-1] + image_tokens + prompt_ids[-1:]
    code_start = len(prefix) + 1
    tokens = prefix + [ANSWER_START] + [MASK] * generation_length + [ANSWER_END]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0), code_start


def _visual_mask_grid(visual_tokens: int, grid_width: int) -> list[int]:
    if visual_tokens < 1 or grid_width < 1 or visual_tokens % grid_width:
        msg = "visual_tokens must be positive and divisible by grid_width"
        raise ValueError(msg)
    output: list[int] = []
    for _ in range(visual_tokens // grid_width):
        output.extend([MASK] * grid_width)
        output.append(NEWLINE)
    return output


def _vqvae_autocast(vqvae: object):
    parameters = getattr(vqvae, "parameters", None)
    if parameters is None:
        return nullcontext()
    try:
        anchor = next(parameters())
    except StopIteration:
        return nullcontext()
    if anchor.device.type not in {"cpu", "cuda"} or anchor.dtype not in {
        torch.float16,
        torch.bfloat16,
    }:
        return nullcontext()
    return torch.autocast(device_type=anchor.device.type, dtype=anchor.dtype)


def _load_rgb_image(path: Path) -> Image.Image:
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as source:
            return source.convert("RGB").copy()
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def build_generation_tokens(
    tokenizer: TokenizerProtocol,
    *,
    prompt: str,
    visual_tokens: int,
    grid_width: int,
) -> tuple[Tensor, Tensor, int]:
    conditional_ids = tokenizer(_prompt("generation", prompt))["input_ids"]
    unconditional_ids = tokenizer(_prompt("generation", "<uncondition>"))["input_ids"]
    visual = _visual_mask_grid(visual_tokens, grid_width)
    suffix = [ANSWER_START, IMAGE_START, *visual, IMAGE_END, ANSWER_END]
    code_start = len(conditional_ids) + 2
    conditional = torch.tensor(conditional_ids + suffix, dtype=torch.long).unsqueeze(0)
    unconditional = torch.tensor(unconditional_ids, dtype=torch.long).unsqueeze(0)
    return conditional, unconditional, code_start


@dataclass(slots=True)
class DimooInferenceAdapter:
    model: nn.Module
    tokenizer: TokenizerProtocol
    vqvae: Any
    route_module: nn.Module | None
    external_repo: Path

    def _import(self, name: str) -> ModuleType:
        repo = str(self.external_repo.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        return importlib.import_module(name)

    def understand(
        self,
        image_path: Path,
        question: str,
        *,
        steps: int = 128,
        generation_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
    ) -> str:
        return self.understand_batch(
            [image_path],
            [question],
            steps=steps,
            generation_length=generation_length,
            block_length=block_length,
            temperature=temperature,
        )[0]

    def understand_batch(
        self,
        image_paths: Sequence[Path],
        questions: Sequence[str],
        *,
        steps: int = 128,
        generation_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        model_batch_size: int = 4,
    ) -> list[str]:
        if not image_paths or len(image_paths) != len(questions):
            message = "understanding batches require paired non-empty inputs"
            raise ValueError(message)
        if model_batch_size < 1:
            message = "model batch size must be positive"
            raise ValueError(message)
        image_utils = self._import("utils.image_utils")
        generator_module = self._import("generators.text_understanding_generator")
        encode = image_utils.encode_img_with_breaks
        prepared: list[tuple[Tensor, int]] = []
        for image_path, question in zip(image_paths, questions, strict=True):
            image = _load_rgb_image(image_path)
            with _vqvae_autocast(self.vqvae):
                image_tokens = cast("list[int]", encode(image, self.vqvae))
            prepared.append(
                build_understanding_tokens(
                    self.tokenizer,
                    question=question,
                    image_tokens=image_tokens,
                    generation_length=generation_length,
                )
            )
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (prompt, code_start) in enumerate(prepared):
            groups[(prompt.shape[1], code_start)].append(index)
        device = next(self.model.parameters()).device
        generate = generator_module.generate_text_understanding
        answers: list[str | None] = [None] * len(prepared)
        for (_, code_start), indices in groups.items():
            for offset in range(0, len(indices), model_batch_size):
                chunk = indices[offset : offset + model_batch_size]
                prompt = torch.cat([prepared[index][0] for index in chunk], dim=0).to(device)
                route_context = (
                    forward_step_routing(
                        self.model,
                        self.route_module,
                        task="understanding",
                        total_steps=steps,
                    )
                    if self.route_module is not None
                    else nullcontext()
                )
                with route_context:
                    output = generate(
                        self.model,
                        prompt,
                        steps=steps,
                        gen_length=generation_length,
                        block_length=block_length,
                        temperature=temperature,
                        code_start=code_start,
                    )
                decoded = self.tokenizer.batch_decode(
                    output[:, code_start:-1],
                    skip_special_tokens=True,
                )
                if len(decoded) != len(chunk):
                    message = "tokenizer returned the wrong batch size"
                    raise ValueError(message)
                for index, answer in zip(chunk, decoded, strict=True):
                    answers[index] = answer
        if any(answer is None for answer in answers):
            message = "understanding batch left an unanswered input"
            raise RuntimeError(message)
        return [cast("str", answer) for answer in answers]

    def generate(
        self,
        prompt: str,
        *,
        height: int = 512,
        width: int = 512,
        steps: int = 64,
        cfg_scale: float = 4.0,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> Image.Image:
        image_utils = self._import("utils.image_utils")
        generator_module = self._import("generators.image_generation_generator")
        scale = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        visual_tokens = (height // scale) * (width // scale)
        grid_width = width // scale
        conditional, unconditional, code_start = build_generation_tokens(
            self.tokenizer,
            prompt=prompt,
            visual_tokens=visual_tokens,
            grid_width=grid_width,
        )
        device = next(self.model.parameters()).device
        conditional = conditional.to(device)
        unconditional = unconditional.to(device)
        random_generator = torch.Generator(device=device).manual_seed(seed)
        output_head = cast("Any", self.model).get_output_embeddings()
        text_vocab_size = int(output_head.out_features) - 8192
        generate = generator_module.generate_image
        calls_per_step = 2 if cfg_scale > 0 else 1
        route_context = (
            forward_step_routing(
                self.model,
                self.route_module,
                task="generation",
                total_steps=steps,
                calls_per_step=calls_per_step,
            )
            if self.route_module is not None
            else nullcontext()
        )
        with route_context:
            codes = generate(
                self.model,
                conditional,
                seq_len=visual_tokens,
                newline_every=grid_width,
                timesteps=steps,
                temperature=temperature,
                cfg_scale=cfg_scale,
                uncon_ids=unconditional,
                code_start=code_start,
                text_vocab_size=text_vocab_size,
                generator=random_generator,
            )
        decode = image_utils.decode_vq_to_image
        return cast(
            "Image.Image",
            decode(
                codes,
                "",
                "",
                height,
                width,
                vqvae=self.vqvae,
            ),
        )
