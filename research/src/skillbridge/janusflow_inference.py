from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch import nn

from skillbridge.janusflow_runtime import JanusFlowObjective, JanusFlowProcessorProtocol
from skillbridge.routing import diffusion_step_to_bucket
from skillbridge.sidecar import sidecar_route


@dataclass(slots=True)
class JanusFlowInferenceAdapter:
    model: nn.Module
    processor: JanusFlowProcessorProtocol
    vae: nn.Module
    route_module: nn.Module | None
    timestep_buckets: int

    @torch.inference_mode()
    def understand(
        self,
        image_path: Path,
        question: str,
        *,
        max_new_tokens: int = 128,
    ) -> str:
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            prepared = self.processor(
                conversations=[
                    {
                        "role": "User",
                        "content": f"<image_placeholder>\n{question}",
                    },
                    {"role": "Assistant", "content": ""},
                ],
                images=[image],
                force_batchify=True,
            )
        inputs = cast("Any", prepared)
        device = next(self.model.parameters()).device
        inputs = inputs.to(device, dtype=next(self.model.parameters()).dtype)
        model = cast("Any", self.model)
        embeddings = model.prepare_inputs_embeds(**inputs)
        route = (
            sidecar_route(
                self.route_module,
                task="understanding",
                timestep_bucket=0,
                shared_enabled=True,
            )
            if self.route_module is not None
            else torch.no_grad()
        )
        tokenizer = cast("Any", self.processor.tokenizer)
        with route:
            output = model.language_model.generate(
                inputs_embeds=embeddings,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        return cast("str", tokenizer.decode(output[0].cpu().tolist(), skip_special_tokens=True))

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        image_size: int = 384,
        steps: int = 30,
        cfg_scale: float = 5.0,
        seed: int = 0,
        carrier_transform: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        step_carrier_transform: Callable[
            [int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ] | None = None,
    ) -> Image.Image:
        if carrier_transform is not None and step_carrier_transform is not None:
            raise ValueError('choose only one carrier transform interface')
        if image_size < 8 or image_size % 8 or steps < 1:
            msg = "image size must be divisible by eight and steps must be positive"
            raise ValueError(msg)
        processor = cast("Any", self.processor)
        text = processor.apply_sft_template_for_multi_turn_prompts(
            [
                {"role": "User", "content": prompt},
                {"role": "Assistant", "content": ""},
            ],
            sft_format=processor.sft_format,
            system_prompt="",
        )
        token_ids = torch.tensor(processor.tokenizer.encode(text + processor.image_gen_tag))
        device = next(self.model.parameters()).device
        conditional = token_ids.to(device)
        unconditional = conditional.clone()
        unconditional[1:-1] = int(processor.pad_id)
        input_ids = torch.stack((conditional, unconditional))
        attention = torch.ones_like(input_ids)
        attention[1, 1:-1] = 0

        generator = torch.Generator(device=device).manual_seed(seed)
        dtype = next(self.model.parameters()).dtype
        latent_size = image_size // 8
        latent = torch.randn((1, 4, latent_size, latent_size), generator=generator, device=device)
        latent = latent.to(dtype=dtype)
        objective = JanusFlowObjective(
            self.route_module or self.model,
            timestep_buckets=self.timestep_buckets,
        )
        for step in range(steps):
            flow_time = torch.full((2,), step / steps, device=device, dtype=dtype)
            bucket = diffusion_step_to_bucket(
                step,
                total_steps=steps,
                buckets=self.timestep_buckets,
            )
            route = (
                sidecar_route(
                    self.route_module,
                    task="generation",
                    timestep_bucket=bucket,
                    shared_enabled=True,
                )
                if self.route_module is not None
                else torch.no_grad()
            )
            with route:
                paired_latent = torch.cat((latent, latent))
                velocity_pair = objective._generation_velocity(  # noqa: SLF001
                    self.model,
                    input_ids=input_ids,
                    prompt_attention_mask=attention,
                    noisy_latents=paired_latent,
                    flow_time=flow_time,
                    carrier_transform=(
                        (lambda hidden: step_carrier_transform(step, hidden, paired_latent, flow_time))
                        if step_carrier_transform is not None else
                        (lambda hidden: carrier_transform(step, hidden))
                        if carrier_transform is not None else None
                    ),
                )
            conditional_velocity, unconditional_velocity = velocity_pair.chunk(2)
            velocity = cfg_scale * conditional_velocity - (cfg_scale - 1.0) * unconditional_velocity
            latent = latent + velocity / steps

        vae = cast("Any", self.vae)
        scaling = float(getattr(vae.config, "scaling_factor", 1.0))
        decoded = vae.decode(latent / scaling).sample
        array = decoded[0].float().clamp(-1, 1).add(1).mul(127.5)
        array = array.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(array)
