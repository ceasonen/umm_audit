from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from skillbridge.objectives import adaptive_weighted_distance, calibration_loss
from skillbridge.route_data import RouteRecord
from skillbridge.sidecar import SidecarTask, sidecars_disabled
from skillbridge.training import SidecarTrainBatch


class JanusFlowTokenizerProtocol(Protocol):
    def encode(self, text: str, **kwargs: object) -> list[int]: ...


class JanusFlowProcessorProtocol(Protocol):
    tokenizer: JanusFlowTokenizerProtocol
    image_gen_tag: str
    sft_format: str

    def apply_sft_template_for_multi_turn_prompts(
        self,
        conversations: list[dict[str, str]],
        *,
        sft_format: str,
        system_prompt: str,
    ) -> str: ...

    def __call__(
        self,
        *,
        conversations: list[dict[str, str]],
        images: list[Image.Image],
        force_batchify: bool,
    ) -> object: ...

    def batchify(self, prepared: list[object]) -> object: ...


@dataclass(frozen=True, slots=True)
class JanusFlowComponents:
    model: nn.Module
    processor: JanusFlowProcessorProtocol
    vae: nn.Module


def load_janusflow_components(
    external_repo: Path,
    *,
    model_id: str,
    model_revision: str,
    vae_id: str,
    vae_revision: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> JanusFlowComponents:
    repo = external_repo.resolve()
    if not (repo / "janus" / "janusflow" / "models" / "modeling_vlm.py").is_file():
        msg = f"official Janus repository is missing at {repo}"
        raise FileNotFoundError(msg)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    dtype = getattr(torch, torch_dtype, None)
    if not isinstance(dtype, torch.dtype):
        msg = f"unknown torch dtype: {torch_dtype}"
        raise ValueError(msg)

    models = importlib.import_module("janus.janusflow.models")
    diffusers = importlib.import_module("diffusers")
    model = models.MultiModalityCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
        revision=model_revision,
    )
    processor = models.VLChatProcessor.from_pretrained(model_id, revision=model_revision)
    vae = diffusers.AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype, revision=vae_revision)
    device = next(model.parameters()).device
    vae = vae.to(device=device, dtype=dtype).eval()
    vae.requires_grad_(requires_grad=False)
    return JanusFlowComponents(model=model, processor=processor, vae=vae)


def _left_pad(rows: Sequence[Tensor], *, value: int) -> Tensor:
    width = max(int(row.numel()) for row in rows)
    output = torch.full((len(rows), width), value, dtype=torch.long)
    for index, row in enumerate(rows):
        output[index, -row.numel() :] = row
    return output


def _common_prefix(left: Tensor, right: Tensor) -> int:
    limit = min(int(left.numel()), int(right.numel()))
    unequal = (left[:limit] != right[:limit]).nonzero()
    return int(unequal[0]) if unequal.numel() else limit


def _understanding_inputs(
    records: Sequence[RouteRecord],
    processor: JanusFlowProcessorProtocol,
) -> Mapping[str, object]:
    prepared: list[object] = []
    label_rows: list[Tensor] = []
    for record in records:
        image_path = Path(record["user_image"])
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            user = {
                "role": "User",
                "content": f"<image_placeholder>\n{record['user_prompt']}",
            }
            prefix = processor(
                conversations=[user, {"role": "Assistant", "content": ""}],
                images=[image],
                force_batchify=False,
            )
            full = processor(
                conversations=[user, {"role": "Assistant", "content": record["answer_text"]}],
                images=[image],
                force_batchify=False,
            )
        prefix_output = cast("Any", prefix)
        full_output = cast("Any", full)
        prefix_ids = cast("Tensor", prefix_output.input_ids)
        full_ids = cast("Tensor", full_output.input_ids)
        labels = full_ids.clone()
        labels[: _common_prefix(prefix_ids, full_ids)] = -100
        prepared.append(full)
        label_rows.append(labels)

    batch = processor.batchify(prepared)
    batch_output = cast("Any", batch)
    input_ids = cast("Tensor", batch_output.input_ids)
    labels = _left_pad(label_rows, value=-100)
    if labels.shape != input_ids.shape:
        msg = "JanusFlow labels do not align with processor input ids"
        raise ValueError(msg)
    return {
        "input_ids": input_ids,
        "pixel_values": batch_output.pixel_values,
        "attention_mask": batch_output.attention_mask,
        "images_seq_mask": batch_output.images_seq_mask,
        "images_emb_mask": batch_output.images_emb_mask,
        "labels": labels,
    }


def _image_tensor(path: Path, *, image_size: int) -> Tensor:
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0)


def _generation_inputs(
    records: Sequence[RouteRecord],
    processor: JanusFlowProcessorProtocol,
    vae: nn.Module,
    *,
    image_size: int,
    stochastic_seed: int,
) -> Mapping[str, object]:
    token_rows: list[Tensor] = []
    images: list[Tensor] = []
    for record in records:
        conversation = [
            {"role": "User", "content": record["user_prompt"]},
            {"role": "Assistant", "content": ""},
        ]
        text = processor.apply_sft_template_for_multi_turn_prompts(
            conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        token_rows.append(torch.tensor(processor.tokenizer.encode(text + processor.image_gen_tag)))
        images.append(_image_tensor(Path(record["answer_image"]), image_size=image_size))

    device = next(vae.parameters()).device
    pixels = torch.stack(images).to(device=device, dtype=next(vae.parameters()).dtype)
    with torch.no_grad():
        encoded = cast("Any", vae).encode(pixels)
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(stochastic_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(stochastic_seed)
            latents = encoded.latent_dist.sample()
    scaling = float(getattr(getattr(vae, "config", object()), "scaling_factor", 1.0))
    attention_rows = [torch.ones_like(row) for row in token_rows]
    return {
        "input_ids": _left_pad(token_rows, value=0),
        "attention_mask": _left_pad(attention_rows, value=0),
        "target_latents": latents * scaling,
    }


def build_janusflow_batches(
    records: Sequence[RouteRecord],
    *,
    processor: JanusFlowProcessorProtocol,
    vae: nn.Module,
    task: SidecarTask,
    batch_size: int,
    timestep_buckets: int,
    image_size: int = 384,
    stochastic_seed: int = 0,
) -> Sequence[SidecarTrainBatch]:
    if batch_size < 1 or timestep_buckets < 1 or image_size < 8:
        msg = "batch size, timestep buckets, and image size must be positive"
        raise ValueError(msg)
    return JanusFlowBatchSequence(
        records,
        processor=processor,
        vae=vae,
        task=task,
        batch_size=batch_size,
        timestep_buckets=timestep_buckets,
        image_size=image_size,
        stochastic_seed=stochastic_seed,
    )


class JanusFlowBatchSequence(Sequence[SidecarTrainBatch]):
    """Materialize image inputs at the optimizer step that consumes them."""

    def __init__(
        self,
        records: Sequence[RouteRecord],
        *,
        processor: JanusFlowProcessorProtocol,
        vae: nn.Module,
        task: SidecarTask,
        batch_size: int,
        timestep_buckets: int,
        image_size: int,
        stochastic_seed: int,
    ) -> None:
        self._records = records
        self._processor = processor
        self._vae = vae
        self._task = task
        self._batch_size = batch_size
        self._timestep_buckets = timestep_buckets
        self._image_size = image_size
        self._stochastic_seed = stochastic_seed

    def __len__(self) -> int:
        return (len(self._records) + self._batch_size - 1) // self._batch_size

    def __getitem__(
        self, index: int | slice
    ) -> SidecarTrainBatch | list[SidecarTrainBatch]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self._batch_size
        chunk = self._records[start : start + self._batch_size]
        if self._task == "understanding":
            inputs = _understanding_inputs(chunk, self._processor)
            bucket = 0
        else:
            inputs = _generation_inputs(
                chunk,
                self._processor,
                self._vae,
                image_size=self._image_size,
                stochastic_seed=self._stochastic_seed + index,
            )
            bucket = index % self._timestep_buckets
        return SidecarTrainBatch(self._task, bucket, inputs)


def _last_hidden_state(output: object) -> Tensor:
    value = output.get("last_hidden_state") if isinstance(output, Mapping) else None
    value = value if isinstance(value, Tensor) else getattr(output, "last_hidden_state", None)
    if not isinstance(value, Tensor):
        msg = "JanusFlow transformer output has no last_hidden_state"
        raise ValueError(msg)
    return value


def _logits(output: object) -> Tensor:
    value = output.get("logits") if isinstance(output, Mapping) else None
    value = value if isinstance(value, Tensor) else getattr(output, "logits", None)
    if not isinstance(value, Tensor):
        msg = "JanusFlow language model output has no logits"
        raise ValueError(msg)
    return value


def _sample_flow_targets(
    target: Tensor,
    *,
    timestep_bucket: int,
    timestep_buckets: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device=target.device)
    generator.manual_seed(seed)
    low = timestep_bucket / timestep_buckets
    flow_time = (
        low
        + torch.rand(
            target.shape[0],
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        / timestep_buckets
    )
    noise = torch.randn(
        target.shape,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    return flow_time, noise


class JanusFlowObjective:
    def __init__(
        self,
        route_module: nn.Module,
        *,
        timestep_buckets: int,
        preservation_weight: float = 0.0,
        temperature: float = 1.0,
        adaptive_preservation_power: float = 0.0,
        adaptive_preservation_epsilon: float = 1e-4,
        stochastic_seed: int = 0,
    ) -> None:
        if timestep_buckets < 1 or preservation_weight < 0:
            msg = "timestep buckets must be positive and preservation weight non-negative"
            raise ValueError(msg)
        self.route_module = route_module
        self.timestep_buckets = timestep_buckets
        self.preservation_weight = preservation_weight
        self.temperature = temperature
        self.adaptive_preservation_power = adaptive_preservation_power
        self.adaptive_preservation_epsilon = adaptive_preservation_epsilon
        self.stochastic_seed = stochastic_seed
        self.generation_call_count = 0

    def _understanding_logits(self, model: nn.Module, inputs: Mapping[str, object]) -> Tensor:
        device = next(model.parameters()).device
        keys = (
            "input_ids",
            "pixel_values",
            "images_seq_mask",
            "images_emb_mask",
        )
        prepared = {key: cast("Tensor", inputs[key]).to(device=device) for key in keys}
        prepared["pixel_values"] = prepared["pixel_values"].to(dtype=next(model.parameters()).dtype)
        janus = cast("Any", model)
        embeddings = janus.prepare_inputs_embeds(**prepared)
        attention_mask = cast("Tensor", inputs["attention_mask"]).to(device=device)
        return _logits(
            janus.language_model(inputs_embeds=embeddings, attention_mask=attention_mask)
        )

    def _generation_velocity(
        self,
        model: nn.Module,
        *,
        input_ids: Tensor,
        prompt_attention_mask: Tensor,
        noisy_latents: Tensor,
        flow_time: Tensor,
        carrier_transform: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        janus = cast("Any", model)
        prompt = janus.language_model.get_input_embeddings()(input_ids)
        if prompt.shape[1] < 2:
            msg = "generation prompt must end in a removable generation tag"
            raise ValueError(msg)
        prompt = prompt[:, :-1]
        prompt_attention_mask = prompt_attention_mask[:, :-1]
        z_emb, t_emb, skips = janus.vision_gen_enc_model(noisy_latents, flow_time * 1000.0)
        height, width = z_emb.shape[-2:]
        visual = z_emb.flatten(2).transpose(1, 2)
        visual = janus.vision_gen_enc_aligner(visual)
        embeddings = torch.cat((prompt, t_emb.unsqueeze(1), visual), dim=1)
        visual_attention = torch.ones(
            (embeddings.shape[0], 1 + visual.shape[1]),
            device=embeddings.device,
            dtype=prompt_attention_mask.dtype,
        )
        attention_mask = torch.cat((prompt_attention_mask, visual_attention), dim=1)
        hidden = _last_hidden_state(
            janus.language_model.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        )
        hidden = hidden[:, -(height * width) :]
        # This interface is visited once per flow step. It carries visual-token
        # states into the frozen decoder aligner; it is not BLIP3-o's static
        # text-conditioning interface. The normal path performs no extra ops.
        if carrier_transform is not None:
            transformed = carrier_transform(hidden)
            if (not isinstance(transformed, Tensor) or transformed.shape != hidden.shape
                    or transformed.dtype != hidden.dtype or transformed.device != hidden.device
                    or not torch.isfinite(transformed).all()):
                raise ValueError('carrier transform must preserve shape, dtype, device, and finite values')
            hidden = transformed
        hidden = janus.vision_gen_dec_aligner_norm(hidden)
        hidden = janus.vision_gen_dec_aligner(hidden)
        hidden = hidden.reshape(hidden.shape[0], height, width, -1).permute(0, 3, 1, 2)
        return cast("Tensor", janus.vision_gen_dec_model(hidden, list(skips), t_emb))

    def __call__(self, model: nn.Module, batch: SidecarTrainBatch) -> Tensor:
        if batch.task == "understanding":
            labels = cast("Tensor", batch.model_inputs["labels"]).to(
                next(model.parameters()).device
            )
            frozen_logits: Tensor | None = None
            if self.preservation_weight:
                with sidecars_disabled(self.route_module), torch.no_grad():
                    frozen_logits = self._understanding_logits(model, batch.model_inputs)
            adapted_logits = self._understanding_logits(model, batch.model_inputs)
            shifted_logits = adapted_logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            task_loss = torch.nn.functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
            if frozen_logits is None:
                return task_loss
            return calibration_loss(
                task_loss,
                adapted_logits=shifted_logits,
                frozen_logits=frozen_logits[:, :-1].contiguous(),
                keep_weight=self.preservation_weight,
                valid_mask=shifted_labels.ne(-100),
                temperature=self.temperature,
                adaptive_keep_power=self.adaptive_preservation_power,
                adaptive_keep_epsilon=self.adaptive_preservation_epsilon,
            )

        device = next(model.parameters()).device
        target = cast("Tensor", batch.model_inputs["target_latents"]).to(device=device)
        input_ids = cast("Tensor", batch.model_inputs["input_ids"]).to(device=device)
        attention_mask = cast("Tensor", batch.model_inputs["attention_mask"]).to(device=device)
        flow_time, noise = _sample_flow_targets(
            target,
            timestep_bucket=batch.timestep_bucket,
            timestep_buckets=self.timestep_buckets,
            seed=self.stochastic_seed + self.generation_call_count,
        )
        self.generation_call_count += 1
        mix = flow_time[:, None, None, None]
        noisy = (1.0 - mix) * noise + mix * target
        velocity_target = target - noise
        frozen_velocity: Tensor | None = None
        if self.preservation_weight:
            with sidecars_disabled(self.route_module), torch.no_grad():
                frozen_velocity = self._generation_velocity(
                    model,
                    input_ids=input_ids,
                    prompt_attention_mask=attention_mask,
                    noisy_latents=noisy,
                    flow_time=flow_time,
                )
        velocity = self._generation_velocity(
            model,
            input_ids=input_ids,
            prompt_attention_mask=attention_mask,
            noisy_latents=noisy,
            flow_time=flow_time,
        )
        task_loss = torch.nn.functional.mse_loss(velocity, velocity_target)
        if frozen_velocity is None:
            return task_loss
        keep = torch.nn.functional.mse_loss(velocity, frozen_velocity)
        return task_loss + self.preservation_weight * adaptive_weighted_distance(
            keep,
            power=self.adaptive_preservation_power,
            epsilon=self.adaptive_preservation_epsilon,
        )
