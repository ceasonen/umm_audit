"""Janus-Pro source-only training with its native image and conversation processor."""
from collections.abc import Sequence
from pathlib import Path
import sys
import torch
from PIL import Image
from skillbridge.training import SidecarTrainBatch


def generation_prompt(processor, prompt):
    conversation = [
        {'role': '<|User|>', 'content': prompt},
        {'role': '<|Assistant|>', 'content': ''},
    ]
    text = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=processor.sft_format, system_prompt='')
    return text + processor.image_start_tag


class JanusProInferenceAdapter:
    def __init__(self, model, processor, route):
        self.model, self.processor, self.route = model, processor, route

    @torch.inference_mode()
    def understand(self, image_path, question, *, max_new_tokens=32):
        from skillbridge.routing import forward_step_routing
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        conversation = [
            {'role': '<|User|>', 'content': '<image_placeholder>\n' + question},
            {'role': '<|Assistant|>', 'content': ''},
        ]
        with Image.open(image_path) as handle:
            image = handle.convert('RGB')
            prepared = self.processor(conversations=conversation, images=[image], force_batchify=True)
        prepared = prepared.to(device, dtype=dtype)
        embeds = self.model.prepare_inputs_embeds(**prepared)
        tokenizer = self.processor.tokenizer
        with forward_step_routing(self.route, self.route, task='understanding', total_steps=max_new_tokens):
            output = self.model.language_model.generate(
                inputs_embeds=embeds, attention_mask=prepared.attention_mask,
                pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True)
        return tokenizer.decode(output[0].detach().cpu().tolist(), skip_special_tokens=True)

    @torch.inference_mode()
    def generate(self, prompt, *, seed, width=384, height=384, cfg_weight=5.0,
                 temperature=1.0, image_token_num_per_image=576):
        from skillbridge.routing import forward_step_routing
        import numpy as np
        if (width, height, image_token_num_per_image) != (384, 384, 576):
            raise ValueError('Janus-Pro native generation requires 384px and 576 image tokens')
        if temperature <= 0:
            raise ValueError('generation temperature must be positive')
        device = next(self.model.parameters()).device
        tokenizer = self.processor.tokenizer
        ids = tokenizer.encode(generation_prompt(self.processor, prompt))
        tokens = torch.tensor([ids, ids], dtype=torch.long, device=device)
        tokens[1, 1:-1] = self.processor.pad_id
        embeds = self.model.language_model.get_input_embeddings()(tokens)
        generated = torch.empty((1, image_token_num_per_image), dtype=torch.int, device=device)
        random = torch.Generator(device=device).manual_seed(seed)
        previous = None
        with forward_step_routing(self.route, self.route, task='generation',
                                  total_steps=image_token_num_per_image):
            for index in range(image_token_num_per_image):
                output = self.model.language_model.model(
                    inputs_embeds=embeds, use_cache=True, past_key_values=previous)
                previous = output.past_key_values
                logits = self.model.gen_head(output.last_hidden_state[:, -1, :])
                guided = logits[1:2] + cfg_weight * (logits[0:1] - logits[1:2])
                probabilities = torch.softmax(guided / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=random)
                generated[:, index] = next_token[:, 0]
                both = next_token.repeat(2, 1)[:, 0]
                embeds = self.model.prepare_gen_img_embeds(both).unsqueeze(1)
        decoded = self.model.gen_vision_model.decode_code(
            generated, shape=[1, 8, height // 16, width // 16])
        pixels = decoded[0].to(torch.float32).cpu().numpy().transpose(1, 2, 0)
        pixels = np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels)


def understanding_inputs(records, processor):
    prepared, labels = [], []
    for record in records:
        if not str(record['answer_text']).strip():
            raise ValueError('Janus-Pro requires a nonempty answer')
        user = dict(role='<|User|>', content='<image_placeholder>\n'+record['user_prompt'])
        with Image.open(record['user_image']) as handle:
            image = handle.convert('RGB')
            prefix = processor(conversations=[user, dict(role='<|Assistant|>', content='')],
                               images=[image], force_batchify=False)
            full = processor(conversations=[user, dict(role='<|Assistant|>', content=record['answer_text'])],
                             images=[image], force_batchify=False)
        n = prefix.input_ids.numel()
        if n >= full.input_ids.numel() or not torch.equal(prefix.input_ids, full.input_ids[:n]):
            raise ValueError('Janus-Pro answer does not preserve the complete prompt prefix')
        y = full.input_ids.clone(); y[:n] = -100
        labels.append(y); prepared.append(full)
    batch = processor.batchify(prepared)
    y = torch.full_like(batch.input_ids, -100)
    for i, row in enumerate(labels):
        y[i, -row.numel():] = row
    y[~batch.attention_mask.bool()] = -100
    return dict(input_ids=batch.input_ids, pixel_values=batch.pixel_values,
                attention_mask=batch.attention_mask, images_seq_mask=batch.images_seq_mask,
                images_emb_mask=batch.images_emb_mask, labels=y)


class JanusProBatches(Sequence):
    def __init__(self, records, processor):
        self.records, self.processor = records, processor
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        if isinstance(index, slice): return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0: index += len(self)
        if index < 0 or index >= len(self): raise IndexError(index)
        return SidecarTrainBatch('understanding', 0,
            understanding_inputs([self.records[index]], self.processor))


def load_components(root, device='cuda'):
    sys.path.insert(0, str(Path(root)/'external/Janus'))
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    path = str(Path(root)/'models/januspro/Janus-Pro-1B')
    processor = VLChatProcessor.from_pretrained(path, local_files_only=True)
    model = MultiModalityCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
        local_files_only=True).to(device).eval()
    return model, processor


def load_training(*, root, records, rank, seed, capability, timestep_buckets,
                  method, image_size, gain_limit, target_suffixes):
    from scripts.train_cross_backbone_pilot import attach_route
    from skillbridge.janusflow_runtime import JanusFlowObjective
    if image_size != 384 or timestep_buckets != 1 or method != 'tied_core':
        raise ValueError('Janus-Pro extension requires native 384px, one bucket, tied_core')
    model, processor = load_components(root)
    model.requires_grad_(False)
    route = model.language_model.model
    attach_route(route, rank=rank, timestep_buckets=1, method=method,
                 gain_limit=gain_limit, target_suffixes=target_suffixes)
    return model, route, JanusProBatches(records, processor), JanusFlowObjective(
        route, timestep_buckets=1, stochastic_seed=seed+1000)
