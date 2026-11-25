"""Native greedy full responses, with answer boundaries and first-step diagnostics."""
from pathlib import Path

import torch
from PIL import Image

from skillbridge.full_answer_audit import generated_span
from skillbridge.sidecar import sidecar_route


@torch.inference_mode()
def generate_answer(adapter, backbone, image_path, question, *, max_new_tokens=32):
    tokenizer = adapter.processor.tokenizer
    device = next(adapter.model.parameters()).device
    with Image.open(Path(image_path)) as handle:
        image = handle.convert('RGB')
    if backbone == 'blip3o':
        template = adapter.processor.apply_chat_template(
            adapter._understanding_messages(image, question), tokenize=False, add_generation_prompt=True)
        inputs = adapter.processor(text=[template], images=[image], videos=None,
                                   padding=True, return_tensors='pt').to(device)
        model, kwargs = adapter.model, dict(inputs)
    elif backbone == 'janusflow':
        inputs = adapter.processor(conversations=[
            {'role': 'User', 'content': f'<image_placeholder>\n{question}'},
            {'role': 'Assistant', 'content': ''}], images=[image], force_batchify=True)
        inputs = inputs.to(device, dtype=next(adapter.model.parameters()).dtype)
        template = inputs.sft_format[0]
        embeddings = adapter.model.prepare_inputs_embeds(**inputs)
        model = adapter.model.language_model
        kwargs = dict(inputs_embeds=embeddings, attention_mask=inputs.attention_mask,
                      pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                      eos_token_id=tokenizer.eos_token_id)
    else:
        raise ValueError('unsupported backbone')
    configuration = model.generation_config.to_dict()
    eos = kwargs.get('eos_token_id', configuration.get('eos_token_id'))
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    with sidecar_route(adapter.route_module, task='understanding', timestep_bucket=0):
        output = model.generate(**kwargs, max_new_tokens=max_new_tokens, do_sample=False,
                                num_beams=1, use_cache=True, return_dict_in_generate=True, output_scores=True)
    if output.sequences.shape[0] != 1 or not output.scores:
        raise ValueError('expected one nonempty greedy answer')
    span = generated_span(output.sequences[0].tolist(), len(output.scores), eos_ids, max_new_tokens)
    scores = output.scores[0][0].float()
    if int(scores.argmax()) != span['generated_token_ids'][0]:
        raise ValueError('first generated token disagrees with recorded score')
    probabilities = torch.log_softmax(scores, dim=-1)
    values, indices = probabilities.topk(8)
    if not torch.isfinite(values).all():
        raise ValueError('invalid first-step score distribution')
    input_ids = inputs['input_ids'][0].tolist()
    return dict(**span, raw_answer=tokenizer.decode(span['generated_token_ids'], skip_special_tokens=True),
                raw_answer_with_special_tokens=tokenizer.decode(span['generated_token_ids'], skip_special_tokens=False),
                max_new_tokens=max_new_tokens, eos_token_ids=eos_ids,
                template=template, input_token_count=len(input_ids),
                input_boundary_token_ids=input_ids[-64:],
                input_boundary_text=tokenizer.decode(input_ids[-64:], skip_special_tokens=False),
                first_step_top=[dict(token_id=i, text=tokenizer.decode([i]), log_probability=v)
                                for i, v in zip(indices.tolist(), values.tolist())],
                generation_configuration=configuration,
                generation_overrides=dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1),
                score_scope='processed generation scores, not raw forward logits or sequence likelihood')
