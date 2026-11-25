"""Answer-only supervision after multimodal token expansion."""
import torch


def encode_answer_training(processor, process_vision_info, record):
    if not str(record['answer_text']).strip():
        raise ValueError('empty training answer')
    user = {'role': 'user', 'content': [
        {'type': 'image', 'image': str(record['user_image'])},
        {'type': 'text', 'text': str(record['user_prompt'])}]}
    messages = [user, {'role': 'assistant', 'content': str(record['answer_text'])}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    kwargs = dict(images=images, videos=videos, padding=True, return_tensors='pt')
    encoded = processor(text=[text], **kwargs)
    prefix = processor(text=[prefix_text], **kwargs)
    ids, prefix_ids = encoded['input_ids'], prefix['input_ids']
    boundary = prefix_ids.shape[1]
    if ids.shape[0] != 1 or prefix_ids.shape[0] != 1 or boundary >= ids.shape[1]:
        raise ValueError('expected one expanded input followed by a nonempty answer')
    if not torch.equal(ids[:, :boundary], prefix_ids):
        raise ValueError('multimodal prefix does not match the training input')
    labels = ids.clone()
    labels[:, :boundary] = -100
    if 'attention_mask' in encoded:
        labels.masked_fill_(encoded['attention_mask'].eq(0), -100)
    if not labels.ne(-100).any():
        raise ValueError('no supervised answer tokens')
    encoded['labels'] = labels
    return encoded
