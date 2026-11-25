"""Explicit image framing and answer boundaries for new DiMOO diagnostics.

Historical adapter behavior remains unchanged. This module records both the
full generated span and its first protocol-delimited answer.
"""


def inspect_image_tokens(tokens):
    if not tokens or tokens[0]!=126349 or tokens[-1]!=126350:
        raise ValueError('image must have exactly one complete wrapper')
    if tokens.count(126349)!=1 or tokens.count(126350)!=1:
        raise ValueError('duplicated image wrapper')
    widths=[];width=0
    for token in tokens[1:-1]:
        if token==126084:
            if width==0: raise ValueError('empty visual-token row')
            widths.append(width);width=0
        elif 126356<=token<126356+8192:
            width+=1
        else:
            raise ValueError('unexpected token inside image')
    if width or not widths or len(set(widths))!=1:
        raise ValueError('image row framing or grid width differs')
    return dict(rows=len(widths),columns=widths[0],visual_tokens=sum(widths),total_tokens=len(tokens))


def answer_span(tokens,*,generation_length):
    if len(tokens)!=generation_length or generation_length<1:
        raise ValueError('generated span length differs')
    raw=list(tokens)
    stop=next((i for i,t in enumerate(raw) if t in (126355,126081,126339)),None)
    answer=raw if stop is None else raw[:stop]
    return dict(raw_tokens=raw,answer_tokens=answer,stop_id=None if stop is None else raw[stop],
                stop_index=stop,tail_tokens=[] if stop is None else raw[stop+1:],
                unresolved_answer_masks=answer.count(126336))


def diagnostic_cases():
    return [dict(name='historical',system='historical',image_size=None),
            dict(name='official_prompt',system='official',image_size=None),
            dict(name='official_prompt_crop1024',system='official',image_size=1024),
            dict(name='official_prompt_crop256',system='official',image_size=256),
            dict(name='training_prompt_crop256',system='training',image_size=256)]


def generate_single_block(model,prompt,*,code_start,generation_length=128,steps=128):
    """Official greedy low-confidence schedule, restricted to one complete block.

    No CFG, temperature, or multi-block early return. The official one-block
    EOS branch references undefined variables after all scheduled steps;
    this implementation returns the completed sequence directly.
    """
    import torch
    if steps<1 or steps>generation_length or prompt.ndim!=2 or code_start<1:
        raise ValueError('invalid fixed single-block generation settings')
    if prompt.shape[1]!=code_start+generation_length+1:
        raise ValueError('answer span or fixed ending differs')
    answer=prompt[:,code_start:code_start+generation_length]
    if not answer.eq(126336).all() or prompt[:,:code_start].eq(126336).any() or not prompt[:,-1].eq(126355).all():
        raise ValueError('only the answer span may be masked')
    with torch.inference_mode():
        x=prompt.clone()
        base,remainder=divmod(generation_length,steps)
        for step in range(steps):
            mask=x.eq(126336)
            logits=model(x,infer=True).logits
            predicted=logits.argmax(dim=-1)
            # Frozen visual/text positions never compete for answer updates.
            # Preserve native float64 probabilities and full-length top-k ties.
            end=code_start+generation_length
            probabilities=torch.softmax(logits[:,code_start:end].to(torch.float64),dim=-1)
            confidence=torch.full(x.shape,-torch.inf,dtype=torch.float64,device=x.device)
            confidence[:,code_start:end]=torch.gather(
                probabilities,-1,predicted[:,code_start:end].unsqueeze(-1)).squeeze(-1)
            predicted=torch.where(mask,predicted,x)
            confidence=torch.where(mask,confidence,-torch.inf)
            count=base+int(step<remainder)
            selected=torch.zeros_like(mask)
            for row in range(x.shape[0]):
                indices=torch.topk(confidence[row],k=count).indices
                selected[row,indices]=True
            x[selected]=predicted[selected]
            del logits,probabilities,confidence,predicted,selected
        return x
