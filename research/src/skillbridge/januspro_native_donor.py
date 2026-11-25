"""Janus-Pro donor changes at the next-image-token prediction interface."""
import numpy as np
import torch
from PIL import Image

from skillbridge.januspro_runtime import generation_prompt


def active_streams(condition):
    streams=('recipient_zero','recipient_full')
    if condition in ('semantic_donor','nuisance_donor'):
        return streams+('donor_zero','donor_full')
    if condition in ('random_sign','conditional_only','unconditional_only','interface_replay'):
        return streams
    raise ValueError('unknown Janus-Pro donor condition')


def intervene_logits(base,full,donor_zero,donor_full,condition,*,sign_seed):
    if condition not in ('semantic_donor','nuisance_donor','random_sign',
                         'conditional_only','unconditional_only','interface_replay'):
        raise ValueError('unknown Janus-Pro donor condition')
    tensors=(base,full,donor_zero,donor_full)
    if any(not isinstance(x,torch.Tensor) or x.shape!=base.shape for x in tensors) or \
            base.ndim!=2 or base.shape[0]!=2:
        raise ValueError('Janus-Pro donor carrier shape differs')
    if any(x.dtype!=base.dtype or x.device!=base.device or not torch.isfinite(x).all() for x in tensors):
        raise ValueError('Janus-Pro donor logits must be finite and share dtype/device')
    reference=full[0].float()-base[0].float();reference_norm=float(reference.norm())
    donor=donor_full[0].float()-donor_zero[0].float();donor_norm=float(donor.norm())
    scale=1.0
    if condition=='interface_replay':changed=full.clone()
    elif condition=='conditional_only':
        changed=base.clone();changed[0]=full[0]
    elif condition=='unconditional_only':
        changed=base.clone();changed[1]=full[1]
    else:
        if condition=='random_sign':
            rng=torch.Generator(device=base.device).manual_seed(sign_seed)
            signs=torch.randint(0,2,reference.shape,device=base.device,generator=rng).mul(2).sub(1)
            delta=reference*signs
        else:
            scale=reference_norm/donor_norm if donor_norm else None
            delta=donor*scale if scale is not None else torch.zeros_like(donor)
        changed=base.clone();changed[0]=(base[0].float()+delta).to(base.dtype)
    actual=float((changed[0].float()-base[0].float()).norm())
    ratio=actual/reference_norm if reference_norm else None
    matched=(abs(ratio-1)<=.2 if ratio is not None else actual==0)
    if condition in ('semantic_donor','nuisance_donor') and scale is None:matched=False
    audit=dict(condition=condition,reference_norm=reference_norm,donor_norm=donor_norm,
               scale=scale,actual_norm=actual,norm_ratio=ratio,norm_matched=matched,
               unconditional_unchanged=bool(torch.equal(changed[1],base[1])),
               interface_replay=condition=='interface_replay')
    return changed,audit


class JanusProDonorGenerator:
    def __init__(self,adapter,full_state):
        self.adapter=adapter;self.full_state=full_state
        self.zero_state={key:torch.zeros_like(value) for key,value in full_state.items()}

    @torch.inference_mode()
    def generate(self,recipient,donor,condition,*,seed,cfg_weight=5.,temperature=1.,
                 image_token_num_per_image=576,sign_seed=1017):
        from skillbridge.readout_probe import set_core_state
        from skillbridge.routing import forward_step_routing
        if image_token_num_per_image!=576 or temperature<=0:
            raise ValueError('Janus-Pro native 576-token generation required')
        adapter=self.adapter;model=adapter.model;processor=adapter.processor;route=adapter.route
        device=next(model.parameters()).device
        def prompt_embeds(prompt):
            ids=processor.tokenizer.encode(generation_prompt(processor,prompt))
            tokens=torch.tensor([ids,ids],device=device,dtype=torch.long)
            tokens[1,1:-1]=processor.pad_id
            return model.language_model.get_input_embeddings()(tokens)
        recipient_embed=prompt_embeds(recipient);donor_embed=prompt_embeds(donor)
        streams={
            'recipient_zero':dict(embeds=recipient_embed,past=None,state=self.zero_state),
            'recipient_full':dict(embeds=recipient_embed,past=None,state=self.full_state),
            'donor_zero':dict(embeds=donor_embed,past=None,state=self.zero_state),
            'donor_full':dict(embeds=donor_embed,past=None,state=self.full_state),
        }
        generated=torch.empty((1,576),dtype=torch.int,device=device)
        random=torch.Generator(device=device).manual_seed(seed)
        audit=[]
        names=active_streams(condition)
        try:
            with forward_step_routing(route,route,task='generation',total_steps=576,
                                      calls_per_step=len(names)):
                for step in range(576):
                    values={}
                    for name in names:
                        stream=streams[name]
                        set_core_state(route,stream['state'])
                        result=model.language_model.model(inputs_embeds=stream['embeds'],
                            use_cache=True,past_key_values=stream['past'])
                        stream['past']=result.past_key_values
                        values[name]=model.gen_head(result.last_hidden_state[:,-1,:])
                    changed,record=intervene_logits(values['recipient_zero'],values['recipient_full'],
                        values.get('donor_zero',values['recipient_zero']),
                        values.get('donor_full',values['recipient_zero']),condition,sign_seed=sign_seed)
                    record['step']=step;audit.append(record)
                    guided=changed[1:2]+cfg_weight*(changed[0:1]-changed[1:2])
                    probs=torch.softmax(guided/temperature,dim=-1)
                    next_token=torch.multinomial(probs,num_samples=1,generator=random)
                    generated[:,step]=next_token[:,0]
                    code=model.prepare_gen_img_embeds(next_token.repeat(2,1)[:,0]).unsqueeze(1)
                    for name in names:streams[name]['embeds']=code
        finally:set_core_state(route,self.zero_state)
        decoded=model.gen_vision_model.decode_code(generated,shape=[1,8,24,24])
        pixels=decoded[0].float().cpu().numpy().transpose(1,2,0)
        pixels=np.clip((pixels+1)*127.5,0,255).astype(np.uint8)
        return Image.fromarray(pixels),audit
