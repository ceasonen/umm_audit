"""DiMOO donor changes on paired masked image-token predictions."""
from types import SimpleNamespace

import torch

from skillbridge.dimoo_inference import MASK,build_generation_tokens


def donor_route(adapter):
    route = adapter.route_module
    if route is None:
        raise ValueError('DiMOO donor requires an attached native route')
    return route


def intervene_mask_logits(base,full,donor_zero,donor_full,condition,*,sign_seed,
                          postcast_calibration=False):
    if condition not in ('semantic_donor','nuisance_donor','random_sign',
                         'conditional_only','unconditional_only','interface_replay'):
        raise ValueError('unknown DiMOO donor condition')
    tensors=(base,full,donor_zero,donor_full)
    if any(not isinstance(x,torch.Tensor) or x.shape!=base.shape for x in tensors) or base.ndim!=2:
        raise ValueError('DiMOO donor masked-logit shape differs')
    if any(x.dtype!=base.dtype or x.device!=base.device or not torch.isfinite(x).all() for x in tensors):
        raise ValueError('DiMOO masked logits must be finite and share dtype/device')
    reference=full.float()-base.float();reference_norm=float(reference.norm())
    donor=donor_full.float()-donor_zero.float();donor_norm=float(donor.norm())
    scale=1.0
    if condition in ('interface_replay','conditional_only'):changed=full.clone()
    elif condition=='unconditional_only':changed=base.clone()
    else:
        if condition=='random_sign':
            rng=torch.Generator(device=base.device).manual_seed(sign_seed)
            signs=torch.randint(0,2,reference.shape,device=base.device,generator=rng).mul(2).sub(1)
            delta=reference*signs
        else:
            scale=reference_norm/donor_norm if donor_norm else None
            delta=donor*scale if scale is not None else torch.zeros_like(donor)
        changed=(base.float()+delta).to(base.dtype)
        if postcast_calibration and condition in ('semantic_donor','nuisance_donor') \
                and scale is not None and reference_norm>0:
            # Native logit quantization can undo float-space norm matching.
            # Recalibrate the same donor direction against the realized norm.
            best=changed;best_scale=scale
            best_error=abs(float((changed.float()-base.float()).norm())/reference_norm-1)
            for _ in range(6):
                realized=float((changed.float()-base.float()).norm())
                if realized==0:scale*=2
                else:scale*=reference_norm/realized
                candidate=(base.float()+donor*scale).to(base.dtype)
                error=abs(float((candidate.float()-base.float()).norm())/reference_norm-1)
                if error<best_error:best,best_scale,best_error=candidate,scale,error
                changed=candidate
                if best_error<=.2:break
            changed=best;scale=best_scale
    actual=float((changed.float()-base.float()).norm())
    ratio=actual/reference_norm if reference_norm else None
    matched=(abs(ratio-1)<=.2 if ratio is not None else actual==0)
    if condition in ('semantic_donor','nuisance_donor') and scale is None:matched=False
    return changed,dict(condition=condition,reference_norm=reference_norm,donor_norm=donor_norm,
                        scale=scale,actual_norm=actual,norm_ratio=ratio,norm_matched=matched)


class DonorModelProxy:
    """Preserve the official MaskGit loop and replace only its masked logits."""
    def __init__(self,model,route,full_state,donor_prompt,recipient_code_start,donor_code_start,
                 donor_initial,condition,vocab_offset,sign_seed,postcast_calibration=False):
        self.module=model;self.route=route;self.full_state=full_state
        self.zero_state={key:torch.zeros_like(value) for key,value in full_state.items()}
        self.donor_initial=donor_initial;self.recipient_code_start=recipient_code_start
        self.donor_code_start=donor_code_start;self.condition=condition
        self.vocab_offset=vocab_offset;self.sign_seed=sign_seed;self.audit=[]
        self.postcast_calibration=postcast_calibration

    def parameters(self):return self.module.parameters()

    def __call__(self,x,**kwargs):
        from skillbridge.readout_probe import set_core_state
        cat=kwargs.get('cat')
        if cat=='uncond':
            state=(self.full_state if self.condition in ('interface_replay','unconditional_only')
                   else self.zero_state)
            set_core_state(self.route,state)
            return self.module(x,**kwargs)
        if cat!='cond':raise ValueError('DiMOO donor requires official CFG branches')
        mask=x==MASK
        donor_x=self.donor_initial.clone()
        donor_x[:,self.donor_code_start-2:]=x[:,self.recipient_code_start-2:]
        donor_mask=donor_x==MASK
        if int(mask.sum())!=int(donor_mask.sum()):raise ValueError('donor masked image positions differ')
        def capture(tokens,state):
            set_core_state(self.route,state)
            result=self.module(tokens,**kwargs).logits
            return result
        base=capture(x,self.zero_state)
        full=capture(x,self.full_state)
        needs_donor=self.condition in ('semantic_donor','nuisance_donor')
        donor_zero=capture(donor_x,self.zero_state) if needs_donor else base
        donor_full=capture(donor_x,self.full_state) if needs_donor else base
        offset=self.vocab_offset;end=offset+8192
        base_mask=base[...,offset:end][mask].view(-1,8192)
        full_mask=full[...,offset:end][mask].view(-1,8192)
        values=(base_mask,full_mask,
                donor_zero[...,offset:end][donor_mask].view(-1,8192) if needs_donor else base_mask,
                donor_full[...,offset:end][donor_mask].view(-1,8192) if needs_donor else base_mask)
        changed,audit=intervene_mask_logits(*values,self.condition,sign_seed=self.sign_seed,
                                           postcast_calibration=self.postcast_calibration)
        audit['unconditional_unchanged']=self.condition not in ('interface_replay','unconditional_only')
        audit['step']=len(self.audit);self.audit.append(audit)
        result=base.clone();view=result[...,offset:end];view[mask]=changed.reshape(-1,8192)
        set_core_state(self.route,self.zero_state)
        return SimpleNamespace(logits=result)


class DimooDonorGenerator:
    def __init__(self,adapter,full_state,postcast_calibration=False):
        self.adapter=adapter;self.full_state=full_state
        self.zero_state={key:torch.zeros_like(value) for key,value in full_state.items()}
        self.postcast_calibration=postcast_calibration

    @torch.inference_mode()
    def generate(self,recipient,donor,condition,*,seed,height=512,width=512,
                 steps=64,cfg_scale=4.,temperature=1.,sign_seed=1017):
        from skillbridge.readout_probe import set_core_state
        from skillbridge.routing import forward_step_routing
        if (height,width,steps,cfg_scale,temperature)!=(512,512,64,4.,1.):
            raise ValueError('registered DiMOO native donor decoder differs')
        adapter=self.adapter;model=adapter.model;route=donor_route(adapter)
        image_utils=adapter._import('utils.image_utils')
        module=adapter._import('generators.image_generation_generator')
        scale=2**(len(adapter.vqvae.config.block_out_channels)-1)
        visual=(height//scale)*(width//scale);grid=width//scale
        recipient_tokens,unconditional,start=build_generation_tokens(adapter.tokenizer,
            prompt=recipient,visual_tokens=visual,grid_width=grid)
        donor_tokens,_,donor_start=build_generation_tokens(adapter.tokenizer,
            prompt=donor,visual_tokens=visual,grid_width=grid)
        device=next(model.parameters()).device
        recipient_tokens=recipient_tokens.to(device);unconditional=unconditional.to(device)
        donor_tokens=donor_tokens.to(device)
        offset=int(model.get_output_embeddings().out_features)-8192
        proxy=DonorModelProxy(model,route,self.full_state,donor,start,donor_start,
                             donor_tokens,condition,offset,sign_seed,self.postcast_calibration)
        rng=torch.Generator(device=device).manual_seed(seed)
        try:
            with forward_step_routing(model,route,task='generation',total_steps=64,
                                      calls_per_step=5 if condition in ('semantic_donor','nuisance_donor') else 3):
                codes=module.generate_image(proxy,recipient_tokens,seq_len=visual,newline_every=grid,
                    timesteps=64,temperature=1.,cfg_scale=4.,uncon_ids=unconditional,
                    code_start=start,text_vocab_size=offset,generator=rng)
        finally:set_core_state(route,self.zero_state)
        image=image_utils.decode_vq_to_image(codes,'','',height,width,vqvae=adapter.vqvae)
        return image,proxy.audit
