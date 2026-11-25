"""Show-o2 donor changes at paired raw velocity branches before CFG."""
import numpy as np
import torch
from PIL import Image


def intervene_velocity(base,full,donor_zero,donor_full,condition,*,sign_seed):
    if condition not in ('semantic_donor','nuisance_donor','random_sign',
                         'conditional_only','unconditional_only','interface_replay'):
        raise ValueError('unknown Show-o2 donor condition')
    tensors=(base,full,donor_zero,donor_full)
    if any(not isinstance(x,torch.Tensor) or x.shape!=base.shape for x in tensors) or \
            base.ndim!=4 or base.shape[0]!=2:
        raise ValueError('Show-o2 donor carrier shape differs')
    if any(x.dtype!=base.dtype or x.device!=base.device or not torch.isfinite(x).all() for x in tensors):
        raise ValueError('Show-o2 donor velocity must be finite and share dtype/device')
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
    return changed,dict(condition=condition,reference_norm=reference_norm,donor_norm=donor_norm,
                        scale=scale,actual_norm=actual,norm_ratio=ratio,norm_matched=matched,
                        unconditional_unchanged=bool(torch.equal(changed[1],base[1])))


class Showo2DonorGenerator:
    def __init__(self,adapter,full_state):
        self.adapter=adapter;self.full_state=full_state
        self.zero_state={key:torch.zeros_like(value) for key,value in full_state.items()}

    @torch.inference_mode()
    def generate(self,recipient,donor,condition,*,seed,guidance_scale=7.5,
                 num_inference_steps=50,sign_seed=1017):
        from models import omni_attn_mask_naive
        from models.misc import prepare_gen_input
        from skillbridge.readout_probe import set_core_state
        from skillbridge.routing import forward_step_routing
        from skillbridge.vendor.showo2_transport import Sampler,create_transport
        if num_inference_steps!=50:raise ValueError('native 50-step Show-o2 protocol required')
        adapter=self.adapter;model=adapter.model;route=adapter.route
        device=next(model.parameters()).device;ids=adapter.ids;tokenizer=adapter.tokenizer
        def input_batch(prompt):
            text,null,positions,null_positions=prepare_gen_input([prompt],tokenizer,730,
                ids['bos_id'],ids['eos_id'],ids['boi_id'],ids['eoi_id'],
                tokenizer.pad_token_id,ids['img_pad_id'],290,device)
            text=torch.cat([text,null],dim=0)
            positions=torch.cat([positions,null_positions],dim=0)
            mask=omni_attn_mask_naive(2,1024,positions,device).to(torch.bfloat16)
            return dict(text_tokens=text,modality_positions=positions,attention_mask=mask)
        inputs={'recipient':input_batch(recipient),'donor':input_batch(donor)}
        rng=torch.Generator(device=device).manual_seed(seed)
        latent=torch.randn((1,16,54,54),device=device,dtype=torch.bfloat16,generator=rng)
        latent=torch.cat([latent,latent],dim=0)
        transport=create_transport(path_type='Linear',prediction='velocity',loss_weight=None,
            train_eps=None,sample_eps=None,snr_type='lognorm',do_shift=True,seq_len=730)
        sample=Sampler(transport).sample_ode(sampling_method='euler',num_steps=50,
            atol=1e-6,rtol=1e-3,reverse=False,time_shifting_factor=3.)
        audit=[]
        needs_donor=condition in ('semantic_donor','nuisance_donor')
        def raw_velocity(x,t,prompt,state):
            set_core_state(route,state)
            adjusted=t.clone()
            if adjusted.shape[-1]!=2:
                t_cond,t_uncond=torch.chunk(adjusted,2)
                t_cond[:-1]=1.;t_uncond[:-1]=1.
                adjusted=torch.cat([t_cond,t_uncond])
            _,velocity=model(inputs[prompt]['text_tokens'],image_latents=x,t=adjusted,
                attention_mask=inputs[prompt]['attention_mask'],
                modality_positions=inputs[prompt]['modality_positions'],
                guidance_scale=guidance_scale,output_hidden_states=True,max_seq_len=1024)
            return velocity
        def donor_velocity(x,t,**_kwargs):
            base=raw_velocity(x,t,'recipient',self.zero_state)
            full=raw_velocity(x,t,'recipient',self.full_state)
            donor_zero=raw_velocity(x,t,'donor',self.zero_state) if needs_donor else base
            donor_full=raw_velocity(x,t,'donor',self.full_state) if needs_donor else base
            changed,record=intervene_velocity(base,full,donor_zero,donor_full,
                                               condition,sign_seed=sign_seed)
            record['step']=len(audit);audit.append(record)
            conditional,unconditional=torch.chunk(changed,2)
            guided=unconditional+guidance_scale*(conditional-unconditional)
            return torch.cat([guided,guided],dim=0)
        try:
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                with forward_step_routing(route,route,task='generation',total_steps=50,
                                          calls_per_step=4 if needs_donor else 2):
                    generated=sample(latent,donor_velocity)[-1]
        finally:set_core_state(route,self.zero_state)
        image=adapter.vae.batch_decode(torch.chunk(generated,2)[0].float().unsqueeze(2)).squeeze(2)
        pixels=np.clip((image[0].float().cpu().permute(1,2,0).numpy()+1)*127.5,0,255).astype(np.uint8)
        return Image.fromarray(pixels),audit
