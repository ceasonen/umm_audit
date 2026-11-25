"""Show-o2 reference-core adapter; native 432px VAE and Qwen chat sequence."""
from collections.abc import Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import torch
from PIL import Image
from skillbridge.training import SidecarTrainBatch


@contextmanager
def native_sdpa_import():
    """Import official modules without compiling the unused BlockMask branch.

    This adapter supplies dense masks, selecting official SDPA. Restore the
    compiler immediately, including on import errors; no inference is patched.
    """
    original = torch.compile
    try:
        torch.compile = lambda function, *args, **kwargs: function
        yield
    finally:
        torch.compile = original


def understanding_tokens(question, answer, tokenizer, ids, image_tokens=730):
    if not str(answer).strip(): raise ValueError('nonempty answer required')
    encode=lambda s: tokenizer(s,add_special_tokens=False)['input_ids']
    a=[ids['bos_id']]+encode('system\nYou are a helpful assistant.<|im_end|>')+encode('\n<|im_start|>user\n')
    prefix=a+[ids['boi_id']]+[ids['img_pad_id']]*image_tokens+[ids['eoi_id']]+encode(question)+encode('\n<|im_start|>assistant\n')
    target=encode(answer)+[tokenizer.eos_token_id]
    return dict(text_tokens=torch.tensor([prefix+target]),
                text_labels=torch.tensor([[-100]*len(prefix)+target]),
                modality_positions=torch.tensor([[[len(a)+1,image_tokens]]]))


def load_components(root, device='cuda'):
    root=Path(root);sys.path.insert(0,str(root/'external/Show-o2'))
    with native_sdpa_import():
        from models import Showo2Qwen2_5, WanVAE
    from models.misc import get_text_tokenizer
    directory=root/'models/showo2'
    config=json.loads((directory/'show-o2-1.5B/config.json').read_text())
    config={k:v for k,v in config.items() if not k.startswith('_')}
    config['llm_model_path']=str(directory/'Qwen2.5-1.5B-Instruct')
    config['clip_pretrained_model_path']=str(directory/'siglip-so400m-patch14-384')
    model=Showo2Qwen2_5(**config)
    weights=torch.load(directory/'show-o2-1.5B/pytorch_model.bin',map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True);del weights
    model=model.to(device=device,dtype=torch.float32).eval()
    tokenizer, ids=get_text_tokenizer(config['llm_model_path'],return_showo_token_ids=True,llm_name='qwen2_5')
    if len(tokenizer)!=config['llm_vocab_size']:raise ValueError('native vocabulary mismatch')
    vae=WanVAE(vae_pth=str(directory/'wan/Wan2.1_VAE.pth'),dtype=torch.float32,device=device)
    return model,tokenizer,ids,vae


class Showo2Batches(Sequence):
    def __init__(self,records,tokenizer,ids,vae,seed):
        self.records,self.tokenizer,self.ids,self.vae,self.seed=records,tokenizer,ids,vae,seed
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        from torchvision import transforms
        from models import omni_attn_mask_naive
        r=self.records[index]
        tokens=understanding_tokens(r['user_prompt'],r['answer_text'],self.tokenizer,self.ids)
        device=torch.device(self.vae.device)
        with Image.open(r['user_image']) as handle:
            image=transforms.Compose([transforms.Resize(432,interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((432,432)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])(handle.convert('RGB'))
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed+index)
            latents=self.vae.sample(image[None,:,None].to(device)).squeeze(2)
        tokens={k:v.to(device) for k,v in tokens.items()}
        mask=omni_attn_mask_naive(1,tokens['text_tokens'].shape[1],tokens['modality_positions'],device,inverted=True).float()
        return SidecarTrainBatch('understanding',0,dict(tokens,image_latents=latents,
            t=torch.ones(1,1,device=device),attention_mask=mask))


class Showo2Objective:
    def __call__(self,model,batch):
        if batch.task!='understanding':raise ValueError('source-only objective')
        return model.forward_und_only(**batch.model_inputs)[1]


class Showo2InferenceAdapter:
    """Official 432px Show-o2 input paths with the trained language route active."""

    def __init__(self, model, tokenizer, ids, vae, route):
        self.model, self.tokenizer, self.ids = model, tokenizer, ids
        self.vae, self.route = vae, route

    @torch.inference_mode()
    def understand(self, image_path, question, *, max_new_tokens=32):
        from torchvision import transforms
        from models import omni_attn_mask_naive
        from skillbridge.routing import forward_step_routing
        device = next(self.model.parameters()).device
        transform = transforms.Compose([
            transforms.Resize(432, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((432, 432)), transforms.ToTensor(),
            transforms.Normalize([.5] * 3, [.5] * 3)])
        with Image.open(image_path) as handle:
            image = transform(handle.convert('RGB'))[None, :, None].to(device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20260924)
            latents = self.vae.sample(image).squeeze(2)
        model = self.model
        image_und = model.image_embedder_und(latents)
        image_gen = model.image_embedder_gen(latents)
        image_und = image_und + model.position_embedding(model.image_position_ids)
        image_und = model.und_trans(image_und)['last_hidden_state']
        image_embeds = model.fusion_proj(torch.cat([image_und, image_gen], dim=-1))
        encode = lambda value: self.tokenizer(value, add_special_tokens=False)['input_ids']
        system = encode('system\nYou are a helpful assistant.<|im_end|>')
        user = encode('\n<|im_start|>user\n')
        assistant = encode('\n<|im_start|>assistant\n')
        a = torch.tensor([[self.ids['bos_id']] + system + user], device=device)
        b = torch.tensor([[self.ids['boi_id'], self.ids['eoi_id']] + encode(question) + assistant], device=device)
        embed = model.showo.model.embed_tokens
        start, end = embed(a), embed(b)
        if model.config.add_time_embeds:
            time = model.time_embed(torch.tensor([[1.0]], device=device), start.dtype)
            if hasattr(model, 'time_embed_proj'):
                time = model.time_embed_proj(time)
            input_embeds = torch.cat([start, end[:, :1], time, image_embeds, end[:, 1:]], dim=1)
            position = a.shape[1] + 2
            image_tokens = 730
        else:
            input_embeds = torch.cat([start, end[:, :1], image_embeds, end[:, 1:]], dim=1)
            position = a.shape[1] + 1
            image_tokens = 729
        modality = torch.tensor([[[position, image_tokens]]], device=device)
        mask = omni_attn_mask_naive(1, input_embeds.shape[1], modality,
                                    device, inverted=True).to(input_embeds.dtype)
        with forward_step_routing(self.route, self.route, task='understanding',
                                  total_steps=max_new_tokens):
            result = model.mmu_generate(input_embeds=input_embeds, attention_mask=mask,
                                        top_k=1, max_new_tokens=max_new_tokens,
                                        eos_token=self.tokenizer.eos_token_id)
        if not result:
            return ''
        tokens = torch.stack(result).reshape(1, -1)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def generate(self, prompt, *, seed, width=432, height=432,
                 guidance_scale=7.5, num_inference_steps=50):
        import numpy as np
        from models import omni_attn_mask_naive
        from models.misc import prepare_gen_input
        from skillbridge.routing import forward_step_routing
        from skillbridge.vendor.showo2_transport import Sampler, create_transport
        if (width, height, num_inference_steps) != (432, 432, 50):
            raise ValueError('Show-o2 official 432px / 50-step protocol required')
        device = next(self.model.parameters()).device
        ids = self.ids
        tokens = prepare_gen_input([prompt], self.tokenizer, 730,
            ids['bos_id'], ids['eos_id'], ids['boi_id'], ids['eoi_id'],
            self.tokenizer.pad_token_id, ids['img_pad_id'], 290, device)
        text, null, positions, null_positions = tokens
        text = torch.cat([text, null], dim=0)
        positions = torch.cat([positions, null_positions], dim=0)
        mask = omni_attn_mask_naive(2, 1024, positions, device).to(torch.bfloat16)
        random = torch.Generator(device=device).manual_seed(seed)
        latent = torch.randn((1, 16, 54, 54), generator=random,
                             device=device, dtype=torch.bfloat16)
        latent = torch.cat([latent, latent], dim=0)
        transport = create_transport(path_type='Linear', prediction='velocity',
            loss_weight=None, train_eps=None, sample_eps=None,
            snr_type='lognorm', do_shift=True, seq_len=730)
        sampler = Sampler(transport)
        sample_fn = sampler.sample_ode(sampling_method='euler', num_steps=50,
            atol=1e-6, rtol=1e-3, reverse=False, time_shifting_factor=3.0)
        kwargs = dict(text_tokens=text, attention_mask=mask,
                      modality_positions=positions, output_hidden_states=True,
                      max_seq_len=1024, guidance_scale=guidance_scale)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with forward_step_routing(self.route, self.route, task='generation', total_steps=50):
                generated = sample_fn(latent, self.model.t2i_generate, **kwargs)[-1]
        image = self.vae.batch_decode(torch.chunk(generated, 2)[0].float().unsqueeze(2)).squeeze(2)
        array = np.clip((image[0].float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5,
                        0, 255).astype(np.uint8)
        return Image.fromarray(array)


def load_training(*,root,records,rank,seed,capability,timestep_buckets,method,image_size,gain_limit,target_suffixes):
    from scripts.train_cross_backbone_pilot import attach_route
    if (image_size,timestep_buckets,method)!=(432,1,'tied_core'):
        raise ValueError('native 432px, one bucket, tied_core required')
    model,tokenizer,ids,vae=load_components(root)
    model.requires_grad_(False)
    route=model.showo.model
    attach_route(route,rank=rank,timestep_buckets=1,method=method,gain_limit=gain_limit,target_suffixes=target_suffixes)
    return model,route,Showo2Batches(records,tokenizer,ids,vae,seed),Showo2Objective()
