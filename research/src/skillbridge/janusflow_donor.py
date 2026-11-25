"""Donor changes evaluated at the evolving recipient latent and flow time."""
from __future__ import annotations

import torch


class SameLatentDonor:
    """Inject one conditional update; capture receives the exact current z and t.

    The capture function must restore the frozen core after every evaluation.
    Failed norm matches remain in the audit and never trigger resampling.
    """

    def __init__(self, capture, recipient, donor, condition, *, sign_seed):
        if condition not in ('semantic_donor', 'nuisance_donor', 'random_sign',
                             'conditional_only', 'unconditional_only', 'interface_replay'):
            raise ValueError('unknown donor condition')
        self.capture = capture
        self.recipient = recipient
        self.donor = donor
        self.condition = condition
        self.sign_seed = sign_seed
        self.audit = []

    def __call__(self, step, base, latent, time):
        if step != len(self.audit):
            raise ValueError('carrier step sequence differs')
        if base.ndim != 3 or base.shape[0] != 2 or not torch.isfinite(base).all():
            raise ValueError('finite two-branch carrier required')

        def capture(prompt, state):
            value = self.capture(prompt, state, latent, time)
            if (value.shape != base.shape or value.dtype != base.dtype
                    or value.device != base.device or not torch.isfinite(value).all()):
                raise ValueError('capture must preserve shape, dtype, device and finite values')
            return value

        full = capture(self.recipient, 'full')
        reference = full[0].float() - base[0].float()
        reference_norm = float(reference.norm())
        scale = 1.0
        donor_norm = None
        if self.condition == 'interface_replay':
            changed = full.clone()
        elif self.condition == 'conditional_only':
            changed = base.clone()
            changed[0] = full[0]
        elif self.condition == 'unconditional_only':
            changed = base.clone()
            changed[1] = full[1]
        else:
            if self.condition == 'random_sign':
                generator = torch.Generator(device=base.device).manual_seed(self.sign_seed)
                signs = torch.randint(0, 2, reference.shape, device=base.device,
                                      generator=generator).mul(2).sub(1)
                delta = reference * signs
            else:
                donor_zero = capture(self.donor, 'zero')
                donor_full = capture(self.donor, 'full')
                delta = donor_full[0].float() - donor_zero[0].float()
                donor_norm = float(delta.norm())
                scale = reference_norm / donor_norm if donor_norm else None
                delta = delta * scale if scale is not None else torch.zeros_like(delta)
            changed = base.clone()
            changed[0] = (base[0].float() + delta).to(base.dtype)

        actual_norm = float((changed[0].float() - base[0].float()).norm())
        ratio = actual_norm / reference_norm if reference_norm else None
        matched = (abs(ratio - 1) <= .2 if ratio is not None else actual_norm == 0)
        if self.condition in ('semantic_donor', 'nuisance_donor') and scale is None:
            matched = False
        if not torch.isfinite(changed).all():
            raise ValueError('injected carrier is not finite')
        self.audit.append(dict(step=step,condition=self.condition,reference_norm=reference_norm,
            donor_norm=donor_norm,scale=scale,actual_conditional_update_norm=actual_norm,
            norm_ratio=ratio,norm_matched=matched,
            unconditional_unchanged=bool(torch.equal(changed[1],base[1]))))
        return changed


def model_capture(adapter, route, full_state):
    """Build native captures without changing the outer recipient flow state."""
    from skillbridge.janusflow_runtime import JanusFlowObjective
    from skillbridge.readout_probe import set_core_state

    zero = {k:torch.zeros_like(v) for k,v in full_state.items()}
    objective = JanusFlowObjective(route, timestep_buckets=adapter.timestep_buckets)
    device = next(adapter.model.parameters()).device
    inputs = {}

    @torch.inference_mode()
    def capture(prompt, state, latent, time):
        if state not in ('zero', 'full'):
            raise ValueError('capture state must be zero or full')
        if prompt not in inputs:
            processor = adapter.processor
            text = processor.apply_sft_template_for_multi_turn_prompts(
                [{'role':'User','content':prompt},{'role':'Assistant','content':''}],
                sft_format=processor.sft_format,system_prompt='')
            conditional = torch.tensor(processor.tokenizer.encode(text+processor.image_gen_tag),device=device)
            unconditional = conditional.clone()
            unconditional[1:-1] = int(processor.pad_id)
            ids = torch.stack((conditional,unconditional))
            attention = torch.ones_like(ids)
            attention[1,1:-1] = 0
            inputs[prompt] = ids, attention
        ids, attention = inputs[prompt]
        captured = []
        def observe(hidden):
            captured.append(hidden.detach().clone())
            return hidden
        try:
            set_core_state(route, full_state if state=='full' else zero)
            objective._generation_velocity(adapter.model,input_ids=ids,prompt_attention_mask=attention,
                noisy_latents=latent,flow_time=time,carrier_transform=observe)
        finally:
            set_core_state(route, zero)
        if len(captured)!=1:
            raise ValueError('one native carrier capture required')
        return captured[0]
    return capture
