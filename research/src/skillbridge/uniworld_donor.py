"""Norm-matched donor changes at UniWorld's native Qwen-to-FLUX interface."""
from difflib import SequenceMatcher
import torch


def align_donor_change(delta, donor_token_ids, recipient_token_ids):
    """Map donor hidden positions onto the recipient's tokenized chat sequence."""
    if (delta.ndim!=3 or delta.shape[0]!=1 or delta.shape[1]!=len(donor_token_ids)
            or not donor_token_ids or not recipient_token_ids):
        raise ValueError('donor token ids must match native carrier length')
    output=torch.zeros((1,len(recipient_token_ids),delta.shape[2]),device=delta.device,dtype=delta.dtype)
    matcher=SequenceMatcher(None,list(donor_token_ids),list(recipient_token_ids),autojunk=False)
    for operation,i0,i1,j0,j1 in matcher.get_opcodes():
        if operation=='equal':
            output[:,j0:j1]=delta[:,i0:i1]
        elif j1>j0:
            source=list(range(i0,i1))
            if not source:
                source=[min(max(i0,0),delta.shape[1]-1)]
            for j in range(j0,j1):
                index=source[min(len(source)-1,int((j-j0)*len(source)/(j1-j0)))]
                output[:,j]=delta[:,index]
        elif i1>i0:
            index=min(j0,len(recipient_token_ids)-1)
            output[:,index]+=delta[:,i0:i1].mean(dim=1)
    return output


def matched_change(recipient_zero, recipient_full, donor_zero, donor_full,
                   *,recipient_token_ids=None,donor_token_ids=None):
    tensors=(recipient_zero,recipient_full,donor_zero,donor_full)
    if any(not isinstance(x,torch.Tensor) for x in tensors) or \
            recipient_full.shape!=recipient_zero.shape or donor_full.shape!=donor_zero.shape or \
            any(x.ndim!=3 or x.shape[0]!=recipient_zero.shape[0] or
                x.shape[2]!=recipient_zero.shape[2] for x in tensors):
        raise ValueError(f'native donor carrier shape differs: {[tuple(x.shape) for x in tensors]}')
    if any(x.dtype!=recipient_zero.dtype or x.device!=recipient_zero.device or
           not torch.isfinite(x).all() for x in tensors):
        raise ValueError('native donor carrier dtype, device or finiteness differs')
    reference=(recipient_full.float()-recipient_zero.float())
    donor=(donor_full.float()-donor_zero.float())
    raw_donor_norm=float(donor.norm())
    if recipient_token_ids is not None or donor_token_ids is not None:
        if recipient_token_ids is None or donor_token_ids is None:
            raise ValueError('both token sequences required for donor alignment')
        donor=align_donor_change(donor,donor_token_ids,recipient_token_ids)
        alignment='token'
    elif donor.shape!=reference.shape:
        raise ValueError('native donor carrier shape differs without token alignment')
    else:alignment='position'
    reference_norm=float(reference.norm());donor_norm=float(donor.norm())
    scale=reference_norm/donor_norm if donor_norm else None
    changed=(recipient_zero.float()+donor*scale).to(recipient_zero.dtype) if scale is not None else recipient_zero.clone()
    actual_norm=float((changed.float()-recipient_zero.float()).norm())
    ratio=actual_norm/reference_norm if reference_norm else None
    audit=dict(reference_norm=reference_norm,donor_norm=donor_norm,scale=scale,
               actual_norm=actual_norm,norm_ratio=ratio,
               raw_donor_norm=raw_donor_norm,alignment=alignment,
               donor_tokens=donor_zero.shape[1],recipient_tokens=recipient_zero.shape[1],
               norm_matched=bool(scale is not None and ratio is not None and abs(ratio-1)<=.2))
    return changed,audit
