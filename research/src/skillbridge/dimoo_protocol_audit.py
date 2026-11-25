"""Read-only decomposition of the native masked-token objective."""
import torch
from torch.nn import functional as F


def partition_loss(logits, inputs, labels, *, mask_id=126336, pad_id=126339, end_id=126355):
    if logits.ndim != 3 or inputs.shape != labels.shape or logits.shape[:2] != labels.shape:
        raise ValueError('logits, inputs and labels must have matching batch/sequence dimensions')
    valid = labels.ne(-100)
    if not valid.any():
        raise ValueError('no supervised tokens')
    masks = dict(all=valid, answer=valid & labels.ne(pad_id) & labels.ne(end_id),
                 padding=valid & labels.eq(pad_id), end=valid & labels.eq(end_id))
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        total = correct = 0
        # Bound temporary float32 allocations for the 126k+ vocabulary.
        indices = mask.flatten().nonzero().flatten()
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.flatten()
        for chunk in indices.split(16):
            if not len(chunk):
                continue
            selected = flat_logits[chunk].float()
            if not torch.isfinite(selected).all():
                raise ValueError('nonfinite supervised logits')
            target = flat_labels[chunk]
            total += float(F.cross_entropy(selected, target, reduction='sum'))
            correct += int(selected.argmax(-1).eq(target).sum())
        result[name] = dict(tokens=count, mean_ce=total/count if count else None,
            accuracy=correct/count if count else None,
            unmasked_supervised=int((mask & inputs.ne(mask_id)).sum()))
    return result
