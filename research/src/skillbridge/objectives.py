from __future__ import annotations

import torch
from torch import Tensor


def adaptive_weighted_distance(
    distance: Tensor,
    *,
    power: float = 0.0,
    epsilon: float = 1e-4,
) -> Tensor:
    """Scale a scalar distance with a detached inverse-error weight.

    ``power=0`` recovers the original distance exactly. The normalized,
    detached weight stays at most one, so reweighting cannot amplify a small
    preservation gradient.
    """
    if power < 0:
        msg = "adaptive preservation power must be non-negative"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = "adaptive preservation epsilon must be positive"
        raise ValueError(msg)
    if power == 0:
        return distance
    normalized_distance = distance.detach() / epsilon
    weight = (1.0 + normalized_distance).pow(-power)
    return weight * distance


def preservation_kl(
    adapted_logits: Tensor,
    frozen_logits: Tensor,
    *,
    valid_mask: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    if adapted_logits.shape != frozen_logits.shape:
        msg = "adapted and frozen logits must have the same shape"
        raise ValueError(msg)
    if temperature <= 0:
        msg = "temperature must be positive"
        raise ValueError(msg)
    student = torch.log_softmax(adapted_logits / temperature, dim=-1)
    teacher = torch.softmax(frozen_logits.detach() / temperature, dim=-1)
    token_kl = torch.sum(teacher * (torch.log(teacher.clamp_min(1e-12)) - student), dim=-1)
    if valid_mask is not None:
        if valid_mask.shape != token_kl.shape:
            msg = "valid_mask must match the non-vocabulary logit dimensions"
            raise ValueError(msg)
        selected = token_kl[valid_mask]
        if selected.numel() == 0:
            return adapted_logits.sum() * 0.0
        token_kl = selected
    return token_kl.mean() * temperature**2


def calibration_loss(
    task_loss: Tensor,
    *,
    adapted_logits: Tensor,
    frozen_logits: Tensor,
    keep_weight: float,
    valid_mask: Tensor | None = None,
    temperature: float = 1.0,
    adaptive_keep_power: float = 0.0,
    adaptive_keep_epsilon: float = 1e-4,
) -> Tensor:
    if keep_weight < 0:
        msg = "keep_weight must be non-negative"
        raise ValueError(msg)
    keep = preservation_kl(
        adapted_logits,
        frozen_logits,
        valid_mask=valid_mask,
        temperature=temperature,
    )
    return task_loss + keep_weight * adaptive_weighted_distance(
        keep,
        power=adaptive_keep_power,
        epsilon=adaptive_keep_epsilon,
    )
