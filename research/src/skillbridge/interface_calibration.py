"""Frozen prompt contrasts at the learned intervention's native budget."""


def cached_readout_values(pairs, keys, values, role):
    """Old official stages retain original IDs; noise follows registered order."""
    if role not in ('recipient', 'donor') or len(keys) != len(pairs) or len(values) != len(keys):
        raise ValueError('cached readout identity differs')
    indexed = {int(k[1]): (float(v), k) for k, v in zip(keys, values, strict=True)}
    expected = [int(p['recipient']['benchmark_id']) for p in pairs]
    if len(indexed) != len(keys) or len(set(expected)) != len(expected) or set(indexed) != set(expected):
        raise ValueError('cached readout identity differs')
    ordered = []
    for i, pair in enumerate(pairs):
        value, key = indexed[expected[i]]
        prompt = pair['recipient' if role == 'recipient' else 'semantic_donor']['user_prompt']
        if key[2] != prompt or key[3] != 8087 + i:
            raise ValueError('cached readout prompt/noise identity differs')
        ordered.append(value)
    return ordered


def validate_generated_identity(row, pair, index, seed, arm):
    prompt = pair['semantic_donor' if arm == 'natural_donor' else 'recipient']['user_prompt']
    expected = dict(split_key=pair['recipient']['split_key'], prompt=prompt,
                    generation_prompt=prompt, seed=8087 + index, condition=arm,
                    pair_index=index, source_seed=seed, image=f'images/{index:05d}.png')
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('generated prediction identity differs')

def prompt_contrast_state(zero, full, donor_zero):
    import torch
    values = (zero, full, donor_zero)
    if (zero.ndim != 3 or zero.shape[0] != 2 or any(
            v.shape != zero.shape or v.dtype != zero.dtype or not torch.isfinite(v).all()
            for v in values)):
        raise ValueError('expected matching finite conditional/unconditional interfaces')
    reference = (full[0].float() - zero[0].float()).square().sum().item()
    delta = donor_zero[0].float() - zero[0].float()
    energy = delta.square().sum().item()
    if reference <= 0 or energy <= 0:
        raise ValueError('reference and prompt contrast require positive energy')
    scale = (reference / energy) ** .5
    state = zero.clone()
    state[0] = (zero[0].float() + scale * delta).to(zero.dtype)
    realized = (state[0].float() - zero[0].float()).square().sum().item()
    ratio = realized / reference
    return state, dict(reference_energy=reference, prompt_energy=energy, scale=scale,
        realized_energy=realized, relative_energy=ratio, passed=.8 <= ratio <= 1.2,
        tolerance=.2, unconditional_preserved=torch.equal(state[1], zero[1]),
        scope='native carrier budget; semantic sensitivity requires measured outputs')
