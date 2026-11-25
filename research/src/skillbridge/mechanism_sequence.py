"""Locked experimental transformations; no target scores enter source selection."""
from __future__ import annotations

import itertools
import json
import random
import re
from contextlib import contextmanager
from pathlib import Path

import torch


def validate_gpu(gpu: int) -> None:
    if gpu not in range(4):
        raise ValueError("only physical GPU 0 through 3 is permitted")


def lock_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"locked design changed: {path}")
        return
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def image_permutation(records: list[dict], *, seed: int) -> list[dict]:
    images = [str(r["user_image"]) for r in records]
    if len(images) < 2 or len({Path(p).name for p in images}) != len(images):
        raise ValueError("image control requires at least two unique images")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    donor = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    return [{**row, "user_image": images[donor[i]],
             "diagnostic_original_image": images[i],
             "diagnostic": "mismatched image; reference remains original, not donor truth"}
            for i, row in enumerate(records)]


def singular_bands(state: dict, *, groups: int = 8) -> list[dict]:
    if not state or groups < 2:
        raise ValueError("nonempty state and at least two groups required")
    bands = [{} for _ in range(groups)]
    for name, value in state.items():
        if value.ndim != 2 or not torch.isfinite(value).all():
            raise ValueError("finite core matrices required")
        u, s, vh = torch.linalg.svd(value.detach().float().cpu(), full_matrices=False)
        if len(s) % groups:
            raise ValueError("core rank must be divisible by groups")
        width = len(s) // groups
        for i in range(groups):
            sl = slice(i * width, (i + 1) * width)
            # Avoid rounding every component separately before summing BF16 cores.
            # set_core_state performs the single native cast at model write time.
            bands[i][name] = (u[:, sl] * s[sl]) @ vh[sl]
    return bands


def selected_band_states(full: dict, bands: list[dict], damages: list[float], *, seed: int, keep: int = 2):
    if len(bands) != len(damages) or not 0 < keep < len(bands):
        raise ValueError("invalid band selection budget")
    if not all(torch.isfinite(torch.tensor(x)) for x in damages):
        raise ValueError("nonfinite source scores")
    chosen = sorted(range(len(bands)), key=lambda i: (-damages[i], i))[:keep]
    def combine(indices):
        return {k: sum((bands[i][k] for i in indices), torch.zeros_like(v, dtype=torch.float32)) for k, v in full.items()}
    selected = combine(chosen)
    states = {"source_selected": selected,
              "source_complement": {k: v - selected[k] for k, v in full.items()}}
    alternatives = [c for c in itertools.combinations(range(len(bands)), keep) if set(c) != set(chosen)]
    if len(alternatives) < 3:
        raise ValueError("three distinct random support controls required")
    random_sets = random.Random(9000 + seed).sample(alternatives, 3)
    for i, indices in enumerate(random_sets):
        states[f"random_bands_{i}"] = combine(indices)
    return states, {"selected": chosen, "source_damages": damages,
                    "random_bands": [list(v) for v in random_sets], "bands": len(bands),
                    "keep": keep, "selection_used_target": False,
                    "scope": "singular-index bands; not individual semantic directions"}


def training_jobs(root: Path, *, seed: int) -> list[dict]:
    campaign = root / "runs/mechanism_controls/mechanism_sequence_v1"
    jobs = []
    for cap, count, gain in (("counting", 8192, .2), ("spatial", 4096, .1)):
        source = campaign / "data" / cap / f"seed-{seed}" / "ordered-source.jsonl"
        for arm, backbone, method, shuffle in (
            ("reference", "blip3o", "tied_core", False),
            ("label_shuffle", "blip3o", "tied_core", True),
            ("additive", "blip3o", "additive_tied_core", False),
            ("replication", "janusflow", "tied_core", False),
        ):
            jobs.append(dict(arm=arm, backbone=backbone, capability=cap, method=method,
                source_records=str(source), seed=seed, steps=count, examples=count,
                rank=64, timestep_buckets=1, image_size=384, gain_limit=gain,
                learning_rate=1e-4, optimizer="adamw", shuffle_source_labels=shuffle,
                output=str(campaign / backbone / cap / f"seed-{seed}" / arm)))
    return jobs


def counting_pairs(records: list[dict]) -> list[dict]:
    """Use existing simple count/object prompts; never synthesize benchmark answers."""
    parsed = []
    for row in records:
        match = re.fullmatch(r"(one|two|three|four|five|six) ([a-z]+)", row['user_prompt'].strip())
        if match:
            parsed.append((row, match.group(1), match.group(2)))
    pairs = []
    for row, count, obj in parsed:
        semantic = [r for r, n, o in parsed if n != count and o == obj]
        nuisance = [r for r, n, o in parsed if n == count and o != obj]
        if semantic and nuisance:
            pairs.append(dict(recipient=row,
                semantic_donor=sorted(semantic, key=lambda r: (r['user_prompt'], r['split_key']))[0],
                nuisance_donor=sorted(nuisance, key=lambda r: (r['user_prompt'], r['split_key']))[0]))
    return pairs


def condition_interventions(zero, full, semantic_zero, semantic_full, nuisance_zero, nuisance_full, *, seed):
    """Patch the conditional interface, holding recipient unconditional input fixed.

    Donor deltas are target-input diagnostics, not portable low-rank checkpoints.
    Norm matching is checked again after native precision rounding.
    """
    from skillbridge.activation_sign_control import signed_native_control
    values = (zero, full, semantic_zero, semantic_full, nuisance_zero, nuisance_full)
    if zero.ndim != 3 or zero.shape[0] != 2 or any(
        v.shape != zero.shape or v.dtype != zero.dtype or not torch.isfinite(v).all() for v in values
    ):
        raise ValueError('expected matching finite conditional/unconditional interface tensors')
    states = {'zero': zero.clone(), 'full': full.clone(), 'conditional_only': zero.clone(),
              'unconditional_only': zero.clone(), 'interface_replay': full.clone()}
    states['conditional_only'][0] = full[0]
    states['unconditional_only'][1] = full[1]
    target = (full[0].float() - zero[0].float()).square().sum().item()
    audit = {}
    for name, a, b in (('semantic_donor', semantic_zero, semantic_full),
                       ('nuisance_donor', nuisance_zero, nuisance_full)):
        delta = b[0].float() - a[0].float()
        energy = delta.square().sum().item()
        scale = (target / energy) ** .5 if energy > 0 else 0.
        state = zero.clone()
        state[0] = (zero[0].float() + scale * delta).to(zero.dtype)
        realized = (state[0].float() - zero[0].float()).square().sum().item()
        ratio = realized / target if target > 0 else None
        states[name] = state
        audit[name] = dict(scale=scale, reference_energy=target, donor_energy=energy,
            realized_energy=realized, relative_energy=ratio,
            passed=ratio is not None and .8 <= ratio <= 1.2 and energy > 0,
            scope='interface-norm diagnostic; no whole-trajectory energy claim')
    signs = torch.randint(0, 2, (zero.shape[-1],), generator=torch.Generator().manual_seed(seed)) * 2 - 1
    signed, report = signed_native_control(zero[0], full[0], signs)
    states['random_sign'] = zero.clone()
    states['random_sign'][0] = signed
    audit['random_sign'] = report
    return states, audit


def capture_condition(adapter, prompt, *, seed, kwargs):
    """Capture the released AR-to-diffusion boundary without running the decoder."""
    pipeline = adapter.diffusion_pipeline
    original = pipeline._prepare_and_encode_inputs
    captured = []
    class ConditionCaptured(Exception):
        pass
    def capture(*args, **kw):
        value = original(*args, **kw)
        captured.append(value.detach().cpu().clone())
        raise ConditionCaptured
    pipeline._prepare_and_encode_inputs = capture
    try:
        adapter.reset_condition_cache()
        try:
            adapter.generate(prompt, seed=seed, **kwargs)
        except ConditionCaptured:
            pass
    finally:
        pipeline._prepare_and_encode_inputs = original
        adapter.reset_condition_cache()
    if len(captured) != 1:
        raise ValueError('expected exactly one captured decoder interface')
    return captured[0]


@contextmanager
def replace_condition(pipeline, value, *, seed, verify_native=False):
    """Use common decoder noise, independent of any encoder RNG consumption."""
    original = pipeline._prepare_and_encode_inputs
    audit = {'calls': 0, 'decoder_seed': seed}
    def replace(inputs, do_classifier_free_guidance=False, **kwargs):
        if not do_classifier_free_guidance or len(inputs) != 1 or value.shape[0] != 2:
            raise ValueError('expected one CFG prompt with two interface branches')
        if verify_native:
            native = original(inputs, do_classifier_free_guidance, **kwargs).detach().cpu().to(value.dtype)
            if not torch.equal(native, value.cpu()):
                raise ValueError('native encoder output disagrees with captured interface')
        audit['calls'] += 1
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return value.clone()
    pipeline._prepare_and_encode_inputs = replace
    try:
        yield audit
        if audit['calls'] != 1:
            raise ValueError('decoder interface was not consumed exactly once')
    finally:
        pipeline._prepare_and_encode_inputs = original
