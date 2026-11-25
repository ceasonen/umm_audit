from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from skillbridge.sidecar import SidecarSite

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SiteEffect:
    site: SidecarSite
    effect: float
    residual_norm: float


@dataclass(frozen=True, slots=True)
class InterventionPlan:
    selected: tuple[SidecarSite, ...]
    random_controls: tuple[tuple[SidecarSite, ...], ...]
    layer_matched: tuple[SidecarSite, ...]
    norm_matched: tuple[SidecarSite, ...]
    development_split_sha256: str


def select_top_sites(effects: list[SiteEffect], *, budget: int) -> list[SiteEffect]:
    if not 0 < budget <= len(effects):
        msg = "budget must select a non-empty subset"
        raise ValueError(msg)
    ordered = sorted(effects, key=lambda item: (-item.effect, item.residual_norm, item.site))
    return ordered[:budget]


def sample_random_site_sets(
    sites: list[SidecarSite],
    *,
    budget: int,
    samples: int,
    seed: int,
) -> list[tuple[SidecarSite, ...]]:
    if not 0 < budget <= len(sites) or samples < 1:
        msg = "invalid random-site sampling request"
        raise ValueError(msg)
    rng = random.Random(seed)
    return [tuple(sorted(rng.sample(sites, budget))) for _ in range(samples)]


def norm_matched_sites(
    selected: list[SiteEffect],
    candidates: list[SiteEffect],
) -> list[SiteEffect]:
    remaining = {
        item.site: item for item in candidates if item.site not in {x.site for x in selected}
    }
    matched: list[SiteEffect] = []
    for target in selected:
        if not remaining:
            msg = "not enough candidates for norm matching"
            raise ValueError(msg)
        best = min(
            remaining.values(),
            key=lambda item: (abs(item.residual_norm - target.residual_norm), item.site),
        )
        matched.append(best)
        del remaining[best.site]
    return matched


def layer_matched_sites(
    selected: list[SiteEffect],
    candidates: list[SiteEffect],
) -> list[SiteEffect]:
    remaining = {
        item.site: item
        for item in candidates
        if item.site not in {value.site for value in selected}
    }
    matched: list[SiteEffect] = []
    for target in selected:
        layer = target.site[0].rsplit(".", 1)[0]
        same_layer = [
            item for item in remaining.values() if item.site[0].rsplit(".", 1)[0] == layer
        ]
        if not same_layer:
            msg = f"no layer-matched control for site: {target.site}"
            raise ValueError(msg)
        best = min(same_layer, key=lambda item: (abs(item.effect), item.site))
        matched.append(best)
        del remaining[best.site]
    return matched


def build_intervention_plan(
    effects: list[SiteEffect],
    *,
    budget: int,
    random_samples: int,
    seed: int,
    development_split_sha256: str,
) -> InterventionPlan:
    if len(development_split_sha256) != 64:
        msg = "development split SHA-256 must contain 64 hexadecimal characters"
        raise ValueError(msg)
    selected = select_top_sites(effects, budget=budget)
    return InterventionPlan(
        selected=tuple(item.site for item in selected),
        random_controls=tuple(
            sample_random_site_sets(
                [item.site for item in effects],
                budget=budget,
                samples=random_samples,
                seed=seed,
            )
        ),
        layer_matched=tuple(item.site for item in layer_matched_sites(selected, effects)),
        norm_matched=tuple(item.site for item in norm_matched_sites(selected, effects)),
        development_split_sha256=development_split_sha256,
    )


def write_intervention_plan(path: Path, plan: InterventionPlan) -> None:
    if path.exists():
        msg = f"refusing to overwrite intervention plan: {path}"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "selected": plan.selected,
        "random_controls": plan.random_controls,
        "layer_matched": plan.layer_matched,
        "norm_matched": plan.norm_matched,
        "development_split_sha256": plan.development_split_sha256,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_intervention_plan(path: Path) -> InterventionPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        msg = f"unsupported intervention plan: {path}"
        raise ValueError(msg)

    def sites(name: str) -> tuple[SidecarSite, ...]:
        values = payload.get(name)
        if not isinstance(values, list):
            msg = f"intervention plan lacks site list: {name}"
            raise ValueError(msg)
        return tuple((str(value[0]), int(value[1])) for value in values)

    random_values = payload.get("random_controls")
    if not isinstance(random_values, list):
        msg = "intervention plan lacks random controls"
        raise ValueError(msg)
    return InterventionPlan(
        selected=sites("selected"),
        random_controls=tuple(
            tuple((str(value[0]), int(value[1])) for value in group) for group in random_values
        ),
        layer_matched=sites("layer_matched"),
        norm_matched=sites("norm_matched"),
        development_split_sha256=str(payload["development_split_sha256"]),
    )


def covariance_matched_noise(
    residuals: NDArray[np.floating],
    *,
    samples: int,
    seed: int,
) -> NDArray[np.float64]:
    if residuals.ndim != 2 or residuals.shape[0] < 2 or samples < 1:
        msg = "residuals must be a two-dimensional sample matrix"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    mean = residuals.mean(axis=0)
    covariance = np.cov(residuals, rowvar=False)
    return rng.multivariate_normal(mean, np.atleast_2d(covariance), size=samples)


def paired_bootstrap_interval(
    adapted: NDArray[np.floating],
    baseline: NDArray[np.floating],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    if adapted.shape != baseline.shape or adapted.ndim != 1 or adapted.size == 0:
        msg = "paired samples must be non-empty one-dimensional arrays with equal shape"
        raise ValueError(msg)
    if samples < 1 or not 0.0 < confidence < 1.0:
        msg = "invalid bootstrap settings"
        raise ValueError(msg)
    differences = np.asarray(adapted - baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    draws = differences[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(differences.mean()), float(low), float(high)


def hierarchical_paired_bootstrap_interval(
    adapted: NDArray[np.floating],
    baseline: NDArray[np.floating],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample training seeds, then paired examples within each sampled seed."""
    if adapted.shape != baseline.shape or adapted.ndim != 2 or 0 in adapted.shape:
        msg = "hierarchical paired samples must be non-empty seed-by-example matrices"
        raise ValueError(msg)
    if samples < 1 or not 0.0 < confidence < 1.0:
        msg = "invalid hierarchical bootstrap settings"
        raise ValueError(msg)
    differences = np.asarray(adapted - baseline, dtype=np.float64)
    seed_count, example_count = differences.shape
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw_index in range(samples):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_means = np.empty(seed_count, dtype=np.float64)
        for output_index, seed_index in enumerate(sampled_seeds):
            examples = rng.integers(0, example_count, size=example_count)
            seed_means[output_index] = differences[seed_index, examples].mean()
        draws[draw_index] = seed_means.mean()
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(differences.mean()), float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    if any(not 0.0 <= value <= 1.0 or not math.isfinite(value) for value in p_values):
        msg = "p-values must be finite and lie in [0, 1]"
        raise ValueError(msg)
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def transfer_efficiency(*, source_gain: float, target_gain: float, epsilon: float = 1e-8) -> float:
    """Normalize target transfer by the matched source-side improvement."""
    if not all(math.isfinite(value) for value in (source_gain, target_gain, epsilon)):
        msg = "transfer-efficiency inputs must be finite"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    return target_gain / max(abs(source_gain), epsilon)
