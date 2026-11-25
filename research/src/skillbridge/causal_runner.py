from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

from skillbridge.causal_analysis import InterventionPlan
from skillbridge.interventions import (
    CoreDeltaBank,
    PatchBank,
    noise_patch_bank,
    rank_norm_matched_core_delta,
    route_stability_metrics,
    select_core_delta,
    select_patch_bank,
    singular_component_core_delta,
    singular_value_concentration,
)
from skillbridge.sidecar import (
    SidecarSite,
    SidecarTask,
    capture_shared_modulation_sites,
    functional_capability_scale,
    functional_capability_site_mask,
    patch_shared_modulation_sites,
    sidecar_route,
)

if TYPE_CHECKING:
    from torch import nn

OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class CausalOutcome:
    score: float
    attributes: dict[str, str]


@dataclass(frozen=True, slots=True)
class CapturedOutcome:
    outcome: CausalOutcome
    patches: PatchBank
    trajectory: TrajectoryMetadata | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryMetadata:
    schedule_sha256: str
    token_counts: dict[SidecarSite, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    recipient_id: str
    donor_id: str
    target_attribute: str
    preserved_attributes: tuple[str, ...]
    recipient_attributes: dict[str, str] | None = None
    donor_attributes: dict[str, str] | None = None


CaptureCallback = Callable[[str, str], CapturedOutcome]
PatchCallback = Callable[[str, str, PatchBank], CausalOutcome]
CoreEvaluationCallback = Callable[[str, CoreDeltaBank, float], CausalOutcome]
OutcomeCallback = Callable[[], CausalOutcome]
CoreRecoveryCallback = Callable[[CoreDeltaBank], CausalOutcome]
PatchRecoveryCallback = Callable[[PatchBank], CausalOutcome]


def capture_route_run(
    model: nn.Module,
    route_module: nn.Module,
    *,
    task: SidecarTask,
    timestep_bucket: int,
    forward: Callable[[], OutputT],
) -> tuple[OutputT, PatchBank]:
    del model
    with (
        sidecar_route(
            route_module,
            task=task,
            timestep_bucket=timestep_bucket,
            shared_enabled=True,
        ),
        capture_shared_modulation_sites(route_module) as captured,
    ):
        output = forward()
    return output, captured


def patch_route_run(
    model: nn.Module,
    route_module: nn.Module,
    patches: PatchBank,
    *,
    task: SidecarTask,
    timestep_bucket: int,
    forward: Callable[[], OutputT],
) -> OutputT:
    del model
    with (
        sidecar_route(
            route_module,
            task=task,
            timestep_bucket=timestep_bucket,
            shared_enabled=True,
        ),
        patch_shared_modulation_sites(route_module, patches, strict=True),
    ):
        return forward()


def intervention_effect(adapted_score: float, patched_score: float) -> float:
    return adapted_score - patched_score


def _outcome_payload(outcome: CausalOutcome) -> dict[str, object]:
    return {"score": outcome.score, "attributes": outcome.attributes}


def _validate_trajectory_metadata(label: str, captured: CapturedOutcome) -> TrajectoryMetadata:
    metadata = captured.trajectory
    if metadata is None:
        msg = f"trajectory metadata is required for {label}"
        raise ValueError(msg)
    if len(metadata.schedule_sha256) != 64:
        msg = f"trajectory schedule SHA-256 is invalid for {label}"
        raise ValueError(msg)
    if metadata.token_counts.keys() != captured.patches.keys():
        msg = f"trajectory token metadata keys mismatch for {label}"
        raise ValueError(msg)
    for site, calls in captured.patches.items():
        token_counts = metadata.token_counts[site]
        if len(token_counts) != len(calls):
            msg = f"trajectory call-count metadata mismatch for {label} at {site}"
            raise ValueError(msg)
        if any(count < 1 for count in token_counts):
            msg = f"trajectory token counts must be positive for {label} at {site}"
            raise ValueError(msg)
    return metadata


def _validate_aligned_captures(captures: dict[str, CapturedOutcome]) -> str:
    metadata = {
        label: _validate_trajectory_metadata(label, captured)
        for label, captured in captures.items()
    }
    schedules = {item.schedule_sha256 for item in metadata.values()}
    if len(schedules) != 1:
        msg = "trajectory schedule mismatch across calibrated, adapted, and donor runs"
        raise ValueError(msg)
    reference_label = next(iter(captures))
    reference = captures[reference_label]
    for label, captured in captures.items():
        if captured.patches.keys() != reference.patches.keys():
            msg = f"trajectory site-key mismatch between {reference_label} and {label}"
            raise ValueError(msg)
        for site, reference_calls in reference.patches.items():
            calls = captured.patches[site]
            if len(calls) != len(reference_calls):
                msg = (
                    f"trajectory call-count mismatch at {site} "
                    f"between {reference_label} and {label}"
                )
                raise ValueError(msg)
            if metadata[label].token_counts[site] != metadata[reference_label].token_counts[site]:
                msg = f"trajectory token correspondence mismatch at {site}"
                raise ValueError(msg)
            for call_index, (reference_value, value) in enumerate(
                zip(reference_calls, calls, strict=True)
            ):
                if reference_value.shape != value.shape:
                    msg = f"trajectory shape mismatch at {site} call {call_index}"
                    raise ValueError(msg)
    return next(iter(schedules))


def _recipient_baseline_plus_donor_delta(
    recipient_calibrated: PatchBank,
    donor_adapted: PatchBank,
    donor_calibrated: PatchBank,
) -> PatchBank:
    """Express a donor source-induced increment in the recipient calibration frame."""
    if not (recipient_calibrated.keys() == donor_adapted.keys() == donor_calibrated.keys()):
        msg = "semantic delta transfer requires identical patch-bank sites"
        raise ValueError(msg)
    output: PatchBank = {}
    for site in sorted(recipient_calibrated):
        recipient_calls = recipient_calibrated[site]
        adapted_calls = donor_adapted[site]
        calibrated_calls = donor_calibrated[site]
        if not len(recipient_calls) == len(adapted_calls) == len(calibrated_calls):
            msg = f"semantic delta transfer call-count mismatch at {site}"
            raise ValueError(msg)
        output[site] = []
        for call_index, (recipient, adapted, calibrated) in enumerate(
            zip(recipient_calls, adapted_calls, calibrated_calls, strict=True)
        ):
            if recipient.shape != adapted.shape or recipient.shape != calibrated.shape:
                msg = f"semantic delta transfer shape mismatch at {site} call {call_index}"
                raise ValueError(msg)
            output[site].append(recipient + adapted.to(recipient) - calibrated.to(recipient))
    return output


def execute_core_interventions(
    core_delta: CoreDeltaBank,
    *,
    selected_layers: tuple[str, ...],
    evaluate: CoreEvaluationCallback,
    calibration_frame_sha256: str,
    development_split_sha256: str,
    held_out_split_sha256: str,
    random_seeds: tuple[int, ...],
    functional_controls: tuple[CoreDeltaBank, ...],
    functional_matching_errors: tuple[float, ...],
    output_path: Path,
) -> dict[str, object]:
    """Run parameter-level necessity, sufficiency, and matched-control tests."""
    hashes = {
        "calibration frame": calibration_frame_sha256,
        "development split": development_split_sha256,
        "held-out split": held_out_split_sha256,
    }
    for label, value in hashes.items():
        if len(value) != 64:
            msg = f"{label} SHA-256 must contain 64 hexadecimal characters"
            raise ValueError(msg)
    if development_split_sha256 == held_out_split_sha256:
        msg = "development and held-out split hashes must differ"
        raise ValueError(msg)
    if not random_seeds or len(set(random_seeds)) != len(random_seeds):
        msg = "matched core controls require unique random seeds"
        raise ValueError(msg)
    if (
        not functional_controls
        or len(functional_controls) != len(functional_matching_errors)
        or any(not math.isfinite(value) or value < 0 for value in functional_matching_errors)
    ):
        msg = "functional controls require one non-negative matching error per control"
        raise ValueError(msg)
    if output_path.exists():
        msg = f"refusing to overwrite core intervention artifact: {output_path}"
        raise FileExistsError(msg)

    selected = select_core_delta(core_delta, selected_layers)
    adapted = evaluate("adapted", selected, 0.0)
    calibrated = evaluate("calibrated", selected, 0.0)
    ablated = evaluate("adapted", selected, -1.0)
    transplanted = evaluate("calibrated", selected, 1.0)
    controls: list[dict[str, object]] = []
    for seed in random_seeds:
        random_delta = rank_norm_matched_core_delta(selected, seed=seed)
        controls.append(
            {
                "seed": seed,
                "ablation": _outcome_payload(evaluate("adapted", random_delta, -1.0)),
                "transplant": _outcome_payload(evaluate("calibrated", random_delta, 1.0)),
            }
        )
    functional_rows = [
        {
            "matching_error": error,
            "ablation": _outcome_payload(evaluate("adapted", control, -1.0)),
            "transplant": _outcome_payload(evaluate("calibrated", control, 1.0)),
        }
        for control, error in zip(
            functional_controls,
            functional_matching_errors,
            strict=True,
        )
    ]

    artifact: dict[str, object] = {
        "format_version": 1,
        "calibration_frame_sha256": calibration_frame_sha256,
        "development_split_sha256": development_split_sha256,
        "held_out_split_sha256": held_out_split_sha256,
        "selected_layers": selected_layers,
        "adapted": _outcome_payload(adapted),
        "calibrated": _outcome_payload(calibrated),
        "core_ablation": _outcome_payload(ablated),
        "core_ablation_effect": intervention_effect(adapted.score, ablated.score),
        "core_transplant": _outcome_payload(transplanted),
        "core_transplant_effect": transplanted.score - calibrated.score,
        "matched_random_controls": controls,
        "functional_energy_controls": functional_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def execute_same_input_dose_response(
    route_module: nn.Module,
    *,
    sites: tuple[SidecarSite, ...],
    scales: tuple[float, ...],
    evaluate: OutcomeCallback,
    development_split_sha256: str,
    held_out_split_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    """Evaluate a scaled candidate write on each layer's current hidden input."""
    if len(development_split_sha256) != 64 or len(held_out_split_sha256) != 64:
        msg = "development and held-out split SHA-256 values must contain 64 characters"
        raise ValueError(msg)
    if development_split_sha256 == held_out_split_sha256:
        msg = "development and held-out split hashes must differ"
        raise ValueError(msg)
    if not sites or not scales or len(scales) != len(set(scales)):
        msg = "same-input dose response requires sites and unique scales"
        raise ValueError(msg)
    if any(not math.isfinite(scale) for scale in scales):
        msg = "same-input dose-response scales must be finite"
        raise ValueError(msg)
    if output_path.exists():
        msg = f"refusing to overwrite dose-response artifact: {output_path}"
        raise FileExistsError(msg)

    rows: list[dict[str, object]] = []
    for scale in scales:
        with functional_capability_scale(route_module, scale=scale, sites=sites):
            outcome = evaluate()
        rows.append({"scale": scale, **_outcome_payload(outcome)})
    artifact: dict[str, object] = {
        "format_version": 1,
        "intervention": "same_input_capability_scale",
        "development_split_sha256": development_split_sha256,
        "held_out_split_sha256": held_out_split_sha256,
        "selected_sites": sites,
        "dose_response": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def execute_minimal_core_recovery(
    core_delta: CoreDeltaBank,
    *,
    ordered_components: tuple[tuple[str, int], ...],
    budgets: tuple[int, ...],
    evaluate: CoreRecoveryCallback,
    full_effect: float,
    recovery_threshold: float,
    development_split_sha256: str,
    held_out_split_sha256: str,
    output_path: Path,
    matched_controls_by_budget: dict[int, tuple[CoreDeltaBank, ...]] | None = None,
) -> dict[str, object]:
    """Measure how many singular directions recover the full candidate effect."""
    if len(development_split_sha256) != 64 or len(held_out_split_sha256) != 64:
        msg = "development and held-out split SHA-256 values must contain 64 characters"
        raise ValueError(msg)
    if development_split_sha256 == held_out_split_sha256:
        msg = "development and held-out split hashes must differ"
        raise ValueError(msg)
    if not budgets or len(budgets) != len(set(budgets)) or any(value < 1 for value in budgets):
        msg = "minimal-state budgets must be unique positive integers"
        raise ValueError(msg)
    if not math.isfinite(full_effect):
        msg = "full candidate effect must be finite"
        raise ValueError(msg)
    if not 0.0 < recovery_threshold <= 1.0:
        msg = "recovery threshold must lie in (0, 1]"
        raise ValueError(msg)
    if output_path.exists():
        msg = f"refusing to overwrite minimal-state artifact: {output_path}"
        raise FileExistsError(msg)

    total_components = len(singular_value_concentration(core_delta))
    if max(budgets) > total_components:
        msg = "minimal-state budget exceeds the available singular directions"
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    qualifying: list[int] = []
    matched_controls_by_budget = matched_controls_by_budget or {}
    for budget in sorted(budgets):
        candidate = singular_component_core_delta(
            core_delta,
            budget=budget,
            ordered_components=ordered_components,
        )
        outcome = evaluate(candidate)
        recovery = outcome.score / full_effect if full_effect > 0 else None
        control_outcomes = [
            evaluate(control) for control in matched_controls_by_budget.get(budget, ())
        ]
        control_margin = (
            outcome.score - max(control.score for control in control_outcomes)
            if control_outcomes
            else None
        )
        rows.append(
            {
                "budget": budget,
                "fraction": budget / total_components,
                "recovery": recovery,
                "matched_controls": [_outcome_payload(item) for item in control_outcomes],
                "control_margin": control_margin,
                **_outcome_payload(outcome),
            }
        )
        if (
            recovery is not None
            and recovery >= recovery_threshold
            and (not control_outcomes or (control_margin is not None and control_margin > 0))
        ):
            qualifying.append(budget)
    artifact: dict[str, object] = {
        "format_version": 1,
        "development_split_sha256": development_split_sha256,
        "held_out_split_sha256": held_out_split_sha256,
        "full_effect": full_effect,
        "target_transfer_gate_pass": full_effect > 0,
        "undefined_reason": None if full_effect > 0 else "nonpositive_target_transfer",
        "recovery_threshold": recovery_threshold,
        "total_singular_components": total_components,
        "development_ranked_components": ordered_components,
        "minimum_budget": min(qualifying) if qualifying else None,
        "recovery_curve": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _validate_ordered_sites(
    patch_bank: PatchBank,
    ordered_sites: tuple[SidecarSite, ...],
) -> None:
    if set(ordered_sites) != set(patch_bank) or len(ordered_sites) != len(set(ordered_sites)):
        msg = "ordered sites must be a permutation of the patch-bank sites"
        raise ValueError(msg)


def execute_minimal_site_recovery(
    patch_bank: PatchBank,
    *,
    ordered_sites: tuple[SidecarSite, ...],
    budgets: tuple[int, ...],
    evaluate: PatchRecoveryCallback,
    full_effect: float,
    recovery_threshold: float,
    development_split_sha256: str,
    held_out_split_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    """Evaluate nested layer-time site budgets fixed on development data."""
    if len(development_split_sha256) != 64 or len(held_out_split_sha256) != 64:
        msg = "development and held-out split SHA-256 values must contain 64 characters"
        raise ValueError(msg)
    if development_split_sha256 == held_out_split_sha256:
        msg = "development and held-out split hashes must differ"
        raise ValueError(msg)
    _validate_ordered_sites(patch_bank, ordered_sites)
    if not budgets or len(budgets) != len(set(budgets)) or any(value < 1 for value in budgets):
        msg = "minimal-site budgets must be unique positive integers"
        raise ValueError(msg)
    if max(budgets) > len(ordered_sites):
        msg = "minimal-site budget exceeds the available layer-time sites"
        raise ValueError(msg)
    if not math.isfinite(full_effect):
        msg = "full candidate effect must be finite"
        raise ValueError(msg)
    if not 0.0 < recovery_threshold <= 1.0:
        msg = "recovery threshold must lie in (0, 1]"
        raise ValueError(msg)
    if output_path.exists():
        msg = f"refusing to overwrite minimal-site artifact: {output_path}"
        raise FileExistsError(msg)

    rows: list[dict[str, object]] = []
    qualifying: list[int] = []
    for budget in sorted(budgets):
        selected_sites = ordered_sites[:budget]
        outcome = evaluate(select_patch_bank(patch_bank, selected_sites))
        recovery = outcome.score / full_effect if full_effect > 0 else None
        rows.append(
            {
                "budget": budget,
                "fraction": budget / len(ordered_sites),
                "selected_sites": selected_sites,
                "recovery": recovery,
                **_outcome_payload(outcome),
            }
        )
        if recovery is not None and recovery >= recovery_threshold:
            qualifying.append(budget)
    artifact: dict[str, object] = {
        "format_version": 1,
        "development_split_sha256": development_split_sha256,
        "held_out_split_sha256": held_out_split_sha256,
        "full_effect": full_effect,
        "target_transfer_gate_pass": full_effect > 0,
        "undefined_reason": None if full_effect > 0 else "nonpositive_target_transfer",
        "recovery_threshold": recovery_threshold,
        "total_sites": len(ordered_sites),
        "minimum_budget": min(qualifying) if qualifying else None,
        "recovery_curve": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def execute_minimal_same_input_site_recovery(  # noqa: C901
    route_module: nn.Module,
    *,
    ordered_sites: tuple[SidecarSite, ...],
    budgets: tuple[int, ...],
    evaluate: OutcomeCallback,
    full_effect: float,
    recovery_threshold: float,
    development_split_sha256: str,
    held_out_split_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    """Recover transfer with nested site masks on each run's current hidden inputs."""
    if len(development_split_sha256) != 64 or len(held_out_split_sha256) != 64:
        msg = "development and held-out split SHA-256 values must contain 64 characters"
        raise ValueError(msg)
    if development_split_sha256 == held_out_split_sha256:
        msg = "development and held-out split hashes must differ"
        raise ValueError(msg)
    if not ordered_sites or len(ordered_sites) != len(set(ordered_sites)):
        msg = "same-input site ordering must contain unique sites"
        raise ValueError(msg)
    if not budgets or len(budgets) != len(set(budgets)) or any(value < 1 for value in budgets):
        msg = "same-input site budgets must be unique positive integers"
        raise ValueError(msg)
    if max(budgets) > len(ordered_sites):
        msg = "same-input site budget exceeds the locked ordering"
        raise ValueError(msg)
    if not math.isfinite(full_effect):
        msg = "full candidate effect must be finite"
        raise ValueError(msg)
    if not 0.0 < recovery_threshold <= 1.0:
        msg = "recovery threshold must lie in (0, 1]"
        raise ValueError(msg)
    if output_path.exists():
        msg = f"refusing to overwrite same-input site artifact: {output_path}"
        raise FileExistsError(msg)

    rows: list[dict[str, object]] = []
    qualifying: list[int] = []
    for budget in sorted(budgets):
        selected = ordered_sites[:budget]
        with functional_capability_site_mask(route_module, active_sites=selected):
            outcome = evaluate()
        recovery = outcome.score / full_effect if full_effect > 0 else None
        rows.append(
            {
                "budget": budget,
                "fraction": budget / len(ordered_sites),
                "selected_sites": selected,
                "recovery": recovery,
                **_outcome_payload(outcome),
            }
        )
        if recovery is not None and recovery >= recovery_threshold:
            qualifying.append(budget)
    artifact: dict[str, object] = {
        "format_version": 1,
        "intervention": "same_input_site_mask",
        "development_split_sha256": development_split_sha256,
        "held_out_split_sha256": held_out_split_sha256,
        "full_effect": full_effect,
        "target_transfer_gate_pass": full_effect > 0,
        "undefined_reason": None if full_effect > 0 else "nonpositive_target_transfer",
        "recovery_threshold": recovery_threshold,
        "total_sites": len(ordered_sites),
        "minimum_budget": min(qualifying) if qualifying else None,
        "recovery_curve": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def execute_intervention_plan(  # noqa: C901, PLR0915
    plan: InterventionPlan,
    *,
    pairs: tuple[CounterfactualPair, ...],
    capture: CaptureCallback,
    patch: PatchCallback,
    held_out_split_sha256: str,
    output_path: Path,
    noise_seed: int,
    minimum_evaluable_fraction: float = 0.8,
) -> dict[str, object]:
    """Execute a locked plan without reopening development-set site selection.

    The callbacks bind this model-agnostic orchestration to understanding or
    generation inference. Patches always recompute downstream activations.
    """
    if len(held_out_split_sha256) != 64:
        msg = "held-out split SHA-256 must contain 64 hexadecimal characters"
        raise ValueError(msg)
    if held_out_split_sha256 == plan.development_split_sha256:
        msg = "development and held-out split hashes must differ"
        raise ValueError(msg)
    if not pairs:
        msg = "at least one counterfactual pair is required"
        raise ValueError(msg)
    if not 0.0 < minimum_evaluable_fraction <= 1.0:
        msg = "minimum evaluable fraction must lie in (0, 1]"
        raise ValueError(msg)
    if output_path.exists():
        msg = f"refusing to overwrite causal protocol artifact: {output_path}"
        raise FileExistsError(msg)

    cache: dict[tuple[str, str], CapturedOutcome] = {}

    def captured(example_id: str, state: str) -> CapturedOutcome:
        key = (example_id, state)
        if key not in cache:
            cache[key] = capture(example_id, state)
        return cache[key]

    rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        recipient_adapted = captured(pair.recipient_id, "adapted")
        recipient_calibrated = captured(pair.recipient_id, "calibrated")
        donor_adapted = captured(pair.donor_id, "adapted")
        donor_calibrated = captured(pair.donor_id, "calibrated")
        schedule_sha256 = _validate_aligned_captures(
            {
                "recipient_adapted": recipient_adapted,
                "recipient_calibrated": recipient_calibrated,
                "donor_adapted": donor_adapted,
                "donor_calibrated": donor_calibrated,
            }
        )

        recipient_attributes = recipient_adapted.outcome.attributes
        donor_attributes = donor_adapted.outcome.attributes
        expected_recipient = pair.recipient_attributes or recipient_attributes
        expected_donor = pair.donor_attributes or donor_attributes
        if (
            pair.target_attribute not in expected_recipient
            or pair.target_attribute not in expected_donor
            or expected_recipient[pair.target_attribute] == expected_donor[pair.target_attribute]
        ):
            msg = "counterfactual data labels must differ on the target attribute"
            raise ValueError(msg)
        invalid_expected_preserved = [
            attribute
            for attribute in pair.preserved_attributes
            if attribute not in expected_recipient
            or attribute not in expected_donor
            or expected_recipient[attribute] != expected_donor[attribute]
        ]
        if invalid_expected_preserved:
            msg = (
                "counterfactual data labels must match on preserved attributes: "
                f"{invalid_expected_preserved}"
            )
            raise ValueError(msg)
        target_precondition = (
            recipient_attributes.get(pair.target_attribute)
            == expected_recipient[pair.target_attribute]
            and donor_attributes.get(pair.target_attribute) == expected_donor[pair.target_attribute]
        )
        invalid_preserved = [
            attribute
            for attribute in pair.preserved_attributes
            if recipient_attributes.get(attribute) != expected_recipient[attribute]
            or donor_attributes.get(attribute) != expected_donor[attribute]
        ]
        precondition_failures: list[str] = []
        if not target_precondition:
            precondition_failures.append("model_outputs_do_not_differ_on_target_attribute")
        if invalid_preserved:
            precondition_failures.append("model_outputs_do_not_match_preserved_attributes")
        evaluable = not precondition_failures

        selected_calibrated = select_patch_bank(recipient_calibrated.patches, plan.selected)
        selected_adapted = select_patch_bank(recipient_adapted.patches, plan.selected)
        selected_patch = patch(pair.recipient_id, "adapted", selected_calibrated)
        transplant = patch(pair.recipient_id, "calibrated", selected_adapted)

        layer_control = patch(
            pair.recipient_id,
            "adapted",
            select_patch_bank(recipient_calibrated.patches, plan.layer_matched),
        )
        norm_control = patch(
            pair.recipient_id,
            "adapted",
            select_patch_bank(recipient_calibrated.patches, plan.norm_matched),
        )
        random_controls = [
            patch(
                pair.recipient_id,
                "adapted",
                select_patch_bank(recipient_calibrated.patches, sites),
            )
            for sites in plan.random_controls
        ]
        noise_control = patch(
            pair.recipient_id,
            "adapted",
            noise_patch_bank(
                selected_adapted,
                seed=noise_seed + pair_index,
                covariance_matched=True,
            ),
        )
        semantic_delta = _recipient_baseline_plus_donor_delta(
            recipient_calibrated.patches,
            donor_adapted.patches,
            donor_calibrated.patches,
        )
        semantic_swap = patch(
            pair.recipient_id,
            "adapted",
            select_patch_bank(semantic_delta, plan.selected),
        )
        donor_target = expected_donor[pair.target_attribute]
        target_matches = (
            donor_target is not None
            and semantic_swap.attributes.get(pair.target_attribute) == donor_target
        )
        preserved_match = all(
            semantic_swap.attributes.get(attribute) == expected_recipient[attribute]
            for attribute in pair.preserved_attributes
        )
        rows.append(
            {
                **asdict(pair),
                "evaluable": evaluable,
                "precondition_failures": precondition_failures,
                "trajectory_schedule_sha256": schedule_sha256,
                "adapted": _outcome_payload(recipient_adapted.outcome),
                "calibrated": _outcome_payload(recipient_calibrated.outcome),
                "adapted_route_stability": route_stability_metrics(recipient_adapted.patches),
                "calibrated_route_stability": route_stability_metrics(recipient_calibrated.patches),
                "selected_patch": _outcome_payload(selected_patch),
                "selected_intervention_effect": intervention_effect(
                    recipient_adapted.outcome.score, selected_patch.score
                ),
                "transplant": _outcome_payload(transplant),
                "layer_matched": _outcome_payload(layer_control),
                "norm_matched": _outcome_payload(norm_control),
                "random_controls": [_outcome_payload(item) for item in random_controls],
                "covariance_noise": _outcome_payload(noise_control),
                "semantic_swap": {
                    **_outcome_payload(semantic_swap),
                    "intervention": "recipient_calibrated+donor_delta",
                    "target_matches_donor": target_matches,
                    "preserved_attributes_match_recipient": preserved_match,
                },
            }
        )

    evaluable_rows = [row for row in rows if row["evaluable"]]
    evaluable_fraction = len(evaluable_rows) / len(rows)
    qualified_rows = [
        row
        for row in evaluable_rows
        if cast("dict[str, object]", row["semantic_swap"])["target_matches_donor"]
        and cast("dict[str, object]", row["semantic_swap"])["preserved_attributes_match_recipient"]
    ]
    coverage_pass = evaluable_fraction >= minimum_evaluable_fraction
    artifact: dict[str, object] = {
        "format_version": 1,
        "development_split_sha256": plan.development_split_sha256,
        "held_out_split_sha256": held_out_split_sha256,
        "selected_sites": plan.selected,
        "layer_matched_sites": plan.layer_matched,
        "norm_matched_sites": plan.norm_matched,
        "random_site_sets": plan.random_controls,
        "noise_seed": noise_seed,
        "minimum_evaluable_fraction": minimum_evaluable_fraction,
        "evaluable_fraction": evaluable_fraction,
        "semantic_gate_status": (
            "unresolved"
            if not coverage_pass
            else "pass"
            if len(qualified_rows) == len(evaluable_rows)
            else "fail"
        ),
        "pairs": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
