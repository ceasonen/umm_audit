from __future__ import annotations

import json
import math
from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from PIL import Image

from skillbridge.baseline import (
    capture_private_shared_lora_sites,
    patch_private_shared_lora_sites,
    private_shared_lora_scale,
)
from skillbridge.evaluation import PredictionRecord, answers_match, evaluate_predictions
from skillbridge.sidecar import (
    capture_shared_modulation_sites,
    functional_capability_scale,
    patch_shared_modulation_sites,
)

if TYPE_CHECKING:
    from torch import nn


def resolve_witness_budgets(
    single: int | None,
    multiple: list[int] | tuple[int, ...] | None,
    *,
    total_components: int,
    fractions: list[float] | tuple[float, ...] | None = None,
) -> tuple[int, ...]:
    supplied = sum(value is not None for value in (single, multiple, fractions))
    if supplied > 1:
        raise ValueError("witness budget forms are mutually exclusive")
    if fractions is not None:
        if not fractions or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions
        ):
            raise ValueError("witness budget fractions must lie in [0, 1]")
        requested = [ceil(value * total_components) for value in fractions]
    else:
        requested = (
            [total_components]
            if single is None and multiple is None
            else ([single] if single is not None else list(multiple or ()))
        )
    budgets = tuple(dict.fromkeys(int(value) for value in requested))
    if not budgets:
        raise ValueError("witness budget schedule must be non-empty")
    if any(value < 0 or value > total_components for value in budgets):
        raise ValueError("witness component budget lies outside the available rank")
    return budgets


class GenerationAdapter(Protocol):
    def generate(self, prompt: str, *, seed: int, **kwargs: object) -> Image.Image: ...


class UnderstandingAdapter(Protocol):
    def understand(self, image_path: Path, question: str, **kwargs: object) -> str: ...


def _reset_condition_cache(adapter: GenerationAdapter) -> None:
    resetter = getattr(adapter, "reset_condition_cache", None)
    if resetter is not None:
        if not callable(resetter):
            message = "generation condition-cache resetter must be callable"
            raise TypeError(message)
        resetter()


class BatchedUnderstandingAdapter(Protocol):
    def understand_batch(
        self,
        image_paths: list[Path],
        questions: list[str],
        **kwargs: object,
    ) -> list[str]: ...


def _load_understanding_predictions(manifest: Path) -> list[PredictionRecord]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        PredictionRecord(
            str(row["split_key"]),
            str(row["capability"]),
            str(row["reference"]),
            str(row["prediction"]),
            bool(row.get("detector_succeeded", True)),
        )
        for row in rows
    ]


def _write_understanding_metrics(manifest: Path, metrics_path: Path) -> dict[str, float]:
    metrics = evaluate_predictions(_load_understanding_predictions(manifest))
    temporary = metrics_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)
    return metrics


def _validated_understanding_rows(
    records: Sequence[dict[str, object]], extra: dict[str, object]
) -> list[dict[str, object]]:
    expected: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in records:
        identifier = str(record.get("split_key", ""))
        question = str(record.get("user_prompt", ""))
        image_path = Path(str(record.get("user_image", "")))
        reference = str(record.get("answer_text", ""))
        capability = str(record.get("capability", ""))
        if (
            not identifier
            or identifier in seen
            or not question
            or not image_path.is_file()
            or not reference
        ):
            message = "understanding records require unique ids, inputs, and answers"
            raise ValueError(message)
        seen.add(identifier)
        expected.append(
            {
                "split_key": identifier,
                "capability": capability,
                "question": question,
                "image": str(image_path),
                "reference": reference,
                **extra,
            }
        )
    return expected


def _load_understanding_partial(
    temporary: Path, expected: Sequence[dict[str, object]]
) -> dict[str, dict[str, object]]:
    if not temporary.is_file():
        return {}
    rows_by_id: dict[str, dict[str, object]] = {}
    expected_by_id = {str(row["split_key"]): row for row in expected}
    for line in temporary.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            message = "understanding partial contains a non-object row"
            raise ValueError(message)
        identifier = str(row.get("split_key", ""))
        base = expected_by_id.get(identifier)
        if base is None or identifier in rows_by_id:
            message = "understanding partial contains an unknown or duplicate id"
            raise ValueError(message)
        if any(row.get(key) != value for key, value in base.items()):
            message = "understanding partial disagrees with requested records"
            raise ValueError(message)
        if not isinstance(row.get("prediction"), str):
            message = "understanding partial lacks a prediction"
            raise ValueError(message)
        rows_by_id[identifier] = row
    return rows_by_id


def _checkpoint_understanding_partial(
    temporary: Path,
    expected: Sequence[dict[str, object]],
    rows_by_id: dict[str, dict[str, object]],
) -> None:
    ordered = [
        rows_by_id[str(row["split_key"])] for row in expected if str(row["split_key"]) in rows_by_id
    ]
    checkpoint = temporary.with_suffix(".partial.tmp")
    checkpoint.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    checkpoint.replace(temporary)


def _scale_label(value: float) -> str:
    return format(value, ".6g").replace("-", "m").replace(".", "p")


def _validated_records(
    records: Sequence[dict[str, object]],
) -> tuple[list[str], list[str]]:
    if not records:
        raise ValueError("generation requires records")
    identifiers = [str(record.get("split_key", "")) for record in records]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("generation requires unique non-empty split keys")
    prompts = [str(record.get("user_prompt", "")) for record in records]
    if any(not prompt for prompt in prompts):
        raise ValueError("generation requires non-empty prompts")
    for record in records:
        mode = str(record.get("target_mode", "generation"))
        if mode not in {"generation", "editing"}:
            raise ValueError(f"unknown visual target mode: {mode}")
        if mode == "editing":
            image_path = Path(str(record.get("user_image", "")))
            if not image_path.is_file():
                raise ValueError("editing requires an existing reference image")
    return identifiers, prompts


def _generate_visual_target(
    adapter: GenerationAdapter,
    record: dict[str, object],
    *,
    seed: int,
    kwargs: dict[str, object],
) -> Image.Image:
    prompt = str(record["user_prompt"])
    if str(record.get("target_mode", "generation")) == "generation":
        return adapter.generate(prompt, seed=seed, **kwargs)
    editor = getattr(adapter, "edit", None)
    if not callable(editor):
        raise TypeError("editing record requires an adapter.edit method")
    return editor(Path(str(record["user_image"])), prompt, seed=seed, **kwargs)


def _visual_target_metadata(record: dict[str, object]) -> dict[str, object]:
    if str(record.get("target_mode", "generation")) != "editing":
        return {}
    return {
        "target_mode": "editing",
        "source_image": str(record["user_image"]),
    }


def generate_fixed_predictions(
    adapter: GenerationAdapter,
    records: Sequence[dict[str, object]],
    output: Path,
    *,
    seed: int,
    generation_kwargs: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    resume: bool = False,
) -> Path:
    """Generate one locked state, used for witness-rank recovery curves."""
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to reuse generation output: {output}")
    identifiers, prompts = _validated_records(records)
    extra = dict(metadata or {})
    reserved = {"split_key", "prompt", "seed", "image"}
    if reserved & extra.keys():
        raise ValueError("fixed-prediction metadata uses a reserved field")
    output.mkdir(parents=True, exist_ok=resume)
    image_root = output / "images"
    image_root.mkdir(exist_ok=resume)
    rows: list[dict[str, object]] = []
    kwargs = dict(generation_kwargs or {})
    for index, (record, identifier, prompt) in enumerate(
        zip(records, identifiers, prompts, strict=True)
    ):
        example_seed = seed + index
        relative = Path("images") / f"{index:05d}.png"
        image_path = output / relative
        valid_existing = False
        if resume and image_path.is_file():
            try:
                with Image.open(image_path) as existing:
                    existing.verify()
                valid_existing = True
            except (OSError, ValueError, SyntaxError):
                valid_existing = False
        if not valid_existing:
            image = _generate_visual_target(
                adapter,
                record,
                seed=example_seed,
                kwargs=kwargs,
            )
            if not isinstance(image, Image.Image):
                raise TypeError("generation adapter must return a PIL image")
            image.save(image_path)
        rows.append(
            {
                "split_key": identifier,
                "prompt": prompt,
                "seed": example_seed,
                "image": str(relative),
                **_visual_target_metadata(record),
                **extra,
            }
        )
    manifest = output / "predictions.jsonl"
    temporary_manifest = manifest.with_suffix(".jsonl.partial")
    temporary_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest)
    return manifest


def generate_understanding_predictions(
    adapter: UnderstandingAdapter,
    records: Sequence[dict[str, object]],
    output: Path,
    *,
    understanding_kwargs: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    resume: bool = False,
    batch_size: int = 1,
) -> dict[str, float]:
    """Evaluate a fixed witness budget on held-out source understanding."""
    manifest = output / "predictions.jsonl"
    metrics_path = output / "metrics.json"
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to reuse understanding output: {output}")
    if resume and manifest.is_file():
        return _write_understanding_metrics(manifest, metrics_path)
    if not records:
        raise ValueError("understanding evaluation requires records")
    if batch_size < 1:
        message = "understanding batch size must be positive"
        raise ValueError(message)
    extra = dict(metadata or {})
    reserved = {
        "split_key",
        "capability",
        "question",
        "image",
        "reference",
        "prediction",
    }
    if reserved & extra.keys():
        raise ValueError("understanding metadata uses a reserved field")
    output.mkdir(parents=True, exist_ok=resume)
    kwargs = dict(understanding_kwargs or {})
    expected = _validated_understanding_rows(records, extra)
    temporary = manifest.with_suffix(".jsonl.partial")
    rows_by_id = _load_understanding_partial(temporary, expected) if resume else {}

    pending = [base for base in expected if str(base["split_key"]) not in rows_by_id]
    batch_method = getattr(adapter, "understand_batch", None)
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        questions = [str(base["question"]) for base in batch]
        image_paths = [Path(str(base["image"])) for base in batch]
        if batch_size > 1 and callable(batch_method):
            predictions = batch_method(image_paths, questions, **kwargs)
        else:
            predictions = [
                adapter.understand(image_path, question, **kwargs)
                for image_path, question in zip(image_paths, questions, strict=True)
            ]
        if len(predictions) != len(batch) or any(
            not isinstance(prediction, str) for prediction in predictions
        ):
            message = "understanding batch returned an invalid prediction list"
            raise ValueError(message)
        for base, prediction in zip(batch, predictions, strict=True):
            identifier = str(base["split_key"])
            rows_by_id[identifier] = {**base, "prediction": prediction}
        _checkpoint_understanding_partial(temporary, expected, rows_by_id)

    _checkpoint_understanding_partial(temporary, expected, rows_by_id)
    temporary.replace(manifest)
    return _write_understanding_metrics(manifest, metrics_path)


def summarize_understanding_budgets(
    output: Path,
    *,
    backbone: str,
    seed: int,
    frozen_reference: Path | None = None,
) -> dict[str, object]:
    """Convert held-out understanding budgets to the shared seed-summary schema."""
    directories = sorted(
        (path for path in output.glob("budget-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-", 1)[1]),
    )
    if not directories:
        raise ValueError("understanding output has no budget conditions")
    loaded: list[tuple[Path, list[dict[str, object]], dict[str, float]]] = []
    total_components: int | None = None
    for directory in directories:
        rows = [
            json.loads(line)
            for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("understanding prediction file is empty")
        metrics = {
            str(key): float(value)
            for key, value in json.loads(
                (directory / "metrics.json").read_text(encoding="utf-8")
            ).items()
        }
        current_total = int(rows[0]["total_components"])
        if total_components is None:
            total_components = current_total
        elif current_total != total_components:
            raise ValueError("understanding budgets disagree on total components")
        loaded.append((directory, rows, metrics))
    assert total_components is not None
    frozen_rows = next(
        (rows for _, rows, _ in loaded if int(rows[0]["component_budget"]) == 0),
        None,
    )
    if frozen_rows is None:
        if frozen_reference is None:
            raise ValueError(
                "understanding budgets require budget 0 or an external frozen reference"
            )
        frozen_rows = [
            json.loads(line)
            for line in frozen_reference.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not frozen_rows:
            raise ValueError("external frozen reference is empty")
    frozen_by_id = {str(row["split_key"]): row for row in frozen_rows}
    conditions: dict[str, dict[str, object]] = {}
    for directory, rows, metrics in loaded:
        budget = int(rows[0]["component_budget"])
        differences: list[int] = []
        for row in rows:
            frozen = frozen_by_id[str(row["split_key"])]
            adapted_correct = answers_match(
                str(row["capability"]),
                str(row["reference"]),
                str(row["prediction"]),
            )
            frozen_correct = answers_match(
                str(frozen["capability"]),
                str(frozen["reference"]),
                str(frozen["prediction"]),
            )
            differences.append(int(adapted_correct) - int(frozen_correct))
        conditions[directory.name] = {
            "budget": budget,
            "budget_fraction": budget / total_components,
            "role": (
                "frozen" if budget == 0 else "full" if budget == total_components else "recovery"
            ),
            "accuracy": metrics["exact_match"],
            "valid_response": metrics.get("valid_response", 1.0),
            "delta_vs_frozen": sum(differences) / len(differences),
            "paired_wins": sum(value > 0 for value in differences),
            "paired_losses": sum(value < 0 for value in differences),
            "paired_ties": sum(value == 0 for value in differences),
        }
    return {
        "format_version": 1,
        "backbone": backbone,
        "seed": seed,
        "prompt_count": len(frozen_rows),
        "inference_unit": "training_seed",
        "conditions": conditions,
    }


def summarize_understanding_doses(output: Path, *, backbone: str, seed: int) -> dict[str, object]:
    """Convert tied-core dose outputs to the shared seed-summary schema."""
    directories = sorted(
        (path for path in output.glob("scale-*") if path.is_dir()),
        key=lambda path: float(path.name.split("-", 1)[1].replace("p", ".")),
    )
    if not directories:
        raise ValueError("understanding output has no scale conditions")
    loaded: list[tuple[Path, float, list[dict[str, object]], dict[str, float]]] = []
    for directory in directories:
        rows = [
            json.loads(line)
            for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("understanding prediction file is empty")
        metrics = {
            str(key): float(value)
            for key, value in json.loads(
                (directory / "metrics.json").read_text(encoding="utf-8")
            ).items()
        }
        scale = float(rows[0]["scale"])
        if any(float(row["scale"]) != scale for row in rows):
            raise ValueError("understanding dose directory mixes scales")
        loaded.append((directory, scale, rows, metrics))
    if loaded[0][1] != 0.0 or loaded[-1][1] != 1.0:
        raise ValueError("understanding doses must include scales zero and one")
    frozen_by_id = {str(row["split_key"]): row for row in loaded[0][2]}
    conditions: dict[str, dict[str, object]] = {}
    for directory, scale, rows, metrics in loaded:
        differences: list[int] = []
        for row in rows:
            frozen = frozen_by_id[str(row["split_key"])]
            adapted_correct = answers_match(
                str(row["capability"]), str(row["reference"]), str(row["prediction"])
            )
            frozen_correct = answers_match(
                str(frozen["capability"]),
                str(frozen["reference"]),
                str(frozen["prediction"]),
            )
            differences.append(int(adapted_correct) - int(frozen_correct))
        conditions[directory.name] = {
            "budget": scale,
            "budget_fraction": scale,
            "role": "frozen" if scale == 0.0 else "full" if scale == 1.0 else "recovery",
            "accuracy": metrics["exact_match"],
            "valid_response": metrics.get("valid_response", 1.0),
            "delta_vs_frozen": sum(differences) / len(differences),
            "paired_wins": sum(value > 0 for value in differences),
            "paired_losses": sum(value < 0 for value in differences),
            "paired_ties": sum(value == 0 for value in differences),
        }
    return {
        "format_version": 1,
        "backbone": backbone,
        "seed": seed,
        "prompt_count": len(loaded[0][2]),
        "inference_unit": "training_seed",
        "conditions": conditions,
    }


def rescore_understanding_budgets(output: Path, *, backbone: str, seed: int) -> dict[str, object]:
    """Rescore completed budget manifests without loading the backbone."""
    directories = sorted(
        (path for path in output.glob("budget-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-", 1)[1]),
    )
    if not directories:
        raise ValueError("understanding output has no budget conditions")
    for directory in directories:
        manifest = directory / "predictions.jsonl"
        if not manifest.is_file():
            raise ValueError(f"understanding budget is incomplete: {directory.name}")
        _write_understanding_metrics(manifest, directory / "metrics.json")
    summary = summarize_understanding_budgets(output, backbone=backbone, seed=seed)
    summary_path = output / "summary.json"
    temporary = summary_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def generate_paired_trajectories(
    adapter: GenerationAdapter,
    route_module: nn.Module,
    records: Sequence[dict[str, object]],
    output: Path,
    *,
    scales: tuple[float, ...],
    seed: int,
    generation_kwargs: dict[str, object] | None = None,
    resume: bool = False,
) -> Path:
    """Generate a same-prompt, same-noise dose trajectory for every record."""
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to reuse paired-generation output: {output}")
    identifiers, prompts = _validated_records(records)
    if not scales or len(set(scales)) != len(scales) or any(not math.isfinite(x) for x in scales):
        raise ValueError("paired generation requires unique finite scales")
    output.mkdir(parents=True, exist_ok=resume)
    image_root = output / "images"
    image_root.mkdir(exist_ok=resume)
    rows: list[dict[str, object]] = []
    kwargs = dict(generation_kwargs or {})
    for index, (record, identifier, prompt) in enumerate(
        zip(records, identifiers, prompts, strict=True)
    ):
        example_seed = seed + index
        for scale in scales:
            relative = Path("images") / f"{index:05d}-scale-{_scale_label(scale)}.png"
            image_path = output / relative
            valid_existing = False
            if resume and image_path.is_file():
                try:
                    with Image.open(image_path) as existing:
                        existing.verify()
                    valid_existing = True
                except (OSError, ValueError, SyntaxError):
                    valid_existing = False
            if not valid_existing:
                with functional_capability_scale(route_module, scale=scale):
                    image = _generate_visual_target(
                        adapter,
                        record,
                        seed=example_seed,
                        kwargs=kwargs,
                    )
                if not isinstance(image, Image.Image):
                    raise TypeError("generation adapter must return a PIL image")
                image.save(image_path)
            rows.append(
                {
                    "split_key": identifier,
                    "prompt": prompt,
                    "seed": example_seed,
                    "scale": scale,
                    "image": str(relative),
                    **_visual_target_metadata(record),
                }
            )
    manifest = output / "predictions.jsonl"
    temporary_manifest = manifest.with_suffix(".jsonl.partial")
    temporary_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest)
    return manifest


def generate_route_interventions(
    adapter: GenerationAdapter,
    route_module: nn.Module,
    records: Sequence[dict[str, object]],
    output: Path,
    *,
    seed: int,
    generation_kwargs: dict[str, object] | None = None,
    resume: bool = False,
    intervention_kind: Literal["tied_core", "shared_lora"] = "tied_core",
) -> dict[str, Path]:
    """Run paired whole-route removal and realized-modulation replay."""
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to reuse route intervention output: {output}")
    identifiers, prompts = _validated_records(records)
    conditions = ("adapted", "removal", "replay")
    output.mkdir(parents=True, exist_ok=resume)
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in conditions}
    kwargs = dict(generation_kwargs or {})
    if intervention_kind == "shared_lora":
        scale_context = private_shared_lora_scale
        capture_context = capture_private_shared_lora_sites
        patch_context = patch_private_shared_lora_sites
    else:
        scale_context = functional_capability_scale
        capture_context = capture_shared_modulation_sites
        patch_context = patch_shared_modulation_sites

    def valid_image(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            with Image.open(path) as existing:
                existing.verify()
        except (OSError, ValueError, SyntaxError):
            return False
        return True

    for index, (record, identifier, prompt) in enumerate(
        zip(records, identifiers, prompts, strict=True)
    ):
        example_seed = seed + index
        paths = {name: output / name / "images" / f"{index:05d}.png" for name in conditions}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=resume)

        complete = resume and all(valid_image(path) for path in paths.values())
        if not complete:
            _reset_condition_cache(adapter)
            with (
                scale_context(route_module, scale=1.0),
                capture_context(route_module, capture_device="cpu") as realized,
            ):
                adapted = _generate_visual_target(
                    adapter,
                    record,
                    seed=example_seed,
                    kwargs=kwargs,
                )
            if not isinstance(adapted, Image.Image):
                raise TypeError("generation adapter must return a PIL image")
            if not (resume and valid_image(paths["adapted"])):
                adapted.save(paths["adapted"])

            if not (resume and valid_image(paths["removal"])):
                _reset_condition_cache(adapter)
                with scale_context(route_module, scale=0.0):
                    removal = _generate_visual_target(
                        adapter,
                        record,
                        seed=example_seed,
                        kwargs=kwargs,
                    )
                if not isinstance(removal, Image.Image):
                    raise TypeError("generation adapter must return a PIL image")
                removal.save(paths["removal"])

            if not (resume and valid_image(paths["replay"])):
                _reset_condition_cache(adapter)
                with (
                    scale_context(route_module, scale=0.0),
                    patch_context(route_module, realized, strict=True),
                ):
                    replay = _generate_visual_target(
                        adapter,
                        record,
                        seed=example_seed,
                        kwargs=kwargs,
                    )
                if not isinstance(replay, Image.Image):
                    raise TypeError("generation adapter must return a PIL image")
                replay.save(paths["replay"])

        for name in conditions:
            rows[name].append(
                {
                    "split_key": identifier,
                    "prompt": prompt,
                    "seed": example_seed,
                    "image": f"images/{index:05d}.png",
                    "route_intervention": name,
                    **_visual_target_metadata(record),
                }
            )

    manifests: dict[str, Path] = {}
    for name in conditions:
        manifest = output / name / "predictions.jsonl"
        temporary = manifest.with_suffix(".jsonl.partial")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[name]),
            encoding="utf-8",
        )
        temporary.replace(manifest)
        manifests[name] = manifest
    return manifests
