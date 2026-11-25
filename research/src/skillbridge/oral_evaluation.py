from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        message = f"JSONL must contain non-empty objects: {path}"
        raise ValueError(message)
    return rows


def _index(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("split_key", ""))
        if not key or key in indexed:
            message = f"{name} requires unique non-empty split keys"
            raise ValueError(message)
        indexed[key] = row
    return indexed


def _link_image(source: Path, destination: Path) -> None:
    if not source.is_file():
        message = f"generated image is missing: {source}"
        raise FileNotFoundError(message)
    if destination.exists():
        message = f"refusing to overwrite staged image: {destination}"
        raise FileExistsError(message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _safe_prompt_filename(prompt: str) -> str:
    collapsed = " ".join(prompt.replace("_", " ").replace("/", " or ").split())
    cleaned = "".join(character for character in collapsed if character not in '\\:*?"<>|')
    if not cleaned or len(cleaned.encode("utf-8")) > 180:
        message = "T2I-CompBench prompt cannot form a safe official filename"
        raise ValueError(message)
    return cleaned


def _t2i_category(benchmark: str) -> str:
    mapping = {
        "t2i_compbench_counting": "numeracy",
        "t2i_compbench_2d": "2d_spatial",
        "t2i_compbench_3d": "3d_spatial",
        "t2i_compbench_nonspatial": "non_spatial",
        "t2i_compbench_color": "color_binding",
        "t2i_compbench_shape": "shape_binding",
        "t2i_compbench_texture": "texture_binding",
    }
    try:
        return mapping[benchmark]
    except KeyError as error:
        message = f"unknown T2I-CompBench subset: {benchmark}"
        raise ValueError(message) from error


def stage_oral_predictions(  # noqa: C901, PLR0912, PLR0915
    records_path: Path, predictions_path: Path, output: Path
) -> dict[str, object]:
    """Join a locked trajectory with target records and stage native evaluator inputs."""
    if output.exists():
        message = f"refusing to reuse evaluator stage: {output}"
        raise FileExistsError(message)
    records = _index(_read_jsonl(records_path), name="target records")
    predictions = _index(_read_jsonl(predictions_path), name="predictions")
    if records.keys() != predictions.keys():
        message = "target records and predictions require identical split keys"
        raise ValueError(message)

    output.mkdir(parents=True)
    prediction_root = predictions_path.parent
    counts: Counter[str] = Counter()
    concept_items: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    dsg_items: list[dict[str, object]] = []
    wise_items: list[dict[str, object]] = []
    text_items: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    world_items: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    index_rows: list[dict[str, object]] = []

    for split_key, record in records.items():
        prediction = predictions[split_key]
        prompt = str(record.get("user_prompt", ""))
        if prompt != str(prediction.get("prompt", "")):
            message = f"prompt mismatch for split key: {split_key}"
            raise ValueError(message)
        source = (prediction_root / str(prediction.get("image", ""))).resolve()
        benchmark = str(record.get("benchmark", ""))
        benchmark_id = str(record.get("benchmark_id", ""))
        metadata = record.get("evaluator_metadata", {})
        if not benchmark_id or not isinstance(metadata, dict):
            message = f"record lacks evaluator identity: {split_key}"
            raise ValueError(message)

        if benchmark == "spatialgeneval":
            relative = Path("spatialgeneval/images") / f"{int(benchmark_id):06d}.png"
        elif benchmark.startswith("t2i_compbench_"):
            category = _t2i_category(benchmark)
            filename = f"{_safe_prompt_filename(prompt)}_{int(benchmark_id):06d}.png"
            relative = Path("t2i_compbench") / category / "samples" / filename
        elif benchmark == "conceptmix" or benchmark.startswith("conceptmix_"):
            difficulty = int(metadata.get("difficulty", 0))
            if difficulty < 1:
                message = f"ConceptMix record lacks difficulty: {split_key}"
                raise ValueError(message)
            relative = Path("conceptmix") / f"k={difficulty}" / "images" / f"{benchmark_id}.png"
            concept_items[difficulty].append(
                {
                    "image_path": str((output / relative).resolve()),
                    "prompt": prompt,
                    "question": metadata.get("questions", []),
                    "benchmark_id": benchmark_id,
                    "split_key": split_key,
                }
            )
        elif benchmark.startswith("dsg1k_"):
            relative = Path("dsg/images") / f"{benchmark_id}.png"
            dsg_items.append(
                {
                    "image_path": str((output / relative).resolve()),
                    "prompt": prompt,
                    "questions": metadata.get("questions", []),
                    "dependencies": metadata.get("dependencies", []),
                    "benchmark_id": benchmark_id,
                    "split_key": split_key,
                }
            )
        elif benchmark == "wise_verified":
            relative = Path("wise_verified/images") / f"{int(benchmark_id)}.png"
            wise_items.append(
                {
                    "benchmark_id": benchmark_id,
                    "category": str(metadata.get("category", "")),
                    "explanation": str(metadata.get("explanation", "")),
                    "image_path": str((output / relative).resolve()),
                    "prompt": prompt,
                    "split_key": split_key,
                }
            )
        elif benchmark.startswith("anytext_"):
            language = benchmark.removeprefix("anytext_")
            relative = Path("anytext") / language / "images" / f"{benchmark_id}.png"
            text_items[("anytext", language)].append(
                {
                    "benchmark_id": benchmark_id,
                    "image_path": str((output / relative).resolve()),
                    "prompt": prompt,
                    "split_key": split_key,
                    "target_text": str(record.get("target_value", "")),
                }
            )
        elif benchmark.startswith("wegenbench_text_"):
            language = benchmark.removeprefix("wegenbench_text_")
            relative = Path("wegenbench/text") / language / "images" / f"{benchmark_id}.png"
            text_items[("wegenbench/text", language)].append(
                {
                    "benchmark_id": benchmark_id,
                    "difficulty": metadata.get("difficulty", ""),
                    "image_path": str((output / relative).resolve()),
                    "prompt": prompt,
                    "split_key": split_key,
                    "target_text": str(record.get("target_value", "")),
                    "type": metadata.get("type", ""),
                }
            )
        elif benchmark == "worldgenbench":
            checklist = metadata.get("checklist", [])
            if not isinstance(checklist, list) or not checklist:
                message = f"WorldGenBench record lacks checklist: {split_key}"
                raise ValueError(message)
            relative = Path("worldgenbench/images") / f"{benchmark_id}.png"
            world_items[benchmark].append(
                {
                    "benchmark_id": benchmark_id,
                    "checklist": checklist,
                    "domain": metadata.get("domain", ""),
                    "image_path": str((output / relative).resolve()),
                    "prompt": prompt,
                    "split_key": split_key,
                }
            )
        elif benchmark == "picworld":
            relative = Path("picworld/images") / f"{benchmark_id}.png"
            world_items[benchmark].append(
                {
                    "benchmark_id": benchmark_id,
                    "category": metadata.get("category", ""),
                    "image_path": str((output / relative).resolve()),
                    "prompt": prompt,
                    "split_key": split_key,
                    "subcategory": metadata.get("subcategory", ""),
                }
            )
        else:
            message = f"no oral evaluator layout for benchmark: {benchmark}"
            raise ValueError(message)
        _link_image(source, output / relative)
        counts[benchmark] += 1
        index_rows.append(
            {
                "split_key": split_key,
                "benchmark": benchmark,
                "benchmark_id": benchmark_id,
                "image": str(relative),
            }
        )

    for difficulty, items in concept_items.items():
        path = output / "conceptmix" / f"k={difficulty}" / "items.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    if dsg_items:
        path = output / "dsg/items.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dsg_items, indent=2) + "\n", encoding="utf-8")
    if wise_items:
        path = output / "wise_verified/items.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(wise_items, indent=2) + "\n", encoding="utf-8")
    for (family, language), items in text_items.items():
        path = output / family / language / "items.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for benchmark, items in world_items.items():
        path = output / benchmark / "items.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )

    report: dict[str, object] = {
        "format_version": 1,
        "records": len(index_rows),
        "benchmarks": dict(sorted(counts.items())),
        "protocols": {
            "spatialgeneval": {
                "judge": "Qwen2.5-VL-72B-Instruct",
                "rollouts": 5,
                "vote_threshold": 4,
                "credential_required": False,
            },
            "t2i_compbench": {
                "judge": "official UniDet/depth or BLIP-VQA pipeline by subset",
                "credential_required": False,
            },
            "conceptmix": {
                "judge": "official GPT grading path",
                "credential_required": True,
            },
            "dsg": {
                "judge": "fixed VQA model with official dependency scoring",
                "credential_required": False,
            },
            "wise_verified": {
                "judge": "Qwen3.5-35B-A3B",
                "credential_required": False,
            },
            "anytext": {
                "judge": "official OCR recognizer",
                "metrics": ["sentence_accuracy", "normalized_edit_distance"],
                "credential_required": False,
            },
            "wegenbench_text": {
                "judge": "released TextPecker and OCR scoring chain",
                "metrics": ["edit_similarity", "sentence_accuracy", "character_f1"],
                "credential_required": False,
            },
            "worldgenbench": {
                "judge": "knowledge checklist score",
                "credential_required": False,
            },
            "picworld": {
                "judge": "PW-Agent specification",
                "official_code_available": False,
                "credential_required": False,
            },
        },
    }
    (output / "staging_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
