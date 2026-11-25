from __future__ import annotations

from skillbridge.artifact_policy import forbidden_hashing as hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from PIL import Image


@dataclass(frozen=True, slots=True)
class EvaluatorLock:
    benchmark: str
    repository_url: str
    commit: str
    required_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluatorRepositoryReport:
    repository: Path
    expected_commit: str
    actual_commit: str | None
    missing_files: tuple[str, ...]
    commit_matches: bool

    @property
    def ready(self) -> bool:
        return not self.missing_files and self.commit_matches


_REQUIRED_FILES = {
    "geneval": (
        "evaluation/evaluate_images.py",
        "evaluation/summary_scores.py",
    ),
    "spatialgeneval": (
        "eval/SpatialGenEval_T2I_Prompts.jsonl",
        "scripts/spatialgeneval_stage1_eval.py",
        "scripts/spatialgeneval_stage2_acc.py",
    ),
    "t2i_compbench": (
        "UniDet_eval/2D_spatial_eval.py",
        "UniDet_eval/3D_spatial_eval.py",
        "UniDet_eval/numeracy_eval.py",
        "BLIPvqa_eval/BLIP_vqa.py",
        "CLIPScore_eval/CLIP_similarity.py",
    ),
    "conceptmix": (
        "dynamix/grading/gpt_grading.py",
        "dynamix/grading/multi_grading.py",
    ),
    "dsg": (
        "dsg/data/dsg-1k-anns.csv",
        "dsg/vqa_utils.py",
        "scripts/t2i_eval_dsg1k.py",
    ),
    "wise_verified": (
        "vllm_eval.py",
        "calculate_verified.py",
        "data_verified/merge.json",
    ),
    "anytext": (
        "eval/eval_ocr.sh",
        "eval/eval_fid.sh",
        "ocr_recog/RecModel.py",
    ),
    "wegenbench_text": (
        "evaluation/Text/recognition/eval_textpecker.py",
        "evaluation/Text/scoring/score_text.py",
        "data/prompts/wegenbench_text_en.jsonl",
        "data/prompts/wegenbench_text_cn.jsonl",
    ),
}

_T2I_COMPBENCH_SUBSETS = {
    "numeracy": (
        "UniDet_eval/numeracy_eval.py",
        "annotation_num/vqa_result.json",
        "--outpath",
        "UniDet_eval",
    ),
    "2d_spatial": (
        "UniDet_eval/2D_spatial_eval.py",
        "labels/annotation_obj_detection_2d/vqa_result.json",
        "--outpath",
        "UniDet_eval",
    ),
    "3d_spatial": (
        "UniDet_eval/3D_spatial_eval.py",
        "labels/annotation_obj_detection_3d/vqa_result.json",
        "--outpath",
        "UniDet_eval",
    ),
    "color_binding": (
        "BLIPvqa_eval/BLIP_vqa.py",
        "annotation_blip/vqa_result.json",
        "--out_dir",
        "BLIPvqa_eval",
    ),
    "shape_binding": (
        "BLIPvqa_eval/BLIP_vqa.py",
        "annotation_blip/vqa_result.json",
        "--out_dir",
        "BLIPvqa_eval",
    ),
    "texture_binding": (
        "BLIPvqa_eval/BLIP_vqa.py",
        "annotation_blip/vqa_result.json",
        "--out_dir",
        "BLIPvqa_eval",
    ),
    "non_spatial": (
        "CLIPScore_eval/CLIP_similarity.py",
        "annotation_clip/vqa_result.json",
        "--outpath",
        "CLIPScore_eval",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes(), usedforsecurity=False).hexdigest()


def _path_sha256(path: Path) -> tuple[str, list[dict[str, str]]]:
    if path.is_file():
        digest = _sha256(path)
        return digest, [{"path": path.name, "sha256": digest}]
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        msg = f"detector path contains no hashable files: {path}"
        raise FileNotFoundError(msg)
    entries = [{"path": str(file.relative_to(path)), "sha256": _sha256(file)} for file in files]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded, usedforsecurity=False).hexdigest(), entries


def load_evaluator_lock(path: Path, benchmark: str) -> EvaluatorLock:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload.get(benchmark) if isinstance(payload, dict) else None
    if not isinstance(item, dict):
        msg = f"evaluator lock lacks benchmark: {benchmark}"
        raise ValueError(msg)
    repository = item.get("repository")
    commit = item.get("commit")
    if not isinstance(repository, str) or not repository.startswith("https://"):
        msg = f"evaluator repository URL is invalid for {benchmark}"
        raise ValueError(msg)
    if not isinstance(commit, str) or len(commit) != 40:
        msg = f"evaluator commit must be a full 40-character hash for {benchmark}"
        raise ValueError(msg)
    required = _REQUIRED_FILES.get(benchmark)
    if required is None:
        msg = f"no executable evaluator bridge is registered for {benchmark}"
        raise ValueError(msg)
    return EvaluatorLock(benchmark, repository, commit, required)


def validate_evaluator_repository(
    repository: Path,
    lock: EvaluatorLock,
    *,
    verify_commit: bool = True,
) -> EvaluatorRepositoryReport:
    repo = repository.resolve()
    missing = tuple(relative for relative in lock.required_files if not (repo / relative).is_file())
    if missing:
        msg = f"evaluator repository is missing required files: {list(missing)}"
        raise FileNotFoundError(msg)
    actual_commit: str | None = None
    commit_matches = True
    if verify_commit:
        git = which("git")
        if git is None:
            msg = "git is required to verify the evaluator revision"
            raise FileNotFoundError(msg)
        result = subprocess.run(  # noqa: S603
            (git, "-C", str(repo), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
        actual_commit = result.stdout.strip() if result.returncode == 0 else None
        commit_matches = actual_commit == lock.commit
        if not commit_matches:
            msg = (
                f"evaluator revision mismatch for {lock.benchmark}: "
                f"expected {lock.commit}, found {actual_commit}"
            )
            raise ValueError(msg)
    return EvaluatorRepositoryReport(repo, lock.commit, actual_commit, missing, commit_matches)


def stage_geneval_sample(image: Image.Image, root: Path, *, benchmark_id: str) -> Path:
    try:
        numeric_id = int(benchmark_id)
    except ValueError as error:
        msg = f"GenEval benchmark id must be numeric: {benchmark_id}"
        raise ValueError(msg) from error
    path = root / f"{numeric_id:05d}" / "samples" / "00000.png"
    if path.exists():
        msg = f"refusing to overwrite staged GenEval sample: {path}"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def write_geneval_metadata(
    root: Path,
    records: Iterable[tuple[str, dict[str, object]]],
) -> Path:
    path = root / "metadata.jsonl"
    if path.exists():
        msg = f"refusing to overwrite GenEval metadata: {path}"
        raise FileExistsError(msg)
    ordered = sorted(records, key=lambda item: int(item[0]))
    if not ordered:
        msg = "GenEval metadata must be non-empty"
        raise ValueError(msg)
    identifiers = [int(identifier) for identifier, _ in ordered]
    if identifiers != list(range(len(ordered))):
        msg = "staged GenEval ids must be contiguous from zero"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(metadata, sort_keys=True) + "\n" for _, metadata in ordered),
        encoding="utf-8",
    )
    return path


def run_geneval_evaluator(
    repository: Path,
    *,
    lock: EvaluatorLock,
    image_root: Path,
    detector_path: Path,
    output_jsonl: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Run the revision-locked official detector and summary scripts."""
    report = validate_evaluator_repository(repository, lock)
    metadata = image_root / "metadata.jsonl"
    if not metadata.is_file() or not detector_path.exists():
        msg = "GenEval scoring requires staged metadata and detector weights"
        raise FileNotFoundError(msg)
    if output_jsonl.exists() or summary_path.exists():
        msg = "refusing to overwrite official evaluator outputs"
        raise FileExistsError(msg)
    image_paths = sorted(image_root.glob("*/samples/*.png"))
    metadata_count = sum(
        bool(line.strip()) for line in metadata.read_text(encoding="utf-8").splitlines()
    )
    staged_ids = sorted({int(path.parent.parent.name) for path in image_paths})
    if metadata_count < 1 or staged_ids != list(range(metadata_count)):
        msg = "GenEval staged image directories must match every contiguous metadata row"
        raise ValueError(msg)
    evaluate = report.repository / "evaluation/evaluate_images.py"
    summarize = report.repository / "evaluation/summary_scores.py"
    subprocess.run(  # noqa: S603
        (
            sys.executable,
            str(evaluate),
            str(image_root),
            "--outfile",
            str(output_jsonl),
            "--model-path",
            str(detector_path),
        ),
        check=True,
        cwd=report.repository,
    )
    summary = subprocess.run(  # noqa: S603
        (sys.executable, str(summarize), str(output_jsonl)),
        check=True,
        capture_output=True,
        text=True,
        cwd=report.repository,
    )
    summary_path.write_text(summary.stdout, encoding="utf-8")
    primary_score = geneval_primary_score(output_jsonl)
    detector_sha256, detector_files = _path_sha256(detector_path)
    return {
        "format_version": 1,
        "benchmark": lock.benchmark,
        "repository": str(report.repository),
        "commit": report.actual_commit,
        "evaluator_files": [
            {"path": relative, "sha256": _sha256(report.repository / relative)}
            for relative in lock.required_files
        ],
        "metadata_sha256": _sha256(metadata),
        "detector_sha256": detector_sha256,
        "detector_files": detector_files,
        "images": [
            {"path": str(path.relative_to(image_root)), "sha256": _sha256(path)}
            for path in image_paths
        ],
        "parameters": {
            "image_root": str(image_root),
            "model_path": str(detector_path),
            "output_jsonl": str(output_jsonl),
            "summary_path": str(summary_path),
        },
        "scores_sha256": _sha256(output_jsonl),
        "summary_sha256": _sha256(summary_path),
        "primary_score": primary_score,
    }


def geneval_primary_score(path: Path) -> float:
    """Aggregate official per-prompt correctness without parsing human-readable stdout."""
    values: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            msg = f"GenEval score line {line_number} must be a JSON object"
            raise ValueError(msg)
        correct = payload.get("correct")
        if isinstance(correct, (bool, int, float)):
            values.append(float(correct))
            continue
        results = payload.get("results")
        if isinstance(results, list):
            values.extend(
                float(result["correct"])
                for result in results
                if isinstance(result, dict)
                and isinstance(result.get("correct"), (bool, int, float))
            )
    if not values:
        msg = f"official GenEval output contains no correctness values: {path}"
        raise ValueError(msg)
    return sum(values) / len(values)


def spatialgeneval_scores(path: Path, *, expected_rows: int | None = 1230) -> dict[str, object]:
    """Aggregate the official ten-question SpatialGenEval records."""
    identifiers: set[str] = set()
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            msg = f"SpatialGenEval line {line_number} must be a JSON object"
            raise ValueError(msg)
        identifier = str(payload.get("id", ""))
        outcomes = payload.get("true-or-false")
        if (
            not identifier
            or identifier in identifiers
            or not isinstance(outcomes, list)
            or len(outcomes) != 10
            or any(not isinstance(value, (bool, int)) for value in outcomes)
        ):
            msg = f"invalid SpatialGenEval record on line {line_number}"
            raise ValueError(msg)
        identifiers.add(identifier)
        rows.append([float(bool(value)) for value in outcomes])
    if not rows or (expected_rows is not None and len(rows) != expected_rows):
        msg = (
            f"SpatialGenEval expected {expected_rows} rows, found {len(rows)}"
            if expected_rows is not None
            else "SpatialGenEval output is empty"
        )
        raise ValueError(msg)
    per_dimension = [sum(column) / len(rows) for column in zip(*rows, strict=True)]
    return {
        "format_version": 1,
        "benchmark": "spatialgeneval",
        "rows": len(rows),
        "average_accuracy": sum(per_dimension) / 10,
        "basic_accuracy": sum(per_dimension[:2]) / 2,
        "spatial_accuracy": sum(per_dimension[2:]) / 8,
        "per_dimension": per_dimension,
    }


_WISE_CATEGORIES = (
    ("CULTURE", range(1, 401), 0.40),
    ("TIME", range(401, 521), 0.12),
    ("SPACE", range(521, 641), 0.12),
    ("BIOLOGY", range(641, 761), 0.12),
    ("PHYSICS", range(761, 881), 0.12),
    ("CHEMISTRY", range(881, 1001), 0.12),
)


def _wise_category(prompt_id: int) -> tuple[str, float]:
    for category, prompt_ids, weight in _WISE_CATEGORIES:
        if prompt_id in prompt_ids:
            return category, weight
    msg = f"WISE prompt id is outside the official range: {prompt_id}"
    raise ValueError(msg)


def wise_verified_scores(  # noqa: C901
    paths: Iterable[Path], *, expected_ids: set[int] | None = None
) -> dict[str, object]:
    """Aggregate binary WISE_Verified judge rows with official category weights."""
    scores: dict[int, float] = {}
    by_category: dict[str, list[float]] = {category: [] for category, _, _ in _WISE_CATEGORIES}
    for path in paths:
        if not path.is_file():
            msg = f"official WISE score file is missing: {path}"
            raise FileNotFoundError(msg)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                msg = f"WISE score line {line_number} must be a JSON object"
                raise ValueError(msg)
            prompt_id = payload.get("prompt_id")
            score = payload.get("score")
            if not isinstance(prompt_id, int) or prompt_id in scores:
                msg = f"invalid or duplicate WISE prompt id on line {line_number}"
                raise ValueError(msg)
            if isinstance(score, bool) or score not in (0, 1, 0.0, 1.0):
                msg = f"WISE score must be binary on line {line_number}"
                raise ValueError(msg)
            category, _ = _wise_category(prompt_id)
            value = float(score)
            scores[prompt_id] = value
            by_category[category].append(value)
    if not scores:
        msg = "official WISE output is empty"
        raise ValueError(msg)
    if expected_ids is not None and scores.keys() != expected_ids:
        msg = "official WISE rows do not match staged image ids"
        raise ValueError(msg)
    categories = {
        category: {"rows": len(values), "score": sum(values) / len(values)}
        for category, values in by_category.items()
        if values
    }
    missing_categories = [
        category for category, _, _ in _WISE_CATEGORIES if category not in categories
    ]
    if missing_categories:
        msg = f"WISE split cannot compute the official weighted score: {missing_categories}"
        raise ValueError(msg)
    wiscore = sum(
        weight * float(categories[category]["score"]) for category, _, weight in _WISE_CATEGORIES
    )
    return {
        "format_version": 1,
        "benchmark": "wise_verified",
        "rows": len(scores),
        "categories": categories,
        "wiscore": wiscore,
    }


def _load_wise_staged_prompts(
    staged_root: Path,
) -> tuple[list[dict[str, object]], set[int], Path]:
    items_path = staged_root / "items.json"
    images = staged_root / "images"
    if not items_path.is_file() or not images.is_dir():
        msg = f"WISE scoring requires staged items and images under {staged_root}"
        raise FileNotFoundError(msg)
    raw_items = json.loads(items_path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list) or not raw_items:
        msg = f"WISE staged items must be a non-empty list: {items_path}"
        raise ValueError(msg)
    prompts: list[dict[str, object]] = []
    expected_ids: set[int] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            msg = f"invalid WISE staged item in {items_path}"
            raise ValueError(msg)
        try:
            prompt_id = int(item["benchmark_id"])
        except (KeyError, TypeError, ValueError) as error:
            msg = f"WISE staged item lacks a numeric benchmark id: {item}"
            raise ValueError(msg) from error
        image = images / f"{prompt_id}.png"
        if prompt_id in expected_ids or not image.is_file():
            msg = f"WISE staged item has a duplicate id or missing image: {prompt_id}"
            raise ValueError(msg)
        expected_ids.add(prompt_id)
        prompts.append(
            {
                "prompt_id": prompt_id,
                "Prompt": str(item.get("prompt", "")),
                "Explanation": str(item.get("explanation", "")),
                "Subcategory": str(item.get("category", "")),
            }
        )
    try:
        image_ids = {int(image.stem) for image in images.glob("*.png")}
    except ValueError as error:
        msg = "WISE staged image lacks a numeric id"
        raise ValueError(msg) from error
    if image_ids != expected_ids:
        msg = "WISE staged item ids do not match staged image ids"
        raise ValueError(msg)
    return prompts, expected_ids, images


def run_wise_verified_evaluator(
    repository: Path,
    *,
    lock: EvaluatorLock,
    staged_root: Path,
    output_root: Path,
    model: str,
    api_base: str,
    python_executable: Path = Path(sys.executable),
    max_workers: int = 16,
) -> dict[str, object]:
    """Run the locked WISE_Verified judge and aggregate its machine output."""
    report = validate_evaluator_repository(repository, lock)
    if lock.benchmark != "wise_verified":
        msg = f"WISE runner received the {lock.benchmark} lock"
        raise ValueError(msg)
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        msg = f"refusing to overwrite official evaluator output: {summary_path}"
        raise FileExistsError(msg)
    prompts, expected_ids, images = _load_wise_staged_prompts(staged_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prompts_path = output_root / "prompts.json"
    prompts_path.write_text(
        json.dumps(sorted(prompts, key=lambda row: int(row["prompt_id"])), indent=2) + "\n",
        encoding="utf-8",
    )
    full_name = "wise_full.json"
    scores_name = "wise_scores.jsonl"
    command = (
        str(python_executable),
        str(report.repository / "vllm_eval.py"),
        "--json_path",
        str(prompts_path),
        "--image_dir",
        str(images),
        "--output_dir",
        str(output_root),
        "--model",
        model,
        "--api_base",
        api_base,
        "--result_full",
        full_name,
        "--result_scores",
        scores_name,
        "--max_workers",
        str(max_workers),
    )
    subprocess.run(command, check=True, cwd=report.repository)  # noqa: S603
    summary = wise_verified_scores((output_root / scores_name,), expected_ids=expected_ids)
    summary["commit"] = report.actual_commit
    summary["judge"] = model
    summary["api_base"] = api_base
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _t2i_staged_ids(samples: Path) -> set[int]:
    identifiers: set[int] = set()
    for image in sorted(samples.glob("*.png")):
        suffix = image.stem.rsplit("_", maxsplit=1)[-1]
        try:
            identifier = int(suffix)
        except ValueError as error:
            msg = f"T2I-CompBench image lacks an integer suffix: {image.name}"
            raise ValueError(msg) from error
        if identifier in identifiers:
            msg = f"duplicate T2I-CompBench image id: {identifier}"
            raise ValueError(msg)
        identifiers.add(identifier)
    if not identifiers:
        msg = f"T2I-CompBench sample directory is empty: {samples}"
        raise ValueError(msg)
    return identifiers


def _t2i_category_scores(category: str, subset: Path, result_path: Path) -> list[float]:
    expected_ids = _t2i_staged_ids(subset / "samples")
    if not result_path.is_file():
        msg = f"official T2I-CompBench output is missing: {result_path}"
        raise FileNotFoundError(msg)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        msg = f"official T2I-CompBench output must be a non-empty list: {result_path}"
        raise ValueError(msg)
    scores_by_id: dict[int, float] = {}
    for row in payload:
        if not isinstance(row, dict):
            msg = f"invalid T2I-CompBench score row in {result_path}"
            raise ValueError(msg)
        identifier = row.get("question_id")
        score = row.get("answer")
        lower_bound = -1.0 if category == "non_spatial" else 0.0
        if (
            not isinstance(identifier, int)
            or identifier in scores_by_id
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not lower_bound <= float(score) <= 1.0
        ):
            msg = f"invalid T2I-CompBench score row in {result_path}"
            raise ValueError(msg)
        scores_by_id[identifier] = float(score)
    if scores_by_id.keys() != expected_ids:
        msg = f"official T2I-CompBench rows do not match staged image ids: {category}"
        raise ValueError(msg)
    return list(scores_by_id.values())


def t2i_compbench_scores(root: Path) -> dict[str, object]:
    """Aggregate official T2I-CompBench outputs and bind them to staged images."""
    categories: dict[str, dict[str, object]] = {}
    all_scores: list[float] = []
    for category, (_, relative_result, _, _) in _T2I_COMPBENCH_SUBSETS.items():
        subset = root / category
        if not subset.exists():
            continue
        result_path = subset / relative_result
        values = _t2i_category_scores(category, subset, result_path)
        all_scores.extend(values)
        categories[category] = {
            "rows": len(values),
            "score": sum(values) / len(values),
            "result": str(result_path),
        }
    if not categories:
        msg = f"no staged T2I-CompBench subsets found under {root}"
        raise FileNotFoundError(msg)
    category_scores = [float(item["score"]) for item in categories.values()]
    return {
        "format_version": 1,
        "benchmark": "t2i_compbench",
        "categories": categories,
        "rows": len(all_scores),
        "micro_average": sum(all_scores) / len(all_scores),
        "macro_average": sum(category_scores) / len(category_scores),
    }


def run_t2i_compbench_evaluator(
    repository: Path,
    *,
    lock: EvaluatorLock,
    staged_root: Path,
    output_json: Path,
    python_executable: Path = Path(sys.executable),
    verify_revision: bool = True,
) -> dict[str, object]:
    """Run every staged subset through the revision-locked official evaluator."""
    report = validate_evaluator_repository(
        repository,
        lock,
        verify_commit=verify_revision,
    )
    if lock.benchmark != "t2i_compbench":
        msg = f"T2I-CompBench runner received the {lock.benchmark} lock"
        raise ValueError(msg)
    if output_json.exists():
        msg = f"refusing to overwrite official evaluator output: {output_json}"
        raise FileExistsError(msg)
    evaluated = 0
    for category, (
        relative_script,
        relative_result,
        output_argument,
        working_directory,
    ) in _T2I_COMPBENCH_SUBSETS.items():
        subset = staged_root / category
        if not subset.exists():
            continue
        result_path = subset / relative_result
        if result_path.exists():
            _t2i_category_scores(category, subset, result_path)
            evaluated += 1
            continue
        _t2i_staged_ids(subset / "samples")
        command = [
            str(python_executable),
            str(report.repository / relative_script),
            output_argument,
            str(subset.resolve()),
        ]
        environment = os.environ.copy()
        repository_path = str(report.repository.resolve())
        evaluator_path = str((report.repository / working_directory).resolve())
        inherited_pythonpath = environment.get("PYTHONPATH")
        python_paths = [evaluator_path, repository_path]
        if inherited_pythonpath:
            python_paths.append(inherited_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        subprocess.run(  # noqa: S603
            tuple(command),
            check=True,
            cwd=report.repository / working_directory,
            env=environment,
        )
        evaluated += 1
    if evaluated == 0:
        msg = f"no staged T2I-CompBench subsets found under {staged_root}"
        raise FileNotFoundError(msg)
    summary = t2i_compbench_scores(staged_root)
    summary["commit"] = report.actual_commit
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
