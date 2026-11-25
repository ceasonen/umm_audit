from __future__ import annotations

from skillbridge.artifact_policy import forbidden_hashing as hashlib
import json
import random
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from PIL import Image, ImageDraw

RouteTask = Literal["understanding", "generation"]
RouteSplit = Literal["train", "development", "test"]
Capability = Literal[
    "counting",
    "spatial",
    "visual_attribute",
    "text",
    "world_knowledge",
]

_CAPABILITIES: tuple[Capability, ...] = ("counting", "spatial", "text")
_SPLIT_OFFSETS: dict[RouteSplit, int] = {
    "train": 0,
    "development": 100_000,
    "test": 200_000,
}


class RouteRecord(TypedDict):
    system_prompt: str
    user_prompt: str
    user_image: str
    answer_text: str
    answer_image: str
    capability: Capability
    split_key: str
    semantic_key: NotRequired[str]
    render_key: NotRequired[str]
    benchmark_id: NotRequired[str]
    benchmark: NotRequired[str]
    evaluator_metadata: NotRequired[dict[str, object]]
    target_value: str


def _semantic_spec(capability: Capability, index: int, rng: random.Random) -> dict[str, object]:
    if capability == "counting":
        return {
            "capability": capability,
            "count": 1 + index % 5,
            "color": ("red", "blue", "green")[rng.randrange(3)],
            "variant": index,
        }
    if capability == "spatial":
        return {
            "capability": capability,
            "relation": ("left", "right", "above", "below")[index % 4],
            "variant": index,
        }
    return {
        "capability": capability,
        "text": f"R{index:06d}",
        "variant": index,
    }


def _split_key(spec: dict[str, object]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload, usedforsecurity=False).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes(), usedforsecurity=False).hexdigest()


def _render(spec: dict[str, object], path: Path) -> None:
    capability = spec["capability"]
    variant_rng = random.Random(cast("int", spec["variant"]))
    background = tuple(240 + variant_rng.randrange(16) for _ in range(3))
    image = Image.new("RGB", (256, 256), background)
    draw = ImageDraw.Draw(image)
    if capability == "counting":
        count = cast("int", spec["count"])
        color = str(spec["color"])
        for index in range(count):
            column = index % 3
            row = index // 3
            left = 21 + variant_rng.randrange(21) + column * 70
            top = 51 + variant_rng.randrange(21) + row * 80
            diameter = 34 + variant_rng.randrange(9)
            width = 1 + variant_rng.randrange(3)
            draw.ellipse(
                (left, top, left + diameter, top + diameter),
                fill=color,
                outline="black",
                width=width,
            )
    elif capability == "spatial":
        relation = str(spec["relation"])
        positions = {
            "left": ((45, 105), (165, 105)),
            "right": ((165, 105), (45, 105)),
            "above": ((105, 45), (105, 165)),
            "below": ((105, 165), (105, 45)),
        }
        red_base, blue_base = positions[relation]
        red = (
            red_base[0] + variant_rng.randrange(-12, 13),
            red_base[1] + variant_rng.randrange(-12, 13),
        )
        blue = (
            blue_base[0] + variant_rng.randrange(-12, 13),
            blue_base[1] + variant_rng.randrange(-12, 13),
        )
        draw.rectangle((*red, red[0] + 42, red[1] + 42), fill="red", outline="black", width=2)
        draw.ellipse((*blue, blue[0] + 42, blue[1] + 42), fill="blue", outline="black", width=2)
    else:
        content = str(spec["text"])
        bounds = draw.textbbox((0, 0), content)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        draw.text(((256 - width) / 2, (256 - height) / 2), content, fill="black")
    image.save(path)


def _prompts(spec: dict[str, object]) -> tuple[str, str]:
    capability = spec["capability"]
    if capability == "counting":
        count = cast("int", spec["count"])
        color = str(spec["color"])
        return f"How many {color} circles are shown?", str(count)
    if capability == "spatial":
        relation = str(spec["relation"])
        return "Where is the red square relative to the blue circle?", relation
    content = str(spec["text"])
    return "Read the text in the image exactly.", content


def _generation_prompt(spec: dict[str, object]) -> str:
    capability = spec["capability"]
    if capability == "counting":
        return f"An image containing exactly {spec['count']} {spec['color']} circles."
    if capability == "spatial":
        return f"A red square {spec['relation']} a blue circle."
    return f"A clean image displaying the exact text {spec['text']}."


def _render_key(path: Path) -> str:
    return hashlib.sha256(path.read_bytes(), usedforsecurity=False).hexdigest()


def generate_route_transfer_split(
    output_dir: Path,
    *,
    task: RouteTask,
    split: RouteSplit,
    examples_per_capability: int,
    seed: int,
    capability: Capability | None = None,
    index_offset: int = 0,
) -> list[RouteRecord]:
    if examples_per_capability < 1:
        msg = "examples_per_capability must be positive"
        raise ValueError(msg)
    if capability is not None and capability not in _CAPABILITIES:
        msg = f"procedural generation does not support capability: {capability}"
        raise ValueError(msg)
    if index_offset < 0:
        msg = "index_offset must be non-negative"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    rng = random.Random(seed + _SPLIT_OFFSETS[split])
    records: list[RouteRecord] = []
    selected = (capability,) if capability is not None else _CAPABILITIES
    for selected_capability in selected:
        for local_index in range(examples_per_capability):
            index = index_offset + _SPLIT_OFFSETS[split] + local_index
            spec = _semantic_spec(selected_capability, index, rng)
            key = _split_key(spec)
            image_path = image_dir / f"{selected_capability}-{key[:12]}.png"
            _render(spec, image_path)
            render_key = _render_key(image_path)
            question, answer = _prompts(spec)
            attributes = {
                selected_capability: answer,
                "color": (
                    str(spec["color"])
                    if selected_capability == "counting"
                    else "red+blue"
                    if selected_capability == "spatial"
                    else "black"
                ),
            }
            record: RouteRecord = {
                "system_prompt": "Follow the multimodal instruction precisely.",
                "user_prompt": question if task == "understanding" else _generation_prompt(spec),
                "user_image": str(image_path) if task == "understanding" else "",
                "answer_text": answer if task == "understanding" else "",
                "answer_image": str(image_path) if task == "generation" else "",
                "capability": selected_capability,
                "split_key": key,
                "semantic_key": key,
                "evaluator_metadata": {"attributes": attributes},
                "render_key": render_key,
                "target_value": answer,
            }
            records.append(record)

    jsonl = "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    records_path = output_dir / "records.jsonl"
    records_path.write_text(jsonl, encoding="utf-8")
    image_paths = sorted(
        {Path(record["user_image"] or record["answer_image"]).resolve() for record in records}
    )
    manifest = {
        "format_version": 1,
        "task": task,
        "split": split,
        "seed": seed,
        "index_offset": index_offset,
        "examples_per_capability": examples_per_capability,
        "capability": capability or "all",
        "records": {"path": records_path.name, "sha256": _file_sha256(records_path)},
        "images": [{"path": str(image), "sha256": _file_sha256(image)} for image in image_paths],
        "count": len(records),
        "split_keys": [record["split_key"] for record in records],
        "semantic_keys": [record["semantic_key"] for record in records],
        "render_keys": [record["render_key"] for record in records],
        "image_sha256": [_file_sha256(image) for image in image_paths],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records


def build_counterfactual_pairs(  # noqa: C901
    records_path: Path,
    *,
    target_attribute: str,
    preserved_attributes: tuple[str, ...],
    output_path: Path,
    maximum_pairs: int,
) -> dict[str, object]:
    if maximum_pairs < 1 or not target_attribute or not preserved_attributes:
        msg = "counterfactual attributes and maximum_pairs must be non-empty and positive"
        raise ValueError(msg)
    if output_path.exists() or output_path.with_name(f"{output_path.stem}.manifest.json").exists():
        msg = f"refusing to overwrite counterfactual pair artifact: {output_path}"
        raise FileExistsError(msg)
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def attributes(record: dict[str, object]) -> dict[str, object]:
        metadata = record.get("evaluator_metadata")
        values = metadata.get("attributes") if isinstance(metadata, dict) else None
        if not isinstance(values, dict):
            msg = "counterfactual records require evaluator_metadata.attributes"
            raise ValueError(msg)
        return cast("dict[str, object]", values)

    pairs: list[dict[str, object]] = []
    ordered = sorted(records, key=lambda record: str(record["split_key"]))
    for recipient in ordered:
        recipient_attributes = attributes(recipient)
        for donor in ordered:
            if recipient is donor:
                continue
            donor_attributes = attributes(donor)
            if (
                target_attribute not in recipient_attributes
                or target_attribute not in donor_attributes
                or recipient_attributes[target_attribute] == donor_attributes[target_attribute]
            ):
                continue
            if any(
                attribute not in recipient_attributes
                or attribute not in donor_attributes
                or recipient_attributes[attribute] != donor_attributes[attribute]
                for attribute in preserved_attributes
            ):
                continue
            pairs.append(
                {
                    "recipient_id": str(recipient["split_key"]),
                    "donor_id": str(donor["split_key"]),
                    "target_attribute": target_attribute,
                    "preserved_attributes": list(preserved_attributes),
                    "recipient_attributes": {
                        key: str(recipient_attributes[key])
                        for key in (target_attribute, *preserved_attributes)
                    },
                    "donor_attributes": {
                        key: str(donor_attributes[key])
                        for key in (target_attribute, *preserved_attributes)
                    },
                }
            )
            break
        if len(pairs) >= maximum_pairs:
            break
    if len(pairs) < maximum_pairs:
        msg = f"only {len(pairs)} valid counterfactual pairs found; requested {maximum_pairs}"
        raise ValueError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact: dict[str, object] = {
        "format_version": 1,
        "records_path": str(records_path),
        "records_sha256": _file_sha256(records_path),
        "pairs_path": str(output_path),
        "pairs_sha256": _file_sha256(output_path),
        "target_attribute": target_attribute,
        "preserved_attributes": list(preserved_attributes),
        "count": len(pairs),
    }
    output_path.with_name(f"{output_path.stem}.manifest.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_counterfactual_pair_manifest(pairs_path: Path, records_path: Path) -> Path:
    manifest_path = pairs_path.with_name(f"{pairs_path.stem}.manifest.json")
    if not manifest_path.is_file():
        msg = f"counterfactual pair manifest is missing: {manifest_path}"
        raise FileNotFoundError(msg)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "counterfactual pair manifest must be a JSON object"
        raise ValueError(msg)
    checks = (
        ("pairs_path", "pairs_sha256", pairs_path),
        ("records_path", "records_sha256", records_path),
    )
    for path_key, hash_key, current_path in checks:
        if payload.get(path_key) != str(current_path) or payload.get(hash_key) != _file_sha256(
            current_path
        ):
            msg = f"counterfactual pair manifest does not bind current {path_key}"
            raise ValueError(msg)
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or payload.get("count") != len(pairs):
        msg = "counterfactual pair count does not match its manifest"
        raise ValueError(msg)
    return manifest_path


def audit_split_manifests(paths: list[Path]) -> dict[str, object]:
    if len(paths) < 2:
        msg = "at least two split manifests are required"
        raise ValueError(msg)
    key_sets: dict[str, set[str]] = {}
    semantic_sets: dict[str, set[str]] = {}
    render_sets: dict[str, set[str]] = {}
    image_sets: dict[str, set[str]] = {}
    manifest_sha256: dict[str, str] = {}
    within_split_duplicates: dict[str, dict[str, int]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        split = str(payload.get("split", path.parent.name))
        task = str(payload.get("task", "unknown"))
        capability = str(payload.get("capability", "all"))
        benchmark = str(payload.get("benchmark", "route_transfer"))
        identity = f"{benchmark}:{task}:{capability}:{split}:{path.parent.resolve()}"
        keys = {str(value) for value in payload.get("split_keys", [])}
        semantic_keys = {
            str(value) for value in payload.get("semantic_keys", payload.get("split_keys", []))
        }
        render_keys = {str(value) for value in payload.get("render_keys", [])}
        image_keys = {str(value) for value in payload.get("image_sha256", [])}
        key_sets[identity] = keys
        semantic_sets[identity] = semantic_keys
        duplicate_counts = {
            "split_keys": len(payload.get("split_keys", [])) - len(keys),
            "semantic_keys": len(payload.get("semantic_keys", payload.get("split_keys", [])))
            - len(semantic_keys),
        }
        if any(duplicate_counts.values()):
            within_split_duplicates[identity] = duplicate_counts
        render_sets[identity] = render_keys
        image_sets[identity] = image_keys
        manifest_sha256[str(path.resolve())] = _file_sha256(path)
    overlaps: dict[str, list[str]] = {}
    semantic_overlaps: dict[str, list[str]] = {}
    render_overlaps: dict[str, list[str]] = {}
    image_overlaps: dict[str, list[str]] = {}
    names = sorted(key_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(key_sets[left] & key_sets[right])
            if shared:
                overlaps[f"{left}:{right}"] = shared
            shared_semantics = sorted(semantic_sets[left] & semantic_sets[right])
            if shared_semantics:
                semantic_overlaps[f"{left}:{right}"] = shared_semantics
            shared_renders = sorted(render_sets[left] & render_sets[right])
            if shared_renders:
                render_overlaps[f"{left}:{right}"] = shared_renders
            shared_images = sorted(image_sets[left] & image_sets[right])
            if shared_images:
                image_overlaps[f"{left}:{right}"] = shared_images
    overlap_count = (
        sum(map(len, overlaps.values()))
        + sum(map(len, semantic_overlaps.values()))
        + sum(map(len, render_overlaps.values()))
        + sum(map(len, image_overlaps.values()))
        + sum(sum(values.values()) for values in within_split_duplicates.values())
    )
    return {
        "splits": {name: len(keys) for name, keys in key_sets.items()},
        "overlap_count": overlap_count,
        "overlaps": overlaps,
        "semantic_overlaps": semantic_overlaps,
        "render_overlaps": render_overlaps,
        "image_overlaps": image_overlaps,
        "within_split_duplicates": within_split_duplicates,
        "manifest_sha256": manifest_sha256,
        "passed": overlap_count == 0,
    }


def validate_preflight_audit(  # noqa: C901
    path: Path,
    *,
    required_records: tuple[Path, ...],
) -> None:
    """Require a passing audit that hashes every dataset manifest consumed by training."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        msg = f"data preflight did not pass: {path}"
        raise ValueError(msg)
    locked = payload.get("manifest_sha256")
    if not isinstance(locked, dict):
        msg = "data preflight lacks manifest hashes"
        raise ValueError(msg)
    for records in required_records:
        manifest = records.with_name("manifest.json").resolve()
        if not manifest.is_file():
            msg = f"training input lacks a manifest: {records}"
            raise FileNotFoundError(msg)
        expected = locked.get(str(manifest))
        actual = _file_sha256(manifest)
        if expected != actual:
            msg = f"training manifest is absent from or changed after preflight: {manifest}"
            raise ValueError(msg)
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        records_entry = manifest_payload.get("records")
        if not isinstance(records_entry, dict) or records_entry.get("sha256") != _file_sha256(
            records
        ):
            msg = f"training records changed after manifest creation: {records}"
            raise ValueError(msg)
        images = manifest_payload.get("images")
        if not isinstance(images, list):
            msg = f"training manifest lacks image content hashes: {manifest}"
            raise ValueError(msg)
        for item in images:
            if not isinstance(item, dict):
                msg = f"training manifest has an invalid image entry: {manifest}"
                raise ValueError(msg)
            image = Path(str(item.get("path", "")))
            if not image.is_file() or item.get("sha256") != _file_sha256(image):
                msg = f"training image changed after manifest creation: {image}"
                raise ValueError(msg)


def audit_capability_exclusion(
    records: list[RouteRecord],
    *,
    excluded: tuple[Capability, ...],
) -> dict[str, object]:
    """Fail capability calibration that includes an excluded capability label."""
    if not excluded:
        msg = "capability exclusion audit requires at least one excluded capability"
        raise ValueError(msg)
    counts = {
        capability: sum(record["capability"] == capability for record in records)
        for capability in excluded
    }
    violations = {capability: count for capability, count in counts.items() if count > 0}
    return {
        "excluded": excluded,
        "counts": violations,
        "passed": not violations,
    }


def validate_capability_audit(
    path: Path,
    *,
    required_records: tuple[Path, ...],
    excluded: tuple[Capability, ...],
) -> None:
    """Require a passing, content-bound exclusion audit before capability training."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        msg = f"capability exclusion audit did not pass: {path}"
        raise ValueError(msg)
    audited_exclusions = payload.get("excluded")
    if not isinstance(audited_exclusions, list) or not set(excluded).issubset(
        {str(value) for value in audited_exclusions}
    ):
        msg = "capability exclusion audit does not cover the run capability"
        raise ValueError(msg)
    hashes = payload.get("records_sha256")
    if not isinstance(hashes, dict):
        msg = "capability exclusion audit lacks record hashes"
        raise ValueError(msg)
    for records in required_records:
        expected = hashes.get(str(records.resolve()))
        actual = _file_sha256(records)
        if expected != actual:
            msg = f"calibration records are absent from or changed after audit: {records}"
            raise ValueError(msg)
