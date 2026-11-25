from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from skillbridge.route_data import RouteRecord, RouteTask

ImageEncoder = Callable[[Path], Mapping[str, object]]


def pretokenize_route_records(
    records: Sequence[RouteRecord],
    output_dir: Path,
    *,
    task: RouteTask,
    encoder: ImageEncoder,
) -> list[RouteRecord]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        msg = f"refusing to overwrite tokenized data: {output_dir}"
        raise FileExistsError(msg)
    files_dir = output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    tokenized: list[RouteRecord] = []
    image_field = "user_image" if task == "understanding" else "answer_image"
    empty_field = "answer_image" if task == "understanding" else "user_image"
    for index, source in enumerate(records):
        if not source[image_field] or source[empty_field]:
            msg = f"record {index} does not match the requested {task} projection"
            raise ValueError(msg)
        encoded = dict(encoder(Path(source[image_field])))
        required = {"input_ids", "height", "width"}
        if not required.issubset(encoded):
            msg = f"encoder output lacks fields: {sorted(required - set(encoded))}"
            raise ValueError(msg)
        token_path = files_dir / f"{index:07d}-{source['split_key'][:12]}.pkl"
        with token_path.open("wb") as handle:
            pickle.dump(encoded, handle, protocol=pickle.HIGHEST_PROTOCOL)
        item = dict(source)
        item[image_field] = str(token_path.resolve())
        tokenized.append(cast("RouteRecord", item))
    records_text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in tokenized)
    records_path = output_dir / "records.jsonl"
    records_path.write_text(records_text, encoding="utf-8")
    manifest = {
        "format_version": 1,
        "task": task,
        "count": len(tokenized),
        "records": "records.jsonl",
        "source_records": [dict(item) for item in records],
        "split_keys": [item["split_key"] for item in tokenized],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tokenized


def load_route_records(path: Path) -> list[RouteRecord]:
    records: list[RouteRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            msg = f"record {line_number} must be a JSON object"
            raise ValueError(msg)
        records.append(cast("RouteRecord", payload))
    return records


def build_dimoo_encoder(
    external_repo: Path,
    *,
    model_id: str,
    target_size: int,
    device: str,
) -> ImageEncoder:
    repo = external_repo.resolve()
    if not (repo / "data" / "item_processor.py").is_file():
        msg = f"official backbone repository is missing at {repo}"
        raise FileNotFoundError(msg)
    sys.path.insert(0, str(repo))
    module = importlib.import_module("data.item_processor")
    processor_class = getattr(module, "DimooItemProcessor", None)
    if processor_class is None:
        msg = "official data.item_processor lacks DimooItemProcessor"
        raise AttributeError(msg)
    processor = processor_class(model_id, model_id, target_size, device=device)

    def encode(path: Path) -> Mapping[str, object]:
        output = processor.process_item("", str(path))
        if not isinstance(output, Mapping):
            msg = f"unexpected tokenizer output for {path}"
            raise TypeError(msg)
        return output

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize RouteTransfer-Syn images")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("understanding", "generation"), required=True)
    parser.add_argument("--external-repo", type=Path, default=Path("external/Lumina-DiMOO"))
    parser.add_argument("--model-id", default="Alpha-VLLM/Lumina-DiMOO")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    encoder = build_dimoo_encoder(
        args.external_repo,
        model_id=args.model_id,
        target_size=args.target_size,
        device=args.device,
    )
    pretokenize_route_records(
        load_route_records(args.input),
        args.output,
        task=args.task,
        encoder=encoder,
    )


if __name__ == "__main__":
    main()
