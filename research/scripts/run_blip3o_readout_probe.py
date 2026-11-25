#!/usr/bin/env python3
"""Paired source retention and local readout audit with fixed interventions."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from scripts.evaluate_cross_backbone_generation import checkpoint_route_spec, load_blip3o
from skillbridge.cross_backbone_evaluation import generate_fixed_predictions, generate_understanding_predictions
from skillbridge.dimoo_data import load_route_records
from skillbridge.readout_probe import CommonInputProbe, core_conditions, set_core_state


class FreshGeneration:
    def __init__(self, adapter):
        self.adapter = adapter

    def generate(self, prompt, *, seed, **kwargs):
        self.adapter.reset_condition_cache()
        return self.adapter.generate(prompt, seed=seed, **kwargs)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--probe-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("understanding", "generation"), required=True)
    parser.add_argument("--source-seed", type=int, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.core, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    conditions = core_conditions(checkpoint["state_dict"], budget=64, seed=1000 + args.source_seed)
    records = load_route_records(args.records)
    probe_records = load_route_records(args.probe_records)[:16]
    if not records or len(probe_records) != 16:
        raise ValueError("missing evaluation records or 16 diagnostic records")
    gain, suffixes = checkpoint_route_spec(args.core)
    adapter, route, kwargs = load_blip3o(args.root, rank=64, timestep_buckets=1,
        method="tied_core", gain_limit=gain, target_suffixes=suffixes, task=args.task)
    fresh = FreshGeneration(adapter)
    order = list(conditions)
    random.Random(args.source_seed).shuffle(order)
    write_json(args.output / "design.json", {
        "task": args.task, "source_seed": args.source_seed, "condition_order": order,
        "records": str(args.records), "rows": len(records),
        "core": str(args.core), "global_component_budget": 64,
        "total_components": sum(min(v.shape) for v in conditions["full"].values()),
        "inference_seed": 8087, "random_direction_seed": 1000 + args.source_seed,
        "cache_policy": "reset before every generated image",
        "probe_usage": "diagnostic only; no selection or rescaling from target scores or activations",
    })
    report_path = args.output / "same-input-activation.json"
    if not report_path.exists():
        set_core_state(route, conditions["zero"])
        probe = CommonInputProbe(conditions)
        with probe.observe(route):
            for index, record in enumerate(probe_records):
                if args.task == "generation":
                    fresh.generate(str(record["user_prompt"]), seed=8087 + index, **kwargs)
                else:
                    adapter.understand(Path(str(record["user_image"])), str(record["user_prompt"]))
                print(f"probe {args.task} {index+1}/16", flush=True)
        write_json(report_path, {"probe_records": str(args.probe_records), "rows": 16, **probe.report()})
    for condition in order:
        set_core_state(route, conditions[condition])
        output = args.output / condition
        metadata = {"intervention": condition, "source_seed": args.source_seed}
        print(f"begin {args.task} {condition} rows={len(records)}", flush=True)
        if args.task == "understanding":
            generate_understanding_predictions(adapter, records, output, metadata=metadata, resume=True)
        else:
            generate_fixed_predictions(fresh, records, output, seed=8087,
                generation_kwargs=kwargs, metadata=metadata, resume=True)
        print(f"complete {args.task} {condition}", flush=True)
    write_json(args.output / "complete.json", {"conditions": order, "rows": len(records)})


if __name__ == "__main__":
    main()
