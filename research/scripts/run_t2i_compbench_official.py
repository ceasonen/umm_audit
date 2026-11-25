#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from skillbridge.official_evaluator import (
    load_evaluator_lock,
    run_t2i_compbench_evaluator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the revision-locked official T2I-CompBench evaluators."
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--no-revision-verification",
        action="store_false",
        dest="verify_revision",
        help="Run the official metric without repository revision verification.",
    )
    parser.set_defaults(verify_revision=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = load_evaluator_lock(args.lock, "t2i_compbench")
    report = run_t2i_compbench_evaluator(
        args.repository,
        lock=lock,
        staged_root=args.staged_root,
        output_json=args.output,
        python_executable=args.python,
        verify_revision=args.verify_revision,
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
