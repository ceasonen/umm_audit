#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from skillbridge.oral_evaluation import stage_oral_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temporary = args.output.with_name(f"{args.output.name}.partial-{os.getpid()}")
    report = stage_oral_predictions(args.records, args.predictions, temporary)
    temporary.replace(args.output)
    sys.stdout.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
