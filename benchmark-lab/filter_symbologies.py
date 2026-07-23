#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from barcode_benchmark.evaluate import canonical_symbology
from barcode_benchmark.io import load_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply an application-declared symbology allowlist to a normalized run",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow", required=True, help="comma-separated symbologies")
    arguments = parser.parse_args()
    allowed = {
        canonical_symbology(item)
        for item in arguments.allow.split(",")
        if item.strip()
    }
    if not allowed:
        raise ValueError("--allow cannot be empty")
    value = load_json(arguments.input)
    filtered = 0
    for image in value.get("images", []):
        for prediction in image.get("predictions", []):
            if (
                prediction.get("payload") is not None
                and canonical_symbology(prediction.get("symbology")) not in allowed
            ):
                prediction["payload"] = None
                prediction["status"] = "localized"
                prediction.setdefault("evidence", {})["payload_rejected"] = (
                    "application_symbology_allowlist"
                )
                filtered += 1
    value.setdefault("run", {}).setdefault("parameters", {})["allowed_symbologies"] = sorted(allowed)
    value["run"]["parameters"]["payloads_rejected_by_allowlist"] = filtered
    write_json(arguments.output, value)
    print(arguments.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
