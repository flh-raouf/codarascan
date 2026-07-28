#!/usr/bin/env python3
"""Convert the fixed 5.3 easy manifest to the neutral localization schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent / "benchmark-lab"
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from barcode_benchmark.io import DATASET_SCHEMA, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ordinary-orientation", action="store_true")
    arguments = parser.parse_args()
    source = json.loads(arguments.source.read_text(encoding="utf-8"))
    images = []
    for record in source["images"]:
        objects = record.get("objects", [])
        if arguments.ordinary_orientation and not all(
            item.get("rotation_deg", 0) in (-90, 0, 90)
            for item in objects
        ):
            continue
        images.append(
            {
                "id": record["id"],
                "path": record["path"],
                "width": record["width"],
                "height": record["height"],
                "source_group": "fast-engine-easy-holdout",
                "split": "test",
                "objects": [
                    {
                        "id": f"object-{index:04d}",
                        "polygon": item["polygon"],
                        "kind": item["kind"],
                        "symbology": item["format"],
                        "payload": item["payload"],
                        "decodable": True,
                    }
                    for index, item in enumerate(objects, start=1)
                ],
            }
        )
    write_json(
        arguments.output,
        {
            "schema_version": DATASET_SCHEMA,
            "dataset": {
                "name": source["name"],
                "source_manifest": str(arguments.source.resolve()),
                "ordinary_orientation_only": arguments.ordinary_orientation,
            },
            "images": images,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
