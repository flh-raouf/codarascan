#!/usr/bin/env python3
"""Convert reviewed pipeline-7 output into a distillation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parents[2] / "benchmarks" / "lab"
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from barcode_benchmark.io import DATASET_SCHEMA, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("detections", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    arguments = parser.parse_args()
    source = json.loads(
        arguments.detections.read_text(encoding="utf-8")
    )
    images = sorted(
        path
        for path in arguments.images.iterdir()
        if path.is_file()
    )
    records = []
    for page in source["pages"]:
        page_number = int(page["page"])
        path = images[page_number - 1].resolve()
        objects = []
        for index, item in enumerate(page.get("detections", [])):
            kind = str(item.get("kind", ""))
            objects.append(
                {
                    "id": f"object-{index + 1:04d}",
                    "polygon": item["quad"],
                    "kind": (
                        "1d"
                        if kind in {"linear", "1d"}
                        else "2d"
                    ),
                    "symbology": (
                        "QR_CODE"
                        if kind == "qr"
                        else "unknown"
                    ),
                    "payload": None,
                    "decodable": False,
                    "attributes": {
                        "teacher": "pipeline-7",
                        "teacher_confidence": item.get("confidence"),
                    },
                }
            )
        records.append(
            {
                "id": f"{arguments.name}/page-{page_number:04d}",
                "path": str(path),
                "width": int(page["image_size"]["width"]),
                "height": int(page["image_size"]["height"]),
                "source_group": arguments.name,
                "split": "development",
                "objects": objects,
            }
        )
    write_json(
        arguments.output,
        {
            "schema_version": DATASET_SCHEMA,
            "dataset": {
                "name": arguments.name,
                "annotation_format": "pipeline-7-distillation",
                "teacher_run": str(
                    arguments.detections.resolve()
                ),
            },
            "images": records,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
