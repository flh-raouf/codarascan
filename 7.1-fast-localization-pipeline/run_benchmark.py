#!/usr/bin/env python3
"""Run one fast-localization candidate using the neutral benchmark schema."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent / "benchmark-lab"
for path in (SCRIPT_DIR, LAB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from barcode_benchmark.io import (  # noqa: E402
    DATASET_SCHEMA,
    PREDICTION_SCHEMA,
    image_index,
    load_json,
    require_schema,
    write_json,
)
from candidates import DEFAULT_MODEL, METHODS, method  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--work-size", type=int, default=256)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--minimum-area", type=float, default=14.0)
    parser.add_argument(
        "--resize",
        choices=("area", "linear", "nearest"),
        default="area",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    arguments = parser.parse_args()

    dataset = load_json(arguments.dataset)
    require_schema(dataset, DATASET_SCHEMA, "dataset")
    records = list(image_index(dataset, "dataset").values())
    implementation = method(
        arguments.method,
        weights=arguments.weights,
        work_size=arguments.work_size,
        confidence=arguments.confidence,
        minimum_area=arguments.minimum_area,
        interpolation={
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
        }[arguments.resize],
    )
    output_images = []
    wall_started = perf_counter()
    for index, record in enumerate(records, start=1):
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(record["path"])
        started = perf_counter()
        predictions = implementation(image)
        output_images.append(
            {
                "id": record["id"],
                "predictions": [
                    {
                        **prediction,
                        "polygon": [
                            [float(x), float(y)]
                            for x, y in prediction["polygon"]
                        ],
                    }
                    for prediction in predictions
                ],
                "timing": {
                    "latency_ms": 1000.0 * (perf_counter() - started),
                    "peak_rss_mb": None,
                },
            }
        )
        if (
            arguments.progress_every > 0
            and (index % arguments.progress_every == 0 or index == len(records))
        ):
            print(
                f"{arguments.method}: {index}/{len(records)} "
                f"({sum(len(item['predictions']) for item in output_images)} proposals)",
                flush=True,
            )
    write_json(
        arguments.output,
        {
            "schema_version": PREDICTION_SCHEMA,
            "run": {
                "pipeline": arguments.method,
                "implementation": "7.1-fast-localization-tournament-v1",
                "dataset_manifest": str(arguments.dataset.resolve()),
                "images": len(records),
                "wall_seconds": perf_counter() - wall_started,
                "parameters": {
                    "weights": str(arguments.weights.resolve()),
                    "work_size": arguments.work_size,
                    "confidence": arguments.confidence,
                    "minimum_area": arguments.minimum_area,
                    "resize": arguments.resize,
                },
                "environment": {
                    "python": platform.python_version(),
                    "opencv": cv2.__version__,
                },
            },
            "images": output_images,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
