#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from barcode_benchmark.fusion import fuse_predictions
from barcode_benchmark.io import (
    DATASET_SCHEMA,
    PREDICTION_SCHEMA,
    image_index,
    load_json,
    require_schema,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse normalized prediction runs")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("supplement", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-threshold", type=float, default=0.45)
    arguments = parser.parse_args()

    dataset = load_json(arguments.dataset)
    require_schema(dataset, DATASET_SCHEMA, "dataset")
    dataset_images = image_index(dataset, "dataset")
    primary = load_json(arguments.primary)
    require_schema(primary, PREDICTION_SCHEMA, "primary")
    primary_images = image_index(primary, "primary")
    supplement_runs = [load_json(path) for path in arguments.supplement]
    for index, run in enumerate(supplement_runs):
        require_schema(run, PREDICTION_SCHEMA, f"supplement:{index}")
    supplement_indices = [image_index(run, "supplement") for run in supplement_runs]

    output_images = []
    for image_id in dataset_images:
        if image_id not in primary_images:
            raise ValueError(f"primary run is missing dataset image: {image_id}")
        primary_image = primary_images[image_id]
        supplement_images = [
            index[image_id] for index in supplement_indices if image_id in index
        ]
        timing_records = [
            item.get("timing", {}) for item in [primary_image, *supplement_images]
        ]
        latencies = [
            float(item["latency_ms"])
            for item in timing_records
            if isinstance(item.get("latency_ms"), (int, float))
        ]
        memory = [
            float(item["peak_rss_mb"])
            for item in timing_records
            if isinstance(item.get("peak_rss_mb"), (int, float))
        ]
        output_images.append({
            "id": image_id,
            "predictions": fuse_predictions(
                primary_image.get("predictions", []),
                [item.get("predictions", []) for item in supplement_images],
                overlap_threshold=arguments.overlap_threshold,
            ),
            "timing": {
                "latency_ms": sum(latencies) if latencies else None,
                "peak_rss_mb": max(memory) if memory else None,
            },
        })

    value = {
        "schema_version": PREDICTION_SCHEMA,
        "run": {
            "pipeline": "evidence-preserving-fusion",
            "timing_scope": "sequential-sum-upper-bound",
            "primary": str(arguments.primary.resolve()),
            "supplements": [str(path.resolve()) for path in arguments.supplement],
            "overlap_threshold": arguments.overlap_threshold,
        },
        "images": output_images,
    }
    write_json(arguments.output, value)
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
