#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from barcode_benchmark.cascade import conditional_fusion, route_small_evidence
from barcode_benchmark.io import (
    DATASET_SCHEMA,
    PREDICTION_SCHEMA,
    image_index,
    load_json,
    require_schema,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conditionally invoke a fallback run from fast-detector evidence"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("fast", type=Path)
    parser.add_argument("fallback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-area-threshold", type=float, required=True)
    parser.add_argument("--overlap-threshold", type=float, default=0.45)
    arguments = parser.parse_args()

    dataset = load_json(arguments.dataset)
    fast = load_json(arguments.fast)
    fallback = load_json(arguments.fallback)
    require_schema(dataset, DATASET_SCHEMA, "dataset")
    require_schema(fast, PREDICTION_SCHEMA, "fast")
    require_schema(fallback, PREDICTION_SCHEMA, "fallback")
    dataset_images = image_index(dataset, "dataset")
    fast_images = image_index(fast, "fast")
    fallback_images = image_index(fallback, "fallback")
    if set(dataset_images) != set(fast_images) or set(dataset_images) != set(fallback_images):
        raise ValueError("dataset, fast, and fallback image IDs must match exactly")

    output_images = []
    routed_count = 0
    for image_id, truth_image in dataset_images.items():
        fast_image = fast_images[image_id]
        fallback_image = fallback_images[image_id]
        routed = route_small_evidence(
            fast_image.get("predictions", []),
            width=int(truth_image["width"]),
            height=int(truth_image["height"]),
            relative_area_threshold=arguments.relative_area_threshold,
        )
        routed_count += int(routed)
        fast_timing = fast_image.get("timing", {})
        fallback_timing = fallback_image.get("timing", {})
        fast_latency = fast_timing.get("latency_ms")
        fallback_latency = fallback_timing.get("latency_ms") if routed else None
        latency_values = [
            float(value)
            for value in (fast_latency, fallback_latency)
            if isinstance(value, (int, float))
        ]
        memory_values = [
            float(value)
            for value in (
                fast_timing.get("peak_rss_mb"),
                fallback_timing.get("peak_rss_mb") if routed else None,
            )
            if isinstance(value, (int, float))
        ]
        output_images.append({
            "id": image_id,
            "predictions": conditional_fusion(
                fast_image.get("predictions", []),
                fallback_image.get("predictions", []),
                routed=routed,
                overlap_threshold=arguments.overlap_threshold,
            ),
            "timing": {
                "latency_ms": sum(latency_values) if latency_values else None,
                "peak_rss_mb": max(memory_values) if memory_values else None,
            },
            "routing": {"fallback_invoked": routed},
        })

    image_count = len(output_images)
    value = {
        "schema_version": PREDICTION_SCHEMA,
        "run": {
            "pipeline": "small-evidence-conditional-cascade",
            "timing_scope": "conditional-sequential-sum",
            "fast": str(arguments.fast.resolve()),
            "fallback": str(arguments.fallback.resolve()),
            "relative_area_threshold": arguments.relative_area_threshold,
            "overlap_threshold": arguments.overlap_threshold,
            "fallback_invocations": routed_count,
            "fallback_fraction": routed_count / image_count if image_count else 0.0,
        },
        "images": output_images,
    }
    write_json(arguments.output, value)
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
