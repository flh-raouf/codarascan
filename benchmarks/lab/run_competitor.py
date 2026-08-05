#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from barcode_benchmark.competitors import COMPETITOR_NAMES, competitor
from barcode_benchmark.io import (
    DATASET_SCHEMA,
    PREDICTION_SCHEMA,
    image_index,
    load_json,
    require_schema,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one independent tournament competitor")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--method", choices=COMPETITOR_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="debug only; output will not cover the full manifest")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--model-path", type=Path, help="optional learned-competitor checkpoint")
    parser.add_argument("--imgsz", type=int, default=1280, help="learned-competitor inference size")
    parser.add_argument("--conf", type=float, default=0.10, help="learned-competitor confidence threshold")
    parser.add_argument("--device", default="cpu", help="learned-competitor inference device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    dataset = load_json(arguments.dataset)
    require_schema(dataset, DATASET_SCHEMA, "dataset")
    images = list(image_index(dataset, "dataset").values())
    if arguments.limit is not None:
        images = images[:arguments.limit]
    implementation = competitor(
        arguments.method,
        model_path=arguments.model_path,
        image_size=arguments.imgsz,
        confidence=arguments.conf,
        device=arguments.device,
    )
    process = psutil.Process()
    output_images = []
    wall_started = time.perf_counter()
    for index, record in enumerate(images, start=1):
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read {record['path']}")
        started = time.perf_counter()
        found = implementation(image)
        latency_ms = 1000.0 * (time.perf_counter() - started)
        output_images.append({
            "id": record["id"],
            "predictions": found,
            "timing": {
                "latency_ms": latency_ms,
                "peak_rss_mb": process.memory_info().rss / (1024.0 * 1024.0),
            },
        })
        if arguments.progress_every > 0 and (index % arguments.progress_every == 0 or index == len(images)):
            print(
                f"{arguments.method}: {index}/{len(images)} images, "
                f"{sum(len(item['predictions']) for item in output_images)} predictions",
                flush=True,
            )
    value = {
        "schema_version": PREDICTION_SCHEMA,
        "run": {
            "pipeline": arguments.method,
            "implementation": "benchmark-lab-independent-v1",
            "dataset_manifest": str(arguments.dataset.expanduser().resolve()),
            "images": len(images),
            "wall_seconds": time.perf_counter() - wall_started,
            "parameters": {
                "model_path": str(arguments.model_path.resolve()) if arguments.model_path else None,
                "image_size": arguments.imgsz,
                "confidence": arguments.conf,
                "device": arguments.device,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "opencv": cv2.__version__,
            },
        },
        "images": output_images,
    }
    write_json(arguments.output, value)
    print(arguments.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
