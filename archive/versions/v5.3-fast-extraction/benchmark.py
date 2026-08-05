#!/usr/bin/env python3
"""Compare fast extraction candidates on one fixed payload manifest."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import zxingcpp

from fast_direct import direct_decode
from pipeline import FastConfig, FastModelExtractor


def _canonical(value: str) -> str:
    return value[1:] if len(value) == 13 and value.startswith("0") else value


def _score(
    records: list[dict[str, Any]],
    implementation: Callable[[Any], list[dict[str, Any]]],
) -> dict[str, Any]:
    exact = wrong = missing = 0
    negative_false_reads = 0
    latencies: list[float] = []
    for record in records:
        started = perf_counter()
        image = cv2.imread(record["path"], cv2.IMREAD_GRAYSCALE)
        predictions = implementation(image)
        latencies.append((perf_counter() - started) * 1000.0)
        truth = {_canonical(str(item["payload"])) for item in record["objects"]}
        found = {_canonical(str(item["text"])) for item in predictions if item.get("text")}
        exact += len(truth & found)
        missing += len(truth - found)
        wrong += len(found - truth)
        if not truth:
            negative_false_reads += len(found)
    return {
        "truth": exact + missing,
        "exact": exact,
        "recall": exact / (exact + missing) if exact + missing else 0.0,
        "wrong": wrong,
        "wrong_per_truth": wrong / (exact + missing) if exact + missing else 0.0,
        "negative_false_reads": negative_false_reads,
        "latency_ms": {
            "median": statistics.median(latencies),
            "mean": statistics.mean(latencies),
            "p95": float(sorted(latencies)[int(0.95 * (len(latencies) - 1))]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated candidate names; empty runs all.",
    )
    parser.add_argument(
        "--ordinary-orientation",
        action="store_true",
        help="Keep pages whose symbols are all at 0 or +/-90 degrees.",
    )
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest["images"]
        if all(item.get("payload") is not None for item in record.get("objects", []))
        and (
            not arguments.ordinary_orientation
            or all(
                item.get("rotation_deg", 0) in (-90, 0, 90)
                for item in record.get("objects", [])
            )
        )
    ]
    retail = zxingcpp.barcode_formats_from_str("EAN13|EAN8|UPCA|UPCE")
    candidates: dict[str, Callable[[Any], list[dict[str, Any]]]] = {
        "zxing-native-default": lambda image: direct_decode(
            image,
            try_downscale=True,
            try_invert=True,
        ),
        "zxing-native-lean": lambda image: direct_decode(image),
        "zxing-1600-lean": lambda image: direct_decode(
            image,
            max_dimension=1600,
        ),
        "zxing-retail-lean": lambda image: direct_decode(
            image,
            formats=retail,
        ),
    }
    report = {
        "manifest": str(arguments.manifest.resolve()),
        "dataset": manifest.get("name"),
        "results": {},
    }
    selected = {
        value.strip()
        for value in arguments.methods.split(",")
        if value.strip()
    }
    if not selected or "clean-model-native-crops" in selected:
        model = FastModelExtractor(FastConfig(confidence_threshold=0.25))
        model.warm()
        candidates["clean-model-native-crops"] = lambda image: model(image)[0]
    for name, implementation in candidates.items():
        if selected and name not in selected:
            continue
        result = _score(records, implementation)
        report["results"][name] = result
        print(
            f"{name}: exact={result['exact']}/{result['truth']} "
            f"recall={100 * result['recall']:.2f}% wrong={result['wrong']} "
            f"median={result['latency_ms']['median']:.2f}ms "
            f"p95={result['latency_ms']['p95']:.2f}ms",
            flush=True,
        )
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
