#!/usr/bin/env python3
"""Neutral localization benchmark for the structure-tensor research pipeline.

Image loading, grayscale conversion, and detector resizing are reported as
preparation. `timing.latency_ms` contains only engine compute so it matches the
prepared-input service boundary used by Codara.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
LAB = REPOSITORY / "benchmarks" / "lab"
for path in (ROOT, LAB):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from barcode_benchmark.evaluate import evaluate
from barcode_benchmark.io import (
    DATASET_SCHEMA,
    PREDICTION_SCHEMA,
    load_json,
    require_schema,
    write_json,
)
from pipeline import (
    Detection,
    PreparedPage,
    TensorConfig,
    linear_metrics,
    locate_prepared,
    prepare_page,
)


METHODS = (
    "sttg-context",
    "sttg-native",
    "tensor-python",
    "opencv-p7",
    "zxing-valid-work",
    "zxing-errors-work",
    "zxing-valid-native",
)


def _linear_manifest(dataset: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dataset)
    for image in value["images"]:
        image["objects"] = [
            item
            for item in image.get("objects", [])
            if str(item.get("kind", "")).strip().lower() in {"1d", "linear", "barcode", "bar_code"}
        ]
    value["dataset"] = {**value.get("dataset", {}), "evaluation_scope": "linear-only"}
    return value


def _prediction(detection: Detection) -> dict[str, Any]:
    return detection.to_prediction()


def _load_pipeline7() -> Any:
    source = REPOSITORY / "archive" / "versions" / "v7-coarse-to-fine" / "pipeline.py"
    specification = importlib.util.spec_from_file_location("_pipeline7_reference", source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _prepare_opencv(image: np.ndarray) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    native = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(native, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA)
    return {"native": native, "work": work, "scale": 0.65}, perf_counter() - started


def _opencv_runner() -> tuple[
    Callable[[np.ndarray], tuple[Any, float]],
    Callable[[Any], tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]],
]:
    p7 = _load_pipeline7()
    config = p7.Config(output=ROOT / "output" / "_unused", kinds="linear")
    scales = p7.UNIVERSAL_LINEAR_SCALES

    def run(prepared: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
        work = prepared["work"]
        native = prepared["native"]
        detector = p7.linear_detector("universal", scales)
        detector.setDownsamplingThreshold(float(max(work.shape)))
        detector_started = perf_counter()
        try:
            found, points = detector.detectMulti(work)
        except cv2.error:
            found, points = False, None
        detector_seconds = perf_counter() - detector_started
        verification_started = perf_counter()
        accepted = []
        if found and points is not None:
            for raw_quad in np.asarray(points, np.float32):
                quad = raw_quad / prepared["scale"]
                candidate = p7.Detection(
                    quad=quad,
                    kind="linear",
                    confidence=0.0,
                    source="opencv-p7-universal",
                )
                long_side, short_side = candidate.long_short()
                if long_side < config.minimum_linear_length or short_side < 5 or long_side / max(short_side, 1e-6) < 1.65:
                    continue
                metrics = p7.linear_metrics(native, quad)
                if (
                    metrics.get("score", 0.0) >= config.minimum_linear_score
                    and metrics.get("orientation", 0.0) >= 0.62
                    and metrics.get("persistence", 0.0) >= 0.045
                    and metrics.get("transitions", 0.0) >= 12.0
                    and metrics.get("scanline_agreement", 0.0) >= 0.50
                    and 0.015 <= metrics.get("transition_rate", 0.0) <= 0.72
                ):
                    candidate.metrics = metrics
                    candidate.confidence = float(metrics["score"])
                    accepted.append(candidate)
        accepted = p7.deduplicate(accepted)
        verification_seconds = perf_counter() - verification_started
        predictions = [
            {
                "polygon": np.asarray(item.quad, np.float32).tolist(),
                "kind": "1d",
                "symbology": "unknown",
                "payload": None,
                "confidence": float(item.confidence),
                "status": "localized",
                "sources": ["opencv-p7-universal"],
            }
            for item in accepted
        ]
        return (
            predictions,
            {
                "raw_candidates": 0 if points is None else len(points),
                "accepted_count": len(predictions),
            },
            {
                "detector_seconds": detector_seconds,
                "verification_seconds": verification_seconds,
                "engine_seconds": detector_seconds + verification_seconds,
            },
        )

    return _prepare_opencv, run


def _zxing_quad(value: Any) -> np.ndarray | None:
    position = value.position
    points = np.asarray(
        [
            [position.top_left.x, position.top_left.y],
            [position.top_right.x, position.top_right.y],
            [position.bottom_right.x, position.bottom_right.y],
            [position.bottom_left.x, position.bottom_left.y],
        ],
        np.float32,
    )
    if abs(float(cv2.contourArea(points))) > 1e-6:
        return points
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    start, end = points[first], points[second]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return None
    normal = np.asarray([-direction[1], direction[0]], np.float32) / length
    half_width = max(2.0, 0.025 * length)
    return np.asarray(
        [
            start - half_width * normal,
            end - half_width * normal,
            end + half_width * normal,
            start + half_width * normal,
        ],
        np.float32,
    )


def _zxing_runner(
    *,
    native: bool,
    errors: bool,
    work_size: int,
) -> tuple[
    Callable[[np.ndarray], tuple[Any, float]],
    Callable[[Any], tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]],
]:
    import zxingcpp

    def prepare(image: np.ndarray) -> tuple[dict[str, Any], float]:
        started = perf_counter()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if native:
            work = gray
        else:
            scale = min(1.0, work_size / max(gray.shape))
            work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
        return {
            "native": gray,
            "work": work,
            "scale_x": work.shape[1] / gray.shape[1],
            "scale_y": work.shape[0] / gray.shape[0],
        }, perf_counter() - started

    def run(prepared: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
        started = perf_counter()
        values = zxingcpp.read_barcodes(
            prepared["work"],
            try_rotate=True,
            try_downscale=False,
            try_invert=False,
            return_errors=errors,
        )
        predictions = []
        for value in values:
            if "QR" in str(value.format) or "Data Matrix" in str(value.format) or "Aztec" in str(value.format):
                continue
            quad = _zxing_quad(value)
            if quad is None:
                continue
            quad[:, 0] /= prepared["scale_x"]
            quad[:, 1] /= prepared["scale_y"]
            predictions.append(
                {
                    "polygon": quad.tolist(),
                    "kind": "1d",
                    "symbology": str(value.format),
                    "payload": None,
                    "confidence": 1.0 if value.valid else 0.35,
                    "status": "localized",
                    "sources": ["zxing-valid" if value.valid else "zxing-error"],
                }
            )
        engine_seconds = perf_counter() - started
        return predictions, {"raw_candidates": len(values), "accepted_count": len(predictions)}, {"engine_seconds": engine_seconds}

    return prepare, run


def _tensor_runner(
    config: TensorConfig,
) -> tuple[
    Callable[[np.ndarray], tuple[PreparedPage, float]],
    Callable[[PreparedPage], tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]],
]:
    def prepare(image: np.ndarray) -> tuple[PreparedPage, float]:
        value = prepare_page(image, config)
        return value, value.preparation_seconds

    def run(prepared: PreparedPage) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
        result = locate_prepared(prepared, config)
        return [_prediction(item) for item in result.detections], result.diagnostics, result.timings

    return prepare, run


def _implementation(
    name: str,
    config: TensorConfig,
) -> tuple[Callable[[np.ndarray], tuple[Any, float]], Callable[[Any], tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]]]:
    if name == "sttg-context":
        return _tensor_runner(
            TensorConfig(**{**config.__dict__, "backend": "native-context"})
        )
    if name == "sttg-native":
        return _tensor_runner(TensorConfig(**{**config.__dict__, "backend": "native"}))
    if name == "tensor-python":
        return _tensor_runner(TensorConfig(**{**config.__dict__, "backend": "python"}))
    if name == "opencv-p7":
        return _opencv_runner()
    if name == "zxing-valid-work":
        return _zxing_runner(native=False, errors=False, work_size=config.work_size)
    if name == "zxing-errors-work":
        return _zxing_runner(native=False, errors=True, work_size=config.work_size)
    if name == "zxing-valid-native":
        return _zxing_runner(native=True, errors=False, work_size=config.work_size)
    raise ValueError(name)


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(median(values)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--work-size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=1)
    parser.add_argument("--windows", default="12")
    parser.add_argument("--orientation-bins", type=int, default=12)
    parser.add_argument("--gradient-threshold", type=float, default=48.0)
    parser.add_argument("--edge-occupancy", type=float, default=0.18)
    parser.add_argument("--coherence", type=float, default=0.92)
    parser.add_argument("--seed-context-coherence", type=float, default=0.82)
    parser.add_argument("--percentile", type=float, default=35.0)
    parser.add_argument("--minimum-cells", type=int, default=3)
    parser.add_argument("--component-density", type=float, default=0.0)
    parser.add_argument("--component-orientation", type=float, default=0.0)
    parser.add_argument("--grow-edge-occupancy", type=float, default=0.07)
    parser.add_argument("--grow-coherence", type=float, default=0.80)
    parser.add_argument("--grow-context-coherence", type=float, default=0.67)
    parser.add_argument("--grow-energy-factor", type=float, default=0.55)
    parser.add_argument("--orientation-tolerance", type=float, default=13.0)
    parser.add_argument("--context-orientation-tolerance", type=float, default=16.9)
    parser.add_argument("--polarity-balance", type=float, default=0.0)
    parser.add_argument("--no-profile-filter", action="store_true")
    parser.add_argument("--profile-transitions", type=int, default=8)
    parser.add_argument("--profile-contrast", type=float, default=20.0)
    parser.add_argument("--profile-agreement", type=float, default=0.45)
    parser.add_argument("--profile-transition-rate", type=float, default=0.72)
    parser.add_argument("--retain-low-transition-rescue", action="store_true")
    parser.add_argument("--split-periodic-bands", action="store_true")
    parser.add_argument("--fragment-gap", type=float, default=2.60)
    parser.add_argument("--maximum-candidates", type=int, default=0)
    parser.add_argument("--minimum-length", type=float, default=45.0)
    parser.add_argument("--native-transitions", type=float, default=12.0)
    parser.add_argument("--progress-every", type=int, default=100)
    arguments = parser.parse_args()

    dataset = load_json(arguments.dataset)
    require_schema(dataset, DATASET_SCHEMA, "dataset")
    dataset = _linear_manifest(dataset)
    config = TensorConfig(
        work_size=arguments.work_size,
        tile_size=arguments.tile_size,
        aggregation_windows=tuple(int(value) for value in arguments.windows.split(",") if value),
        orientation_bins=arguments.orientation_bins,
        gradient_threshold=arguments.gradient_threshold,
        minimum_edge_occupancy=arguments.edge_occupancy,
        minimum_coherence=arguments.coherence,
        seed_context_coherence=arguments.seed_context_coherence,
        response_percentile=arguments.percentile,
        minimum_component_cells=arguments.minimum_cells,
        minimum_component_density=arguments.component_density,
        minimum_component_orientation=arguments.component_orientation,
        grow_edge_occupancy=arguments.grow_edge_occupancy,
        grow_coherence=arguments.grow_coherence,
        grow_context_coherence=arguments.grow_context_coherence,
        grow_energy_factor=arguments.grow_energy_factor,
        orientation_tolerance_degrees=arguments.orientation_tolerance,
        context_orientation_tolerance_degrees=arguments.context_orientation_tolerance,
        minimum_polarity_balance=arguments.polarity_balance,
        profile_filter=not arguments.no_profile_filter,
        minimum_profile_transitions=arguments.profile_transitions,
        minimum_profile_contrast=arguments.profile_contrast,
        minimum_profile_agreement=arguments.profile_agreement,
        maximum_profile_transition_rate=arguments.profile_transition_rate,
        retain_low_transition_rescue=arguments.retain_low_transition_rescue,
        split_periodic_bands=arguments.split_periodic_bands,
        fragment_join_max_gap_short=arguments.fragment_gap,
        maximum_candidates=arguments.maximum_candidates,
        minimum_source_length=arguments.minimum_length,
        minimum_native_transitions=arguments.native_transitions,
    )
    prepare, run = _implementation(arguments.method, config)
    preparations: list[float] = []
    engine_times: list[float] = []
    stage_times: dict[str, list[float]] = {}
    output_images = []
    images = dataset["images"]

    # Warm caches without using a labeled result.
    if images:
        probe = cv2.imread(str(images[0]["path"]), cv2.IMREAD_GRAYSCALE)
        if probe is None:
            raise RuntimeError(images[0]["path"])
        prepared_probe, _ = prepare(probe)
        run(prepared_probe)

    wall_started = perf_counter()
    for index, record in enumerate(images, start=1):
        load_started = perf_counter()
        image = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        load_seconds = perf_counter() - load_started
        if image is None:
            raise RuntimeError(record["path"])
        prepared, preparation_seconds = prepare(image)
        predictions, diagnostics, timings = run(prepared)
        engine_seconds = float(timings["engine_seconds"])
        preparations.append(preparation_seconds)
        engine_times.append(engine_seconds)
        for key, value in timings.items():
            stage_times.setdefault(key, []).append(1000.0 * float(value))
        output_images.append(
            {
                "id": record["id"],
                "predictions": predictions,
                "timing": {"latency_ms": 1000.0 * engine_seconds, "peak_rss_mb": None},
                "diagnostics": {
                    **diagnostics,
                    "load_ms": 1000.0 * load_seconds,
                    "preparation_ms": 1000.0 * preparation_seconds,
                    "engine_ms": 1000.0 * engine_seconds,
                },
            }
        )
        if arguments.progress_every and (index % arguments.progress_every == 0 or index == len(images)):
            print(
                f"{arguments.method}: {index}/{len(images)} "
                f"predictions={sum(len(item['predictions']) for item in output_images)} "
                f"engine_median={1000.0 * median(engine_times):.3f}ms",
                flush=True,
            )

    predictions = {
        "schema_version": PREDICTION_SCHEMA,
        "run": {
            "pipeline": arguments.method,
            "implementation": "structure-tensor-linear-v1",
            "dataset_manifest": str(arguments.dataset.resolve()),
            "evaluation_scope": "linear-only",
            "parameters": config.__dict__,
            "wall_seconds": perf_counter() - wall_started,
            "preparation_ms": _summary([1000.0 * value for value in preparations]),
            "engine_ms": _summary([1000.0 * value for value in engine_times]),
            "stages_ms": {key: _summary(values) for key, values in stage_times.items()},
            "environment": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "opencv_threads": cv2.getNumThreads(),
            },
        },
        "images": output_images,
    }
    write_json(arguments.output, predictions)
    report = evaluate(dataset, predictions)
    report_path = arguments.report or arguments.output.with_name(f"{arguments.output.stem}-report.json")
    write_json(report_path, report)
    print(json.dumps({
        "method": arguments.method,
        "images": len(images),
        "predictions": sum(len(item["predictions"]) for item in output_images),
        "engine_ms": predictions["run"]["engine_ms"],
        "localization": report["localization"],
        "output": str(arguments.output.resolve()),
        "report": str(report_path.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
