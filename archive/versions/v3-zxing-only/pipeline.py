#!/usr/bin/env python3
"""ZXing-C++-only barcode localization and decoding pipeline.

This package builds on the repository's shared PDF/result utilities and the
deterministic OpenCV locator, but it has no commercial decoder path. Easy
symbols use a short fast path. Only unresolved 1-D proposals receive geometry,
sampling-phase, anisotropic-scale, and scanline-consensus recovery. Matrix
codes use deterministic two-directional-gradient proposals followed by a
bounded, context-aware ZXing crop portfolio.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import zxingcpp

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deterministic_barcode_locator.barcode_locator import Detection as LocatorDetection  # noqa: E402
from deterministic_barcode_locator.barcode_locator import expand_quad, locate_page, order_quad, rectify_quad  # noqa: E402
from hybrid_barcode_pipeline.pipeline import (  # noqa: E402
    LINEAR_FORMATS,
    MATRIX_FORMATS,
    PipelineConfig,
    Result,
    annotate,
    collect_pages,
    decode_candidate,
    merge_results,
    normalize_format,
    parse_page_selection,
    whole_page_results,
    write_result_crops,
    zxing_quad,
    zxing_read,
)


@dataclass
class ZXingPipelineConfig:
    output: Path
    render_dpi: int = 300
    pages: list[int] | None = None
    angle_step: int = 0
    minimum_score: float = 0.52
    minimum_linear_length: float = 100.0
    tensor_fallback: bool = False
    matrix_recovery: bool = True
    matrix_proposal_limit: int = 14
    deep_matrix_recovery: bool = False
    save_crops: bool = True
    include_unresolved: bool = True


def rotate_same_canvas(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-6:
        return image
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    white = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=white,
    )


def prepare_variant(
    image: np.ndarray,
    detection: LocatorDetection,
    pad_long: float,
    pad_short: float,
    angle: float,
    scale_x: float,
    scale_y: float,
    interpolation: int,
    preprocessing: str,
    phase_x: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    source_quad = np.asarray(detection.quad, dtype=np.float32)
    if pad_long or pad_short:
        source_quad = expand_quad(source_quad, 1.0 + 2.0 * pad_long, 1.0 + 2.0 * pad_short)
    source = order_quad(source_quad)
    top = np.linalg.norm(source[1] - source[0])
    bottom = np.linalg.norm(source[2] - source[3])
    left = np.linalg.norm(source[3] - source[0])
    right = np.linalg.norm(source[2] - source[1])
    width = max(2, int(round(max(top, bottom))))
    height = max(2, int(round(max(left, right))))
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    page_to_variant = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    crop = cv2.warpPerspective(
        image,
        page_to_variant,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if crop.shape[0] > crop.shape[1]:
        old_height = crop.shape[0]
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        rotation_90 = np.asarray([[0.0, -1.0, old_height - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        page_to_variant = rotation_90 @ page_to_variant
    if abs(angle) >= 1e-6:
        height, width = crop.shape[:2]
        affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        crop = cv2.warpAffine(
            crop,
            affine,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        page_to_variant = np.vstack([affine, [0.0, 0.0, 1.0]]) @ page_to_variant
    if phase_x:
        transform = np.asarray([[1.0, 0.0, phase_x], [0.0, 1.0, 0.0]], dtype=np.float32)
        white = (255, 255, 255) if crop.ndim == 3 else 255
        crop = cv2.warpAffine(
            crop,
            transform,
            (crop.shape[1], crop.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=white,
        )
        page_to_variant = np.asarray(
            [[1.0, 0.0, phase_x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        ) @ page_to_variant
    if scale_x != 1.0 or scale_y != 1.0:
        crop = cv2.resize(crop, None, fx=scale_x, fy=scale_y, interpolation=interpolation)
        page_to_variant = np.asarray(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        ) @ page_to_variant
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if preprocessing == "otsu":
        output = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return output, np.linalg.inv(page_to_variant)
    if preprocessing == "clahe":
        output = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return output, np.linalg.inv(page_to_variant)
    if preprocessing == "unsharp":
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        output = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
        return output, np.linalg.inv(page_to_variant)
    return gray, np.linalg.inv(page_to_variant)


def map_barcode_quad_to_page(barcode: Any, variant_to_page: np.ndarray) -> np.ndarray:
    local = zxing_quad(barcode).reshape(1, -1, 2).astype(np.float32)
    return cv2.perspectiveTransform(local, variant_to_page.astype(np.float64))[0]


def scanline_consensus_variants(image: np.ndarray, detection: LocatorDetection) -> Iterable[tuple[str, np.ndarray]]:
    """Create clean synthetic bar fields from the most consistent scanlines."""
    crop = rectify_quad(image, detection.quad, pad_long=0.30, pad_short=0.25)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    height, width = gray.shape[:2]
    if height < 8 or width < 60:
        return
    top = int(round(height * 0.08))
    bottom = max(top + 4, int(round(height * 0.78)))
    band = gray[top:bottom]
    if band.shape[0] < 4:
        return

    sharpness = np.mean(np.abs(cv2.Scharr(band, cv2.CV_32F, 1, 0)), axis=1)
    selected = band[np.argsort(sharpness)[-min(24, band.shape[0]) :]]
    profiles = {
        "median": np.median(selected, axis=0).astype(np.uint8),
        "mean": np.mean(selected, axis=0).astype(np.uint8),
    }
    for name, profile in profiles.items():
        profile = cv2.medianBlur(profile.reshape(1, -1), 3).ravel()
        stacked = np.repeat(profile.reshape(1, -1), 64, axis=0)
        for scale, interpolation, interpolation_name in (
            (3.0, cv2.INTER_CUBIC, "cubic3"),
            (4.0, cv2.INTER_LANCZOS4, "lanczos4"),
        ):
            enlarged = cv2.resize(stacked, None, fx=scale, fy=1.0, interpolation=interpolation)
            yield f"consensus-{name}-{interpolation_name}", enlarged
            binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            yield f"consensus-{name}-{interpolation_name}-otsu", binary


def decode_advanced_linear(image: np.ndarray, detection: LocatorDetection, previous: Result) -> Result:
    """Escalate only an unresolved, structurally accepted 1-D proposal."""
    result = previous
    recipes = [
        # Geometry correction for thin, slightly misoriented Code 39 fields.
        ("angle-nearest-otsu", 0.00, 0.05, (-4, -2, 2, 4), 1.5, 1.5, cv2.INTER_NEAREST, "otsu", (0.0,)),
        # Small Code 128: module-aware horizontal scale without vertical blur.
        ("horizontal-lanczos-otsu", 0.00, 0.05, (0,), 4.0, 1.0, cv2.INTER_LANCZOS4, "otsu", (0.0, 0.25, 0.5, 0.75)),
        # Overlapping/parent barcode: restore missing long-axis context.
        ("wide-parent-cubic", 0.40, 0.05, (-4, -3, -2, 0, 2, 3, 4), 3.0, 1.0, cv2.INTER_CUBIC, "gray", (0.0, 0.5)),
        ("wide-parent-clahe", 0.60, 0.15, (-3, 0, 3), 3.0, 1.5, cv2.INTER_CUBIC, "clahe", (0.0, 0.5)),
    ]
    binarizers = (
        zxingcpp.Binarizer.LocalAverage,
        zxingcpp.Binarizer.GlobalHistogram,
        zxingcpp.Binarizer.FixedThreshold,
    )
    for name, pad_long, pad_short, angles, sx, sy, interpolation, preprocessing, phases in recipes:
        for angle in angles:
            for phase in phases:
                variant, variant_to_page = prepare_variant(
                    image,
                    detection,
                    pad_long,
                    pad_short,
                    angle,
                    sx,
                    sy,
                    interpolation,
                    preprocessing,
                    phase,
                )
                for binarizer in binarizers:
                    attempt = f"zxing-advanced:{name}:a{angle}:phase{phase}:{binarizer}"
                    result.attempts.append(attempt)
                    valid = [
                        barcode
                        for barcode in zxing_read(
                            variant,
                            formats=LINEAR_FORMATS,
                            binarizer=binarizer,
                            try_downscale=False,
                            return_errors=True,
                        )
                        if barcode.valid
                    ]
                    if valid:
                        barcode = valid[0]
                        result.text = str(barcode.text)
                        result.format = normalize_format(str(barcode.format))
                        result.confidence = 1.0
                        result.status = "decoded"
                        result.quad = map_barcode_quad_to_page(barcode, variant_to_page)
                        result.sources.add(f"zxing:advanced:{name}")
                        return result

    for name, variant in scanline_consensus_variants(image, detection):
        result.attempts.append(f"zxing-advanced:{name}")
        valid = [
            barcode
            for barcode in zxing_read(
                variant,
                formats=LINEAR_FORMATS,
                binarizer=zxingcpp.Binarizer.LocalAverage,
                try_downscale=False,
                return_errors=True,
            )
            if barcode.valid
        ]
        if valid:
            barcode = valid[0]
            result.text = str(barcode.text)
            result.format = normalize_format(str(barcode.format))
            result.confidence = 1.0
            result.status = "decoded"
            result.sources.add(f"zxing:advanced:{name}")
            return result
    return result


def matrix_proposals(image: np.ndarray, limit: int) -> list[dict[str, float]]:
    """Find regions with strong local transitions in both directions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Scharr(gray, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Scharr(gray, cv2.CV_32F, 0, 1))
    raw: list[dict[str, float]] = []
    for local_size in (9, 15, 25, 41):
        energy_x = cv2.boxFilter(gx, cv2.CV_32F, (local_size, local_size), normalize=True)
        energy_y = cv2.boxFilter(gy, cv2.CV_32F, (local_size, local_size), normalize=True)
        response = np.sqrt(energy_x * energy_y)
        threshold = float(np.percentile(response, 99.5))
        mask = (response >= threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            aspect = width / max(1.0, float(height))
            if not (12 <= width <= 420 and 12 <= height <= 420 and 0.25 <= aspect <= 4.0):
                continue
            raw.append(
                {
                    "score": float(np.mean(response[y : y + height, x : x + width])),
                    "cx": x + width / 2.0,
                    "cy": y + height / 2.0,
                    "width": float(width),
                    "height": float(height),
                    "scale": float(local_size),
                }
            )
    raw.sort(key=lambda item: item["score"], reverse=True)
    kept: list[dict[str, float]] = []
    for candidate in raw:
        if all(
            (candidate["cx"] - prior["cx"]) ** 2 + (candidate["cy"] - prior["cy"]) ** 2 > 70**2
            for prior in kept
        ):
            kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


def context_crop(
    image: np.ndarray,
    center: tuple[float, float],
    width: int,
    height: int,
    fraction_x: float,
    fraction_y: float,
) -> tuple[np.ndarray, int, int]:
    cx, cy = center
    x1 = int(round(cx - fraction_x * width))
    y1 = int(round(cy - fraction_y * height))
    x1 = min(max(0, x1), max(0, image.shape[1] - width))
    y1 = min(max(0, y1), max(0, image.shape[0] - height))
    x2 = min(image.shape[1], x1 + width)
    y2 = min(image.shape[0], y1 + height)
    return image[y1:y2, x1:x2], x1, y1


def decode_matrix_proposals(image: np.ndarray, proposals: Sequence[dict[str, float]], deep: bool) -> list[Result]:
    compact_recipes = [
        (500, 500, 0.80, 0.75, 1.0),
        (500, 500, 0.65, 0.60, 1.5),
        (700, 700, 0.80, 0.60, 1.5),
        (850, 800, 0.80, 0.60, 1.5),
    ]
    deep_recipes = [
        (700, 700, 0.50, 0.50, 1.0),
        (700, 700, 0.65, 0.75, 2.0),
        (850, 800, 0.50, 0.60, 2.0),
        (1000, 900, 0.80, 0.75, 1.5),
    ]
    recipes = compact_recipes + (deep_recipes if deep else [])
    output: list[Result] = []
    for proposal_index, proposal in enumerate(proposals):
        center = (proposal["cx"], proposal["cy"])
        decoded_this_proposal = False
        for width, height, fraction_x, fraction_y, scale in recipes:
            crop, offset_x, offset_y = context_crop(image, center, width, height, fraction_x, fraction_y)
            if crop.size == 0:
                continue
            variant = crop if scale == 1.0 else cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            for binarizer in (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram):
                reads = zxing_read(
                    variant,
                    formats=MATRIX_FORMATS,
                    binarizer=binarizer,
                    try_downscale=False,
                    return_errors=True,
                )
                for barcode in reads:
                    if not barcode.valid:
                        continue
                    output.append(
                        Result(
                            quad=zxing_quad(barcode, scale=scale, offset=(offset_x, offset_y)),
                            text=str(barcode.text),
                            format=normalize_format(str(barcode.format)),
                            confidence=1.0,
                            status="decoded",
                            sources={"zxing:matrix-context-recovery"},
                            attempts=[
                                f"proposal:{proposal_index}:window{width}x{height}:"
                                f"fraction{fraction_x},{fraction_y}:scale{scale}:{binarizer}"
                            ],
                        )
                    )
                    decoded_this_proposal = True
                    break
                if decoded_this_proposal:
                    break
            if decoded_this_proposal:
                break
    return output


def quad_geometry(quad: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    rectangle = cv2.minAreaRect(np.asarray(quad, dtype=np.float32))
    width, height = (float(value) for value in rectangle[1])
    if width >= height:
        angle = float(rectangle[2])
        long_side, short_side = width, height
    else:
        angle = float(rectangle[2]) + 90.0
        long_side, short_side = height, width
    return np.asarray(rectangle[0], dtype=np.float32), long_side, short_side, angle % 180.0


def repair_thin_decoder_quads(results: Sequence[Result], detections: Sequence[LocatorDetection]) -> None:
    """Replace scanline-only ZXing positions with corroborating visual quads."""
    for result in results:
        if not result.decoded or result.kind == "matrix":
            continue
        center, long_side, short_side, angle = quad_geometry(result.quad)
        matches: list[tuple[float, LocatorDetection, float]] = []
        for detection in detections:
            candidate_center, candidate_long, candidate_short, candidate_angle = quad_geometry(detection.quad)
            angle_delta = abs(angle - candidate_angle)
            angle_delta = min(angle_delta, 180.0 - angle_delta)
            center_delta = float(np.linalg.norm(center - candidate_center))
            length_ratio = long_side / max(1.0, candidate_long)
            if angle_delta <= 14.0 and center_delta <= 0.38 * candidate_long and 0.45 <= length_ratio <= 1.65:
                score = center_delta / max(1.0, candidate_long) + angle_delta / 30.0
                matches.append((score, detection, candidate_short))
        if not matches:
            continue
        _, detection, candidate_short = min(matches, key=lambda item: item[0])
        # ZXing often reports a single scanline for a valid 1-D read. Repair
        # only those degenerate positions; a thin but non-degenerate mapped
        # quad can be more accurate than the original visual proposal.
        if short_side < 8.0:
            rectangle = cv2.minAreaRect(np.asarray(result.quad, dtype=np.float32))
            width, height = (float(value) for value in rectangle[1])
            repaired_short = max(8.0, 0.75 * candidate_short)
            repaired_size = (width, repaired_short) if width >= height else (repaired_short, height)
            result.quad = cv2.boxPoints((rectangle[0], repaired_size, rectangle[2])).astype(np.float32)
            result.sources.add("geometry:deterministic-quad-repair")


def process_page(
    page_number: int,
    page_path: Path,
    source_mode: str,
    config: ZXingPipelineConfig,
) -> dict[str, Any]:
    page_started = perf_counter()
    image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read {page_path}")
    timings: dict[str, float] = {}

    started = perf_counter()
    anchors = whole_page_results(image, None)
    timings["whole_page_decode_seconds"] = perf_counter() - started

    started = perf_counter()
    located, rejected = locate_page(
        image,
        angle_step=max(0, config.angle_step),
        minimum_score=config.minimum_score,
        use_tensor_fallback=config.tensor_fallback,
        allow_tensor_only=False,
        minimum_linear_length=config.minimum_linear_length,
    )
    timings["linear_localization_seconds"] = perf_counter() - started

    started = perf_counter()
    linear_results: list[Result] = []
    advanced_attempted = 0
    for detection in located:
        if detection.kind != "linear":
            continue
        result = decode_candidate(image, detection, None)
        if not result.decoded:
            advanced_attempted += 1
            result = decode_advanced_linear(image, detection, result)
        linear_results.append(result)
    timings["linear_decode_seconds"] = perf_counter() - started

    started = perf_counter()
    proposals = matrix_proposals(image, config.matrix_proposal_limit) if config.matrix_recovery else []
    matrix_results = decode_matrix_proposals(image, proposals, config.deep_matrix_recovery) if proposals else []
    timings["matrix_recovery_seconds"] = perf_counter() - started

    results = merge_results([*anchors, *linear_results, *matrix_results])
    repair_thin_decoder_quads(results, located)
    if not config.include_unresolved:
        results = [result for result in results if result.decoded]
    results.sort(key=lambda item: (round(float(item.center()[1]) / 20), float(item.center()[0])))

    overlay = annotate(image, results)
    overlay_path = config.output / "overlays" / f"page-{page_number:04d}.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)
    if config.save_crops:
        write_result_crops(image, results, config.output, page_number)

    timings["total_seconds"] = perf_counter() - page_started
    decoded_count = sum(result.decoded for result in results)
    return {
        "page": page_number,
        "source_image": str(page_path),
        "source_mode": source_mode,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "decoded_count": decoded_count,
        "unresolved_count": len(results) - decoded_count,
        "results": [result.to_json() for result in results],
        "diagnostics": {
            "accepted_linear_proposals": len(located),
            "rejected_linear_proposals": len(rejected),
            "advanced_linear_candidates": advanced_attempted,
            "matrix_proposals": len(proposals),
        },
        "overlay": str(overlay_path),
        "timings": {key: round(value, 4) for key, value in timings.items()},
    }


def run_pipeline(input_path: Path, config: ZXingPipelineConfig) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    config.output = config.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    config.output.mkdir(parents=True, exist_ok=True)

    common_config = PipelineConfig(
        output=config.output,
        render_dpi=config.render_dpi,
        pages=config.pages,
        angle_step=config.angle_step,
        minimum_score=config.minimum_score,
        minimum_linear_length=config.minimum_linear_length,
        tensor_fallback=config.tensor_fallback,
        engine="zxing",
        matrix_sweep=False,
        save_crops=config.save_crops,
        include_unresolved=config.include_unresolved,
    )
    started = perf_counter()
    pages = collect_pages(input_path, common_config)
    extraction_seconds = perf_counter() - started
    payloads: list[dict[str, Any]] = []
    for page_number, page_path, source_mode in pages:
        payload = process_page(page_number, page_path, source_mode, config)
        payloads.append(payload)
        print(
            f"page {page_number:04d}: {payload['decoded_count']} decoded, "
            f"{payload['unresolved_count']} unresolved, {payload['timings']['total_seconds']:.2f}s",
            flush=True,
        )
    decoded = sum(page["decoded_count"] for page in payloads)
    unresolved = sum(page["unresolved_count"] for page in payloads)
    payload = {
        "input": str(input_path),
        "method": "ZXing-only native raster + deterministic 1-D proposals + adaptive linear and matrix recovery",
        "uses_commercial_decoder": False,
        "engines": {"zxing_cpp": True},
        "configuration": {
            "render_dpi": config.render_dpi,
            "angle_step": config.angle_step,
            "minimum_score": config.minimum_score,
            "minimum_linear_length": config.minimum_linear_length,
            "tensor_fallback": config.tensor_fallback,
            "matrix_recovery": config.matrix_recovery,
            "matrix_proposal_limit": config.matrix_proposal_limit,
            "deep_matrix_recovery": config.deep_matrix_recovery,
        },
        "summary": {
            "pages": len(payloads),
            "decoded": decoded,
            "unresolved": unresolved,
            "extraction_seconds": round(extraction_seconds, 4),
            "processing_seconds": round(sum(page["timings"]["total_seconds"] for page in payloads), 4),
        },
        "pages": payloads,
    }
    destination = config.output / "detections.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {destination}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("zxing_only_barcode_pipeline/output"))
    parser.add_argument("--pages", type=parse_page_selection)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--angle-step", type=int, default=0)
    parser.add_argument("--minimum-score", type=float, default=0.52)
    parser.add_argument("--minimum-linear-length", type=float, default=100.0)
    parser.add_argument("--tensor-fallback", action="store_true")
    parser.add_argument("--no-matrix-recovery", dest="matrix_recovery", action="store_false")
    parser.add_argument("--matrix-proposal-limit", type=int, default=14)
    parser.add_argument("--deep-matrix-recovery", action="store_true")
    parser.add_argument("--no-crops", dest="save_crops", action="store_false")
    parser.add_argument("--decoded-only", dest="include_unresolved", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = ZXingPipelineConfig(
        output=arguments.output,
        render_dpi=arguments.render_dpi,
        pages=arguments.pages,
        angle_step=arguments.angle_step,
        minimum_score=arguments.minimum_score,
        minimum_linear_length=arguments.minimum_linear_length,
        tensor_fallback=arguments.tensor_fallback,
        matrix_recovery=arguments.matrix_recovery,
        matrix_proposal_limit=arguments.matrix_proposal_limit,
        deep_matrix_recovery=arguments.deep_matrix_recovery,
        save_crops=arguments.save_crops,
        include_unresolved=arguments.include_unresolved,
    )
    try:
        run_pipeline(arguments.input, config)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
