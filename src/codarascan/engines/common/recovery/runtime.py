#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Accuracy-gated, throughput-optimized barcode pipeline.

The implementation preserves the validated ZXing/OpenCV recovery cascade with
per-format selection.  Formats are auto-categorized into matrix vs linear so
the correct detector path is always used.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import zxingcpp

from codarascan.engines.common.classical_localizer import (  # noqa: E402
    data_matrix_threshold_mask,
    local_qr_finder_centers,
    qr_finder_ratio_hits,
    refine_data_matrix_candidate,
    score_data_matrix_mask,
    square_warp,
)
from codarascan.engines.common.recovery.hybrid import (  # noqa: E402
    PipelineConfig,
    collect_pages,
    decode_candidate,
    normalize_format,
)
from codarascan.engines.common.recovery.locator import (
    EPS,
    SCALE_PROFILES,
    accept_linear,
    barcode_structure_metrics,
    deduplicate,
    make_detector,
    order_quad,
    quad_overlap,
    rectify_quad,
)
from codarascan.engines.common.recovery.locator import (  # noqa: E402
    Detection as LocatorDetection,
)
from codarascan.engines.common.recovery.matrix import (  # noqa: E402
    MatrixAnchor,
    Result,
    annotate,
    barcode_quad,
    merge_anchors,
    merge_results,
    prepare_work_directory,
    recover_invalid_anchor,
    safe_zxing_read,
    sha256_file,
    unresolved_from_anchor,
    utc_now,
    visual_matrix_proposals,
    write_crop,
)
from codarascan.engines.common.recovery.zxing import decode_advanced_linear  # noqa: E402

from .adaptive import (
    adaptive_short_linear_accept,
    color_linear_decode_views,
    dense_layout_candidate_review_allowed,
    dense_layout_short_review_allowed,
    ean13_checksum_valid,
    estimate_linear_module_metrics,
    input_quality_diagnostics,
    opencv_qr_geometry_proposals,
    proposal_scale_plan,
    relative_matrix_contexts,
    repeated_ean13_template_evidence,
    split_dense_linear_quad,
    unique_scales,
)

UNION_LINEAR_SCALES = tuple(
    sorted({value for _name, scales in SCALE_PROFILES for value in scales})
)

_MATRIX_FORMAT_TOKENS = frozenset({
    "datamatrix", "data matrix", "qr code", "qrcode", "micro qr",
    "micro qr code", "rmqr", "aztec", "pdf417", "micro pdf417",
    "maxicode",
})

_META_FORMAT_NAMES = frozenset({
    "all", "all readable", "all creatable", "all linear", "all matrix",
    "all gs1", "all retail", "all industrial", "other barcode",
})


def category_from_name(name: str) -> str | None:
    lower = name.lower().replace("barcodeformat.", "")
    if lower in _META_FORMAT_NAMES:
        return None
    if any(token in lower for token in _MATRIX_FORMAT_TOKENS):
        return "matrix"
    return "linear"


def categorize_formats(
    barcode_formats: Any,
) -> tuple[Any, Any]:
    selected = list(barcode_formats) if barcode_formats is not None else []
    if not selected:
        return (
            zxingcpp.barcode_formats_from_str("AllMatrix"),
            zxingcpp.barcode_formats_from_str("AllLinear"),
        )
    matrix: list[str] = []
    linear: list[str] = []
    for fmt in selected:
        category = category_from_name(str(fmt))
        if category == "matrix":
            matrix.append(str(fmt))
        elif category == "linear":
            linear.append(str(fmt))
    return (
        zxingcpp.barcode_formats_from_str(",".join(matrix)),
        zxingcpp.barcode_formats_from_str(",".join(linear)),
    )


@dataclass
class Config:
    output: Path
    barcode_formats: Any = None
    render_dpi: int = 300
    pages: list[int] | None = None
    overwrite: bool = False
    save_crops: bool = True
    save_overlays: bool = True
    residual_proposals: bool = True
    include_review_candidates: bool = True
    keep_unresolved_qr: bool = False
    # Unresolved matrix geometry is user-visible evidence.  Keep it behind
    # format-specific physical verification by default; valid ZXing decodes
    # are not affected by this setting.
    strict_matrix_acceptance: bool = True
    minimum_linear_score: float = 0.52
    minimum_linear_length: float = 100.0
    workers: int = 0
    linear_detector: str = "optimized"
    batch_pdf_extraction: bool = True
    adaptive_generalization: bool = True


def collect_pages_extended(input_path: Path, config: Config, work: Path) -> list[tuple[int, Path, str]]:
    supported_images = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if input_path.is_file() and input_path.suffix.lower() in supported_images:
        pages_dir = work / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        destination = pages_dir / f"page-0001{input_path.suffix.lower()}"
        shutil.copy2(input_path, destination)
        return [(1, destination, "web-upload-image")]
    shared = PipelineConfig(
        output=work,
        render_dpi=config.render_dpi,
        pages=config.pages,
        angle_step=0,
        minimum_score=config.minimum_linear_score,
        minimum_linear_length=config.minimum_linear_length,
        tensor_fallback=False,
        engine="zxing",
        matrix_sweep=False,
        save_crops=config.save_crops,
        include_unresolved=config.include_review_candidates,
    )
    return collect_pages(input_path, shared)


def _linear_detector_configs(mode: str) -> list[tuple[str, Sequence[float], float]]:
    if mode == "exhaustive":
        return [
            (f"{name}:g{int(threshold)}", scales, threshold)
            for name, scales in SCALE_PROFILES
            for threshold in (48.0, 64.0)
        ]
    return [("union:g48", UNION_LINEAR_SCALES, 48.0)]


def visual_linear_proposals(
    gray: np.ndarray,
    mode: str,
    adaptive_generalization: bool = True,
) -> list[LocatorDetection]:
    proposals: list[LocatorDetection] = []
    plans = proposal_scale_plan(gray.shape, adaptive_generalization)
    for plan in plans:
        work = (
            gray
            if plan.scale == 1.0
            else cv2.resize(gray, None, fx=plan.scale, fy=plan.scale, interpolation=plan.interpolation)
        )
        for name, scales, threshold in _linear_detector_configs(mode):
            detector = make_detector(scales, work.shape, threshold)
            try:
                found, points = detector.detectMulti(work)
            except cv2.error:
                continue
            if not found or points is None:
                continue
            for quad in np.asarray(points, dtype=np.float32):
                mapped = quad / plan.scale
                detection = LocatorDetection(
                    quad=mapped,
                    sources={f"opencv:optimized:{name}:scale-{plan.scale:g}"},
                    proposal_score=0.48 if plan.scale == 1.0 else 0.46,
                )
                long_side, short_side = detection.long_short()
                if long_side < 14 or short_side < 3 or long_side / (short_side + EPS) < 1.15:
                    continue
                proposals.append(detection)
    merged = deduplicate(proposals)
    # Splitting is a dense-sheet refinement, not a generic barcode operation.
    # A long isolated symbol may contain legitimate internal quiet-looking
    # stretches, so never segment sparse pages this way.
    if len(merged) < 12:
        return merged
    refined: list[LocatorDetection] = []
    for detection in merged:
        children = split_dense_linear_quad(gray, detection.quad)
        if not children:
            refined.append(detection)
            continue
        for index, child in enumerate(children, start=1):
            refined.append(
                LocatorDetection(
                    quad=child,
                    sources=set(detection.sources) | {f"geometry:dense-linear-split:{index}/{len(children)}"},
                    proposal_score=max(0.0, detection.proposal_score - 0.01),
                )
            )
    return deduplicate(refined)


def verify_linear_proposals(
    gray: np.ndarray,
    proposals: Sequence[LocatorDetection],
    config: Config,
) -> tuple[list[LocatorDetection], list[LocatorDetection]]:
    eligible: list[LocatorDetection] = []
    rejected: list[LocatorDetection] = []
    decode_floor = max(0.35, config.minimum_linear_score - 0.16)
    for detection in proposals:
        detection.metrics = barcode_structure_metrics(gray, detection.quad)
        if config.adaptive_generalization:
            detection.metrics.update(
                estimate_linear_module_metrics(gray, detection.quad, rectify_quad)
            )
        detection.structural_score = detection.metrics.get("score", 0.0)
        legacy_accepted = accept_linear(
            detection,
            config.minimum_linear_score,
            allow_tensor_only=False,
            minimum_linear_length=config.minimum_linear_length,
        )
        adaptive_accepted = bool(
            config.adaptive_generalization
            and adaptive_short_linear_accept(
                detection,
                config.minimum_linear_score,
                config.minimum_linear_length,
            )
        )
        # Short adaptive candidates may unlock the decoder, but an undecoded
        # one is not exposed as a confirmed review box. On heterogeneous web
        # images this avoids turning small text/QR fragments into linear codes.
        detection.accepted = legacy_accepted
        detection.metrics["legacy_length_gate_passed"] = float(legacy_accepted)
        detection.metrics["adaptive_short_recovery_candidate"] = float(adaptive_accepted)
        if detection.accepted or adaptive_accepted or detection.structural_score >= decode_floor:
            eligible.append(detection)
        else:
            rejected.append(detection)
    return eligible, rejected


def _scan_matrix(
    image: np.ndarray,
    formats: Any,
    binarizer: Any,
    allow_upscale_fallback: bool = False,
) -> list[tuple[Any, float]]:
    native = safe_zxing_read(
        image,
        formats,
        binarizer,
        try_downscale=True,
        return_errors=True,
    )
    output = [(barcode, 1.0) for barcode in native]
    if any(barcode.valid and barcode.text for barcode in native):
        return output
    if allow_upscale_fallback:
        for plan in proposal_scale_plan(image.shape, enabled=True)[1:]:
            enlarged = cv2.resize(
                image,
                None,
                fx=plan.scale,
                fy=plan.scale,
                interpolation=plan.interpolation,
            )
            reads = safe_zxing_read(
                enlarged,
                formats,
                binarizer,
                try_downscale=False,
                return_errors=True,
            )
            output.extend((barcode, plan.scale) for barcode in reads)
            if any(barcode.valid and barcode.text for barcode in reads):
                break
    return output


def _scan_linear(
    image: np.ndarray,
    formats: Any,
    allow_upscale_fallback: bool = False,
) -> list[tuple[Any, float]]:
    native = safe_zxing_read(
        image,
        formats,
        zxingcpp.Binarizer.LocalAverage,
        try_downscale=False,
        return_errors=False,
    )
    if native:
        return [(barcode, 1.0) for barcode in native]

    # Small web uploads and screenshots often leave a narrow barcode at only
    # one or two pixels per module. ZXing does not upscale during its
    # `try_downscale` search, so use a bounded fallback and retain the scale so
    # positions can be mapped back onto the original page.
    if allow_upscale_fallback and image.size <= 4_000_000:
        for scale in (1.5, 2.0, 3.0):
            enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            reads = safe_zxing_read(
                enlarged,
                formats,
                zxingcpp.Binarizer.LocalAverage,
                try_downscale=False,
                return_errors=False,
            )
            # Upscaling can make short alternating text strokes look like ITF.
            # ITF has no mandatory checksum, so reject only these especially
            # weak, upscaled-only two/four-digit reads. Native reads and normal
            # industrial payload lengths remain unaffected.
            reads = [
                barcode
                for barcode in reads
                if str(barcode.format) != "ITF" or len(str(barcode.text)) >= 6
            ]
            if reads:
                return [(barcode, scale) for barcode in reads]
    return []


def fast_stage(
    gray: np.ndarray,
    config: Config,
    matrix_formats: Any,
    linear_formats: Any,
    parallel: bool,
    allow_linear_upscale: bool = False,
) -> tuple[dict[str, list[Any]], list[LocatorDetection], float]:
    jobs: dict[str, Callable[[], Any]] = {}
    if matrix_formats:
        jobs["matrix-local"] = lambda: _scan_matrix(
            gray,
            matrix_formats,
            zxingcpp.Binarizer.LocalAverage,
            allow_upscale_fallback=allow_linear_upscale and config.adaptive_generalization,
        )
        jobs["matrix-global"] = lambda: _scan_matrix(
            gray,
            matrix_formats,
            zxingcpp.Binarizer.GlobalHistogram,
            allow_upscale_fallback=allow_linear_upscale and config.adaptive_generalization,
        )
    if linear_formats:
        jobs["linear-zxing"] = lambda: _scan_linear(gray, linear_formats, allow_linear_upscale)
        jobs["linear-opencv"] = lambda: visual_linear_proposals(
            gray,
            config.linear_detector,
            config.adaptive_generalization,
        )
    started = perf_counter()
    if parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {name: executor.submit(function) for name, function in jobs.items()}
            values = {name: future.result() for name, future in futures.items()}
    else:
        values = {name: function() for name, function in jobs.items()}
    proposals = values.pop("linear-opencv", [])
    return values, proposals, perf_counter() - started


def parse_matrix_reads(reads_by_name: dict[str, list[Any]]) -> tuple[list[Result], list[MatrixAnchor], dict[str, Any]]:
    decoded: list[Result] = []
    invalid: list[MatrixAnchor] = []
    diagnostics: dict[str, Any] = {"reads": {}, "invalid_errors": {}}
    for key, source_name in (("matrix-local", "local-average"), ("matrix-global", "global-histogram")):
        reads = reads_by_name.get(key, [])
        diagnostics["reads"][source_name] = len(reads)
        for item in reads:
            barcode, scale = item if isinstance(item, tuple) else (item, 1.0)
            quad = barcode_quad(barcode, scale=scale)
            if abs(float(cv2.contourArea(quad))) < 9.0:
                continue
            if barcode.valid and barcode.text:
                decoded.append(
                    Result(
                        quad=quad,
                        text=str(barcode.text),
                        raw_bytes=bytes(barcode.bytes),
                        format=normalize_format(str(barcode.format)),
                        status="decoded",
                        confidence=1.0,
                        sources={f"zxing:matrix-whole-page:{source_name}:scale-{scale:g}"},
                        attempts=[f"whole-page:{source_name}:scale-{scale:g}"],
                    )
                )
            else:
                error = str(getattr(barcode, "error", None) or "invalid")
                error_key = error.split(" @ ", 1)[0]
                diagnostics["invalid_errors"][error_key] = diagnostics["invalid_errors"].get(error_key, 0) + 1
                invalid.append(
                    MatrixAnchor(
                        quad=quad,
                        format=normalize_format(str(barcode.format)),
                        error=error,
                        sources={f"zxing:invalid-matrix:{source_name}:scale-{scale:g}"},
                        attempts=[f"whole-page:{source_name}:scale-{scale:g}:return-errors"],
                    )
                )
    decoded = merge_results(decoded)
    invalid = [
        anchor
        for anchor in merge_anchors(invalid)
        if not any(quad_overlap(anchor.quad, result.quad) >= 0.28 for result in decoded)
    ]
    diagnostics["decoded_unique"] = len(decoded)
    diagnostics["invalid_unique_unexplained"] = len(invalid)
    return decoded, invalid, diagnostics


def parse_linear_reads(reads: Sequence[Any]) -> list[Result]:
    output: list[Result] = []
    for item in reads:
        barcode, scale = item if isinstance(item, tuple) else (item, 1.0)
        if not barcode.valid or not barcode.text:
            continue
        output.append(
            Result(
                quad=barcode_quad(barcode, scale=scale),
                text=str(barcode.text),
                raw_bytes=bytes(barcode.bytes),
                format=normalize_format(str(barcode.format)),
                status="decoded",
                confidence=1.0,
                sources={"zxing:linear-whole-page" if scale == 1.0 else "zxing:linear-whole-page:upscaled"},
                attempts=[f"whole-page:linear:local-average:scale-{scale:g}"],
            )
        )
    return merge_results(output)


def _context_crop(
    image: np.ndarray,
    center: np.ndarray,
    width: int,
    height: int,
    fraction_x: float,
    fraction_y: float,
) -> tuple[np.ndarray, int, int]:
    x1 = int(round(float(center[0]) - fraction_x * width))
    y1 = int(round(float(center[1]) - fraction_y * height))
    x1 = min(max(0, x1), max(0, image.shape[1] - width))
    y1 = min(max(0, y1), max(0, image.shape[0] - height))
    return image[y1 : min(image.shape[0], y1 + height), x1 : min(image.shape[1], x1 + width)], x1, y1


def recover_matrix_anchor(
    image: np.ndarray,
    anchor: MatrixAnchor,
    matrix_formats: Any,
    adaptive_generalization: bool = True,
) -> Result | None:
    result = recover_invalid_anchor(image, anchor)
    if result is not None:
        return result
    recipes = (
        relative_matrix_contexts(image.shape, anchor.quad)
        if adaptive_generalization
        else [
            (500, 500, 0.80, 0.75, 1.0, "offset500-native"),
            (500, 500, 0.65, 0.60, 1.5, "offset500-cubic15"),
            (700, 700, 0.80, 0.60, 1.5, "offset700-cubic15"),
        ]
    )
    for width, height, fraction_x, fraction_y, scale, name in recipes:
        crop, offset_x, offset_y = _context_crop(image, anchor.center(), width, height, fraction_x, fraction_y)
        variant = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for binarizer_name, binarizer in (
            ("local-average", zxingcpp.Binarizer.LocalAverage),
            ("global-histogram", zxingcpp.Binarizer.GlobalHistogram),
        ):
            reads = safe_zxing_read(
                variant,
                matrix_formats,
                binarizer,
                try_downscale=True,
                return_errors=False,
            )
            mapped: list[tuple[float, Any, np.ndarray]] = []
            for barcode in reads:
                if not barcode.valid or not barcode.text:
                    continue
                fmt = normalize_format(str(barcode.format))
                if anchor.format and fmt != anchor.format:
                    continue
                quad = barcode_quad(barcode, scale=scale, offset=(offset_x, offset_y))
                distance = float(np.linalg.norm(quad.mean(axis=0) - anchor.center()))
                mapped.append((distance, barcode, quad))
            if not mapped:
                continue
            distance, barcode, quad = min(mapped, key=lambda item: item[0])
            ordered_anchor = order_quad(np.asarray(anchor.quad, dtype=np.float32))
            anchor_width = max(
                float(np.linalg.norm(ordered_anchor[1] - ordered_anchor[0])),
                float(np.linalg.norm(ordered_anchor[2] - ordered_anchor[3])),
            )
            anchor_height = max(
                float(np.linalg.norm(ordered_anchor[3] - ordered_anchor[0])),
                float(np.linalg.norm(ordered_anchor[2] - ordered_anchor[1])),
            )
            maximum_distance = max(45.0, 0.90 * max(anchor_width, anchor_height))
            if distance > maximum_distance:
                continue
            return Result(
                quad=quad,
                text=str(barcode.text),
                format=normalize_format(str(barcode.format)),
                status="decoded",
                confidence=1.0,
                sources=set(anchor.sources) | {"zxing:matrix-offset-context-recovery"},
                attempts=anchor.attempts + [f"invalid-anchor:{name}:{binarizer_name}:downscale"],
                evidence={"prior_error": anchor.error, **anchor.evidence},
            )
    return None


def unresolved_matrix_from_anchor(
    image: np.ndarray,
    anchor: MatrixAnchor,
    keep_weak_qr: bool,
    strict_matrix_acceptance: bool = True,
) -> Result | None:
    if strict_matrix_acceptance:
        return strict_unresolved_matrix_from_anchor(image, anchor)

    # Legacy escape hatch for controlled comparisons.  This preserves the
    # former generic square/transition gate, but is intentionally not the
    # production default because it promotes text, MRZ fragments, stamps and
    # form graphics to visible 2-D locations.
    if anchor.error == "opencv-qr-geometry":
        structural = unresolved_from_anchor(image, anchor)
        evidence = {
            key: value
            for key, value in anchor.evidence.items()
            if key != "_strict_matrix_gate"
        }
        if structural is not None:
            evidence.update(structural.evidence)
        return Result(
            quad=np.asarray(anchor.quad, dtype=np.float32),
            text=None,
            format="QR Code",
            status="localized_unresolved_matrix",
            confidence=max(0.90, float(anchor.confidence)),
            sources=set(anchor.sources) | {"verification:opencv-qr-finder-geometry"},
            attempts=list(anchor.attempts),
            evidence={"decoder_error": anchor.error, **evidence},
        )
    result = unresolved_from_anchor(image, anchor)
    if result is None or anchor.format != "QR Code" or keep_weak_qr:
        return result
    structure = result.evidence.get("structure", {})
    if (
        float(structure.get("dark_fraction", 0.0)) < 0.30
        or float(structure.get("square_score", 0.0)) < 0.75
        or float(structure.get("transition_score", 0.0)) < 0.55
    ):
        return None
    result.sources.add("verification:qr-module-occupancy")
    return result


def _matrix_format_kind(value: str | None) -> str | None:
    """Return the only two matrix families with a physical verifier here."""
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if normalized in {"datamatrix", "datamatrixecc200"}:
        return "data-matrix"
    if normalized in {"qrcode", "qr"}:
        return "qr"
    return None


def _numeric_evidence(values: dict[str, Any]) -> dict[str, Any]:
    """Convert NumPy scalar metrics before attaching them to API JSON."""
    output: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (bool, str, int, float)):
            output[key] = value
        else:
            try:
                output[key] = float(value)
            except (TypeError, ValueError):
                output[key] = str(value)
    return output


def _qr_finder_layout(centers: Sequence[tuple[float, float]]) -> dict[str, Any] | None:
    """Require three finder centers to form QR's right-triangle layout.

    Counting nested squares alone is not enough: a form can contain several
    stamps or boxed fields.  After the candidate is perspective-warped, the
    three QR finder centers should form two comparable legs and a diagonal.
    The generous bounds tolerate skew and damaged scans while rejecting a
    line of unrelated nested boxes.
    """
    if len(centers) < 3:
        return None
    points = np.asarray(centers, dtype=np.float32)
    for first, second, third in combinations(range(len(points)), 3):
        selected = points[[first, second, third]]
        distances = sorted(
            float(np.linalg.norm(selected[index] - selected[other]))
            for index, other in ((0, 1), (0, 2), (1, 2))
        )
        short_leg, long_leg, diagonal = distances
        if short_leg < 1.0:
            continue
        leg_ratio = long_leg / short_leg
        right_triangle_ratio = diagonal / max(
            EPS,
            math.hypot(short_leg, long_leg),
        )
        if (
            0.45 <= leg_ratio <= 2.20
            and 0.70 <= right_triangle_ratio <= 1.30
        ):
            return {
                "valid": True,
                "leg_ratio": round(leg_ratio, 4),
                "right_triangle_ratio": round(right_triangle_ratio, 4),
                "finder_centers": [
                    [round(float(point[0]), 2), round(float(point[1]), 2)]
                    for point in selected
                ],
            }
    return None


def strict_unresolved_matrix_from_anchor(
    image: np.ndarray,
    anchor: MatrixAnchor,
) -> Result | None:
    """Promote an invalid matrix anchor only with format-specific evidence.

    The old path treated balanced gradients, a square crop, and many
    transitions as sufficient.  Those are useful proposal features, but they
    are also common in text, MRZs, stamps, table intersections and UI artwork.
    This gate deliberately mirrors the robust classical 2-D extractor:

    * Data Matrix requires an ECC 200 solid/timing border and a plausible
      module lattice.
    * QR requires three independent nested finder patterns.  Scanline
      1:1:3:1:1 evidence is retained as diagnostics only: stamps, forms, and
      repeated text can accidentally produce many such runs without being a
      QR symbol.
    * An unknown/unsupported format is never promoted on generic texture.

    A valid payload never enters this function, so tightening it cannot reduce
    successful decodes.  The trade-off is intentional: a damaged symbol with
    no surviving format-specific evidence remains unresolved by omission
    instead of becoming a false location.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    expected_kind = _matrix_format_kind(anchor.format)
    gate: dict[str, Any] = {
        "policy": "format-specific-physical-v1",
        "expected_format": anchor.format,
        "accepted": False,
    }

    # A caller can inspect why an anchor disappeared through diagnostics, while
    # accepted results receive the same evidence in their public JSON.
    def reject(reason: str) -> None:
        gate["reason"] = reason
        anchor.evidence["_strict_matrix_gate"] = gate

    def accept(format_name: str, kind: str, details: dict[str, Any]) -> Result:
        gate.update(
            {
                "accepted": True,
                "format": format_name,
                "verifier": kind,
                "details": details,
            }
        )
        anchor.evidence["_strict_matrix_gate"] = gate
        evidence = {
            key: value
            for key, value in anchor.evidence.items()
            if key != "_strict_matrix_gate"
        }
        evidence["decoder_error"] = anchor.error
        evidence["matrix_format_gate"] = gate
        return Result(
            quad=np.asarray(anchor.quad, dtype=np.float32),
            text=None,
            format=format_name,
            status="localized_unresolved_matrix",
            confidence=max(float(anchor.confidence), 0.90),
            sources=set(anchor.sources)
            | {"verification:matrix-format-structure", f"verification:{kind}"},
            attempts=list(anchor.attempts),
            evidence=evidence,
        )

    if expected_kind in (None, "data-matrix"):
        # Most false anchors are tiny, already axis-aligned fragments.  A
        # single normalized Otsu/lattice score is enough to reject them and
        # avoids generating up to twelve border hypotheses.  The threshold is
        # deliberately below the final ECC-200 threshold so borderline,
        # damaged symbols still receive the full verifier.
        try:
            direct_patch = square_warp(
                gray,
                np.asarray(anchor.quad, dtype=np.float32),
                160,
            )
            direct_mask = data_matrix_threshold_mask(
                direct_patch,
                adaptive=False,
            )
            dark_points = cv2.findNonZero(direct_mask)
            if dark_points is None:
                direct_score, direct_metrics = 0.0, {"grid_score": 0.0}
            else:
                # Remove the quiet-zone margin before the inexpensive lattice
                # score.  This preserves the module scale for small symbols;
                # scoring the whole padded quad can dilute a real ECC-200
                # border below the prefilter even when the full verifier can
                # prove it.
                x, y, width, height = cv2.boundingRect(dark_points)
                content = direct_mask[y : y + height, x : x + width]
                content = cv2.resize(
                    content,
                    (160, 160),
                    interpolation=cv2.INTER_NEAREST,
                )
                direct_score, direct_metrics = score_data_matrix_mask(content)
            direct_metrics = dict(
                direct_metrics,
                grid_score=direct_score,
                prefilter="single-warp-otsu",
            )
        except (cv2.error, ValueError, TypeError):
            direct_score, direct_metrics = 0.0, {"grid_score": 0.0}

        # Keep a narrow borderline band for decoder-error anchors. A damaged
        # Data Matrix can lose just enough contrast to land at 0.62–0.65 even
        # though its solid/timing borders remain physically identifiable.
        if direct_score < 0.62:
            refined_quad, metrics = None, direct_metrics
        else:
            try:
                refined_quad, metrics = refine_data_matrix_candidate(
                    gray,
                    np.asarray(anchor.quad, dtype=np.float32),
                )
            except (cv2.error, ValueError, TypeError):
                refined_quad, metrics = None, direct_metrics
            if refined_quad is None and direct_metrics:
                # Preserve the cheap score in diagnostics when the expensive
                # hypotheses did not find a better border.
                metrics = dict(direct_metrics, **metrics)
        dm_details = _numeric_evidence(metrics)
        gate["data_matrix"] = dm_details
        if refined_quad is not None:
            return accept("Data Matrix", "data-matrix-ecc200-lattice", dm_details)
        if expected_kind == "data-matrix":
            error_text = str(anchor.error or "").lower()
            damaged_decoder_anchor = bool(
                direct_score >= 0.62
                and any(
                    token in error_text
                    for token in ("checksum", "format", "invalid", "error")
                )
                and float(metrics.get("finder_score", 0.0)) >= 0.50
                and float(metrics.get("solid_left", 0.0)) >= 0.58
                and float(metrics.get("solid_bottom", 0.0)) >= 0.78
                and float(metrics.get("timing_top", 0.0)) >= 0.50
                and float(metrics.get("timing_right", 0.0)) >= 0.50
                and float(metrics.get("cell_purity", 0.0)) >= 0.52
                and 0.16
                <= float(metrics.get("module_occupancy", 0.0))
                <= 0.84
            )
            if damaged_decoder_anchor:
                return accept(
                    "Data Matrix",
                    "data-matrix-damaged-decoder-anchor",
                    dm_details,
                )
            reject("data-matrix-lattice-not-proven")
            return None

    if expected_kind in (None, "qr"):
        try:
            finder_centers = local_qr_finder_centers(gray, anchor.quad)
            finder_count = len(finder_centers)
            ratio_hits = int(qr_finder_ratio_hits(gray, anchor.quad))
        except (cv2.error, ValueError, TypeError):
            finder_centers = []
            finder_count, ratio_hits = 0, 0
        finder_layout = _qr_finder_layout(finder_centers)
        qr_details = {
            "finder_patterns": finder_count,
            "finder_ratio_hits": ratio_hits,
            "acceptance_rule": "three-independent-finders",
            "finder_layout": finder_layout or {"valid": False},
        }
        gate["qr"] = qr_details
        if finder_layout is not None:
            return accept("QR Code", "qr-finder-patterns", qr_details)
        if expected_kind == "qr":
            reject("qr-finder-geometry-not-proven")
            return None

    reject("format-unknown-or-unsupported")
    return None


def legacy_local_linear_decode(
    image: np.ndarray,
    detection: LocatorDetection,
    linear_formats: Any,
) -> Result | None:
    patch = rectify_quad(image, detection.quad, pad_long=0.08, pad_short=0.24)
    patch = cv2.copyMakeBorder(patch, 16, 16, 20, 20, cv2.BORDER_CONSTANT, value=255)
    try:
        barcode = zxingcpp.read_barcode(
            patch,
            formats=linear_formats,
            try_rotate=True,
            try_downscale=False,
            try_invert=True,
            return_errors=False,
        )
    except Exception:
        barcode = None
    if barcode is None or not barcode.valid or not barcode.text:
        return None
    return Result(
        quad=np.asarray(detection.quad, dtype=np.float32),
        text=str(barcode.text),
        raw_bytes=bytes(barcode.bytes),
        format=normalize_format(str(barcode.format)),
        status="decoded",
        confidence=1.0,
        sources=set(detection.sources) | {"deterministic:linear-proposal", "zxing:legacy-deskewed-candidate"},
        attempts=["candidate:legacy-deskewed-native"],
        evidence={"structural_metrics": dict(detection.metrics)},
    )


def _convert_hybrid_result(result: Any) -> Result:
    return Result(
        quad=np.asarray(result.quad, dtype=np.float32),
        text=result.text,
        raw_bytes=getattr(result, "raw_bytes", None),
        format=result.format,
        status="decoded" if result.decoded else "review_candidate",
        confidence=float(result.confidence),
        sources=set(result.sources),
        attempts=list(result.attempts),
        evidence={"structural_metrics": dict(result.structural_metrics)},
    )


def linear_paths(
    image: np.ndarray,
    color_image: np.ndarray | None,
    whole_page: Sequence[Result],
    proposals: Sequence[LocatorDetection],
    config: Config,
    linear_formats: int,
) -> tuple[list[Result], list[Result], dict[str, Any]]:
    eligible, rejected = verify_linear_proposals(image, proposals, config)
    decoded: list[Result] = list(whole_page)
    review: list[Result] = []
    dense_review_enabled = bool(
        config.adaptive_generalization
        and dense_layout_short_review_allowed(
            proposals,
            sum(bool(detection.accepted) for detection in eligible),
        )
    )
    explained = 0
    rejected_by_physics = 0
    rejected_after_decode = 0
    advanced_attempted = 0
    color_attempted = 0
    color_recovered = 0
    repeated_ean13_recovered = 0
    ean13_templates = [
        item
        for item in whole_page
        if re.sub(r"[^a-z0-9]", "", str(item.format).lower()) == "ean13"
        and item.text is not None
        and ean13_checksum_valid(item.text)
    ]
    for detection in eligible:
        if any(quad_overlap(detection.quad, item.quad) >= 0.28 for item in whole_page):
            explained += 1
            continue

        direct = legacy_local_linear_decode(image, detection, linear_formats)
        if direct is not None:
            decoded.append(direct)
            continue

        result = decode_candidate(image, detection, None, formats=linear_formats)
        if not result.decoded:
            advanced_attempted += 1
            result = decode_advanced_linear(image, detection, result, formats=linear_formats)
        converted = _convert_hybrid_result(result)
        if converted.decoded:
            decoded.append(converted)
            continue
        if color_image is not None:
            color_attempted += 1
            color_result = decode_color_linear_candidate(
                color_image,
                detection,
                linear_formats,
            )
            if color_result is not None:
                decoded.append(color_result)
                color_recovered += 1
                continue
        dense_candidate_review = bool(
            dense_review_enabled
            and dense_layout_candidate_review_allowed(detection)
        )
        if dense_candidate_review and ean13_templates:
            template_matches: list[tuple[float, Result, dict[str, float]]] = []
            for template in ean13_templates:
                evidence_options = [
                    repeated_ean13_template_evidence(
                        source,
                        detection.quad,
                        template.text or "",
                        rectify_quad,
                    )
                    for source in (
                        [color_image, image]
                        if color_image is not None
                        else [image]
                    )
                ]
                valid_evidence = [item for item in evidence_options if item is not None]
                evidence = max(
                    valid_evidence,
                    key=lambda item: (
                        item["ean13_template_correlation"]
                        + item["ean13_template_margin"]
                    ),
                    default=None,
                )
                if evidence is not None:
                    template_matches.append(
                        (float(evidence["ean13_template_correlation"]), template, evidence)
                    )
            template_matches.sort(key=lambda item: item[0], reverse=True)
            if template_matches and (
                len(template_matches) == 1
                or template_matches[0][0] - template_matches[1][0] >= 0.02
            ):
                correlation, template, evidence = template_matches[0]
                decoded.append(
                    Result(
                        quad=np.asarray(detection.quad, dtype=np.float32),
                        text=template.text,
                        format=template.format,
                        status="decoded",
                        confidence=min(
                            0.97,
                            0.70
                            + 0.25 * correlation
                            + 0.50 * float(evidence["ean13_template_margin"]),
                        ),
                        sources=set(detection.sources)
                        | {
                            "deterministic:ean13-repeated-template",
                            "context:dense-barcode-layout",
                        },
                        attempts=["deterministic:ean13-template-fit"],
                        evidence={
                            "structural_metrics": dict(detection.metrics),
                            "template_anchor_source": "independent-zxing-page-read",
                            **evidence,
                        },
                    )
                )
                repeated_ean13_recovered += 1
                continue
        if not detection.accepted and not dense_candidate_review:
            rejected_after_decode += 1
            continue
        metrics = dict(result.structural_metrics)
        if (
            float(metrics.get("dark_fraction", 0.0)) < 0.08
            or float(metrics.get("transition_rate", 0.0)) < 0.03
        ) and not dense_candidate_review:
            rejected_by_physics += 1
            continue
        if config.include_review_candidates:
            if dense_candidate_review:
                converted.sources.add("context:dense-barcode-layout")
                converted.evidence["dense_layout_candidate_review"] = True
                if ean13_templates:
                    long_side, _short_side = detection.long_short()
                    upper_bound = float(long_side) / 95.0
                    converted.evidence["resolution_diagnostics"] = {
                        "candidate_pixels_per_ean13_module_upper_bound": round(upper_bound, 4),
                        "likely_undersampled": bool(upper_bound < 1.25),
                        "reason": (
                            "candidate width is too close to the 95-module EAN-13 minimum "
                            "for reliable independent payload recovery"
                        ),
                    }
            review.append(converted)

    decoded = [
        item
        for item in merge_results(decoded)
        # ITF has no required checksum and extremely short reads are a common
        # false positive on document text. Industrial ITF payloads shorter
        # than six digits are outside this application's supported profile.
        if item.format != "Itf" or len(item.text or "") >= 6
    ]
    review = [item for item in review if not any(quad_overlap(item.quad, dec.quad) >= 0.70 for dec in decoded)]
    return decoded, review, {
        "raw_visual_proposals": len(proposals),
        "eligible_visual_proposals": len(eligible),
        "locator_rejected": len(rejected) + rejected_after_decode,
        "linear_whole_page_decoded": len(whole_page),
        "linear_decoded": len(decoded),
        "linear_review": len(review),
        "linear_rejected_by_physics": rejected_by_physics,
        "linear_proposals_explained_by_anchor": explained,
        "advanced_linear_candidates": advanced_attempted,
        "color_linear_candidates": color_attempted,
        "color_linear_recovered": color_recovered,
        "repeated_ean13_template_recovered": repeated_ean13_recovered,
        "dense_layout_short_review_enabled": dense_review_enabled,
        "dense_split_proposals": sum(
            any(source.startswith("geometry:dense-linear-split:") for source in detection.sources)
            for detection in proposals
        ),
        "detector_mode": config.linear_detector,
        "detector_calls": len(_linear_detector_configs(config.linear_detector)),
}


def decode_color_linear_candidate(
    color_image: np.ndarray,
    detection: LocatorDetection,
    linear_formats: Any,
) -> Result | None:
    """Retry a visually valid candidate using chromatic, not luminance, contrast."""
    for view_name, view in color_linear_decode_views(
        color_image,
        detection.quad,
        rectify_quad,
    ):
        for scale in (2.0, 3.0):
            work = cv2.resize(
                view,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            quiet = max(12, int(round(work.shape[0] * 0.35)))
            work = cv2.copyMakeBorder(
                work,
                quiet,
                quiet,
                quiet,
                quiet,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            for binarizer in (
                zxingcpp.Binarizer.LocalAverage,
                zxingcpp.Binarizer.GlobalHistogram,
            ):
                reads = safe_zxing_read(
                    work,
                    linear_formats,
                    binarizer,
                    try_downscale=False,
                    return_errors=False,
                )
                for barcode in reads:
                    if not barcode.valid or not barcode.text:
                        continue
                    return Result(
                        quad=np.asarray(detection.quad, dtype=np.float32),
                        text=str(barcode.text),
                        format=normalize_format(str(barcode.format)),
                        status="decoded",
                        confidence=1.0,
                        sources=set(detection.sources)
                        | {f"zxing:color-contrast:{view_name}:scale-{scale:g}"},
                        attempts=[f"color-contrast:{view_name}:scale-{scale:g}:{binarizer}"],
                        evidence={"structural_metrics": dict(detection.metrics)},
                    )
    return None


def process_page(
    page_number: int,
    page_path: Path,
    source_mode: str,
    config: Config,
    work: Path,
    matrix_formats: Any | None = None,
    linear_formats: Any | None = None,
    stage_parallel: bool = True,
) -> dict[str, Any]:
    if matrix_formats is None and linear_formats is None:
        matrix_formats, linear_formats = categorize_formats(config.barcode_formats)

    page_started = perf_counter()
    started = perf_counter()
    color_image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if color_image is None:
        raise RuntimeError(f"unable to read {page_path}")
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    timings: dict[str, float] = {"load_seconds": perf_counter() - started}
    quality_diagnostics = input_quality_diagnostics(image)

    reads, proposals, fast_seconds = fast_stage(
        image,
        config,
        matrix_formats,
        linear_formats,
        stage_parallel,
        allow_linear_upscale=source_mode == "web-upload-image",
    )
    timings["parallel_fast_stage_seconds"] = fast_seconds
    decoded: list[Result] = []
    unresolved: list[Result] = []
    review: list[Result] = []
    diagnostics: dict[str, Any] = {}

    if matrix_formats:
        matrix_decoded, invalid_anchors, matrix_diagnostics = parse_matrix_reads(reads)
        decoded.extend(matrix_decoded)
        started = perf_counter()
        qr_formats_active = any(
            str(value) in {"QR Code", "All Matrix", "All Readable"}
            for value in matrix_formats
        )
        qr_geometry_anchors: list[MatrixAnchor] = []
        if source_mode == "web-upload-image" and qr_formats_active:
            covered = [np.asarray(item.quad, dtype=np.float32) for item in [*decoded, *invalid_anchors]]
            for quad, scale in opencv_qr_geometry_proposals(image, covered):
                qr_geometry_anchors.append(
                    MatrixAnchor(
                        quad=quad,
                        format="QR Code",
                        error="opencv-qr-geometry",
                        sources={f"opencv:qr-geometry:scale-{scale:g}"},
                        attempts=[f"opencv:qr-detect-multi:scale-{scale:g}"],
                        confidence=0.90,
                        evidence={"geometry_only": True, "proposal_scale": scale},
                    )
                )
            invalid_anchors = merge_anchors([*invalid_anchors, *qr_geometry_anchors])
        recovered = 0
        invalid_qr_discarded = 0
        strict_matrix_accepted = 0
        strict_matrix_rejected = 0
        strict_matrix_inferred = 0
        for anchor in invalid_anchors:
            result = recover_matrix_anchor(
                image,
                anchor,
                matrix_formats,
                adaptive_generalization=config.adaptive_generalization,
            )
            if result is not None:
                decoded.append(result)
                recovered += 1
                continue
            unresolved_result = unresolved_matrix_from_anchor(
                image,
                anchor,
                keep_weak_qr=config.keep_unresolved_qr,
                strict_matrix_acceptance=config.strict_matrix_acceptance,
            )
            if unresolved_result is not None:
                unresolved.append(unresolved_result)
                if config.strict_matrix_acceptance:
                    strict_matrix_accepted += 1
                    if _matrix_format_kind(anchor.format) is None:
                        strict_matrix_inferred += 1
            elif anchor.format == "QR Code":
                invalid_qr_discarded += 1
                if config.strict_matrix_acceptance:
                    strict_matrix_rejected += 1
            elif config.strict_matrix_acceptance and anchor.evidence.get("_strict_matrix_gate"):
                strict_matrix_rejected += 1

        residual_count = residual_decoded = residual_unresolved = 0
        residual_effective = bool(
            config.residual_proposals
            and source_mode == "web-upload-image"
            and image.size <= 4_000_000
            and max(image.shape[:2]) <= 1800
        )
        if residual_effective:
            covered = [np.asarray(item.quad, dtype=np.float32) for item in [*decoded, *unresolved]]
            residual = visual_matrix_proposals(image, covered)
            residual_count = len(residual)
            for anchor in residual:
                result = recover_matrix_anchor(
                    image,
                    anchor,
                    matrix_formats,
                    adaptive_generalization=config.adaptive_generalization,
                )
                if result is not None:
                    decoded.append(result)
                    residual_decoded += 1
                # A generic bidirectional texture proposal is a recovery
                # target, not proof of a matrix symbol. Only a checksum/error
                # anchor produced by a format-aware decoder may survive as an
                # unresolved result.
        timings["matrix_recovery_seconds"] = perf_counter() - started
        diagnostics["matrix"] = {
            **matrix_diagnostics,
            "error_guided_recovered": recovered,
            "invalid_qr_not_promoted": invalid_qr_discarded,
            "strict_matrix_acceptance": config.strict_matrix_acceptance,
            "strict_matrix_accepted": strict_matrix_accepted,
            "strict_matrix_rejected": strict_matrix_rejected,
            "strict_matrix_format_inferred": strict_matrix_inferred,
            "opencv_qr_geometry_proposals": len(qr_geometry_anchors),
            "localized_unresolved": len(unresolved),
            "residual_proposals_enabled": config.residual_proposals,
            "residual_proposals_effective": residual_effective,
            "residual_proposals": residual_count,
            "residual_decoded": residual_decoded,
            "residual_unresolved": residual_unresolved,
            "whole_page_scales": unique_scales(
                scale
                for key, values in reads.items()
                if key.startswith("matrix-")
                for item in values
                for scale in [item[1] if isinstance(item, tuple) else 1.0]
            ),
        }

    if linear_formats:
        started = perf_counter()
        linear_whole = parse_linear_reads(reads.get("linear-zxing", []))
        linear_decoded, linear_review, linear_diagnostics = linear_paths(
            image,
            color_image,
            linear_whole,
            proposals,
            config,
            linear_formats,
        )
        timings["linear_verify_decode_seconds"] = perf_counter() - started
        decoded.extend(linear_decoded)
        review.extend(linear_review)
        diagnostics["linear"] = linear_diagnostics

    decoded = merge_results(decoded)
    unresolved = [item for item in merge_results(unresolved) if not any(quad_overlap(item.quad, dec.quad) >= 0.28 for dec in decoded)]
    review = [item for item in review if not any(quad_overlap(item.quad, dec.quad) >= 0.70 for dec in decoded)]
    decoded.sort(key=lambda item: (round(float(item.center()[1]) / 20), float(item.center()[0])))
    unresolved.sort(key=lambda item: (float(item.center()[1]), float(item.center()[0])))
    review.sort(key=lambda item: (float(item.center()[1]), float(item.center()[0])))

    started = perf_counter()
    if config.save_crops:
        for category, values in (("decoded", decoded), ("unresolved", unresolved), ("review", review)):
            for index, result in enumerate(values, start=1):
                write_crop(image, result, work, page_number, category, index)
    overlay_relative: str | None = None
    if config.save_overlays:
        overlay_path = Path("overlays") / f"page-{page_number:04d}.png"
        (work / overlay_path).parent.mkdir(parents=True, exist_ok=True)
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(work / overlay_path), annotate(color, decoded, unresolved, review))
        overlay_relative = overlay_path.as_posix()
    timings["artifact_seconds"] = perf_counter() - started
    timings["total_seconds"] = perf_counter() - page_started

    payload: dict[str, Any] = {
        "page": page_number,
        "source_image": page_path.relative_to(work).as_posix(),
        "source_mode": source_mode,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "decoded_count": len(decoded),
        "localized_unresolved_matrix_count": len(unresolved),
        "review_candidate_count": len(review),
        "results": [item.to_json() for item in decoded],
        "unresolved_matrix": [item.to_json() for item in unresolved],
        "review_candidates": [item.to_json() for item in review],
        "diagnostics": {
            "input_quality": quality_diagnostics,
            "adaptive_generalization": {
                "enabled": config.adaptive_generalization,
                "proposal_scales": [plan.scale for plan in proposal_scale_plan(image.shape, config.adaptive_generalization)],
                "minimum_linear_length_legacy_px": config.minimum_linear_length,
                "residual_matrix_proposals_enabled": config.residual_proposals,
            },
            **diagnostics,
        },
        "timings": {key: round(float(value), 4) for key, value in timings.items()},
    }
    if overlay_relative is not None:
        payload["overlay"] = overlay_relative
    return payload


def validate_run(work: Path, pages: Sequence[dict[str, Any]], config: Config) -> dict[str, int]:
    missing: list[str] = []
    expected_crops: set[str] = set()
    for page in pages:
        if not (work / page["source_image"]).is_file():
            missing.append(page["source_image"])
        if config.save_overlays and not (work / page["overlay"]).is_file():
            missing.append(page["overlay"])
        for key in ("results", "unresolved_matrix", "review_candidates"):
            for result in page[key]:
                crop = result.get("crop")
                if crop:
                    expected_crops.add(crop)
                    if not (work / crop).is_file():
                        missing.append(crop)
    actual_crops = {path.relative_to(work).as_posix() for path in (work / "crops").rglob("*.png")} if (work / "crops").exists() else set()
    orphan = actual_crops - expected_crops
    if missing or orphan or (config.save_crops and actual_crops != expected_crops):
        raise RuntimeError(f"run integrity failure: missing={missing[:8]} orphan={sorted(orphan)[:8]}")
    return {"expected_crops": len(expected_crops), "actual_crops": len(actual_crops), "orphan_crops": 0}


def _resolved_workers(requested: int, page_count: int) -> int:
    if requested > 0:
        return max(1, min(requested, page_count))
    return max(1, min(8, os.cpu_count() or 1, page_count))


def run_pipeline(input_path: Path, config: Config) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output = config.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    work, run_id = prepare_work_directory(output)
    run_started = perf_counter()
    started_at = utc_now()

    matrix_formats, linear_formats = categorize_formats(config.barcode_formats)

    try:
        extraction_started = perf_counter()
        sources = collect_pages_extended(input_path, config, work)
        extraction_seconds = perf_counter() - extraction_started
        workers = _resolved_workers(config.workers, len(sources))
        stage_parallel = workers == 1

        arguments = [
            (page, path, mode, config, work, matrix_formats, linear_formats, stage_parallel)
            for page, path, mode in sources
        ]
        if workers == 1:
            pages = [process_page(*values) for values in arguments]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="barcode-page") as executor:
                futures = [executor.submit(process_page, *values) for values in arguments]
                pages = [future.result() for future in futures]
        pages.sort(key=lambda item: item["page"])
        task_timing_note = (
            "overlaps other page tasks"
            if workers > 1
            else "page tasks are sequential; internal detector scans may overlap"
        )
        for page in pages:
            print(
                f"page {page['page']:04d}: {page['decoded_count']} decoded, "
                f"{page['localized_unresolved_matrix_count']} unresolved, "
                f"{page['review_candidate_count']} review, "
                f"task={page['timings']['total_seconds']:.3f}s ({task_timing_note})",
                flush=True,
            )

        integrity = validate_run(work, pages, config)
        processing_sum = sum(page["timings"]["total_seconds"] for page in pages)
        wall_seconds = perf_counter() - run_started
        formats_label = str(config.barcode_formats) if list(config.barcode_formats or []) else "all"
        payload = {
            "schema_version": 3,
            "run": {
                "id": run_id,
                "started_at": started_at,
                "completed_at": utc_now(),
                "input": str(input_path),
                "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
                "pipeline_sha256": sha256_file(Path(__file__)),
                "output_root": ".",
            },
            "method": "parallel native-raster ZXing + fused proposal-only OpenCV + targeted recovery",
            "configuration": {
                "formats": formats_label,
                "matrix_formats_active": bool(matrix_formats),
                "linear_formats_active": bool(linear_formats),
                "render_dpi": config.render_dpi,
                "pages": config.pages,
                "workers_requested": config.workers,
                "workers_used": workers,
                "linear_detector": config.linear_detector,
                "batch_pdf_extraction": config.batch_pdf_extraction,
                "save_crops": config.save_crops,
                "save_overlays": config.save_overlays,
                "residual_proposals": config.residual_proposals,
                "keep_unresolved_qr": config.keep_unresolved_qr,
                "strict_matrix_acceptance": config.strict_matrix_acceptance,
                "include_review_candidates": config.include_review_candidates,
                "minimum_linear_score": config.minimum_linear_score,
                "minimum_linear_length": config.minimum_linear_length,
                "adaptive_generalization": config.adaptive_generalization,
            },
            "summary": {
                "pages": len(pages),
                "decoded": sum(page["decoded_count"] for page in pages),
                "localized_unresolved_matrix": sum(page["localized_unresolved_matrix_count"] for page in pages),
                "review_candidates": sum(page["review_candidate_count"] for page in pages),
                "extraction_seconds": round(extraction_seconds, 4),
                "page_task_elapsed_sum_seconds": round(processing_sum, 4),
                "mean_page_task_elapsed_seconds": round(processing_sum / max(1, len(pages)), 4),
                "processing_cpu_wall_sum_seconds": round(processing_sum, 4),
                "wall_seconds": round(wall_seconds, 4),
                "effective_wall_seconds_per_page": round(wall_seconds / max(1, len(pages)), 4),
                "throughput_pages_per_second": round(len(pages) / max(wall_seconds, 1e-9), 3),
            },
            "integrity": integrity,
            "pages": pages,
        }
        (work / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        os.replace(work, output)
        print(
            f"summary: {len(pages)} pages completed in {wall_seconds:.3f}s wall time "
            f"with {workers} worker(s); throughput-equivalent "
            f"{1000.0 * wall_seconds / max(1, len(pages)):.1f} ms/page; "
            f"{'overlapping ' if workers > 1 else ''}page-task sum {processing_sum:.3f}s",
            flush=True,
        )
        print(f"wrote {output / 'detections.json'}")
        return payload
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
