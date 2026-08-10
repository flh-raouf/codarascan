#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Format-aware, 2D-first barcode localization and decoding pipeline.

The default path is deliberately small: two checksum-protected matrix reads at
native page resolution with ZXing's scale search enabled, followed by bounded
recovery around decoder-reported invalid positions. The 1D locator is disabled
unless the caller explicitly selects ``--formats 1d`` or ``--formats all``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import zxingcpp

from codarascan.engines.common.recovery.hybrid import (  # noqa: E402
    PipelineConfig,
    collect_pages,
    decode_candidate,
    normalize_format,
    parse_page_selection,
)
from codarascan.engines.common.recovery.locator import (
    locate_page,
    order_quad,
    quad_overlap,
    rectify_quad,
)
from codarascan.engines.common.recovery.zxing import decode_advanced_linear  # noqa: E402

MATRIX_FORMATS = zxingcpp.barcode_formats_from_str("DataMatrix,QRCode")
LINEAR_FORMATS = zxingcpp.barcode_formats_from_str("Code39,Code128")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_matrix_format(name: str | None) -> bool:
    value = (name or "").replace("_", " ").lower()
    return any(
        token in value
        for token in (
            "data matrix",
            "datamatrix",
            "qr code",
            "qrcode",
            "rmqr",
            "aztec",
            "pdf417",
            "maxicode",
        )
    )


def barcode_quad(barcode: Any, scale: float = 1.0, offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    position = barcode.position
    ox, oy = offset
    return np.asarray(
        [
            [position.top_left.x / scale + ox, position.top_left.y / scale + oy],
            [position.top_right.x / scale + ox, position.top_right.y / scale + oy],
            [position.bottom_right.x / scale + ox, position.bottom_right.y / scale + oy],
            [position.bottom_left.x / scale + ox, position.bottom_left.y / scale + oy],
        ],
        dtype=np.float32,
    )


def safe_zxing_read(
    image: np.ndarray,
    formats: Any,
    binarizer: Any,
    *,
    try_downscale: bool,
    return_errors: bool,
) -> list[Any]:
    try:
        return list(
            zxingcpp.read_barcodes(
                image,
                formats=formats,
                try_rotate=True,
                try_downscale=try_downscale,
                try_invert=True,
                binarizer=binarizer,
                return_errors=return_errors,
            )
        )
    except Exception:
        return []


@dataclass
class MatrixAnchor:
    quad: np.ndarray
    format: str | None
    error: str
    sources: set[str] = field(default_factory=set)
    attempts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def center(self) -> np.ndarray:
        return np.asarray(self.quad, dtype=np.float32).mean(axis=0)


@dataclass
class Result:
    quad: np.ndarray
    text: str | None
    format: str | None
    status: str
    confidence: float
    raw_bytes: bytes | None = None
    sources: set[str] = field(default_factory=set)
    attempts: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    crop: str | None = None

    @property
    def decoded(self) -> bool:
        return bool(self.text and self.format and self.status == "decoded")

    @property
    def kind(self) -> str:
        return "matrix" if is_matrix_format(self.format) or self.status == "localized_unresolved_matrix" else "linear"

    def center(self) -> np.ndarray:
        return np.asarray(self.quad, dtype=np.float32).mean(axis=0)

    def to_json(self) -> dict[str, Any]:
        quad = np.asarray(self.quad, dtype=np.float32)
        minimum = quad.min(axis=0)
        maximum = quad.max(axis=0)
        payload = {
            "status": self.status,
            "kind": self.kind,
            "decoded": self.decoded,
            "text": self.text,
            "format": self.format,
            "confidence": round(float(self.confidence), 4),
            "sources": sorted(self.sources),
            "attempts": self.attempts,
            "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in quad],
            "aabb": {
                "x": round(float(minimum[0]), 2),
                "y": round(float(minimum[1]), 2),
                "width": round(float(maximum[0] - minimum[0]), 2),
                "height": round(float(maximum[1] - minimum[1]), 2),
            },
            "evidence": self.evidence,
        }
        if self.raw_bytes is not None:
            payload["raw_bytes_base64"] = base64.b64encode(self.raw_bytes).decode("ascii")
        if self.crop:
            payload["crop"] = self.crop
        return payload


@dataclass
class Config:
    output: Path
    formats: str = "2d"
    render_dpi: int = 300
    pages: list[int] | None = None
    overwrite: bool = False
    save_crops: bool = True
    residual_proposals: bool = False
    include_review_candidates: bool = True
    minimum_linear_score: float = 0.52
    minimum_linear_length: float = 100.0


def same_result(a: Result, b: Result) -> bool:
    overlap = quad_overlap(np.asarray(a.quad, np.float32), np.asarray(b.quad, np.float32))
    if a.decoded and b.decoded and a.text == b.text and a.format == b.format:
        return overlap >= 0.08 or float(np.linalg.norm(a.center() - b.center())) <= 40.0
    return overlap >= 0.66 and (not a.decoded or not b.decoded or a.text == b.text)


def merge_results(results: Iterable[Result]) -> list[Result]:
    merged: list[Result] = []
    for candidate in sorted(results, key=lambda item: (item.decoded, item.confidence), reverse=True):
        existing = next((item for item in merged if same_result(item, candidate)), None)
        if existing is None:
            merged.append(candidate)
            continue
        existing.sources.update(candidate.sources)
        existing.attempts.extend(value for value in candidate.attempts if value not in existing.attempts)
        if candidate.decoded and not existing.decoded:
            existing.text = candidate.text
            existing.format = candidate.format
            existing.status = candidate.status
            existing.confidence = candidate.confidence
            existing.quad = candidate.quad
    return merged


def same_anchor(a: MatrixAnchor, b: MatrixAnchor) -> bool:
    if quad_overlap(a.quad, b.quad) >= 0.38:
        return True
    long_a = max(cv2.minAreaRect(np.asarray(a.quad, np.float32))[1])
    long_b = max(cv2.minAreaRect(np.asarray(b.quad, np.float32))[1])
    return float(np.linalg.norm(a.center() - b.center())) <= 0.45 * max(20.0, min(long_a, long_b))


def merge_anchors(anchors: Iterable[MatrixAnchor]) -> list[MatrixAnchor]:
    merged: list[MatrixAnchor] = []
    for candidate in anchors:
        existing = next((item for item in merged if same_anchor(item, candidate)), None)
        if existing is None:
            merged.append(candidate)
        else:
            existing.sources.update(candidate.sources)
            existing.attempts.extend(value for value in candidate.attempts if value not in existing.attempts)
            if len(candidate.error) > len(existing.error):
                existing.error = candidate.error
            if candidate.confidence > existing.confidence:
                existing.confidence = candidate.confidence
                existing.evidence = candidate.evidence
    return merged


def matrix_fast_pass(image: np.ndarray) -> tuple[list[Result], list[MatrixAnchor], dict[str, Any]]:
    decoded: list[Result] = []
    invalid: list[MatrixAnchor] = []
    diagnostics: dict[str, Any] = {"reads": {}, "invalid_errors": {}}
    recipes = (
        ("local-average", zxingcpp.Binarizer.LocalAverage),
        ("global-histogram", zxingcpp.Binarizer.GlobalHistogram),
    )
    for name, binarizer in recipes:
        reads = safe_zxing_read(
            image,
            MATRIX_FORMATS,
            binarizer,
            try_downscale=True,
            return_errors=True,
        )
        diagnostics["reads"][name] = len(reads)
        for barcode in reads:
            quad = barcode_quad(barcode)
            if abs(float(cv2.contourArea(quad))) < 9.0:
                continue
            if barcode.valid and barcode.text:
                decoded.append(
                    Result(
                        quad=quad,
                        text=str(barcode.text),
                        format=normalize_format(str(barcode.format)),
                        status="decoded",
                        confidence=1.0,
                        sources={f"zxing:matrix-whole-page:{name}:downscale"},
                        attempts=[f"whole-page:{name}:downscale"],
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
                        sources={f"zxing:invalid-matrix:{name}:downscale"},
                        attempts=[f"whole-page:{name}:downscale:return-errors"],
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


def bounded_crop(image: np.ndarray, center: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    size = min(size, width, height)
    x1 = int(round(float(center[0]) - size / 2.0))
    y1 = int(round(float(center[1]) - size / 2.0))
    x1 = min(max(0, x1), max(0, width - size))
    y1 = min(max(0, y1), max(0, height - size))
    return image[y1 : y1 + size, x1 : x1 + size], x1, y1


def recover_invalid_anchor(image: np.ndarray, anchor: MatrixAnchor) -> Result | None:
    recipes = (
        (500, 1.5, cv2.INTER_CUBIC, "context500-cubic15"),
        (300, 2.0, cv2.INTER_CUBIC, "context300-cubic20"),
        (300, 2.0, cv2.INTER_NEAREST, "context300-nearest20"),
    )
    binarizers = (
        ("local-average", zxingcpp.Binarizer.LocalAverage),
        ("global-histogram", zxingcpp.Binarizer.GlobalHistogram),
    )
    for crop_size, scale, interpolation, recipe_name in recipes:
        crop, offset_x, offset_y = bounded_crop(image, anchor.center(), crop_size)
        variant = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interpolation)
        for binarizer_name, binarizer in binarizers:
            attempt = f"invalid-anchor:{recipe_name}:{binarizer_name}:downscale"
            reads = safe_zxing_read(
                variant,
                MATRIX_FORMATS,
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
            if distance > max(140.0, crop_size * 0.42):
                continue
            return Result(
                quad=quad,
                text=str(barcode.text),
                format=normalize_format(str(barcode.format)),
                status="decoded",
                confidence=1.0,
                sources=set(anchor.sources) | {"zxing:matrix-error-guided-recovery"},
                attempts=anchor.attempts + [attempt],
                evidence={"prior_error": anchor.error, **anchor.evidence},
            )
    return None


def visual_matrix_proposals(
    image: np.ndarray,
    covered_quads: Sequence[np.ndarray],
    limit: int = 12,
) -> list[MatrixAnchor]:
    """Return conservative residual 2D proposals outside known symbols.

    The work image is bounded to 1600 pixels, proposals are ranked by balanced
    two-directional energy and square support, and a small spatial quota keeps a
    dense form region from consuming the complete budget.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    # Large document pages stay bounded for speed. Small web images are
    # enlarged before the energy map so a QR/Data Matrix module is not forced
    # to compete as a sub-pixel feature. Coordinates are mapped back below.
    scale = min(1.0, 1600.0 / longest)
    if longest <= 900 and height * width <= 900_000:
        scale = min(3.0, max(1.5, 1200.0 / max(1, longest)))
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
    else:
        work = gray
    gx = np.abs(cv2.Scharr(work, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Scharr(work, cv2.CV_32F, 0, 1))
    raw: list[MatrixAnchor] = []
    for local_size in (9, 17, 29):
        energy_x = cv2.boxFilter(gx, cv2.CV_32F, (local_size, local_size), normalize=True)
        energy_y = cv2.boxFilter(gy, cv2.CV_32F, (local_size, local_size), normalize=True)
        response = np.sqrt(energy_x * energy_y)
        threshold = float(np.percentile(response, 99.55))
        normalizer = max(1e-6, float(np.percentile(response, 99.90)))
        mask = (response >= threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
            if not (14 <= candidate_width <= 360 and 14 <= candidate_height <= 360):
                continue
            aspect = candidate_width / max(1.0, float(candidate_height))
            if not 0.40 <= aspect <= 2.50:
                continue
            region_x = energy_x[y : y + candidate_height, x : x + candidate_width]
            region_y = energy_y[y : y + candidate_height, x : x + candidate_width]
            mean_x = float(np.mean(region_x))
            mean_y = float(np.mean(region_y))
            balance = min(mean_x, mean_y) / max(1e-6, max(mean_x, mean_y))
            square_score = max(0.0, 1.0 - abs(np.log(aspect)) / np.log(2.5))
            area_score = min(1.0, np.sqrt(candidate_width * candidate_height) / 95.0)
            response_score = min(1.0, float(np.mean(response[y : y + candidate_height, x : x + candidate_width])) / normalizer)
            confidence = 0.34 * balance + 0.28 * square_score + 0.20 * area_score + 0.18 * response_score
            if confidence < 0.48:
                continue
            quad = np.asarray(
                [[x, y], [x + candidate_width, y], [x + candidate_width, y + candidate_height], [x, y + candidate_height]],
                dtype=np.float32,
            ) / scale
            if any(quad_overlap(quad, covered) >= 0.20 for covered in covered_quads):
                continue
            raw.append(
                MatrixAnchor(
                    quad=quad,
                    format=None,
                    error="visual-matrix-proposal",
                    sources={f"visual:matrix-energy:l{local_size}"},
                    attempts=[f"visual-proposal:l{local_size}"],
                    confidence=float(confidence),
                    evidence={
                        "proposal_score": round(float(confidence), 4),
                        "edge_balance": round(float(balance), 4),
                        "square_score": round(float(square_score), 4),
                        "area_score": round(float(area_score), 4),
                    },
                )
            )

    ranked = sorted(raw, key=lambda item: item.confidence, reverse=True)
    kept: list[MatrixAnchor] = []
    tile_counts: dict[tuple[int, int], int] = {}
    for candidate in ranked:
        center = candidate.center()
        tile = (min(2, int(center[0] / max(1.0, width / 3.0))), min(3, int(center[1] / max(1.0, height / 4.0))))
        if tile_counts.get(tile, 0) >= 2:
            continue
        if any(same_anchor(candidate, prior) for prior in kept):
            continue
        kept.append(candidate)
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
        if len(kept) >= limit:
            break
    return kept


def matrix_structure_metrics(image: np.ndarray, quad: np.ndarray) -> dict[str, float]:
    ordered = order_quad(np.asarray(quad, dtype=np.float32))
    sides = [float(np.linalg.norm(ordered[(index + 1) % 4] - ordered[index])) for index in range(4)]
    width = max(24, int(round((sides[0] + sides[2]) / 2.0)))
    height = max(24, int(round((sides[1] + sides[3]) / 2.0)))
    destination = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(ordered, destination)
    patch = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
    gx = float(np.mean(np.abs(cv2.Scharr(gray, cv2.CV_32F, 1, 0))))
    gy = float(np.mean(np.abs(cv2.Scharr(gray, cv2.CV_32F, 0, 1))))
    balance = min(gx, gy) / max(1e-6, max(gx, gy))
    dark_fraction = float(np.mean(binary))
    aspect = width / max(1.0, float(height))
    square_score = max(0.0, 1.0 - abs(np.log(max(1e-6, aspect))) / np.log(2.2))
    row_transitions = []
    column_transitions = []
    for position in np.linspace(0.15, 0.85, 7):
        row = binary[min(height - 1, int(round(position * (height - 1))))]
        column = binary[:, min(width - 1, int(round(position * (width - 1))))]
        row_transitions.append(int(np.count_nonzero(np.diff(row.astype(np.int8)))))
        column_transitions.append(int(np.count_nonzero(np.diff(column.astype(np.int8)))))
    transition_score = min(1.0, min(float(np.median(row_transitions)), float(np.median(column_transitions))) / 10.0)
    occupancy_score = max(0.0, 1.0 - abs(dark_fraction - 0.45) / 0.45)
    score = 0.30 * balance + 0.25 * square_score + 0.30 * transition_score + 0.15 * occupancy_score
    return {
        "score": round(float(score), 4),
        "edge_balance": round(float(balance), 4),
        "dark_fraction": round(float(dark_fraction), 4),
        "square_score": round(float(square_score), 4),
        "transition_score": round(float(transition_score), 4),
        "width": float(width),
        "height": float(height),
    }


def unresolved_from_anchor(image: np.ndarray, anchor: MatrixAnchor) -> Result | None:
    metrics = matrix_structure_metrics(image, anchor.quad)
    minimum_score = 0.72 if "visual-matrix-proposal" in anchor.error else 0.58
    if metrics["score"] < minimum_score or not (0.10 <= metrics["dark_fraction"] <= 0.82):
        return None
    return Result(
        quad=np.asarray(anchor.quad, dtype=np.float32),
        text=None,
        format=anchor.format,
        status="localized_unresolved_matrix",
        confidence=float(metrics["score"]),
        sources=set(anchor.sources) | {"verification:matrix-structure"},
        attempts=list(anchor.attempts),
        evidence={"decoder_error": anchor.error, "structure": metrics, **anchor.evidence},
    )


def linear_fast_pass(image: np.ndarray) -> list[Result]:
    output: list[Result] = []
    reads = safe_zxing_read(
        image,
        LINEAR_FORMATS,
        zxingcpp.Binarizer.LocalAverage,
        try_downscale=False,
        return_errors=False,
    )
    for barcode in reads:
        if not barcode.valid or not barcode.text:
            continue
        output.append(
            Result(
                quad=barcode_quad(barcode),
                text=str(barcode.text),
                format=normalize_format(str(barcode.format)),
                status="decoded",
                confidence=1.0,
                sources={"zxing:linear-whole-page"},
                attempts=["whole-page:linear:local-average"],
            )
        )
    return merge_results(output)


def linear_paths(image: np.ndarray, config: Config) -> tuple[list[Result], list[Result], dict[str, Any]]:
    whole_page = linear_fast_pass(image)
    located, rejected = locate_page(
        image,
        angle_step=0,
        minimum_score=config.minimum_linear_score,
        use_tensor_fallback=False,
        allow_tensor_only=False,
        minimum_linear_length=config.minimum_linear_length,
    )
    decoded: list[Result] = list(whole_page)
    review: list[Result] = []
    rejected_by_physics = 0
    proposals_explained_by_anchor = 0
    for detection in located:
        if detection.kind != "linear":
            continue
        if any(quad_overlap(np.asarray(detection.quad, np.float32), item.quad) >= 0.28 for item in whole_page):
            proposals_explained_by_anchor += 1
            continue
        result = decode_candidate(image, detection, None)
        if not result.decoded:
            result = decode_advanced_linear(image, detection, result)
        if result.decoded:
            decoded.append(
                Result(
                    quad=np.asarray(result.quad, dtype=np.float32),
                    text=result.text,
                    format=result.format,
                    status="decoded",
                    confidence=float(result.confidence),
                    sources=set(result.sources),
                    attempts=list(result.attempts),
                    evidence={"structural_metrics": result.structural_metrics},
                )
            )
            continue
        metrics = dict(result.structural_metrics)
        dark_fraction = float(metrics.get("dark_fraction", 0.0))
        transition_rate = float(metrics.get("transition_rate", 0.0))
        if dark_fraction < 0.08 or transition_rate < 0.03:
            rejected_by_physics += 1
            continue
        if config.include_review_candidates:
            review.append(
                Result(
                    quad=np.asarray(result.quad, dtype=np.float32),
                    text=None,
                    format=None,
                    status="review_candidate",
                    confidence=float(result.confidence),
                    sources=set(result.sources),
                    attempts=list(result.attempts),
                    evidence={"structural_metrics": metrics},
                )
            )
    diagnostics = {
        "located_total": len(located),
        "linear_located": sum(item.kind == "linear" for item in located),
        "locator_rejected": len(rejected),
        "linear_whole_page_decoded": len(whole_page),
        "linear_decoded": len(decoded),
        "linear_review": len(review),
        "linear_rejected_by_physics": rejected_by_physics,
        "linear_proposals_explained_by_anchor": proposals_explained_by_anchor,
    }
    decoded = merge_results(decoded)
    review = [item for item in review if not any(quad_overlap(item.quad, dec.quad) >= 0.70 for dec in decoded)]
    diagnostics["linear_decoded"] = len(decoded)
    diagnostics["linear_review"] = len(review)
    return decoded, review, diagnostics


def annotate(image: np.ndarray, decoded: Sequence[Result], unresolved: Sequence[Result], review: Sequence[Result]) -> np.ndarray:
    output = image.copy()
    groups = [
        (decoded, (255, 80, 255), "D"),
        (unresolved, (255, 210, 30), "U"),
        (review, (0, 165, 255), "R"),
    ]
    index = 0
    for results, color, prefix in groups:
        for result in results:
            index += 1
            if result.kind == "linear" and result.decoded:
                color = (40, 190, 40)
            quad = np.round(result.quad).astype(np.int32)
            cv2.polylines(output, [quad], True, color, 3, cv2.LINE_AA)
            anchor = np.min(quad, axis=0)
            label = f"{prefix}{index} {result.text or result.status}"
            cv2.putText(
                output,
                label[:42],
                (int(anchor[0]), max(24, int(anchor[1]) - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
    return output


def write_crop(image: np.ndarray, result: Result, work: Path, page_number: int, category: str, index: int) -> None:
    directory = work / "crops" / category
    directory.mkdir(parents=True, exist_ok=True)
    if result.kind == "matrix":
        minimum = np.floor(np.asarray(result.quad).min(axis=0) - 40).astype(int)
        maximum = np.ceil(np.asarray(result.quad).max(axis=0) + 40).astype(int)
        x1, y1 = np.maximum(minimum, 0)
        x2 = min(image.shape[1], int(maximum[0]))
        y2 = min(image.shape[0], int(maximum[1]))
        crop = image[y1:y2, x1:x2]
    else:
        crop = rectify_quad(image, result.quad, pad_long=0.12, pad_short=0.18)
    relative = Path("crops") / category / f"page-{page_number:04d}-{category}-{index:03d}.png"
    if crop.size and cv2.imwrite(str(work / relative), crop):
        result.crop = relative.as_posix()


def process_page(page_number: int, page_path: Path, source_mode: str, config: Config, work: Path) -> dict[str, Any]:
    started_page = perf_counter()
    image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read {page_path}")
    timings: dict[str, float] = {}

    decoded: list[Result] = []
    unresolved: list[Result] = []
    review: list[Result] = []
    diagnostics: dict[str, Any] = {}

    if config.formats in {"2d", "all"}:
        started = perf_counter()
        matrix_decoded, invalid_anchors, matrix_diagnostics = matrix_fast_pass(image)
        timings["matrix_fast_seconds"] = perf_counter() - started
        decoded.extend(matrix_decoded)

        started = perf_counter()
        recovered = 0
        for anchor in invalid_anchors:
            result = recover_invalid_anchor(image, anchor)
            if result is not None:
                decoded.append(result)
                recovered += 1
                continue
            unresolved_result = unresolved_from_anchor(image, anchor)
            if unresolved_result is not None:
                unresolved.append(unresolved_result)

        residual_count = 0
        residual_decoded = 0
        residual_unresolved = 0
        if config.residual_proposals:
            covered = [np.asarray(item.quad, dtype=np.float32) for item in [*decoded, *unresolved]]
            residual = visual_matrix_proposals(image, covered)
            residual_count = len(residual)
            for anchor in residual:
                result = recover_invalid_anchor(image, anchor)
                if result is not None:
                    decoded.append(result)
                    residual_decoded += 1
                    continue
                unresolved_result = unresolved_from_anchor(image, anchor)
                if unresolved_result is not None:
                    unresolved.append(unresolved_result)
                    residual_unresolved += 1
        timings["matrix_error_recovery_seconds"] = perf_counter() - started
        diagnostics["matrix"] = {
            **matrix_diagnostics,
            "error_guided_recovered": recovered,
            "localized_unresolved": len(unresolved),
            "residual_proposals_enabled": config.residual_proposals,
            "residual_proposals": residual_count,
            "residual_decoded": residual_decoded,
            "residual_unresolved": residual_unresolved,
        }

    if config.formats in {"1d", "all"}:
        started = perf_counter()
        linear_decoded, linear_review, linear_diagnostics = linear_paths(image, config)
        timings["linear_seconds"] = perf_counter() - started
        decoded.extend(linear_decoded)
        review.extend(linear_review)
        diagnostics["linear"] = linear_diagnostics

    decoded = merge_results(decoded)
    unresolved = [item for item in merge_results(unresolved) if not any(quad_overlap(item.quad, dec.quad) >= 0.28 for dec in decoded)]
    review = [item for item in review if not any(quad_overlap(item.quad, dec.quad) >= 0.70 for dec in decoded)]
    decoded.sort(key=lambda item: (round(float(item.center()[1]) / 20), float(item.center()[0])))
    unresolved.sort(key=lambda item: (float(item.center()[1]), float(item.center()[0])))
    review.sort(key=lambda item: (float(item.center()[1]), float(item.center()[0])))

    if config.save_crops:
        for category, values in (("decoded", decoded), ("unresolved", unresolved), ("review", review)):
            for index, result in enumerate(values, start=1):
                write_crop(image, result, work, page_number, category, index)

    overlay_path = Path("overlays") / f"page-{page_number:04d}.png"
    (work / overlay_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(work / overlay_path), annotate(image, decoded, unresolved, review))

    source_relative = page_path.relative_to(work).as_posix()
    timings["total_seconds"] = perf_counter() - started_page
    return {
        "page": page_number,
        "source_image": source_relative,
        "source_mode": source_mode,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "decoded_count": len(decoded),
        "localized_unresolved_matrix_count": len(unresolved),
        "review_candidate_count": len(review),
        "results": [item.to_json() for item in decoded],
        "unresolved_matrix": [item.to_json() for item in unresolved],
        "review_candidates": [item.to_json() for item in review],
        "diagnostics": diagnostics,
        "overlay": overlay_path.as_posix(),
        "timings": {key: round(float(value), 4) for key, value in timings.items()},
    }


def validate_run(work: Path, pages: Sequence[dict[str, Any]], save_crops: bool) -> dict[str, int]:
    missing_paths: list[str] = []
    expected_crops: set[str] = set()
    for page in pages:
        for key in ("source_image", "overlay"):
            path = page[key]
            if not (work / path).is_file():
                missing_paths.append(path)
        for result_key in ("results", "unresolved_matrix", "review_candidates"):
            for result in page[result_key]:
                crop = result.get("crop")
                if crop:
                    expected_crops.add(crop)
                    if not (work / crop).is_file():
                        missing_paths.append(crop)
    actual_crops = {path.relative_to(work).as_posix() for path in (work / "crops").rglob("*.png")} if (work / "crops").exists() else set()
    orphan_crops = actual_crops - expected_crops
    if missing_paths or orphan_crops or (save_crops and actual_crops != expected_crops):
        raise RuntimeError(
            f"run integrity failure: missing={missing_paths[:8]} orphan_crops={sorted(orphan_crops)[:8]}"
        )
    return {"expected_crops": len(expected_crops), "actual_crops": len(actual_crops), "orphan_crops": 0}


def prepare_work_directory(output: Path) -> tuple[Path, str]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    work = output.parent / f".{output.name}.tmp-{run_id}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    return work, run_id


def run_pipeline(input_path: Path, config: Config) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output = config.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"output directory is not empty/new: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    work, run_id = prepare_work_directory(output)
    run_started = perf_counter()
    started_at = utc_now()
    try:
        shared_config = PipelineConfig(
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
        extraction_started = perf_counter()
        page_sources = collect_pages(input_path, shared_config)
        extraction_seconds = perf_counter() - extraction_started
        pages: list[dict[str, Any]] = []
        for page_number, page_path, source_mode in page_sources:
            payload = process_page(page_number, page_path, source_mode, config, work)
            pages.append(payload)
            print(
                f"page {page_number:04d}: {payload['decoded_count']} decoded, "
                f"{payload['localized_unresolved_matrix_count']} unresolved matrix, "
                f"{payload['review_candidate_count']} review, {payload['timings']['total_seconds']:.2f}s",
                flush=True,
            )

        integrity = validate_run(work, pages, config.save_crops)
        payload = {
            "schema_version": 2,
            "run": {
                "id": run_id,
                "started_at": started_at,
                "completed_at": utc_now(),
                "input": str(input_path),
                "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
                "pipeline_sha256": sha256_file(Path(__file__)),
                "output_root": ".",
            },
            "method": "ZXing matrix scale search + error-guided recovery + format-routed linear fallback",
            "configuration": {
                "formats": config.formats,
                "render_dpi": config.render_dpi,
                "pages": config.pages,
                "save_crops": config.save_crops,
                "residual_proposals": config.residual_proposals,
                "include_review_candidates": config.include_review_candidates,
                "minimum_linear_score": config.minimum_linear_score,
                "minimum_linear_length": config.minimum_linear_length,
            },
            "summary": {
                "pages": len(pages),
                "decoded": sum(page["decoded_count"] for page in pages),
                "localized_unresolved_matrix": sum(page["localized_unresolved_matrix_count"] for page in pages),
                "review_candidates": sum(page["review_candidate_count"] for page in pages),
                "extraction_seconds": round(extraction_seconds, 4),
                "processing_seconds": round(sum(page["timings"]["total_seconds"] for page in pages), 4),
                "wall_seconds": round(perf_counter() - run_started, 4),
            },
            "integrity": integrity,
            "pages": pages,
        }
        (work / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        os.replace(work, output)
        print(f"wrote {output / 'detections.json'}")
        return payload
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/codarascan/matrix/run"),
    )
    parser.add_argument("--formats", choices=("2d", "1d", "all"), default="2d")
    parser.add_argument("--pages", type=parse_page_selection)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-crops", dest="save_crops", action="store_false")
    parser.add_argument("--residual-proposals", action="store_true")
    parser.add_argument("--no-review-candidates", dest="include_review_candidates", action="store_false")
    parser.add_argument("--minimum-linear-score", type=float, default=0.52)
    parser.add_argument("--minimum-linear-length", type=float, default=100.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = Config(
        output=arguments.output,
        formats=arguments.formats,
        render_dpi=arguments.render_dpi,
        pages=arguments.pages,
        overwrite=arguments.overwrite,
        save_crops=arguments.save_crops,
        residual_proposals=arguments.residual_proposals,
        include_review_candidates=arguments.include_review_candidates,
        minimum_linear_score=arguments.minimum_linear_score,
        minimum_linear_length=arguments.minimum_linear_length,
    )
    try:
        run_pipeline(arguments.input, config)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
