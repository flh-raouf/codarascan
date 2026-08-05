#!/usr/bin/env python3
"""CPU-optimized, deterministic barcode localization without decoding.

The pipeline deliberately has no barcode decoder dependency and never attempts
to recover a payload. Its default linear path is coarse-to-fine: OpenCV's
classical directional-coherence detector proposes regions on a reduced image
using an ablated scale band, then strict structural verification reads only the
small proposed regions from the native pixels. QR symbols must exhibit their
nested finder-pattern geometry. Data Matrix symbols are proposed by
bidirectional energy, then accepted only after an ECC 200 finder/timing-border
and module-grid verifier succeeds. No GPU is used.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


EPS = 1e-6
DEFAULT_OPENCV_THREADS = max(1, cv2.getNumThreads())
# This project is intentionally CPU-only. ndarray/Mat operations do not use
# OpenCL in normal OpenCV builds, but disabling it explicitly keeps the runtime
# contract unambiguous and reproducible.
cv2.ocl.setUseOpenCL(False)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
LEGACY_LINEAR_SCALES = tuple(
    sorted(
        {
            0.0005,
            0.001,
            0.002,
            0.003,
            0.004,
            0.006,
            0.008,
            0.01,
            0.016,
            0.02,
            0.032,
            0.04,
            0.064,
            0.08,
        }
    )
)

# Scale ablation across clean, mixed, and deliberately harsh benchmark inputs
# showed that the two smallest and four largest legacy scales added cost without
# adding a validated localization. Keeping the middle band also improved grouping: the
# detector returns complete symbols instead of short fragments on the synthetic
# pages.  The legacy profile remains available for audits and unfamiliar data.
UNIVERSAL_LINEAR_SCALES = (0.002, 0.003, 0.004, 0.006, 0.008, 0.01)
FAST_LINEAR_SCALES = (0.003, 0.004, 0.006)
LINEAR_PROFILES: dict[str, tuple[float, tuple[float, ...]]] = {
    "universal": (0.65, UNIVERSAL_LINEAR_SCALES),
    "fast": (0.45, FAST_LINEAR_SCALES),
    "legacy": (0.65, LEGACY_LINEAR_SCALES),
}
DETECTOR_LOCAL = threading.local()

# ECC 200 symbol dimensions. The local verifier deliberately recognizes the
# physical Data Matrix finder/timing structure; it does not decode payloads.
DATA_MATRIX_SQUARE_SIZES = (
    10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 36, 40, 44, 48, 52, 64,
    72, 80, 88, 96, 104, 120, 132, 144,
)
DATA_MATRIX_RECTANGULAR_SIZES = (
    (8, 18), (8, 32), (12, 26), (12, 36), (16, 36), (16, 48),
)
# A 160 px normalized patch cannot safely prove a grid finer than 64 modules.
# Larger symbols need a higher-resolution proposal and therefore remain
# proposals rather than being incorrectly accepted as Data Matrix.
DATA_MATRIX_VERIFIABLE_SIZES = tuple(
    (size, size) for size in DATA_MATRIX_SQUARE_SIZES if size <= 64
) + DATA_MATRIX_RECTANGULAR_SIZES
MINIMUM_DATA_MATRIX_GRID_SCORE = 0.68
DATA_MATRIX_TIMING_PATTERNS = {
    length: np.arange(length, dtype=np.float32) % 2.0
    for length in {
        dimension
        for rows, columns in DATA_MATRIX_VERIFIABLE_SIZES
        for dimension in (rows, columns)
    }
}


@dataclass
class Config:
    output: Path
    kinds: str = "linear"
    pages: list[int] | None = None
    render_dpi: int = 300
    workers: int = 0
    overwrite: bool = False
    save_overlays: bool = False
    save_crops: bool = False
    minimum_linear_score: float = 0.52
    minimum_linear_length: float = 100.0
    linear_profile: str = "universal"
    linear_work_scale: float | None = None
    matrix_work_size: int = 1400
    empty_page_gate: bool = True
    empty_gate_matrix_work_size: int = 700
    empty_gate_qr_work_size: int = 900


@dataclass
class Detection:
    quad: np.ndarray
    kind: str
    confidence: float
    source: str
    metrics: dict[str, float] = field(default_factory=dict)

    def center(self) -> np.ndarray:
        return np.asarray(self.quad, dtype=np.float32).mean(axis=0)

    def area(self) -> float:
        return float(abs(cv2.contourArea(np.asarray(self.quad, dtype=np.float32))))

    def long_short(self) -> tuple[float, float]:
        width, height = cv2.minAreaRect(np.asarray(self.quad, dtype=np.float32))[1]
        return max(float(width), float(height)), min(float(width), float(height))

    def angle(self) -> float:
        points = order_quad(self.quad)
        top = points[1] - points[0]
        left = points[3] - points[0]
        vector = top if np.linalg.norm(top) >= np.linalg.norm(left) else left
        return float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0)

    def to_json(self) -> dict[str, Any]:
        quad = np.asarray(self.quad, dtype=np.float32)
        low = quad.min(axis=0)
        high = quad.max(axis=0)
        long_side, short_side = self.long_short()
        return {
            "kind": self.kind,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in quad],
            "aabb": {
                "x": round(float(low[0]), 2),
                "y": round(float(low[1]), 2),
                "width": round(float(high[0] - low[0]), 2),
                "height": round(float(high[1] - low[1]), 2),
            },
            "long_axis_angle_degrees": round(self.angle(), 2),
            "long_side_pixels": round(long_side, 2),
            "short_side_pixels": round(short_side, 2),
            "metrics": {key: round(float(value), 4) for key, value in self.metrics.items()},
        }


def parse_page_selection(value: str) -> list[int]:
    pages: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise argparse.ArgumentTypeError(f"invalid page range: {token}")
            pages.update(range(start, end + 1))
        else:
            page = int(token)
            if page < 1:
                raise argparse.ArgumentTypeError("page numbers begin at 1")
            pages.add(page)
    if not pages:
        raise argparse.ArgumentTypeError("empty page selection")
    return sorted(pages)


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) != 4:
        points = cv2.boxPoints(cv2.minAreaRect(points))
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    if len({tuple(point) for point in ordered}) != 4:
        center = points.mean(axis=0)
        cycle = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
        ordered = np.roll(cycle, -int(np.argmin(cycle.sum(axis=1))), axis=0)
        first, second = ordered[1] - ordered[0], ordered[2] - ordered[1]
        if float(first[0] * second[1] - first[1] * second[0]) > 0:
            ordered = ordered[[0, 3, 2, 1]]
    return ordered


def quad_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first = cv2.convexHull(np.asarray(first, dtype=np.float32)).reshape(-1, 2)
    second = cv2.convexHull(np.asarray(second, dtype=np.float32)).reshape(-1, 2)
    first_area = abs(cv2.contourArea(first))
    second_area = abs(cv2.contourArea(second))
    if first_area < EPS or second_area < EPS:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(first, second)
    except cv2.error:
        return 0.0
    return float(intersection) / min(float(first_area), float(second_area))


def same_detection(first: Detection, second: Detection) -> bool:
    if quad_overlap(first.quad, second.quad) >= 0.52:
        return True
    first_long, first_short = first.long_short()
    second_long, second_short = second.long_short()
    distance = float(np.linalg.norm(first.center() - second.center()))
    angle = abs(first.angle() - second.angle())
    angle = min(angle, 180.0 - angle)
    return bool(
        distance < 0.16 * min(first_long, second_long)
        and angle < 8.0
        and abs(first_short - second_short) < 0.9 * max(first_short, second_short)
    )


def deduplicate(detections: Iterable[Detection]) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: (item.confidence, item.area()), reverse=True):
        prior = next((item for item in kept if item.kind == detection.kind and same_detection(item, detection)), None)
        if prior is None:
            kept.append(detection)
        elif detection.confidence > prior.confidence:
            prior.quad = detection.quad
            prior.confidence = detection.confidence
            prior.metrics = detection.metrics
            prior.source = f"{prior.source}+{detection.source}"
        elif detection.source not in prior.source:
            prior.source = f"{prior.source}+{detection.source}"
    return kept


def rectify(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    source = order_quad(quad)
    top = np.linalg.norm(source[1] - source[0])
    bottom = np.linalg.norm(source[2] - source[3])
    left = np.linalg.norm(source[3] - source[0])
    right = np.linalg.norm(source[2] - source[1])
    width = max(2, int(round(max(top, bottom))))
    height = max(2, int(round(max(left, right))))
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    patch = cv2.warpPerspective(
        gray,
        cv2.getPerspectiveTransform(source, destination),
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE) if patch.shape[0] > patch.shape[1] else patch


def binary_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8).reshape(1, -1)
    return cv2.threshold(values, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1].ravel() > 0


def count_transition_runs(binary: np.ndarray) -> tuple[int, float]:
    if len(binary) < 2:
        return 0, 0.0
    runs: list[tuple[bool, int]] = []
    state, length = bool(binary[0]), 1
    for raw_value in binary[1:]:
        value = bool(raw_value)
        if value == state:
            length += 1
        else:
            runs.append((state, length))
            state, length = value, 1
    runs.append((state, length))
    compact: list[tuple[bool, int]] = []
    index = 0
    while index < len(runs):
        state, length = runs[index]
        if 0 < index < len(runs) - 1 and length == 1 and runs[index - 1][0] == runs[index + 1][0]:
            if compact:
                previous_state, previous_length = compact[-1]
                compact[-1] = (previous_state, previous_length + 1 + runs[index + 1][1])
            index += 2
            continue
        compact.append((state, length))
        index += 1
    return max(0, len(compact) - 1), float(np.mean(binary))


def linear_metrics(gray: np.ndarray, quad: np.ndarray) -> dict[str, float]:
    patch = rectify(gray, quad)
    height, width = patch.shape[:2]
    if width < 18 or height < 5:
        return {"score": 0.0}
    top = max(0, int(round(height * 0.10)))
    bottom = min(height, max(top + 3, int(round(height * 0.82))))
    strip = patch[top:bottom]
    gx = np.abs(cv2.Scharr(strip, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Scharr(strip, cv2.CV_32F, 0, 1))
    x_energy, y_energy = float(np.mean(gx)), float(np.mean(gy))
    orientation = x_energy / (x_energy + y_energy + EPS)
    threshold = float(np.percentile(gx, 72.0))
    if threshold <= EPS:
        threshold = float(np.mean(gx) + np.std(gx))
    support = np.mean(gx >= max(threshold, EPS), axis=0)
    persistent = support >= 0.43
    persistence = float(np.mean(persistent))
    support_strength = float(np.mean(support[persistent])) if np.any(persistent) else 0.0
    consensus = np.median(strip, axis=0).astype(np.uint8)
    binary = binary_signal(consensus)
    transitions, dark_fraction = count_transition_runs(binary)
    transition_rate = transitions / max(1, width)
    sites = np.flatnonzero(np.diff(binary.astype(np.int8)) != 0)
    agreements: list[float] = []
    if len(sites):
        expanded = np.zeros(width, dtype=bool)
        for offset in range(-2, 3):
            expanded[np.clip(sites + offset, 0, width - 1)] = True
        for row in np.linspace(0, strip.shape[0] - 1, min(9, strip.shape[0]), dtype=int):
            row_binary = binary_signal(strip[row])
            agreements.append(float(np.mean(row_binary[expanded] == binary[expanded])))
    agreement = float(np.median(agreements)) if agreements else 0.0
    aspect = width / max(1.0, height)
    aspect_score = min(1.0, max(0.0, (aspect - 1.15) / 2.2))
    alternation = min(1.0, transitions / 18.0) * min(1.0, transition_rate / 0.055)
    persistence_score = min(1.0, persistence / 0.16) * min(1.0, support_strength / 0.62)
    dark_score = max(0.0, 1.0 - abs(dark_fraction - 0.43) / 0.43)
    score = (
        0.22 * orientation
        + 0.30 * persistence_score
        + 0.25 * alternation
        + 0.16 * agreement
        + 0.04 * aspect_score
        + 0.03 * dark_score
    )
    return {
        "score": score,
        "orientation": orientation,
        "persistence": persistence,
        "support_strength": support_strength,
        "scanline_agreement": agreement,
        "transitions": float(transitions),
        "transition_rate": transition_rate,
        "dark_fraction": dark_fraction,
        "aspect": aspect,
    }


def linear_profile(config: Config) -> tuple[float, tuple[float, ...]]:
    default_scale, scales = LINEAR_PROFILES[config.linear_profile]
    requested_scale = default_scale if config.linear_work_scale is None else float(config.linear_work_scale)
    return min(1.0, max(0.35, requested_scale)), scales


def linear_detector(profile: str, scales: tuple[float, ...]) -> cv2.barcode.BarcodeDetector:
    """Return one detector per worker thread; OpenCV instances are not shared."""
    cache = getattr(DETECTOR_LOCAL, "barcode_detectors", None)
    if cache is None:
        cache = {}
        DETECTOR_LOCAL.barcode_detectors = cache
    detector = cache.get(profile)
    if detector is None:
        detector = cv2.barcode.BarcodeDetector()
        detector.setGradientThreshold(48.0)
        detector.setDetectorScales(np.asarray(scales, dtype=np.float32))
        cache[profile] = detector
    return detector


def locate_linear(
    gray: np.ndarray,
    config: Config,
    prepared_work: np.ndarray | None = None,
) -> tuple[list[Detection], dict[str, Any], dict[str, float]]:
    scale, scales = linear_profile(config)
    if prepared_work is None:
        resize_started = perf_counter()
        work = (
            cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else gray
        )
        resize_seconds = perf_counter() - resize_started
    else:
        work = np.asarray(prepared_work)
        if work.ndim != 2 or work.dtype != np.uint8:
            raise ValueError(
                "prepared linear work image must be a 2-D uint8 grayscale array"
            )
        expected_width = max(1, int(round(gray.shape[1] * scale)))
        expected_height = max(1, int(round(gray.shape[0] * scale)))
        if abs(work.shape[1] - expected_width) > 1 or abs(
            work.shape[0] - expected_height
        ) > 1:
            raise ValueError(
                "prepared linear work image dimensions do not match the "
                f"configured scale: got {work.shape[1]}x{work.shape[0]}, "
                f"expected approximately {expected_width}x{expected_height}"
            )
        resize_seconds = 0.0
    detector = linear_detector(config.linear_profile, scales)
    # An application can scan a small ROI instead of a complete page. OpenCV requires
    # this threshold to be at least 64 even when the ROI itself is smaller.
    detector.setDownsamplingThreshold(max(64.0, float(max(work.shape[:2]))))
    detector_started = perf_counter()
    try:
        found, points = detector.detectMulti(work)
    except cv2.error:
        found, points = False, None
    detector_seconds = perf_counter() - detector_started
    verification_started = perf_counter()
    proposals: list[Detection] = []
    if found and points is not None:
        for raw_quad in np.asarray(points, dtype=np.float32):
            quad = raw_quad / scale
            candidate = Detection(
                quad=quad,
                kind="linear",
                confidence=0.0,
                source=f"opencv:barcode-detector:{config.linear_profile}",
            )
            long_side, short_side = candidate.long_short()
            if long_side < config.minimum_linear_length or short_side < 5 or long_side / (short_side + EPS) < 1.65:
                continue
            metrics = linear_metrics(gray, quad)
            accepted = bool(
                metrics.get("score", 0.0) >= config.minimum_linear_score
                and metrics.get("orientation", 0.0) >= 0.62
                and metrics.get("persistence", 0.0) >= 0.045
                and metrics.get("transitions", 0.0) >= 12.0
                and metrics.get("scanline_agreement", 0.0) >= 0.50
                and 0.015 <= metrics.get("transition_rate", 0.0) <= 0.72
            )
            if accepted:
                candidate.metrics = metrics
                candidate.confidence = float(metrics["score"])
                proposals.append(candidate)
    accepted = deduplicate(proposals)
    verification_seconds = perf_counter() - verification_started
    return (
        accepted,
        {
            "raw_linear_proposals": 0 if points is None else len(points),
            "accepted_linear": len(accepted),
            "linear_profile": config.linear_profile,
            "linear_work_scale": scale,
            "linear_detector_scales": list(scales),
            "linear_work_pixels": int(work.size),
            "native_page_pixels": int(gray.size),
            "linear_work_prepared": int(prepared_work is not None),
        },
        {
            "linear_resize_seconds": resize_seconds,
            "linear_detector_seconds": detector_seconds,
            "linear_verification_seconds": verification_seconds,
        },
    )


def locate_qr(gray: np.ndarray) -> list[Detection]:
    detections: list[Detection] = []

    detector = getattr(DETECTOR_LOCAL, "qr_detector", None)
    if detector is None:
        detector = cv2.QRCodeDetector()
        DETECTOR_LOCAL.qr_detector = detector

    def detect(image: np.ndarray, scale: float, source: str) -> None:
        try:
            found, points = detector.detectMulti(image)
        except cv2.error:
            found, points = False, None
        if not found or points is None:
            return
        for raw_quad in np.asarray(points, dtype=np.float32):
            quad = raw_quad / scale
            area = abs(cv2.contourArea(quad))
            long_side, short_side = Detection(quad, "qr", 0.0, source).long_short()
            if area < 100 or short_side < 10 or long_side / (short_side + EPS) > 1.8:
                continue
            detections.append(Detection(quad=quad, kind="qr", confidence=0.96, source=source))

    detect(gray, 1.0, "opencv:qr-detector")
    return deduplicate(detections)


def qr_finder_chain(
    contours: Sequence[np.ndarray],
    hierarchy: np.ndarray,
    index: int,
    image_pixels: int,
) -> bool:
    """Recognize one nested, concentric QR finder-pattern hypothesis."""
    boxes: list[tuple[int, int, int, int]] = []
    current = index
    for _ in range(3):
        if current < 0:
            return False
        x, y, width, height = cv2.boundingRect(contours[current])
        short_side, long_side = min(width, height), max(width, height)
        if short_side < 3 or long_side / max(1.0, short_side) > 1.35:
            return False
        boxes.append((x, y, width, height))
        current = int(hierarchy[current][2])

    centers = [
        (x + width / 2.0, y + height / 2.0)
        for x, y, width, height in boxes
    ]
    tolerance = 0.16 * max(boxes[0][2], boxes[0][3])
    if any(
        abs(center_x - centers[0][0]) > tolerance
        or abs(center_y - centers[0][1]) > tolerance
        for center_x, center_y in centers[1:]
    ):
        return False

    outer_side = max(boxes[0][2], boxes[0][3])
    middle_side = max(boxes[1][2], boxes[1][3])
    inner_side = max(boxes[2][2], boxes[2][3])
    outer_to_middle = middle_side / max(1.0, outer_side)
    middle_to_inner = inner_side / max(1.0, middle_side)
    outer_area = boxes[0][2] * boxes[0][3]
    return bool(
        0.35 <= outer_to_middle <= 0.88
        and 0.35 <= middle_to_inner <= 0.88
        and outer_area < 0.03 * image_pixels
    )


def quick_qr_finder_evidence(
    gray: np.ndarray,
    work_size: int,
) -> tuple[bool, dict[str, Any]]:
    """Cheap one-sided gate: absence may suppress the final full QR scan.

    It does not localize or accept a QR symbol. It searches two deterministic
    binary views for nested, square, concentric contours compatible with the QR
    finder pattern. Any uncertainty escalates to OpenCV's full native detector.
    """
    height, width = gray.shape
    scale = min(1.0, float(work_size) / max(height, width))
    work = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    candidates = 0
    contours_examined = 0
    # Create the adaptive view lazily. A clear finder returns from the cheaper
    # Otsu view, avoiding a second full-thumbnail threshold pass.
    masks = [cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]]
    for mask_index in range(2):
        if mask_index == 1:
            masks.append(
                cv2.adaptiveThreshold(
                    work,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    31,
                    7,
                )
            )
        mask = masks[mask_index]
        contours, raw_hierarchy = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours_examined += len(contours)
        if raw_hierarchy is None:
            continue
        hierarchy = raw_hierarchy[0]
        for index in range(len(contours)):
            if qr_finder_chain(contours, hierarchy, index, work.size):
                candidates += 1
                # One plausible finder is enough to prevent an unsafe skip.
                return True, {
                    "quick_qr_finder_candidates": candidates,
                    "quick_qr_contours_examined": contours_examined,
                    "quick_qr_work_pixels": int(work.size),
                    "quick_qr_work_scale": scale,
                }
    return False, {
        "quick_qr_finder_candidates": candidates,
        "quick_qr_contours_examined": contours_examined,
        "quick_qr_work_pixels": int(work.size),
        "quick_qr_work_scale": scale,
    }


def longest_true_runs(values: np.ndarray) -> np.ndarray:
    """Longest consecutive true run for every row, without Python row loops."""
    values = np.asarray(values, dtype=bool)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.zeros(values.shape[0] if values.ndim else 0, dtype=np.int16)
    columns = np.arange(values.shape[1], dtype=np.int16)[None, :]
    last_false = np.maximum.accumulate(np.where(values, -1, columns), axis=1)
    lengths = np.where(values, columns - last_false, 0)
    return np.max(lengths, axis=1)


def matrix_proposal_metrics(gray: np.ndarray, quad: np.ndarray) -> dict[str, float]:
    """Permissive format-neutral metrics used only to prune energy proposals.

    Directional energy is intentionally not sufficient for final acceptance:
    text, stamps and table intersections can all satisfy these measurements.
    Final Data Matrix and QR decisions are made by their dedicated physical
    structure verifiers below.
    """
    patch = rectify(gray, quad)
    binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    height, width = patch.shape
    gradient_x = float(np.mean(np.abs(cv2.Scharr(patch, cv2.CV_32F, 1, 0))))
    gradient_y = float(np.mean(np.abs(cv2.Scharr(patch, cv2.CV_32F, 0, 1))))
    edge_balance = min(gradient_x, gradient_y) / max(EPS, max(gradient_x, gradient_y))
    aspect = width / max(1.0, float(height))
    square_score = max(0.0, 1.0 - abs(np.log(max(EPS, aspect))) / np.log(2.2))
    source_binary = binary > 0
    row_transitions: list[int] = []
    column_transitions: list[int] = []
    for position in np.linspace(0.15, 0.85, 7):
        row = source_binary[min(height - 1, int(round(position * (height - 1))))]
        column = source_binary[:, min(width - 1, int(round(position * (width - 1))))]
        row_transitions.append(int(np.count_nonzero(np.diff(row.astype(np.int8)))))
        column_transitions.append(int(np.count_nonzero(np.diff(column.astype(np.int8)))))
    transition_score = min(
        1.0,
        min(float(np.median(row_transitions)), float(np.median(column_transitions))) / 10.0,
    )
    binary = cv2.resize(binary, (96, 96), interpolation=cv2.INTER_NEAREST) > 0
    dark_fraction = float(np.mean(binary))
    occupancy_score = max(0.0, 1.0 - abs(dark_fraction - 0.45) / 0.45)
    base_score = (
        0.30 * edge_balance
        + 0.25 * square_score
        + 0.30 * transition_score
        + 0.15 * occupancy_score
    )
    grid_occupancy = binary.reshape(6, 16, 6, 16).mean(axis=(1, 3)).ravel()
    l_border_score = 0.0
    for rotation in range(4):
        rotated = np.rot90(binary, rotation)
        left = rotated[8:88, :30].T
        bottom = rotated[95 - np.arange(30), 8:88]
        left_scores = 0.55 * left.mean(axis=1) + 0.45 * longest_true_runs(left) / 80.0
        bottom_scores = 0.55 * bottom.mean(axis=1) + 0.45 * longest_true_runs(bottom) / 80.0
        l_border_score = max(l_border_score, float(np.max(np.minimum(left_scores, bottom_scores))))
    component_count, _labels, statistics, _centers = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    areas = sorted((int(value) for value in statistics[1:, cv2.CC_STAT_AREA]), reverse=True)
    largest_component = float(areas[0] / max(1, int(np.count_nonzero(binary)))) if areas else 0.0
    grid_q20 = float(np.quantile(grid_occupancy, 0.20))
    grid_std = float(np.std(grid_occupancy))
    components = component_count - 1
    # Three independent physical signatures cover clean, connected matrices,
    # fragmented/damaged matrices, and strong L-border matrices.  The gates are
    # deliberately conjunctive; text blocks and table intersections fail them.
    connected_matrix = largest_component >= 0.80 and l_border_score >= 0.50 and grid_q20 >= 0.14
    fragmented_matrix = (
        components >= 30
        and largest_component >= 0.35
        and grid_q20 >= 0.12
        and dark_fraction >= 0.25
    )
    finder_matrix = l_border_score >= 0.72 and dark_fraction >= 0.35
    regular_matrix = (
        l_border_score >= 0.42
        and largest_component >= 0.35
        and grid_q20 >= 0.14
        and dark_fraction >= 0.35
        and components >= 3
    )
    # This first, format-neutral gate is crucial.  It cheaply rejects regions
    # that only have horizontal *or* vertical structure before the more
    # specific matrix signatures are considered.
    qr_like_matrix = (
        base_score >= 0.90
        and edge_balance >= 0.85
        and square_score >= 0.85
        and 0.25 <= dark_fraction <= 0.40
        and 5 <= components <= 25
        and 0.55 <= largest_component <= 0.80
        and 0.40 <= l_border_score <= 0.60
        and grid_q20 >= 0.11
    )
    accepted = (
        base_score >= 0.72
        and 0.10 <= dark_fraction <= 0.82
        and (connected_matrix or fragmented_matrix or finder_matrix or regular_matrix or qr_like_matrix)
    )
    confidence = min(
        0.94,
        0.24
        + 0.30 * l_border_score
        + 0.20 * min(1.0, largest_component / 0.80)
        + 0.16 * min(1.0, grid_q20 / 0.25)
        + 0.10 * min(1.0, dark_fraction / 0.45),
    )
    return {
        "accepted": float(accepted),
        "score": confidence,
        "base_score": base_score,
        "edge_balance": edge_balance,
        "square_score": square_score,
        "transition_score": transition_score,
        "dark_fraction": dark_fraction,
        "grid_q20": grid_q20,
        "grid_std": grid_std,
        "l_border_score": l_border_score,
        "component_count": float(components),
        "largest_component_fraction": largest_component,
    }


def quad_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(np.asarray(first, dtype=np.float32)).reshape(-1, 2)
    second_hull = cv2.convexHull(np.asarray(second, dtype=np.float32)).reshape(-1, 2)
    first_area = abs(cv2.contourArea(first_hull))
    second_area = abs(cv2.contourArea(second_hull))
    if min(first_area, second_area) < EPS:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    except cv2.error:
        return 0.0
    union = first_area + second_area - float(intersection)
    return float(intersection) / max(EPS, union)


def square_warp(gray: np.ndarray, quad: np.ndarray, size: int = 160) -> np.ndarray:
    source = order_quad(quad)
    destination = np.asarray(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    return cv2.warpPerspective(
        gray,
        cv2.getPerspectiveTransform(source, destination),
        (size, size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def alternating_module_score(
    values: np.ndarray,
    pattern: np.ndarray | None = None,
) -> float:
    values = np.asarray(values, dtype=np.float32)
    if pattern is None:
        pattern = DATA_MATRIX_TIMING_PATTERNS.get(values.size)
    if pattern is None:
        pattern = np.arange(values.size, dtype=np.float32) % 2.0
    direct = 1.0 - float(np.mean(np.abs(values - pattern)))
    inverse = 1.0 - float(np.mean(np.abs(values - (1.0 - pattern))))
    return max(direct, inverse)


def data_matrix_lattice_score(binary: np.ndarray) -> tuple[float, dict[str, float]]:
    """Score the two solid and two alternating borders of an ECC 200 grid."""
    source = (np.asarray(binary) > 0).astype(np.float32)
    best_score = 0.0
    best_metrics: dict[str, float] = {}
    for rows, columns in DATA_MATRIX_VERIFIABLE_SIZES:
        # Require enough normalized samples to distinguish adjacent modules.
        if min(source.shape) / max(rows, columns) < 1.35:
            continue
        cells = cv2.resize(source, (columns, rows), interpolation=cv2.INTER_AREA)
        top, right = cells[0, :], cells[:, -1]
        bottom, left = cells[-1, :], cells[:, 0]
        solids = tuple(float(np.mean(edge)) for edge in (top, right, bottom, left))
        column_pattern = DATA_MATRIX_TIMING_PATTERNS[columns]
        row_pattern = DATA_MATRIX_TIMING_PATTERNS[rows]
        timings = (
            alternating_module_score(top, column_pattern),
            alternating_module_score(right, row_pattern),
            alternating_module_score(bottom, column_pattern),
            alternating_module_score(left, row_pattern),
        )
        # Entries are solid-left, solid-bottom, timing-top, timing-right for
        # each clockwise orientation, without materializing rotated arrays.
        arrangements = (
            (3, 2, 0, 1),
            (2, 1, 3, 0),
            (1, 0, 2, 3),
            (0, 3, 1, 2),
        )
        rotation, finder_score = max(
            enumerate(
                min(
                    solids[solid_left],
                    solids[solid_bottom],
                    timings[timing_top],
                    timings[timing_right],
                )
                for solid_left, solid_bottom, timing_top, timing_right in arrangements
            ),
            key=lambda item: item[1],
        )
        solid_left, solid_bottom, timing_top, timing_right = arrangements[rotation]
        cell_purity = float(np.mean(np.abs(cells - 0.5) * 2.0))
        interior = cells[1:-1, 1:-1]
        occupancy = float(np.mean(interior)) if interior.size else 0.0
        occupancy_score = max(0.0, 1.0 - abs(occupancy - 0.5) / 0.5)
        score = 0.68 * finder_score + 0.22 * cell_purity + 0.10 * occupancy_score
        if score <= best_score:
            continue
        best_score = score
        best_metrics = {
            "symbol_rows": float(cells.shape[0] if rotation % 2 == 0 else cells.shape[1]),
            "symbol_columns": float(cells.shape[1] if rotation % 2 == 0 else cells.shape[0]),
            "finder_score": finder_score,
            "solid_left": solids[solid_left],
            "solid_bottom": solids[solid_bottom],
            "timing_top": timings[timing_top],
            "timing_right": timings[timing_right],
            "cell_purity": cell_purity,
            "module_occupancy": occupancy,
        }
    return best_score, best_metrics


def score_data_matrix_mask(mask: np.ndarray) -> tuple[float, dict[str, float]]:
    best_score = 0.0
    best_metrics: dict[str, float] = {}
    # The second view moves the sampling grid just inside a thick printed
    # border. It is not a separate detector pass.
    for inset in (0, 2):
        view = mask[
            inset:mask.shape[0] - inset or None,
            inset:mask.shape[1] - inset or None,
        ]
        if min(view.shape) < 100:
            continue
        score, metrics = data_matrix_lattice_score(view)
        if score > best_score:
            best_score = score
            best_metrics = dict(metrics, sampling_inset=float(inset))
    return best_score, best_metrics


def data_matrix_threshold_mask(
    patch: np.ndarray,
    *,
    adaptive: bool,
) -> np.ndarray:
    """Threshold one cached normalized patch using only the requested method."""
    if adaptive:
        return cv2.adaptiveThreshold(
            patch,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
    return cv2.threshold(
        patch,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]


def expanded_candidate_roi(
    gray: np.ndarray,
    rough_quad: np.ndarray,
    factor: float,
) -> tuple[np.ndarray, int, int]:
    x, y, width, height = cv2.boundingRect(np.asarray(rough_quad, dtype=np.int32))
    padding = int(round(factor * max(width, height)))
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2 = min(gray.shape[1], x + width + padding)
    y2 = min(gray.shape[0], y + height + padding)
    return gray[y1:y2, x1:x2], x1, y1


def data_matrix_border_hypotheses(
    gray: np.ndarray,
    rough_quad: np.ndarray,
) -> list[np.ndarray]:
    """Recover precise local border hypotheses around one energy proposal."""
    rough_quad = order_quad(rough_quad)
    hypotheses = [rough_quad]
    x, y, width, height = cv2.boundingRect(np.asarray(rough_quad, dtype=np.int32))
    hypotheses.append(
        np.asarray(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        )
    )
    rough_short, rough_long = min(width, height), max(width, height)
    for roi_factor in (0.20, 0.65):
        roi, offset_x, offset_y = expanded_candidate_roi(gray, rough_quad, roi_factor)
        if roi.size == 0:
            continue
        masks = (
            cv2.threshold(
                roi,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1],
            cv2.adaptiveThreshold(
                roi,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                7,
            ),
        )
        for base in masks:
            for kernel_size in (0, 3, 5):
                mask = (
                    base
                    if kernel_size == 0
                    else cv2.morphologyEx(
                        base,
                        cv2.MORPH_CLOSE,
                        np.ones((kernel_size, kernel_size), dtype=np.uint8),
                    )
                )
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for contour in contours:
                    contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(contour)
                    contour_short = min(contour_width, contour_height)
                    contour_long = max(contour_width, contour_height)
                    if (
                        contour_short >= max(28.0, 0.30 * rough_short)
                        and contour_long <= 2.8 * rough_long
                        and contour_long / max(1.0, contour_short) <= 4.5
                    ):
                        axis_quad = np.asarray(
                            [
                                [contour_x + offset_x, contour_y + offset_y],
                                [contour_x + contour_width + offset_x, contour_y + offset_y],
                                [
                                    contour_x + contour_width + offset_x,
                                    contour_y + contour_height + offset_y,
                                ],
                                [contour_x + offset_x, contour_y + contour_height + offset_y],
                            ],
                            dtype=np.float32,
                        )
                        if quad_overlap(axis_quad, rough_quad) >= 0.10:
                            hypotheses.append(axis_quad)
                    rectangle = cv2.minAreaRect(contour)
                    candidate_width, candidate_height = (float(value) for value in rectangle[1])
                    short_side = min(candidate_width, candidate_height)
                    long_side = max(candidate_width, candidate_height)
                    if (
                        short_side < max(28.0, 0.30 * rough_short)
                        or long_side > 2.8 * rough_long
                        or long_side / max(1.0, short_side) > 4.5
                    ):
                        continue
                    quad = cv2.boxPoints(rectangle).astype(np.float32)
                    quad[:, 0] += offset_x
                    quad[:, 1] += offset_y
                    if quad_overlap(quad, rough_quad) >= 0.10:
                        hypotheses.append(order_quad(quad))
    deduplicated: list[np.ndarray] = []
    for quad in hypotheses:
        # One-pixel border shifts materially change module alignment, so only
        # collapse almost identical quadrilaterals.
        if not any(quad_iou(quad, prior) >= 0.985 for prior in deduplicated):
            deduplicated.append(quad)
        if len(deduplicated) >= 64:
            break
    rough_area = max(EPS, abs(cv2.contourArea(rough_quad)))
    deduplicated.sort(
        key=lambda quad: abs(
            math.log(max(EPS, abs(cv2.contourArea(quad))) / rough_area)
        )
    )
    # Difficult symbols can require a winning border beyond the first nine
    # area-consistent hypotheses. Twelve is the validated conservative bound
    # across the expanded regression corpus.
    return deduplicated[:12]


def refine_data_matrix_candidate(
    gray: np.ndarray,
    rough_quad: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, float]]:
    hypotheses = data_matrix_border_hypotheses(gray, rough_quad)
    otsu_ranked: list[
        tuple[float, np.ndarray, np.ndarray, dict[str, float]]
    ] = []
    for quad in hypotheses:
        # Warp each geometry once. The old path generated an unused adaptive
        # mask for every hypothesis, then warped the strongest two a second
        # time. Caching the patch preserves the exact masks and scores while
        # removing both forms of duplicate work.
        patch = square_warp(gray, quad)
        otsu = data_matrix_threshold_mask(patch, adaptive=False)
        score, metrics = score_data_matrix_mask(otsu)
        otsu_ranked.append(
            (score, quad, patch, dict(metrics, threshold_view=0.0))
        )
    if not otsu_ranked:
        return None, {"grid_score": 0.0, "border_hypotheses": 0.0}
    otsu_ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_quad, _best_patch, best_metrics = otsu_ranked[0]
    # Adaptive thresholding is more expensive and most useful on the strongest
    # geometries. Restrict it to two hypotheses while retaining the Otsu score
    # for every hypothesis.
    for _otsu_score, quad, patch, _otsu_metrics in otsu_ranked[:2]:
        adaptive = data_matrix_threshold_mask(patch, adaptive=True)
        score, metrics = score_data_matrix_mask(adaptive)
        if score > best_score:
            best_score = score
            best_quad = quad
            best_metrics = dict(metrics, threshold_view=1.0)
    metrics = dict(
        best_metrics,
        grid_score=best_score,
        border_hypotheses=float(len(hypotheses)),
    )
    if best_score < MINIMUM_DATA_MATRIX_GRID_SCORE:
        return None, metrics
    return order_quad(best_quad), metrics


def local_qr_finder_chain(
    contours: Sequence[np.ndarray],
    hierarchy: np.ndarray,
    index: int,
) -> tuple[float, float] | None:
    boxes: list[tuple[int, int, int, int]] = []
    current = index
    for _ in range(3):
        if current < 0:
            return None
        x, y, width, height = cv2.boundingRect(contours[current])
        short_side, long_side = min(width, height), max(width, height)
        if short_side < 3 or long_side / max(1.0, short_side) > 1.40:
            return None
        boxes.append((x, y, width, height))
        current = int(hierarchy[current][2])
    centers = np.asarray(
        [(x + width / 2.0, y + height / 2.0) for x, y, width, height in boxes],
        dtype=np.float32,
    )
    tolerance = 0.18 * max(boxes[0][2], boxes[0][3])
    if float(np.max(np.linalg.norm(centers - centers[0], axis=1))) > tolerance:
        return None
    outer, middle, inner = (
        max(width, height) for _x, _y, width, height in boxes
    )
    if not (
        0.30 <= middle / max(1.0, outer) <= 0.88
        and 0.30 <= inner / max(1.0, middle) <= 0.88
    ):
        return None
    return float(centers[0][0]), float(centers[0][1])


def local_qr_finder_centers(
    gray: np.ndarray,
    quad: np.ndarray,
) -> list[tuple[float, float]]:
    """Return independent nested-finder centers in a rectified QR candidate."""
    patch = square_warp(gray, quad, 220)
    masks = (
        cv2.threshold(
            patch,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1],
        cv2.adaptiveThreshold(
            patch,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        ),
    )
    best: list[tuple[float, float]] = []
    for mask in masks:
        contours, raw_hierarchy = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if raw_hierarchy is None:
            continue
        hierarchy = raw_hierarchy[0]
        centers: list[np.ndarray] = []
        for index in range(len(contours)):
            center = local_qr_finder_chain(contours, hierarchy, index)
            if center is None:
                continue
            point = np.asarray(center, dtype=np.float32)
            if not any(float(np.linalg.norm(point - prior)) < 12.0 for prior in centers):
                centers.append(point)
        if len(centers) > len(best):
            best = [
                (float(point[0]), float(point[1]))
                for point in centers
            ]
    return best


def local_qr_finder_count(gray: np.ndarray, quad: np.ndarray) -> int:
    return len(local_qr_finder_centers(gray, quad))


def qr_finder_ratio_hits(gray: np.ndarray, quad: np.ndarray) -> int:
    """Count scanline evidence for QR's black/white 1:1:3:1:1 finder ratio."""
    patch = square_warp(gray, quad, 210)
    masks = (
        cv2.threshold(
            patch,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1] > 0,
        cv2.adaptiveThreshold(
            patch,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        ) > 0,
    )
    best = 0
    for mask in masks:
        hits = 0
        for line in (*mask, *mask.T):
            signal = np.asarray(line, dtype=np.uint8)
            changes = np.flatnonzero(np.diff(signal) != 0) + 1
            starts = np.concatenate(([0], changes))
            ends = np.concatenate((changes, [len(signal)]))
            lengths = ends - starts
            values = signal[starts]
            for index in range(len(lengths) - 4):
                if tuple(values[index:index + 5]) != (1, 0, 1, 0, 1):
                    continue
                runs = lengths[index:index + 5].astype(np.float32)
                unit = float(np.mean(runs[[0, 1, 3, 4]]))
                if (
                    unit >= 1.0
                    and 2.0 * unit <= runs[2] <= 4.5 * unit
                    and float(np.max(np.abs(runs[[0, 1, 3, 4]] - unit))) <= 0.9 * unit
                ):
                    hits += 1
        best = max(best, hits)
    return best


def locate_qr_in_candidate(
    gray: np.ndarray,
    rough_quad: np.ndarray,
) -> Detection | None:
    x, y, width, height = cv2.boundingRect(np.asarray(rough_quad, dtype=np.int32))
    detector = getattr(DETECTOR_LOCAL, "qr_detector", None)
    if detector is None:
        detector = cv2.QRCodeDetector()
        DETECTOR_LOCAL.qr_detector = detector
    # A tight crop is best for small complete proposals; a wider crop recovers
    # QR proposals that contain only the busy center and omit all three finders.
    for expansion_index, expansion in enumerate((0.30, 1.0)):
        padding = int(round(expansion * max(width, height)))
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2 = min(gray.shape[1], x + width + padding)
        y2 = min(gray.shape[0], y + height + padding)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # Preserve the proven grayscale -> Otsu -> adaptive order, but create
        # each threshold view only if all cheaper preceding views failed. A
        # contour prerequisite was tested here and rejected: degraded QR codes
        # in VN02 and Kaggle can gain their first visible finder only after the
        # detector's perspective proposal, so pre-screening them is not safe.
        for view_index in range(3):
            if view_index == 0:
                view = crop
            elif view_index == 1:
                view = cv2.threshold(
                    crop,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )[1]
            else:
                view = cv2.adaptiveThreshold(
                    crop,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    5,
                )
            try:
                found, points = detector.detect(view)
            except cv2.error:
                found, points = False, None
            if not found or points is None:
                continue
            quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
            quad[:, 0] += x1
            quad[:, 1] += y1
            candidate = Detection(quad, "qr", 0.0, "opencv:qr-local-finder")
            long_side, short_side = candidate.long_short()
            if (
                candidate.area() < 100.0
                or short_side < 10.0
                or long_side / max(EPS, short_side) > 1.8
                or quad_overlap(quad, rough_quad) < 0.20
            ):
                continue
            finder_count = local_qr_finder_count(gray, quad)
            ratio_hits = qr_finder_ratio_hits(gray, quad)
            # Three nested finders are conclusive. On heavily damaged scans,
            # one or two contours may survive; require abundant independent
            # 1:1:3:1:1 scanline evidence in that case.
            if finder_count < 3 and not (finder_count >= 1 and ratio_hits >= 180):
                continue
            candidate.confidence = 0.95
            candidate.metrics = {
                "finder_patterns": float(finder_count),
                "finder_ratio_hits": float(ratio_hits),
                "local_threshold_view": float(view_index),
                "local_expansion": float(expansion_index),
            }
            return candidate
    return None


def locate_data_matrix(
    gray: np.ndarray,
    covered: Sequence[Detection],
    work_size: int,
    stop_after_first: bool = False,
    include_rejected: bool = False,
) -> tuple[list[Detection], list[Detection], dict[str, Any]]:
    height, width = gray.shape
    scale = min(1.0, float(work_size) / max(height, width))
    work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    gradient_x = np.abs(cv2.Scharr(work, cv2.CV_32F, 1, 0))
    gradient_y = np.abs(cv2.Scharr(work, cv2.CV_32F, 0, 1))
    raw: list[Detection] = []
    contour_count = 0
    responses: list[tuple[int, np.ndarray]] = []
    evaluated_geometry: set[tuple[float, ...]] = set()

    def consume(local_size: int, response: np.ndarray, percentiles: Sequence[float]) -> bool:
        nonlocal contour_count
        contours: list[np.ndarray] = []
        thresholds = np.atleast_1d(np.percentile(response, percentiles))
        for threshold in thresholds:
            mask = (response >= threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
            level_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(level_contours)
        contour_count += len(contours)
        for contour in contours:
            rectangle = cv2.minAreaRect(contour)
            candidate_width, candidate_height = (float(value) for value in rectangle[1])
            short_side, long_side = min(candidate_width, candidate_height), max(candidate_width, candidate_height)
            if short_side < 14 or long_side > 360 or long_side / max(1.0, short_side) > 4.5:
                continue
            quad = order_quad(cv2.boxPoints(rectangle).astype(np.float32) / scale)
            if any(quad_overlap(quad, item.quad) >= 0.20 for item in covered):
                continue
            # Nested percentile masks frequently return the exact same
            # rectangle. Perspective rectification and proposal metrics are
            # deterministic for identical geometry, so evaluate it once.
            geometry_key = tuple(
                round(float(coordinate), 3)
                for coordinate in quad.reshape(-1)
            )
            if geometry_key in evaluated_geometry:
                continue
            evaluated_geometry.add(geometry_key)
            metrics = matrix_proposal_metrics(gray, quad)
            if not metrics["accepted"]:
                continue
            raw.append(
                Detection(
                    quad=quad,
                    # This remains a proposal until a format-specific physical
                    # verifier accepts it below.
                    kind="matrix_2d",
                    confidence=float(metrics["score"]),
                    source=f"opencv:2d-energy-proposal:l{local_size}",
                    metrics={key: value for key, value in metrics.items() if key != "accepted"},
                )
            )
            if stop_after_first:
                return True
        return False

    for local_size in (9, 17, 29):
        energy_x = cv2.boxFilter(gradient_x, cv2.CV_32F, (local_size, local_size), normalize=True)
        energy_y = cv2.boxFilter(gradient_y, cv2.CV_32F, (local_size, local_size), normalize=True)
        response = np.sqrt(energy_x * energy_y)
        responses.append((local_size, response))
        found_evidence = consume(local_size, response, (99.55, 99.25, 99.0))
        if found_evidence:
            proposals = deduplicate(raw)
            return proposals, [], {
                "matrix_contours": contour_count,
                "matrix_unique_geometry": len(evaluated_geometry),
                "matrix_proposals": len(proposals),
                "accepted_matrix_2d": 0,
                "recovered_local_qr": 0,
                "matrix_dense_retry": 0,
            }

    proposals = deduplicate(raw)
    # Closely packed pages need one wider cut.  Sparse/empty pages skip it,
    # which keeps the common path fast and avoids widening the false-positive
    # surface on ordinary forms.
    dense_retry = len(covered) + len(proposals) >= 6
    if dense_retry:
        for local_size, response in responses:
            consume(local_size, response, (98.75,))
        proposals = deduplicate(raw)

    matrices: list[Detection] = []
    local_qr: list[Detection] = []
    rejected_candidates: list[Detection] = []
    rejected = 0
    hypotheses_examined = 0
    for proposal in proposals:
        # QR has a highly specific three-finder signature and is cheap to test
        # locally. Resolve it first so QR regions never pay for the more
        # expensive Data Matrix border/grid search.
        qr_detection = locate_qr_in_candidate(gray, proposal.quad)
        if qr_detection is not None:
            local_qr.append(qr_detection)
            continue
        refined_quad, metrics = refine_data_matrix_candidate(gray, proposal.quad)
        hypotheses_examined += int(metrics.get("border_hypotheses", 0.0))
        if refined_quad is not None:
            matrices.append(
                Detection(
                    quad=refined_quad,
                    kind="matrix_2d",
                    confidence=float(metrics["grid_score"]),
                    source="opencv:data-matrix-finder-grid",
                    metrics=metrics,
                )
            )
            continue
        if include_rejected:
            rejected_candidates.append(
                Detection(
                    quad=np.asarray(proposal.quad, dtype=np.float32),
                    kind="matrix_2d_rejected",
                    confidence=float(metrics.get("grid_score", 0.0)),
                    source="opencv:data-matrix-damaged-grid",
                    metrics={
                        **{
                            f"proposal_{key}": float(value)
                            for key, value in proposal.metrics.items()
                            if isinstance(value, (int, float, np.number))
                        },
                        **{
                            key: float(value)
                            for key, value in metrics.items()
                            if isinstance(value, (int, float, np.number))
                        },
                    },
                )
            )
        rejected += 1

    matrices = deduplicate(matrices)
    local_qr = deduplicate(local_qr)
    info: dict[str, Any] = {
        "matrix_contours": contour_count,
        "matrix_unique_geometry": len(evaluated_geometry),
        "matrix_proposals": len(proposals),
        "matrix_border_hypotheses": hypotheses_examined,
        "rejected_matrix_proposals": rejected,
        "accepted_matrix_2d": len(matrices),
        "recovered_local_qr": len(local_qr),
        "matrix_dense_retry": int(dense_retry),
    }
    if include_rejected:
        info["_rejected_candidates"] = rejected_candidates
    return matrices, local_qr, info


def annotate(gray: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = {"linear": (0, 180, 0), "qr": (220, 70, 20), "matrix_2d": (180, 0, 180)}
    for index, detection in enumerate(detections, start=1):
        points = np.rint(detection.quad).astype(np.int32)
        color = colors[detection.kind]
        cv2.polylines(output, [points], True, color, 4, cv2.LINE_AA)
        origin = tuple(points[np.argmin(points[:, 0] + points[:, 1])])
        cv2.putText(
            output,
            f"{index}:{detection.kind}",
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def write_crops(gray: np.ndarray, detections: Sequence[Detection], directory: Path, page: int) -> list[str]:
    paths: list[str] = []
    directory.mkdir(parents=True, exist_ok=True)
    for index, detection in enumerate(detections, start=1):
        destination = directory / f"page-{page:04d}-{index:02d}-{detection.kind}.png"
        cv2.imwrite(str(destination), rectify(gray, detection.quad))
        paths.append(destination.name)
    return paths


def locate_regions(
    gray: np.ndarray,
    config: Config,
    *,
    linear_work: np.ndarray | None = None,
    external_linear_evidence: Sequence[Detection] | None = None,
) -> tuple[list[Detection], dict[str, Any], dict[str, float]]:
    """Localize barcode regions in one grayscale image.

    This is the shared in-memory entry point used by both the standalone
    benchmark and the application. File loading and artifact generation intentionally
    remain outside the measured localization interval.
    """
    started = perf_counter()
    timings: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    linear: list[Detection] = []
    qr: list[Detection] = []
    matrices: list[Detection] = []
    recovered_local_qr: list[Detection] = []

    if config.kinds in {"linear", "all"}:
        stage = perf_counter()
        linear, info, stage_timings = locate_linear(
            gray,
            config,
            prepared_work=linear_work,
        )
        timings["linear_seconds"] = perf_counter() - stage
        timings.update(stage_timings)
        diagnostics.update(info)

    if config.kinds in {"2d", "all"}:
        run_full_matrix = True
        run_full_qr = True
        gate_evaluated = False
        external_gate_linear_evidence = (
            config.kinds == "2d"
            and external_linear_evidence is not None
            and bool(external_linear_evidence)
        )
        gate_linear_evidence = bool(linear) or external_gate_linear_evidence
        gate_work_size = max(
            config.empty_gate_matrix_work_size,
            config.empty_gate_qr_work_size,
        )
        explicit_2d_cascade = (
            config.empty_page_gate
            and config.kinds == "2d"
            and max(gray.shape) > gate_work_size
        )

        # Mixed-format mode can reuse final linear results as evidence. In
        # explicit 2-D mode, run the coarse 2-D screens first; dense positive
        # pages can escalate immediately without paying for a hidden linear
        # scan. Only a double-negative 2-D decision unlocks that final safety
        # check, which protects small symbols on mixed barcode pages. When a
        # page already fits inside both gate work sizes, the screen is not
        # coarse and cannot save pixel work, so preserve the original direct
        # path instead.
        should_screen_2d = (
            config.empty_page_gate
            and (
                (explicit_2d_cascade and not gate_linear_evidence)
                or (config.kinds == "all" and not gate_linear_evidence)
            )
        )
        if external_gate_linear_evidence:
            diagnostics["empty_gate_linear_source"] = "external"
        if should_screen_2d:
            gate_evaluated = True
            stage = perf_counter()
            coarse_matrices, _coarse_qr, coarse_info = locate_data_matrix(
                gray,
                (),
                config.empty_gate_matrix_work_size,
                stop_after_first=True,
            )
            finder_evidence = False
            finder_info: dict[str, Any] = {"quick_qr_gate_evaluated": 0}
            if not coarse_matrices:
                finder_evidence, finder_info = quick_qr_finder_evidence(
                    gray,
                    config.empty_gate_qr_work_size,
                )
                finder_info["quick_qr_gate_evaluated"] = 1
            diagnostics.update(
                {f"empty_gate_{key}": value for key, value in coarse_info.items()}
            )
            diagnostics.update(finder_info)
            two_d_evidence = bool(coarse_matrices) or finder_evidence
            if not two_d_evidence and config.kinds == "2d":
                linear_stage = perf_counter()
                if external_linear_evidence is None:
                    (
                        gate_linear,
                        gate_linear_info,
                        _gate_linear_timings,
                    ) = locate_linear(gray, config)
                    gate_linear_source = "pipeline-7"
                else:
                    gate_linear = list(external_linear_evidence)
                    gate_linear_info = {}
                    gate_linear_source = "external"
                timings["empty_gate_linear_seconds"] = perf_counter() - linear_stage
                gate_linear_evidence = bool(gate_linear)
                diagnostics.update(
                    {
                        f"empty_gate_linear_{key}": value
                        for key, value in gate_linear_info.items()
                    }
                )
                diagnostics["empty_gate_linear_source"] = gate_linear_source
                two_d_evidence = gate_linear_evidence
            timings["empty_page_gate_seconds"] = perf_counter() - stage
            run_full_matrix = two_d_evidence
            run_full_qr = two_d_evidence
        diagnostics["empty_gate_linear_evidence"] = int(gate_linear_evidence)

        diagnostics["empty_page_gate_enabled"] = int(config.empty_page_gate)
        diagnostics["empty_page_gate_evaluated"] = int(gate_evaluated)
        diagnostics["matrix_stage_skipped_by_empty_gate"] = int(not run_full_matrix)
        diagnostics["qr_stage_skipped_by_empty_gate"] = int(not run_full_qr)
        diagnostics["two_d_stages_skipped_by_empty_gate"] = int(
            not run_full_matrix and not run_full_qr
        )

        if run_full_matrix:
            stage = perf_counter()
            matrices, recovered_local_qr, info = locate_data_matrix(
                gray,
                linear,
                config.matrix_work_size,
            )
            timings["data_matrix_seconds"] = perf_counter() - stage
            diagnostics.update(info)
        else:
            timings["data_matrix_seconds"] = 0.0
            diagnostics.update(
                {
                    "matrix_contours": 0,
                    "matrix_unique_geometry": 0,
                    "matrix_proposals": 0,
                    "matrix_border_hypotheses": 0,
                    "rejected_matrix_proposals": 0,
                    "accepted_matrix_2d": 0,
                    "recovered_local_qr": 0,
                    "matrix_dense_retry": 0,
                }
            )

        # Small explicit-2D inputs deliberately retain the original ordering:
        # run the native matrix pass once, then use the finder screen only to
        # decide whether the final QR detector is justified.
        if (
            config.empty_page_gate
            and config.kinds == "2d"
            and not explicit_2d_cascade
            and not matrices
            and not recovered_local_qr
        ):
            gate_evaluated = True
            stage = perf_counter()
            finder_evidence, finder_info = quick_qr_finder_evidence(
                gray,
                config.empty_gate_qr_work_size,
            )
            timings["empty_page_gate_seconds"] = (
                timings.get("empty_page_gate_seconds", 0.0)
                + perf_counter()
                - stage
            )
            diagnostics.update(finder_info)
            run_full_qr = finder_evidence
            diagnostics["empty_page_gate_evaluated"] = 1
            diagnostics["qr_stage_skipped_by_empty_gate"] = int(not run_full_qr)

        if run_full_qr:
            stage = perf_counter()
            qr = deduplicate([*recovered_local_qr, *locate_qr(gray)])
            timings["qr_seconds"] = perf_counter() - stage
            # Matrix proposals are intentionally format-neutral. Replace any
            # proposal overlapping a confirmed QR location with the more
            # specific QR result, preventing duplicate cross-kind output.
            matrices = [
                matrix
                for matrix in matrices
                if not any(quad_overlap(matrix.quad, item.quad) >= 0.20 for item in qr)
            ]
        else:
            qr = recovered_local_qr
            timings["qr_seconds"] = 0.0
        diagnostics["accepted_qr"] = len(qr)
        diagnostics["accepted_matrix_2d"] = len(matrices)

    detections = [*linear, *qr, *matrices]
    detections = deduplicate(detections)
    detections.sort(key=lambda item: (round(float(item.center()[1]) / 20.0), float(item.center()[0])))
    timings["localization_seconds"] = perf_counter() - started
    return detections, diagnostics, timings


def process_page(
    page: int,
    path: Path,
    source_mode: str,
    config: Config,
) -> tuple[dict[str, Any], list[Detection], Path]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"unable to read image: {path}")
    detections, diagnostics, timings = locate_regions(gray, config)
    counts = {kind: sum(item.kind == kind for item in detections) for kind in ("linear", "qr", "matrix_2d")}
    payload = {
        "page": page,
        "source_mode": source_mode,
        "image_size": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
        "count": len(detections),
        "counts": counts,
        "detections": [item.to_json() for item in detections],
        "diagnostics": diagnostics,
        "timings": {key: round(value, 6) for key, value in timings.items()},
        "overlay": None,
        "crops": [],
    }
    return payload, detections, path


def generate_artifacts(
    payload: dict[str, Any],
    detections: Sequence[Detection],
    source_path: Path,
    config: Config,
    work: Path,
) -> None:
    started = perf_counter()
    gray = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"unable to reload image for artifacts: {source_path}")
    page = int(payload["page"])
    if config.save_overlays:
        overlay_path = work / "overlays" / f"page-{page:04d}.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), annotate(gray, detections), [cv2.IMWRITE_JPEG_QUALITY, 88])
        payload["overlay"] = str(overlay_path.relative_to(work))
    if config.save_crops:
        crop_directory = work / "crops"
        payload["crops"] = [f"crops/{name}" for name in write_crops(gray, detections, crop_directory, page)]
    payload["timings"]["artifact_seconds"] = round(perf_counter() - started, 6)


def pdf_page_count(path: Path) -> int:
    completed = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
    match = re.search(r"^Pages:\s+(\d+)", completed.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("unable to determine PDF page count")
    return int(match.group(1))


def image_number(path: Path) -> int:
    match = re.search(r"-(\d+)(?:\.[^.]+)?$", path.name)
    return int(match.group(1)) if match else 10**9


def pdf_image_rows(path: Path) -> list[dict[str, int]]:
    completed = subprocess.run(["pdfimages", "-list", str(path)], check=True, capture_output=True, text=True)
    rows: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        rows.append(
            {"page": int(parts[0]), "number": int(parts[1]), "width": int(parts[3]), "height": int(parts[4])}
        )
    return rows


def extract_pdf(path: Path, selected: Sequence[int], work: Path, dpi: int) -> list[tuple[int, Path, str]]:
    extraction = work / ".native"
    pages_directory = work / ".pages"
    extraction.mkdir(parents=True, exist_ok=True)
    pages_directory.mkdir(parents=True, exist_ok=True)
    first, last = min(selected), max(selected)
    rows = [row for row in pdf_image_rows(path) if first <= row["page"] <= last]
    prefix = extraction / "image"
    try:
        subprocess.run(
            ["pdfimages", "-f", str(first), "-l", str(last), "-j", str(path), str(prefix)],
            check=True,
            capture_output=True,
        )
        files = sorted((item for item in extraction.glob("image-*") if item.is_file()), key=image_number)
        if len(files) != len(rows):
            raise RuntimeError("native image manifest mismatch")
        best: dict[int, tuple[int, Path]] = {}
        for row, image_path in zip(rows, files):
            area = row["width"] * row["height"]
            if row["page"] in selected and area >= 1_000_000 and area > best.get(row["page"], (0, image_path))[0]:
                best[row["page"]] = (area, image_path)
        output: list[tuple[int, Path, str]] = []
        for page in selected:
            if page in best:
                source = best[page][1]
                destination = pages_directory / f"page-{page:04d}{source.suffix.lower()}"
                shutil.copy2(source, destination)
                output.append((page, destination, "native-embedded-raster"))
                continue
            prefix_path = pages_directory / f"render-{page:04d}"
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-r",
                    str(dpi),
                    "-gray",
                    "-jpeg",
                    "-singlefile",
                    str(path),
                    str(prefix_path),
                ],
                check=True,
                capture_output=True,
            )
            output.append((page, prefix_path.with_suffix(".jpg"), f"rendered-{dpi}-dpi"))
        return output
    finally:
        shutil.rmtree(extraction, ignore_errors=True)


def collect_pages(path: Path, config: Config, work: Path) -> list[tuple[int, Path, str]]:
    if path.is_dir():
        images = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
        selected = config.pages or list(range(1, len(images) + 1))
        if any(page > len(images) for page in selected):
            raise ValueError("selected page exceeds image count")
        return [(page, images[page - 1], "image-directory") for page in selected]
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        if config.pages and config.pages != [1]:
            raise ValueError("a single image only has page 1")
        return [(1, path, "image-file")]
    if path.suffix.lower() != ".pdf":
        raise ValueError("input must be a PDF, an image, or an image directory")
    page_count = pdf_page_count(path)
    selected = config.pages or list(range(1, page_count + 1))
    if any(page > page_count for page in selected):
        raise ValueError(f"selected page exceeds PDF page count {page_count}")
    return extract_pdf(path, selected, work, config.render_dpi)


def effective_workers(requested: int, page_count: int) -> int:
    if requested > 0:
        return max(1, min(requested, page_count))
    return max(1, min(8, page_count, os.cpu_count() or 1))


def run_pipeline(input_path: Path, config: Config) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output = config.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output}")
    work = output.parent / f".{output.name}.work-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True)
    wall_started = perf_counter()
    try:
        extraction_started = perf_counter()
        pages = collect_pages(input_path, config, work)
        extraction_seconds = perf_counter() - extraction_started
        workers = effective_workers(config.workers, len(pages))
        if config.workers == 0 and config.kinds != "linear":
            # Matrix response maps are memory-bandwidth heavy; beyond four
            # concurrent pages this machine slowed down instead of speeding up.
            workers = min(workers, 4)
        # OpenCV already parallelizes detector internals. With one page, those
        # native threads minimize latency. With several concurrent pages, one
        # native thread per task avoids nested oversubscription.
        opencv_threads = (
            1
            if workers > 1 and config.kinds == "linear"
            else DEFAULT_OPENCV_THREADS
        )
        cv2.setNumThreads(opencv_threads)
        processing_started = perf_counter()
        if workers == 1:
            processed = [process_page(page, path, mode, config) for page, path, mode in pages]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="localize") as executor:
                futures = [executor.submit(process_page, page, path, mode, config) for page, path, mode in pages]
                processed = [future.result() for future in futures]
        processing_wall_seconds = perf_counter() - processing_started
        artifact_wall_seconds = 0.0
        if config.save_overlays or config.save_crops:
            artifact_started = perf_counter()
            if workers == 1:
                for page_payload, detections, source_path in processed:
                    generate_artifacts(page_payload, detections, source_path, config, work)
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="artifact") as executor:
                    futures = [
                        executor.submit(generate_artifacts, page_payload, detections, source_path, config, work)
                        for page_payload, detections, source_path in processed
                    ]
                    for future in futures:
                        future.result()
            artifact_wall_seconds = perf_counter() - artifact_started
        payloads = [item[0] for item in processed]
        payloads.sort(key=lambda item: item["page"])
        for page in payloads:
            print(
                f"page {page['page']:04d}: {page['count']} localized, "
                f"localization={page['timings']['localization_seconds']:.3f}s"
                + (" (overlaps other page tasks)" if workers > 1 else ""),
                flush=True,
            )
        shutil.rmtree(work / ".pages", ignore_errors=True)
        counts = {
            kind: sum(page["counts"][kind] for page in payloads)
            for kind in ("linear", "qr", "matrix_2d")
        }
        resolved_linear_scale, resolved_linear_scales = linear_profile(config)
        timing_keys = sorted(
            {
                key
                for page in payloads
                for key in page["timings"]
                if key != "artifact_seconds"
            }
        )
        stage_seconds_sum = {
            key: round(sum(float(page["timings"].get(key, 0.0)) for page in payloads), 6)
            for key in timing_keys
        }
        payload: dict[str, Any] = {
            "input": str(input_path),
            "method": "CPU coarse proposal plus native-pixel deterministic verification; no decoding",
            "decoding_performed": False,
            "gpu_used": False,
            "configuration": {
                "kinds": config.kinds,
                "workers": workers,
                "opencv_threads": opencv_threads,
                "render_dpi": config.render_dpi,
                "minimum_linear_score": config.minimum_linear_score,
                "minimum_linear_length": config.minimum_linear_length,
                "linear_profile": config.linear_profile,
                "linear_work_scale": resolved_linear_scale,
                "linear_detector_scales": list(resolved_linear_scales),
                "matrix_work_size": config.matrix_work_size,
                "empty_page_gate": config.empty_page_gate,
                "empty_gate_matrix_work_size": config.empty_gate_matrix_work_size,
                "empty_gate_qr_work_size": config.empty_gate_qr_work_size,
                "save_overlays": config.save_overlays,
                "save_crops": config.save_crops,
            },
            "summary": {
                "pages": len(payloads),
                "localized": sum(page["count"] for page in payloads),
                "counts": counts,
                "extraction_seconds": round(extraction_seconds, 6),
                "processing_wall_seconds": round(processing_wall_seconds, 6),
                "artifact_wall_seconds": round(artifact_wall_seconds, 6),
                "page_localization_seconds_sum": round(
                    sum(page["timings"]["localization_seconds"] for page in payloads), 6
                ),
                "page_artifact_seconds_sum": round(
                    sum(page["timings"].get("artifact_seconds", 0.0) for page in payloads), 6
                ),
                "stage_seconds_sum": stage_seconds_sum,
                "empty_page_gate_evaluated_pages": sum(
                    int(page["diagnostics"].get("empty_page_gate_evaluated", 0))
                    for page in payloads
                ),
                "qr_stage_skipped_pages": sum(
                    int(page["diagnostics"].get("qr_stage_skipped_by_empty_gate", 0))
                    for page in payloads
                ),
                "matrix_stage_skipped_pages": sum(
                    int(page["diagnostics"].get("matrix_stage_skipped_by_empty_gate", 0))
                    for page in payloads
                ),
                "all_2d_stages_skipped_pages": sum(
                    int(page["diagnostics"].get("two_d_stages_skipped_by_empty_gate", 0))
                    for page in payloads
                ),
                "average_processing_wall_seconds_per_page": round(processing_wall_seconds / max(1, len(payloads)), 6),
                "wall_seconds": round(perf_counter() - wall_started, 6),
            },
            "pages": payloads,
        }
        (work / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        work.rename(output)
        print(
            f"summary: {payload['summary']['localized']} localized across {len(payloads)} pages; "
            f"processing wall={processing_wall_seconds:.3f}s "
            f"({processing_wall_seconds / max(1, len(payloads)):.3f}s/page), "
            f"artifacts={artifact_wall_seconds:.3f}s, workers={workers}",
            flush=True,
        )
        print(f"wrote {output / 'detections.json'}", flush=True)
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
        default=Path("output/barcode-detection/classical-localizer/run"),
    )
    parser.add_argument("--kinds", choices=("linear", "2d", "all"), default="linear")
    parser.add_argument("--pages", type=parse_page_selection)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--workers", type=int, default=0, help="0=automatic; use 1 for minimum single-page latency")
    parser.add_argument("--minimum-linear-score", type=float, default=0.52)
    parser.add_argument("--minimum-linear-length", type=float, default=100.0)
    parser.add_argument(
        "--linear-profile",
        choices=tuple(LINEAR_PROFILES),
        default="universal",
        help="universal=validated default; fast=aggressive opt-in; legacy=all original detector scales",
    )
    parser.add_argument(
        "--linear-work-scale",
        type=float,
        default=None,
        help="override the profile proposal scale; verification still uses original pixels",
    )
    parser.add_argument("--matrix-work-size", type=int, default=1400)
    parser.add_argument(
        "--no-empty-page-gate",
        dest="empty_page_gate",
        action="store_false",
        help="always run the full matrix and QR stages instead of using the fast-negative cascade",
    )
    parser.add_argument(
        "--empty-gate-matrix-work-size",
        type=int,
        default=700,
        help="longest thumbnail side used by the conservative matrix-energy gate",
    )
    parser.add_argument(
        "--empty-gate-qr-work-size",
        type=int,
        default=900,
        help="longest thumbnail side used by the conservative QR finder-pattern gate",
    )
    parser.add_argument("--overlays", dest="save_overlays", action="store_true")
    parser.add_argument("--crops", dest="save_crops", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = Config(
        output=arguments.output,
        kinds=arguments.kinds,
        pages=arguments.pages,
        render_dpi=arguments.render_dpi,
        workers=arguments.workers,
        overwrite=arguments.overwrite,
        save_overlays=arguments.save_overlays,
        save_crops=arguments.save_crops,
        minimum_linear_score=arguments.minimum_linear_score,
        minimum_linear_length=arguments.minimum_linear_length,
        linear_profile=arguments.linear_profile,
        linear_work_scale=arguments.linear_work_scale,
        matrix_work_size=arguments.matrix_work_size,
        empty_page_gate=arguments.empty_page_gate,
        empty_gate_matrix_work_size=arguments.empty_gate_matrix_work_size,
        empty_gate_qr_work_size=arguments.empty_gate_qr_work_size,
    )
    try:
        run_pipeline(arguments.input, config)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
