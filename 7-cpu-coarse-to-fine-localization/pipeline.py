#!/usr/bin/env python3
"""CPU-optimized, deterministic barcode localization without decoding.

The pipeline deliberately has no barcode decoder dependency and never attempts
to recover a payload. Its default linear path is coarse-to-fine: OpenCV's
classical directional-coherence detector proposes regions on a reduced image
using an ablated scale band, then strict structural verification reads only the
small proposed regions from the native pixels. QR symbols use OpenCV's
detector-only API, while Data Matrix-like symbols use a small
bidirectional-energy/L-border verifier. No GPU is used.
"""

from __future__ import annotations

import argparse
import json
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
) -> tuple[list[Detection], dict[str, Any], dict[str, float]]:
    scale, scales = linear_profile(config)
    resize_started = perf_counter()
    work = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    resize_seconds = perf_counter() - resize_started
    detector = linear_detector(config.linear_profile, scales)
    detector.setDownsamplingThreshold(float(max(work.shape[:2])))
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


def matrix_metrics(gray: np.ndarray, quad: np.ndarray) -> dict[str, float]:
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


def locate_data_matrix(
    gray: np.ndarray,
    covered: Sequence[Detection],
    work_size: int,
    stop_after_first: bool = False,
) -> tuple[list[Detection], dict[str, int]]:
    height, width = gray.shape
    scale = min(1.0, float(work_size) / max(height, width))
    work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    gradient_x = np.abs(cv2.Scharr(work, cv2.CV_32F, 1, 0))
    gradient_y = np.abs(cv2.Scharr(work, cv2.CV_32F, 0, 1))
    raw: list[Detection] = []
    contour_count = 0
    responses: list[tuple[int, np.ndarray]] = []

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
            if short_side < 14 or long_side > 360 or long_side / max(1.0, short_side) > 2.5:
                continue
            quad = cv2.boxPoints(rectangle).astype(np.float32) / scale
            if any(quad_overlap(quad, item.quad) >= 0.20 for item in covered):
                continue
            metrics = matrix_metrics(gray, quad)
            if not metrics["accepted"]:
                continue
            raw.append(
                Detection(
                    quad=quad,
                    # Energy/grid evidence proves a 2-D matrix region, but
                    # without decoding it cannot safely name the symbology.
                    kind="matrix_2d",
                    confidence=float(metrics["score"]),
                    source=f"opencv:matrix-energy:l{local_size}",
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
            accepted = deduplicate(raw)
            return accepted, {
                "matrix_contours": contour_count,
                "accepted_matrix_2d": len(accepted),
                "matrix_dense_retry": 0,
            }

    accepted = deduplicate(raw)
    # Closely packed pages need one wider cut.  Sparse/empty pages skip it,
    # which keeps the common path fast and avoids widening the false-positive
    # surface on ordinary forms.
    dense_retry = len(covered) + len(accepted) >= 6
    if dense_retry:
        for local_size, response in responses:
            consume(local_size, response, (98.75,))
        accepted = deduplicate(raw)
    return accepted, {
        "matrix_contours": contour_count,
        "accepted_matrix_2d": len(accepted),
        "matrix_dense_retry": int(dense_retry),
    }


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


def process_page(
    page: int,
    path: Path,
    source_mode: str,
    config: Config,
) -> tuple[dict[str, Any], list[Detection], Path]:
    started = perf_counter()
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"unable to read image: {path}")
    timings: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    linear: list[Detection] = []
    qr: list[Detection] = []
    matrices: list[Detection] = []

    if config.kinds in {"linear", "all"}:
        stage = perf_counter()
        linear, info, stage_timings = locate_linear(gray, config)
        timings["linear_seconds"] = perf_counter() - stage
        timings.update(stage_timings)
        diagnostics.update(info)

    if config.kinds in {"2d", "all"}:
        run_full_matrix = True
        run_full_qr = True
        gate_evaluated = False

        # In mixed-format mode the validated linear pass is already complete.
        # If it is empty, ask two small, independent 2-D screens whether a
        # native matrix/QR scan is justified. Only a double-negative decision
        # can suppress both expensive stages. Explicit 2-D mode avoids this
        # additional matrix pass because its caller expects a 2-D workload.
        if config.empty_page_gate and config.kinds == "all" and not linear:
            gate_evaluated = True
            stage = perf_counter()
            coarse_matrices, coarse_info = locate_data_matrix(
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
            timings["empty_page_gate_seconds"] = perf_counter() - stage
            diagnostics.update(
                {f"empty_gate_{key}": value for key, value in coarse_info.items()}
            )
            diagnostics.update(finder_info)
            two_d_evidence = bool(coarse_matrices) or finder_evidence
            run_full_matrix = two_d_evidence
            run_full_qr = two_d_evidence

        diagnostics["empty_page_gate_enabled"] = int(config.empty_page_gate)
        diagnostics["empty_page_gate_evaluated"] = int(gate_evaluated)
        diagnostics["matrix_stage_skipped_by_empty_gate"] = int(not run_full_matrix)
        diagnostics["qr_stage_skipped_by_empty_gate"] = int(not run_full_qr)
        diagnostics["two_d_stages_skipped_by_empty_gate"] = int(
            not run_full_matrix and not run_full_qr
        )

        if run_full_matrix:
            stage = perf_counter()
            matrices, info = locate_data_matrix(gray, linear, config.matrix_work_size)
            timings["data_matrix_seconds"] = perf_counter() - stage
            diagnostics.update(info)
        else:
            timings["data_matrix_seconds"] = 0.0
            diagnostics.update(
                {
                    "matrix_contours": 0,
                    "accepted_matrix_2d": 0,
                    "matrix_dense_retry": 0,
                }
            )

        # In explicit 2-D mode the full matrix pass doubles as the first QR
        # screen. If it is empty, a cheap finder check can still suppress the
        # final native QR detector without duplicating matrix work.
        if config.empty_page_gate and config.kinds == "2d" and not matrices:
            gate_evaluated = True
            stage = perf_counter()
            finder_evidence, finder_info = quick_qr_finder_evidence(
                gray,
                config.empty_gate_qr_work_size,
            )
            timings["empty_page_gate_seconds"] = perf_counter() - stage
            diagnostics.update(finder_info)
            run_full_qr = finder_evidence
            diagnostics["empty_page_gate_evaluated"] = 1
            diagnostics["qr_stage_skipped_by_empty_gate"] = int(not run_full_qr)

        if run_full_qr:
            stage = perf_counter()
            qr = locate_qr(gray)
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
            timings["qr_seconds"] = 0.0
        diagnostics["accepted_qr"] = len(qr)
        diagnostics["accepted_matrix_2d"] = len(matrices)

    detections = [*linear, *qr, *matrices]
    detections = deduplicate(detections)
    detections.sort(key=lambda item: (round(float(item.center()[1]) / 20.0), float(item.center()[0])))
    timings["localization_seconds"] = perf_counter() - started
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
        default=Path("7-cpu-coarse-to-fine-localization/output/run"),
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
