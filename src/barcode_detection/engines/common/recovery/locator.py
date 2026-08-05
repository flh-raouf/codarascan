#!/usr/bin/env python3
"""Deterministic, rotation-tolerant barcode localization for document pages.

This program deliberately contains no learned model.  It combines classical
directional-coherence proposals, an independent structure-tensor fallback,
local barcode-geometry validation, and optional ZXing-C++ decoding.

The output keeps an oriented quadrilateral for every detection.  That matters:
an axis-aligned rectangle around a 45 degree label is mostly background and
neighbouring document content.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

try:
    import zxingcpp
except ImportError:  # The visual localizer remains usable without the decoder.
    zxingcpp = None


EPS = 1e-6
@dataclass
class Detection:
    """An oriented candidate in original image coordinates."""

    quad: np.ndarray
    sources: set[str] = field(default_factory=set)
    proposal_score: float = 0.0
    structural_score: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    decoded_text: str | None = None
    barcode_format: str | None = None
    kind: str = "linear"
    accepted: bool = False

    def center(self) -> np.ndarray:
        return np.asarray(self.quad, dtype=np.float32).mean(axis=0)

    def area(self) -> float:
        return float(abs(cv2.contourArea(np.asarray(self.quad, dtype=np.float32))))

    def long_short(self) -> tuple[float, float]:
        rect = cv2.minAreaRect(np.asarray(self.quad, dtype=np.float32))
        width, height = rect[1]
        return max(float(width), float(height)), min(float(width), float(height))

    def angle_degrees(self) -> float:
        """Direction of the barcode's long axis, modulo 180 degrees."""
        points = order_quad(self.quad)
        top = points[1] - points[0]
        left = points[3] - points[0]
        vector = top if np.linalg.norm(top) >= np.linalg.norm(left) else left
        return float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0)

    def to_json(self) -> dict[str, Any]:
        long_side, short_side = self.long_short()
        x_min, y_min = np.min(self.quad, axis=0)
        x_max, y_max = np.max(self.quad, axis=0)
        return {
            "kind": self.kind,
            "confidence": round(float(max(self.proposal_score, self.structural_score)), 4),
            "accepted": self.accepted,
            "sources": sorted(self.sources),
            "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in self.quad],
            "aabb": {
                "x": round(float(x_min), 2),
                "y": round(float(y_min), 2),
                "width": round(float(x_max - x_min), 2),
                "height": round(float(y_max - y_min), 2),
            },
            "long_axis_angle_degrees": round(self.angle_degrees(), 2),
            "long_side_pixels": round(long_side, 2),
            "short_side_pixels": round(short_side, 2),
            "decoded": {
                "text": self.decoded_text,
                "format": self.barcode_format,
            }
            if self.barcode_format
            else None,
            "structural_metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
        }


def as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def order_quad(points: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL for a convex quadrilateral.

    The sum/difference method is stable for the non-self-intersecting rotated
    rectangles this program produces.  We first reduce arbitrary detector
    points to an OpenCV rotated rectangle so their order is always convex.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) != 4:
        points = cv2.boxPoints(cv2.minAreaRect(points))
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()  # y - x
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    # Very narrow, nearly 45-degree boxes can make the sum method ambiguous.
    if len({tuple(point) for point in ordered}) != 4:
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        cycle = points[np.argsort(angles)]
        start = int(np.argmin(cycle.sum(axis=1)))
        ordered = np.roll(cycle, -start, axis=0)
        # The cycle may be TL, BL, BR, TR. Flip it into TL, TR, BR, BL.
        first_edge = ordered[1] - ordered[0]
        second_edge = ordered[2] - ordered[1]
        cross_z = float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])
        if cross_z > 0:
            ordered = ordered[[0, 3, 2, 1]]
    return ordered


def expand_quad(quad: np.ndarray, x_factor: float, y_factor: float) -> np.ndarray:
    """Expand a quad along its local long and short axes."""
    quad = order_quad(quad)
    width_a = np.linalg.norm(quad[1] - quad[0])
    width_b = np.linalg.norm(quad[2] - quad[3])
    height_a = np.linalg.norm(quad[3] - quad[0])
    height_b = np.linalg.norm(quad[2] - quad[1])
    width = max(width_a, width_b)
    height = max(height_a, height_b)
    if height > width:
        # Swap the semantics so x_factor always expands the barcode's long axis.
        x_factor, y_factor = y_factor, x_factor
    center = quad.mean(axis=0)
    vectors = quad - center
    # For a rectangle, independent x/y scaling in its basis is exactly an
    # isotropic scaling of vertex vectors only when factors match.  Build an
    # explicit local basis instead.
    long_axis = quad[1] - quad[0]
    short_axis = quad[3] - quad[0]
    if np.linalg.norm(long_axis) < np.linalg.norm(short_axis):
        long_axis, short_axis = short_axis, long_axis
    long_axis = long_axis / (np.linalg.norm(long_axis) + EPS)
    short_axis = short_axis / (np.linalg.norm(short_axis) + EPS)
    projected_long = vectors @ long_axis
    projected_short = vectors @ short_axis
    return center + np.outer(projected_long * x_factor, long_axis) + np.outer(projected_short * y_factor, short_axis)


def rectify_quad(
    image: np.ndarray, quad: np.ndarray, pad_long: float = 0.0, pad_short: float = 0.0
) -> np.ndarray:
    """Perspective-rectify a candidate; bars end up vertical in a wide crop."""
    if pad_long or pad_short:
        quad = expand_quad(quad, 1.0 + 2.0 * pad_long, 1.0 + 2.0 * pad_short)
    source = order_quad(quad)
    top = np.linalg.norm(source[1] - source[0])
    bottom = np.linalg.norm(source[2] - source[3])
    left = np.linalg.norm(source[3] - source[0])
    right = np.linalg.norm(source[2] - source[1])
    width = max(2, int(round(max(top, bottom))))
    height = max(2, int(round(max(left, right))))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    patch = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    if patch.shape[0] > patch.shape[1]:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    return patch


def quad_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over smaller area: better than IoU for nested proposals."""
    a = cv2.convexHull(np.asarray(a, dtype=np.float32)).reshape(-1, 2)
    b = cv2.convexHull(np.asarray(b, dtype=np.float32)).reshape(-1, 2)
    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))
    if area_a < EPS or area_b < EPS:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    return float(intersection) / min(float(area_a), float(area_b))


def similar_detection(a: Detection, b: Detection) -> bool:
    overlap = quad_overlap(a.quad, b.quad)
    if overlap >= 0.52:
        return True
    # The detector can return two slightly shifted, thin strips for one label.
    long_a, short_a = a.long_short()
    long_b, short_b = b.long_short()
    center_distance = float(np.linalg.norm(a.center() - b.center()))
    angles = abs(a.angle_degrees() - b.angle_degrees())
    angles = min(angles, 180.0 - angles)
    return (
        center_distance < 0.16 * min(long_a, long_b)
        and angles < 8.0
        and abs(short_a - short_b) < 0.9 * max(short_a, short_b)
    )


def merge_detection(existing: Detection, incoming: Detection) -> None:
    existing.sources.update(incoming.sources)
    existing.proposal_score = max(existing.proposal_score, incoming.proposal_score)
    if incoming.barcode_format and not existing.barcode_format:
        existing.quad = incoming.quad
        existing.decoded_text = incoming.decoded_text
        existing.barcode_format = incoming.barcode_format
        existing.kind = incoming.kind
    elif incoming.barcode_format and existing.barcode_format:
        # Prefer a decoder position when it encloses at least as much bar area.
        if incoming.area() > existing.area() * 0.72:
            existing.quad = incoming.quad
            existing.decoded_text = incoming.decoded_text
            existing.barcode_format = incoming.barcode_format
            existing.kind = incoming.kind


def deduplicate(detections: Iterable[Detection]) -> list[Detection]:
    # First keep decoder positions; they are exact semantic anchors.  Then use
    # largest structural proposal so a decoder's very tight quad cannot hide
    # the visible bar field.
    result: list[Detection] = []
    ranked = sorted(
        detections,
        key=lambda item: (bool(item.barcode_format), item.proposal_score, item.area()),
        reverse=True,
    )
    for detection in ranked:
        found = next((prior for prior in result if similar_detection(prior, detection)), None)
        if found is None:
            result.append(detection)
        else:
            merge_detection(found, detection)
    return result


def make_detector(scales: Sequence[float], image_shape: tuple[int, ...], gradient_threshold: float) -> Any:
    detector = cv2.barcode.BarcodeDetector()
    # The stock threshold is 512.  On a 200-ppi document page that would erase
    # the 2-4 pixel modules before the algorithm sees them.
    #
    # OpenCV asserts thresh >= 64, which a small input can fall under — a tightly
    # cropped region of interest, or a small uploaded image. Clamping preserves the
    # intent, since anything already smaller than 64 is below any legal threshold
    # and will not be downsampled either way.
    detector.setDownsamplingThreshold(max(64.0, float(max(image_shape[:2]))))
    detector.setGradientThreshold(float(gradient_threshold))
    detector.setDetectorScales(np.asarray(scales, dtype=np.float32))
    return detector


SCALE_PROFILES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("tiny", (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016)),
    ("document", (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)),
    ("wide", (0.003, 0.006, 0.01, 0.02, 0.04, 0.08)),
)


def detector_proposals(gray: np.ndarray, angle_step: int) -> list[Detection]:
    """Use OpenCV's non-ML directional-coherence locator at several scales.

    The native pass is the main proposal source.  Rotated auxiliary passes
    deliberately stop at 90 degrees: a 1-D barcode has 180-degree symmetry.
    They are a recall safety net for a weak label rather than the primary
    solution; local detector quads provide sub-degree final orientation.
    """
    proposals: list[Detection] = []

    def detect_one(
        image: np.ndarray, source: str, inverse_transform: np.ndarray | None = None, profiles: Sequence[tuple[str, Sequence[float]]] = SCALE_PROFILES
    ) -> None:
        for profile_name, scales in profiles:
            for threshold in (48.0, 64.0):
                detector = make_detector(scales, image.shape, threshold)
                try:
                    found, points = detector.detectMulti(image)
                except cv2.error:
                    continue
                if not found or points is None:
                    continue
                for quad in np.asarray(points, dtype=np.float32):
                    if inverse_transform is not None:
                        quad = cv2.transform(quad.reshape(1, -1, 2), inverse_transform)[0]
                    long_side, short_side = Detection(quad=quad).long_short()
                    if long_side < 18 or short_side < 5 or long_side / (short_side + EPS) < 1.15:
                        continue
                    proposals.append(
                        Detection(
                            quad=quad,
                            sources={f"opencv:{source}:{profile_name}:g{int(threshold)}"},
                            proposal_score=0.48,
                        )
                    )

    detect_one(gray, "native")

    # A small amount of explicit deskew gives very oblique, low-contrast labels
    # an axis-aligned chance. We do not use it to decide a final orientation.
    if angle_step > 0:
        for angle in range(angle_step, 90, angle_step):
            rotated, inverse = rotate_bound(gray, angle)
            # One broader profile keeps this fallback inexpensive and avoids
            # multiplying interpolation artefacts into many nearly identical boxes.
            detect_one(
                rotated,
                f"deskew{angle}",
                inverse,
                profiles=(("deskew", SCALE_PROFILES[1][1]),),
            )
    return proposals


def rotate_bound(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate without clipping content and return transform back to original."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    transform = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(transform[0, 0]), abs(transform[0, 1])
    bound_width = int(math.ceil(height * sine + width * cosine))
    bound_height = int(math.ceil(height * cosine + width * sine))
    transform[0, 2] += bound_width / 2.0 - center[0]
    transform[1, 2] += bound_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        transform,
        (bound_width, bound_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated, cv2.invertAffineTransform(transform)


def tensor_proposals(gray: np.ndarray, max_dimension: int = 1900) -> list[Detection]:
    """Independent rotation-invariant fallback built from the structure tensor.

    The response is high only when a neighbourhood contains *many* strong,
    similarly oriented gradients. Tables have long lines but low local edge
    density; text has high density but weak directional coherence.
    """
    height, width = gray.shape[:2]
    scale = min(1.0, max_dimension / float(max(height, width)))
    if scale < 1.0:
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = gray
    blur = cv2.GaussianBlur(work, (0, 0), 0.8)
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    proposals: list[Detection] = []

    for local_size, pool_size in ((7, 19), (13, 35), (21, 61)):
        cxx = cv2.boxFilter(gx * gx, cv2.CV_32F, (local_size, local_size), normalize=True)
        cyy = cv2.boxFilter(gy * gy, cv2.CV_32F, (local_size, local_size), normalize=True)
        cxy = cv2.boxFilter(gx * gy, cv2.CV_32F, (local_size, local_size), normalize=True)
        trace = cxx + cyy
        coherence = np.sqrt((cxx - cyy) ** 2 + 4.0 * cxy**2) / (trace + EPS)
        # Robust energy normalisation: a few black table borders cannot set the
        # entire page's scale, while a very pale yellow label still contributes.
        energy = np.sqrt(trace)
        high = float(np.percentile(energy, 99.2))
        if high <= EPS:
            continue
        edge_evidence = coherence * np.minimum(energy / high, 1.0)
        response = cv2.boxFilter(edge_evidence, cv2.CV_32F, (pool_size, pool_size), normalize=True)
        orientation = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
        threshold = max(0.07, float(np.percentile(response, 99.45)))
        # Direction bins prevent adjacent labels at different rotations from
        # becoming one huge component after the pooling step.
        for bin_index in range(18):
            target = -math.pi / 2.0 + (bin_index + 0.5) * math.pi / 18.0
            distance = np.abs(np.angle(np.exp(2j * (orientation - target)))) / 2.0
            mask = ((response >= threshold) & (coherence >= 0.52) & (distance <= math.radians(7.0))).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 45:
                    continue
                rect = cv2.minAreaRect(contour)
                long_side, short_side = max(rect[1]), min(rect[1])
                if short_side < 6 or long_side < 24 or long_side / (short_side + EPS) < 1.25:
                    continue
                quad = cv2.boxPoints(rect) / scale
                proposals.append(
                    Detection(
                        quad=quad.astype(np.float32),
                        sources={f"tensor:l{local_size}:p{pool_size}"},
                        proposal_score=0.36,
                    )
                )
    return proposals


def is_matrix_format(format_name: str) -> bool:
    normalized = format_name.lower()
    return any(name in normalized for name in ("data matrix", "qr", "aztec", "maxicode"))


def zxing_position_quad(barcode: Any) -> np.ndarray:
    position = barcode.position
    return np.array(
        [
            [position.top_left.x, position.top_left.y],
            [position.top_right.x, position.top_right.y],
            [position.bottom_right.x, position.bottom_right.y],
            [position.bottom_left.x, position.bottom_left.y],
        ],
        dtype=np.float32,
    )


def decoder_proposals(image: np.ndarray) -> list[Detection]:
    """Return high-confidence detections supplied by the open-source decoder."""
    if zxingcpp is None:
        return []
    proposals: list[Detection] = []
    try:
        barcodes = zxingcpp.read_barcodes(
            image,
            try_rotate=True,
            try_downscale=False,
            try_invert=True,
            return_errors=False,
        )
    except Exception:
        return []
    for barcode in barcodes:
        format_name = str(barcode.format)
        proposals.append(
            Detection(
                quad=zxing_position_quad(barcode),
                sources={"zxing:whole-page"},
                proposal_score=1.0,
                decoded_text=str(barcode.text),
                barcode_format=format_name,
                kind="matrix" if is_matrix_format(format_name) else "linear",
                accepted=True,
            )
        )
    return proposals


def binary_signal(signal: np.ndarray) -> np.ndarray:
    """Otsu-binarize a one-dimensional consensus scanline."""
    signal = np.asarray(signal, dtype=np.uint8).reshape(1, -1)
    # A 1-D median removes isolated paper/noise specks without changing bars.
    if signal.shape[1] >= 5:
        signal = cv2.medianBlur(signal, 3)
    _, binary = cv2.threshold(signal, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary.ravel() > 0


def count_transition_runs(binary: np.ndarray) -> tuple[int, float]:
    """Count meaningful alternating runs and the fraction of dark pixels."""
    if len(binary) < 2:
        return 0, 0.0
    runs: list[tuple[bool, int]] = []
    state = bool(binary[0])
    length = 1
    for value in binary[1:]:
        value = bool(value)
        if value == state:
            length += 1
        else:
            runs.append((state, length))
            state, length = value, 1
    runs.append((state, length))
    # Ignore one-pixel faults only when they are sandwiched between equal runs;
    # 1-pixel modules are legitimate at native document resolution.
    compact: list[tuple[bool, int]] = []
    for index, run in enumerate(runs):
        if 0 < index < len(runs) - 1 and run[1] == 1 and runs[index - 1][0] == runs[index + 1][0]:
            if compact:
                previous_state, previous_length = compact[-1]
                compact[-1] = (previous_state, previous_length + 1 + runs[index + 1][1])
            continue
        if index > 0 and runs[index - 1][1] == 1 and runs[index - 1][0] == run[0]:
            continue
        compact.append(run)
    transitions = max(0, len(compact) - 1)
    dark_fraction = float(np.mean(binary))
    return transitions, dark_fraction


def barcode_structure_metrics(gray: np.ndarray, quad: np.ndarray) -> dict[str, float]:
    """Measure the visual facts a 1-D barcode must satisfy after deskewing."""
    patch = as_gray(rectify_quad(gray, quad, pad_long=0.0, pad_short=0.0))
    height, width = patch.shape[:2]
    if width < 18 or height < 5:
        return {"score": 0.0}
    # The human-readable characters normally live below the bars.  The middle
    # band makes the verifier insensitive to them and to a small bounding-box
    # height error.
    top = max(0, int(round(height * 0.10)))
    bottom = min(height, max(top + 3, int(round(height * 0.82))))
    strip = patch[top:bottom]
    if strip.shape[0] < 3:
        strip = patch
    gx = np.abs(cv2.Scharr(strip, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Scharr(strip, cv2.CV_32F, 0, 1))
    x_energy = float(np.mean(gx))
    y_energy = float(np.mean(gy))
    orientation = x_energy / (x_energy + y_energy + EPS)

    # High-gradient columns should remain present through many rows. A table
    # rule has persistence but too few transition columns; text has transitions
    # but its edge columns stop at glyph boundaries. Combining both is key.
    threshold = float(np.percentile(gx, 72.0))
    if threshold <= EPS:
        threshold = float(np.mean(gx) + np.std(gx))
    edge_mask = gx >= max(threshold, EPS)
    column_support = np.mean(edge_mask, axis=0)
    persistent_columns = column_support >= 0.43
    persistence = float(np.mean(persistent_columns))
    support_strength = float(np.mean(column_support[persistent_columns])) if np.any(persistent_columns) else 0.0

    # A median across the barcode height is a deterministic "multi-scanline
    # consensus". At true bars it retains every dark/light transition; text,
    # signatures, and table content vanish because they do not persist.
    consensus = np.median(strip, axis=0).astype(np.uint8)
    binary = binary_signal(consensus)
    transitions, dark_fraction = count_transition_runs(binary)
    transition_rate = transitions / max(1, width)

    # Check agreement of several independent scanlines near transition sites.
    transition_sites = np.flatnonzero(np.diff(binary.astype(np.int8)) != 0)
    if len(transition_sites):
        expanded = np.zeros(width, dtype=bool)
        for offset in range(-2, 3):
            expanded[np.clip(transition_sites + offset, 0, width - 1)] = True
        scanline_agreements: list[float] = []
        rows = np.linspace(0, strip.shape[0] - 1, min(9, strip.shape[0]), dtype=int)
        for row in rows:
            row_binary = binary_signal(strip[row])
            if len(row_binary) == len(binary) and np.any(expanded):
                scanline_agreements.append(float(np.mean(row_binary[expanded] == binary[expanded])))
        agreement = float(np.median(scanline_agreements)) if scanline_agreements else 0.0
    else:
        agreement = 0.0

    aspect = width / max(1.0, height)
    aspect_score = min(1.0, max(0.0, (aspect - 1.15) / 2.2))
    alternation_score = min(1.0, transitions / 18.0) * min(1.0, transition_rate / 0.055)
    persistence_score = min(1.0, persistence / 0.16) * min(1.0, support_strength / 0.62)
    dark_score = max(0.0, 1.0 - abs(dark_fraction - 0.43) / 0.43)
    score = (
        0.22 * orientation
        + 0.30 * persistence_score
        + 0.25 * alternation_score
        + 0.16 * agreement
        + 0.04 * aspect_score
        + 0.03 * dark_score
    )
    return {
        "score": float(score),
        "orientation": float(orientation),
        "persistence": float(persistence),
        "support_strength": float(support_strength),
        "scanline_agreement": float(agreement),
        "transitions": float(transitions),
        "transition_rate": float(transition_rate),
        "dark_fraction": float(dark_fraction),
        "aspect": float(aspect),
    }


def decode_locally(image: np.ndarray, detection: Detection) -> None:
    """Deskew then decode a candidate. Success improves confidence, never gates it."""
    if zxingcpp is None or detection.kind != "linear":
        return
    patch = rectify_quad(image, detection.quad, pad_long=0.08, pad_short=0.24)
    patch = cv2.copyMakeBorder(patch, 16, 16, 20, 20, cv2.BORDER_CONSTANT, value=255)
    try:
        barcodes = zxingcpp.read_barcodes(
            patch,
            try_rotate=True,
            try_downscale=False,
            try_invert=True,
            return_errors=False,
        )
    except Exception:
        return
    if not barcodes:
        return
    # A linear candidate can sometimes include a nearby 2-D sticker. Prefer a
    # decoded linear symbology, then fall back to the first valid result.
    barcode = next((item for item in barcodes if not is_matrix_format(str(item.format))), barcodes[0])
    detection.decoded_text = str(barcode.text)
    detection.barcode_format = str(barcode.format)
    detection.kind = "matrix" if is_matrix_format(detection.barcode_format) else "linear"
    detection.sources.add("zxing:deskewed-candidate")


def accept_linear(
    detection: Detection,
    minimum_score: float,
    allow_tensor_only: bool,
    minimum_linear_length: float,
) -> bool:
    if detection.barcode_format and detection.kind == "linear":
        return True
    metrics = detection.metrics
    long_side, short_side = detection.long_short()
    aspect = long_side / (short_side + EPS)
    # This dossier's 1-D symbols are materially longer than they are tall. A
    # very short, almost-square "barcode" proposal is nearly always a glyph
    # cluster, a printed hatch, or a piece of a real barcode rather than an
    # independent symbol. Decoded symbols bypass this guard.
    if long_side < minimum_linear_length or aspect < 1.65:
        return False
    has_opencv_proposal = any(source.startswith("opencv:") for source in detection.sources)
    # The custom tensor route is deliberately conservative by default. It is a
    # superb recall tool, but an uncorroborated tensor component can also arise
    # from a form heading (for example "ID | Libelle"). High-recall review mode
    # can opt into those candidates without pretending they are certain.
    if not has_opencv_proposal and not allow_tensor_only:
        return False
    # Reject a few parallel table rules even if their average gradient happens
    # to be high. All limits are physical, not learned from labels.
    return bool(
        metrics.get("score", 0.0) >= minimum_score
        and metrics.get("orientation", 0.0) >= 0.62
        and metrics.get("persistence", 0.0) >= 0.045
        and metrics.get("transitions", 0.0) >= 12.0
        and metrics.get("scanline_agreement", 0.0) >= 0.50
        and 0.015 <= metrics.get("transition_rate", 0.0) <= 0.72
    )


def locate_page(
    image: np.ndarray,
    angle_step: int,
    minimum_score: float,
    use_tensor_fallback: bool,
    allow_tensor_only: bool = False,
    minimum_linear_length: float = 90.0,
) -> tuple[list[Detection], list[Detection]]:
    gray = as_gray(image)
    candidates = decoder_proposals(image) + detector_proposals(gray, angle_step)
    if use_tensor_fallback:
        candidates += tensor_proposals(gray)
    candidates = deduplicate(candidates)

    accepted: list[Detection] = []
    rejected: list[Detection] = []
    for candidate in candidates:
        if candidate.kind == "matrix" and candidate.barcode_format:
            candidate.accepted = True
            accepted.append(candidate)
            continue
        candidate.metrics = barcode_structure_metrics(gray, candidate.quad)
        candidate.structural_score = candidate.metrics.get("score", 0.0)
        # Decode after visual verification, except for already decoded global
        # candidates. This keeps the decoder from becoming a costly page search.
        if candidate.structural_score >= max(0.35, minimum_score - 0.16) or candidate.barcode_format:
            decode_locally(image, candidate)
        candidate.accepted = accept_linear(
            candidate,
            minimum_score,
            allow_tensor_only=allow_tensor_only,
            minimum_linear_length=minimum_linear_length,
        )
        if candidate.accepted:
            accepted.append(candidate)
        else:
            rejected.append(candidate)
    # Local decode can make two formerly distinct visual candidates equivalent.
    return deduplicate(accepted), rejected


def render_pdf(pdf_path: Path, output_pages: Path, dpi: int) -> list[Path]:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("PDF input requires Poppler's pdftoppm. Install Poppler or pass page images instead.")
    output_pages.mkdir(parents=True, exist_ok=True)
    prefix = output_pages / "page"
    command = ["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(prefix)]
    subprocess.run(command, check=True)
    return sorted(output_pages.glob("page-*.png"), key=natural_page_key)


def natural_page_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return int(stem.rsplit("-", 1)[-1]), stem
    except ValueError:
        return 0, stem


def collect_input_pages(input_path: Path, work_dir: Path, dpi: int) -> list[Path]:
    if input_path.suffix.lower() == ".pdf":
        return render_pdf(input_path, work_dir / "pages", dpi)
    if input_path.is_dir():
        images = [item for item in input_path.iterdir() if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]
        return sorted(images, key=natural_page_key)
    return [input_path]


def annotate(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    output = image.copy()
    for index, detection in enumerate(detections, start=1):
        if detection.kind == "matrix":
            color = (255, 100, 255)
        elif detection.barcode_format:
            color = (50, 220, 50)
        else:
            color = (0, 180, 255)
        quad = np.round(detection.quad).astype(np.int32)
        cv2.polylines(output, [quad], True, color, 3, cv2.LINE_AA)
        anchor = tuple(np.min(quad, axis=0).tolist())
        label = f"{index}"
        if detection.decoded_text:
            label += f" {detection.decoded_text[:18]}"
        cv2.putText(output, label, (anchor[0], max(24, anchor[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return output


def write_crops(image: np.ndarray, detections: Sequence[Detection], directory: Path, page_number: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, detection in enumerate(detections, start=1):
        patch = rectify_quad(image, detection.quad, pad_long=0.08, pad_short=0.18)
        cv2.imwrite(str(directory / f"page-{page_number:02d}-barcode-{index:02d}.png"), patch)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PDF, page image, or directory of page images")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/barcode-detection/locator"),
        help="result directory",
    )
    parser.add_argument("--dpi", type=int, default=200, help="rendering DPI for PDF input; 200 preserves this dossier's native raster")
    parser.add_argument(
        "--angle-step",
        type=int,
        default=15,
        help="deskew fallback spacing in degrees (0 disables it; 5 maximizes recall at higher cost)",
    )
    parser.add_argument("--minimum-score", type=float, default=0.52, help="structural acceptance threshold for undecoded 1-D candidates")
    parser.add_argument(
        "--high-recall",
        action="store_true",
        help="include uncorroborated structure-tensor candidates; use diagnostics/overlays for review",
    )
    parser.add_argument(
        "--minimum-linear-length",
        type=float,
        default=90.0,
        help="minimum long side in pixels for an undecoded 1-D candidate (default: 90)",
    )
    parser.add_argument(
        "--tensor-fallback",
        action="store_true",
        help="also run the independent full-page structure-tensor proposal route (slower; useful for recall experiments)",
    )
    parser.add_argument("--no-crops", action="store_true", help="do not save deskewed barcode crops")
    parser.add_argument("--include-rejected", action="store_true", help="write rejected candidates and metrics to diagnostics.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    input_path = arguments.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Input does not exist: {input_path}", file=sys.stderr)
        return 2
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pages = collect_input_pages(input_path, output / "rendered-pages", arguments.dpi)
    if not pages:
        print("No readable page images were found.", file=sys.stderr)
        return 2

    all_pages: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    total = 0
    for page_number, page_path in enumerate(pages, start=1):
        image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable page: {page_path}", file=sys.stderr)
            continue
        accepted, rejected = locate_page(
            image,
            angle_step=max(0, arguments.angle_step),
            minimum_score=arguments.minimum_score,
            use_tensor_fallback=arguments.tensor_fallback or arguments.high_recall,
            allow_tensor_only=arguments.high_recall,
            minimum_linear_length=max(1.0, arguments.minimum_linear_length),
        )
        accepted.sort(key=lambda item: (round(float(item.center()[1]) / 20), float(item.center()[0])))
        annotated = annotate(image, accepted)
        overlay_path = output / "overlays" / f"page-{page_number:02d}.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), annotated)
        if not arguments.no_crops:
            write_crops(image, accepted, output / "crops", page_number)
        all_pages.append(
            {
                "page": page_number,
                "source_image": str(page_path),
                "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
                "detections": [item.to_json() for item in accepted],
            }
        )
        if arguments.include_rejected:
            diagnostics.append({"page": page_number, "rejected": [item.to_json() for item in rejected]})
        total += len(accepted)
        decoded = sum(bool(item.barcode_format) for item in accepted)
        print(f"page {page_number:02d}: {len(accepted)} located ({decoded} decoded)")

    payload = {
        "input": str(input_path),
        "method": "classical directional coherence + structure tensor + scanline verification",
        "uses_learned_model": False,
        "opencv_version": cv2.__version__,
        "zxing_available": zxingcpp is not None,
        "pages": all_pages,
        "summary": {"pages": len(all_pages), "detections": total},
    }
    (output / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if arguments.include_rejected:
        (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"\nWrote {total} oriented detections to {output / 'detections.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
