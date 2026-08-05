#!/usr/bin/env python3
"""Fast CPU-only 1D barcode localization from tiled structure tensors.

The runtime contract deliberately separates preparation from engine compute:

    prepared = prepare_page(image)
    result = locate_prepared(prepared)

`prepare_page` owns grayscale conversion and the one detector resize. The timed
engine starts with an already prepared native grayscale image and detector
thumbnail. No barcode decoder, ZXing, trained model, or GPU is used.

The proposal stage is inspired by Sörös and Flörkemeier's structure-matrix
localizer, but uses a tile pyramid rather than dense full-resolution
multi-window maps. Orientation-aware connected components keep text and table
edges from merging into a single proposal. Every final box must independently
pass the native-pixel physical verifier carried over from Pipeline 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

try:
    import _sttg_native
except ImportError:
    _sttg_native = None


EPS = 1e-6


@dataclass(frozen=True)
class TensorConfig:
    backend: str = "native-context"
    work_size: int = 1600
    tile_size: int = 1
    aggregation_windows: tuple[int, ...] = (12,)
    orientation_bins: int = 12
    gradient_threshold: float = 48.0
    minimum_edge_occupancy: float = 0.18
    minimum_coherence: float = 0.92
    seed_context_coherence: float = 0.82
    response_percentile: float = 35.0
    minimum_component_cells: int = 3
    minimum_component_density: float = 0.0
    minimum_component_orientation: float = 0.0
    grow_edge_occupancy: float = 0.07
    grow_coherence: float = 0.80
    grow_context_coherence: float = 0.67
    grow_energy_factor: float = 0.55
    orientation_tolerance_degrees: float = 13.0
    context_orientation_tolerance_degrees: float = 16.9
    minimum_polarity_balance: float = 0.0
    profile_filter: bool = True
    minimum_profile_transitions: int = 8
    minimum_profile_contrast: float = 20.0
    minimum_profile_agreement: float = 0.45
    maximum_profile_transition_rate: float = 0.72
    retain_low_transition_rescue: bool = False
    rescue_minimum_profile_transitions: int = 4
    rescue_minimum_aspect: float = 1.20
    rescue_maximum_aspect: float = 3.00
    rescue_minimum_orientation: float = 0.995
    rescue_minimum_contrast: float = 150.0
    rescue_minimum_agreement: float = 0.65
    rescue_minimum_cells: int = 40
    rescue_maximum_cells: int = 500
    split_periodic_bands: bool = False
    fragment_join_max_gap_short: float = 2.60
    maximum_candidates: int = 0
    minimum_source_length: float = 45.0
    minimum_candidate_aspect: float = 1.20
    maximum_candidate_aspect: float = 30.0
    long_padding_tiles: float = 1.0
    short_padding_tiles: float = 1.0
    minimum_linear_score: float = 0.52
    minimum_native_transitions: float = 12.0
    maximum_edge_direction_deviation_degrees: float = 90.0


@dataclass(frozen=True)
class PreparedPage:
    native_gray: np.ndarray
    work_gray: np.ndarray
    scale_x: float
    scale_y: float
    preparation_seconds: float


@dataclass
class Detection:
    quad: np.ndarray
    confidence: float
    source: str
    metrics: dict[str, float] = field(default_factory=dict)

    def area(self) -> float:
        return abs(float(cv2.contourArea(np.asarray(self.quad, np.float32))))

    def center(self) -> np.ndarray:
        return np.asarray(self.quad, np.float32).mean(axis=0)

    def long_short(self) -> tuple[float, float]:
        width, height = cv2.minAreaRect(np.asarray(self.quad, np.float32))[1]
        return max(float(width), float(height)), min(float(width), float(height))

    def to_prediction(self) -> dict[str, Any]:
        return {
            "polygon": np.asarray(self.quad, np.float32).tolist(),
            "kind": "1d",
            "symbology": "unknown",
            "payload": None,
            "confidence": float(np.clip(self.confidence, 0.0, 1.0)),
            "status": "localized",
            "sources": [self.source],
            "evidence": {key: float(value) for key, value in self.metrics.items()},
        }


@dataclass
class LocalizationResult:
    detections: list[Detection]
    timings: dict[str, float]
    diagnostics: dict[str, Any]


def profile_config(
    profile: str,
    *,
    work_size: int | None = None,
    tile: int | None = None,
) -> TensorConfig:
    """Return one of the two frozen, benchmarked operating profiles."""
    if profile == "document":
        return replace(
            TensorConfig(),
            work_size=work_size or 1600,
            aggregation_windows=(tile or 7,),
            fragment_join_max_gap_short=5.0,
            minimum_native_transitions=24.0,
            maximum_edge_direction_deviation_degrees=18.0,
            retain_low_transition_rescue=True,
            split_periodic_bands=True,
        )
    if profile == "general":
        return replace(
            TensorConfig(),
            work_size=work_size or 1600,
            aggregation_windows=(tile or 12,),
            fragment_join_max_gap_short=2.6,
            minimum_native_transitions=12.0,
        )
    raise ValueError(f"unknown operating profile: {profile}")


def as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape: {image.shape}")


def prepare_page(image: np.ndarray, config: TensorConfig = TensorConfig()) -> PreparedPage:
    started = perf_counter()
    native = as_gray(image)
    height, width = native.shape
    scale = min(1.0, float(config.work_size) / max(height, width))
    if scale < 1.0:
        work_width = max(1, int(round(width * scale)))
        work_height = max(1, int(round(height * scale)))
        work = cv2.resize(native, (work_width, work_height), interpolation=cv2.INTER_AREA)
    else:
        work = native
    return PreparedPage(
        native_gray=native,
        work_gray=work,
        scale_x=work.shape[1] / width,
        scale_y=work.shape[0] / height,
        preparation_seconds=perf_counter() - started,
    )


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
    return ordered


def _rectify_geometry(
    gray: np.ndarray,
    quad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    source = order_quad(quad)
    top = np.linalg.norm(source[1] - source[0])
    bottom = np.linalg.norm(source[2] - source[3])
    left = np.linalg.norm(source[3] - source[0])
    right = np.linalg.norm(source[2] - source[1])
    width = max(2, int(round(max(top, bottom))))
    height = max(2, int(round(max(left, right))))
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    patch = cv2.warpPerspective(
        gray,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    rotated = patch.shape[0] > patch.shape[1]
    if rotated:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    return patch, np.linalg.inv(transform), height, rotated


def rectify(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    return _rectify_geometry(gray, quad)[0]


def _map_rectified_quad(
    points: np.ndarray,
    inverse_transform: np.ndarray,
    original_height: int,
    rotated: bool,
) -> np.ndarray:
    rectified = np.asarray(points, np.float32).reshape(-1, 2)
    if rotated:
        # cv2.ROTATE_90_CLOCKWISE maps (x, y) to
        # (original_height - 1 - y, x). Undo that before applying the inverse
        # perspective transform.
        rectified = np.column_stack(
            (
                rectified[:, 1],
                original_height - 1 - rectified[:, 0],
            )
        ).astype(np.float32)
    return cv2.perspectiveTransform(
        rectified.reshape(1, -1, 2),
        inverse_transform.astype(np.float64),
    )[0].astype(np.float32)


def _binary_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8).reshape(1, -1)
    return cv2.threshold(values, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1].ravel() > 0


def _count_transition_runs(binary: np.ndarray) -> tuple[int, float]:
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
    """Native-pixel barcode physics verifier used by Pipeline 7."""
    patch = rectify(gray, quad)
    height, width = patch.shape[:2]
    if width < 18 or height < 5:
        return {"score": 0.0}
    top = max(0, int(round(height * 0.10)))
    bottom = min(height, max(top + 3, int(round(height * 0.82))))
    strip = patch[top:bottom]
    signed_gx = cv2.Scharr(strip, cv2.CV_32F, 1, 0)
    signed_gy = cv2.Scharr(strip, cv2.CV_32F, 0, 1)
    gx = np.abs(signed_gx)
    gy = np.abs(signed_gy)
    x_energy, y_energy = float(np.mean(gx)), float(np.mean(gy))
    orientation = x_energy / (x_energy + y_energy + EPS)
    magnitude = np.hypot(signed_gx, signed_gy)
    edge_threshold = float(np.percentile(magnitude, 75.0))
    strong_edges = magnitude >= max(edge_threshold, EPS)
    if np.any(strong_edges):
        edge_angles = np.arctan2(
            signed_gy[strong_edges],
            signed_gx[strong_edges],
        )
        edge_weights = magnitude[strong_edges]
        doubled_angles = 2.0 * edge_angles
        resultant_x = float(
            np.sum(edge_weights * np.cos(doubled_angles))
        )
        resultant_y = float(
            np.sum(edge_weights * np.sin(doubled_angles))
        )
        weight_sum = float(np.sum(edge_weights))
        edge_direction_coherence = math.hypot(
            resultant_x,
            resultant_y,
        ) / max(weight_sum, EPS)
        dominant_doubled_angle = math.atan2(
            resultant_y,
            resultant_x,
        )
        edge_direction_deviation = 0.5 * np.abs(
            np.arctan2(
                np.sin(doubled_angles - dominant_doubled_angle),
                np.cos(doubled_angles - dominant_doubled_angle),
            )
        )
        edge_direction_p80_degrees = float(
            np.degrees(np.percentile(edge_direction_deviation, 80.0))
        )
    else:
        edge_direction_coherence = 0.0
        edge_direction_p80_degrees = 90.0
    threshold = float(np.percentile(gx, 72.0))
    if threshold <= EPS:
        threshold = float(np.mean(gx) + np.std(gx))
    support = np.mean(gx >= max(threshold, EPS), axis=0)
    persistent = support >= 0.43
    persistence = float(np.mean(persistent))
    support_strength = float(np.mean(support[persistent])) if np.any(persistent) else 0.0
    consensus = np.median(strip, axis=0).astype(np.uint8)
    binary = _binary_signal(consensus)
    transitions, dark_fraction = _count_transition_runs(binary)
    transition_rate = transitions / max(1, width)
    sites = np.flatnonzero(np.diff(binary.astype(np.int8)) != 0)
    agreements: list[float] = []
    if len(sites):
        expanded = np.zeros(width, dtype=bool)
        for offset in range(-2, 3):
            expanded[np.clip(sites + offset, 0, width - 1)] = True
        for row in np.linspace(0, strip.shape[0] - 1, min(9, strip.shape[0]), dtype=int):
            row_binary = _binary_signal(strip[row])
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
        "edge_direction_coherence": edge_direction_coherence,
        "edge_direction_p80_degrees": edge_direction_p80_degrees,
        "persistence": persistence,
        "support_strength": support_strength,
        "scanline_agreement": agreement,
        "transitions": float(transitions),
        "transition_rate": transition_rate,
        "dark_fraction": dark_fraction,
        "aspect": aspect,
    }


def _accepted_linear_metrics(
    metrics: dict[str, float],
    config: TensorConfig,
) -> bool:
    return bool(
        metrics.get("score", 0.0) >= config.minimum_linear_score
        and metrics.get("orientation", 0.0) >= 0.62
        and metrics.get("persistence", 0.0) >= 0.045
        and metrics.get("transitions", 0.0)
        >= config.minimum_native_transitions
        and metrics.get("scanline_agreement", 0.0) >= 0.50
        and 0.015 <= metrics.get("transition_rate", 0.0) <= 0.72
        and metrics.get("edge_direction_p80_degrees", 90.0)
        <= config.maximum_edge_direction_deviation_degrees
    )


def _bridge_short_false_runs(values: np.ndarray, maximum_gap: int) -> np.ndarray:
    output = np.asarray(values, bool).copy()
    if maximum_gap <= 0 or output.size < 3:
        return output
    padded = np.concatenate(([True], output, [True]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(changes[::2], changes[1::2]):
        if end - start <= maximum_gap:
            output[start:end] = True
    return output


def _periodic_band_candidates(
    native: np.ndarray,
    proposal: Detection,
    config: TensorConfig,
) -> list[Detection]:
    """Split an oversized tensor component using native-pixel row periodicity.

    The structure-tensor stage sometimes groups a barcode with its printed
    digits or groups two nearby, parallel barcodes. A real barcode creates a
    contiguous short-axis band whose individual rows repeatedly alternate
    across the long axis. This helper searches only inside an already accepted
    tensor component; it never performs a second full-page scan.
    """
    patch, inverse_transform, original_height, rotated = _rectify_geometry(
        native,
        proposal.quad,
    )
    height, width = patch.shape
    if width < 45 or height < 20:
        return []

    row_minimum = patch.min(axis=1).astype(np.float32)
    row_maximum = patch.max(axis=1).astype(np.float32)
    row_contrast = row_maximum - row_minimum
    thresholds = 0.5 * (row_minimum + row_maximum)
    binary = patch < thresholds[:, None]
    cleaned = binary.copy()
    if width >= 3:
        cleaned[:, 1:-1] = np.where(
            (binary[:, :-2] == binary[:, 2:])
            & (binary[:, 1:-1] != binary[:, :-2]),
            binary[:, :-2],
            binary[:, 1:-1],
        )
    transitions = np.count_nonzero(
        cleaned[:, 1:] != cleaned[:, :-1],
        axis=1,
    )
    active = (
        (transitions >= int(math.ceil(config.minimum_native_transitions)))
        & (row_contrast >= 40.0)
    )
    active = _bridge_short_false_runs(active, maximum_gap=3)
    padded = np.concatenate(([False], active, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    minimum_band_height = max(3, int(math.ceil(0.10 * height)))
    children: list[Detection] = []
    for start, end in zip(starts, ends):
        if end - start < minimum_band_height:
            continue
        top = max(0, int(start) - 2)
        bottom = min(height, int(end) + 2)
        rectified_quad = np.asarray(
            [
                [0, top],
                [width - 1, top],
                [width - 1, bottom - 1],
                [0, bottom - 1],
            ],
            np.float32,
        )
        source_quad = _map_rectified_quad(
            rectified_quad,
            inverse_transform,
            original_height,
            rotated,
        )
        metrics = linear_metrics(native, source_quad)
        if not _accepted_linear_metrics(metrics, config):
            continue
        children.append(
            Detection(
                quad=source_quad,
                confidence=float(metrics["score"]),
                source=f"{proposal.source}:periodic-band",
                metrics={
                    **proposal.metrics,
                    **metrics,
                    "periodic_band_split": 1.0,
                    "periodic_band_top": float(top),
                    "periodic_band_bottom": float(bottom),
                    "periodic_band_peak_transitions": float(
                        transitions[start:end].max()
                    ),
                    "periodic_parent_height": float(height),
                },
            )
        )
    return _deduplicate(children, threshold=0.70)


def _quad_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(np.asarray(first, np.float32))
    second_hull = cv2.convexHull(np.asarray(second, np.float32))
    first_area = abs(float(cv2.contourArea(first_hull)))
    second_area = abs(float(cv2.contourArea(second_hull)))
    if min(first_area, second_area) <= EPS:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    return max(0.0, float(intersection)) / min(first_area, second_area)


def _deduplicate(detections: Iterable[Detection], threshold: float = 0.62) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: (item.confidence, item.area()), reverse=True):
        if any(_quad_overlap(candidate.quad, prior.quad) >= threshold for prior in kept):
            continue
        kept.append(candidate)
    return kept


def _oriented_quad(
    xs: np.ndarray,
    ys: np.ndarray,
    theta: float,
    tile_size: int,
    long_padding_tiles: float,
    short_padding_tiles: float,
) -> np.ndarray:
    points = np.column_stack(((xs.astype(np.float32) + 0.5) * tile_size, (ys.astype(np.float32) + 0.5) * tile_size))
    direction = np.asarray([math.cos(theta), math.sin(theta)], np.float32)
    normal = np.asarray([-direction[1], direction[0]], np.float32)
    along = points @ direction
    across = points @ normal
    along_low = float(along.min() - long_padding_tiles * tile_size)
    along_high = float(along.max() + long_padding_tiles * tile_size)
    across_low = float(across.min() - short_padding_tiles * tile_size)
    across_high = float(across.max() + short_padding_tiles * tile_size)
    return np.asarray(
        [
            direction * along_low + normal * across_low,
            direction * along_high + normal * across_low,
            direction * along_high + normal * across_high,
            direction * along_low + normal * across_high,
        ],
        np.float32,
    )


def _tile_tensor_proposals(
    work: np.ndarray,
    prepared: PreparedPage,
    config: TensorConfig,
) -> tuple[list[Detection], dict[str, Any], dict[str, float]]:
    gradient_started = perf_counter()
    gx = cv2.Scharr(work, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(work, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    edge = magnitude >= config.gradient_threshold
    # Match OpenCV's polarity normalization before accumulating orientation:
    # a negative horizontal derivative flips both derivative components.
    flip = gx < 0
    gx = np.where(edge, np.where(flip, -gx, gx), 0.0)
    gy = np.where(edge, np.where(flip, -gy, gy), 0.0)
    gradient_seconds = perf_counter() - gradient_started

    tensor_started = perf_counter()
    height, width = work.shape
    packed = np.dstack((edge.astype(np.float32), gx * gx, gy * gy, gx * gy))
    tensor_seconds = perf_counter() - tensor_started

    grouping_started = perf_counter()
    proposals: list[Detection] = []
    components_considered = 0
    active_cells_by_window: dict[str, int] = {}
    grid_width = grid_height = 0

    for window in config.aggregation_windows:
        window = max(1, int(window))
        cropped_height = (height // window) * window
        cropped_width = (width // window) * window
        if cropped_height < window or cropped_width < window:
            continue
        grid_height = cropped_height // window
        grid_width = cropped_width // window
        pooled = cv2.resize(
            packed[:cropped_height, :cropped_width],
            (grid_width, grid_height),
            interpolation=cv2.INTER_AREA,
        )
        occupancy, jxx, jyy, jxy = cv2.split(pooled)
        trace = jxx + jyy
        discriminant = cv2.sqrt(cv2.max((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
        coherence = discriminant / np.maximum(trace, EPS)
        theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        active = (occupancy >= config.minimum_edge_occupancy) & (coherence >= config.minimum_coherence)
        active_cells_by_window[str(window)] = int(np.count_nonzero(active))
        if not np.any(active):
            continue
        # Quantized orientation masks approximate OpenCV's orientation-aware
        # region growing in compiled primitives. Without this separation,
        # barcode cells can merge into a nearby text/table component and the
        # resulting rectangle becomes too large to verify.
        normalized_theta = np.mod(theta + 0.5 * math.pi, math.pi)
        quantized = np.floor(normalized_theta * config.orientation_bins / math.pi).astype(np.int16)
        for orientation_bin in range(config.orientation_bins):
            mask = (active & (quantized == orientation_bin)).astype(np.uint8)
            if not np.any(mask):
                continue
            neighbor_count = cv2.filter2D(mask, cv2.CV_8U, np.ones((3, 3), np.uint8))
            mask = ((mask > 0) & (neighbor_count >= 3)).astype(np.uint8)
            if not np.any(mask):
                continue
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, count):
                cell_count = int(stats[label, cv2.CC_STAT_AREA])
                if cell_count < config.minimum_component_cells:
                    continue
                components_considered += 1
                component_mask = labels == label
                ys, xs = np.where(component_mask)
                box_area = max(
                    1,
                    int(stats[label, cv2.CC_STAT_WIDTH]) * int(stats[label, cv2.CC_STAT_HEIGHT]),
                )
                density = cell_count / box_area
                if density < config.minimum_component_density:
                    continue
                weights = occupancy[ys, xs]
                double_x = float(np.sum(weights * np.cos(2.0 * theta[ys, xs])))
                double_y = float(np.sum(weights * np.sin(2.0 * theta[ys, xs])))
                weight_sum = float(np.sum(weights)) + EPS
                orientation_resultant = math.hypot(double_x, double_y) / weight_sum
                if orientation_resultant < config.minimum_component_orientation:
                    continue
                component_theta = 0.5 * math.atan2(double_y, double_x)
                quad_work = _oriented_quad(
                    xs,
                    ys,
                    component_theta,
                    window,
                    config.long_padding_tiles,
                    config.short_padding_tiles,
                )
                quad_source = quad_work.copy()
                quad_source[:, 0] /= prepared.scale_x
                quad_source[:, 1] /= prepared.scale_y
                candidate = Detection(
                    quad=quad_source,
                    confidence=orientation_resultant * float(np.mean(weights)) * math.sqrt(cell_count),
                    source=f"integral-tensor:w{window}:o{orientation_bin}",
                    metrics={
                        "proposal_cells": float(cell_count),
                        "proposal_density": float(density),
                        "proposal_coherence": float(np.mean(coherence[ys, xs])),
                        "proposal_orientation_resultant": orientation_resultant,
                        "proposal_edge_occupancy": float(np.mean(weights)),
                        "proposal_window": float(window),
                        "proposal_orientation_bin": float(orientation_bin),
                    },
                )
                long_side, short_side = candidate.long_short()
                aspect = long_side / max(short_side, EPS)
                if (
                    long_side < config.minimum_source_length
                    or short_side < 5.0
                    or not config.minimum_candidate_aspect <= aspect <= config.maximum_candidate_aspect
                ):
                    continue
                proposals.append(candidate)

    proposals = _deduplicate(proposals, threshold=0.58)
    proposals.sort(key=lambda item: item.confidence, reverse=True)
    if config.maximum_candidates > 0:
        proposals = proposals[: config.maximum_candidates]
    grouping_seconds = perf_counter() - grouping_started
    return (
        proposals,
        {
            "grid_width": grid_width,
            "grid_height": grid_height,
            "components_considered": components_considered,
            "candidate_count": len(proposals),
            "active_cells_by_window": active_cells_by_window,
        },
        {
            "gradient_seconds": gradient_seconds,
            "tensor_pool_seconds": tensor_seconds,
            "grouping_seconds": grouping_seconds,
        },
    )


def _native_sttg_proposals(
    prepared: PreparedPage,
    config: TensorConfig,
) -> tuple[list[Detection], dict[str, Any], dict[str, float]]:
    if _sttg_native is None:
        raise RuntimeError(
            "native STTG extension is unavailable; run "
            "`venv/bin/python setup.py build_ext --inplace` in this pipeline folder"
        )

    started = perf_counter()
    proposals: list[Detection] = []
    raw_by_window: dict[str, int] = {}
    for window in config.aggregation_windows:
        quads, metrics = _sttg_native.propose(
            prepared.work_gray,
            window=int(window),
            gradient_threshold=float(config.gradient_threshold),
            seed_occupancy=float(config.minimum_edge_occupancy),
            seed_coherence=float(config.minimum_coherence),
            grow_occupancy=float(config.grow_edge_occupancy),
            grow_coherence=float(config.grow_coherence),
            angle_tolerance_degrees=float(config.orientation_tolerance_degrees),
            minimum_cells=int(config.minimum_component_cells),
            minimum_density=float(config.minimum_component_density),
            minimum_orientation=float(config.minimum_component_orientation),
            minimum_polarity=float(config.minimum_polarity_balance),
            long_padding=float(config.long_padding_tiles),
            short_padding=float(config.short_padding_tiles),
        )
        raw_by_window[str(window)] = len(quads)
        for quad_work, values in zip(np.asarray(quads, np.float32), np.asarray(metrics, np.float32)):
            quad_source = quad_work.copy()
            quad_source[:, 0] /= prepared.scale_x
            quad_source[:, 1] /= prepared.scale_y
            candidate = Detection(
                quad=quad_source,
                confidence=float(values[0]),
                source=f"sttg-native:w{window}",
                metrics={
                    "proposal_score": float(values[0]),
                    "proposal_cells": float(values[1]),
                    "proposal_seeds": float(values[2]),
                    "proposal_density": float(values[3]),
                    "proposal_orientation_resultant": float(values[4]),
                    "proposal_polarity_balance": float(values[5]),
                    "proposal_edge_occupancy": float(values[6]),
                    "proposal_coherence": float(values[7]),
                    "proposal_window": float(window),
                },
            )
            long_side, short_side = candidate.long_short()
            aspect = long_side / max(short_side, EPS)
            if (
                long_side < config.minimum_source_length
                or short_side < 5.0
                or not config.minimum_candidate_aspect <= aspect <= config.maximum_candidate_aspect
            ):
                continue
            proposals.append(candidate)
    proposals = _deduplicate(proposals, threshold=0.58)
    proposals.sort(key=lambda item: item.confidence, reverse=True)
    overflow = bool(config.maximum_candidates > 0 and len(proposals) > config.maximum_candidates)
    if config.maximum_candidates > 0:
        proposals = proposals[: config.maximum_candidates]
    proposal_seconds = perf_counter() - started
    return (
        proposals,
        {
            "backend": "native-sttg",
            "raw_candidates_by_window": raw_by_window,
            "candidate_count": len(proposals),
            "candidate_overflow": overflow,
        },
        {
            "native_proposal_seconds": proposal_seconds,
        },
    )


def _native_context_proposals(
    prepared: PreparedPage,
    config: TensorConfig,
) -> tuple[list[Detection], dict[str, Any], dict[str, float]]:
    if _sttg_native is None:
        raise RuntimeError(
            "native STTG extension is unavailable; run "
            "`venv/bin/python setup.py build_ext --inplace` in this pipeline folder"
        )

    started = perf_counter()
    proposals: list[Detection] = []
    raw_by_window: dict[str, int] = {}
    for window in config.aggregation_windows:
        window_proposals: list[Detection] = []
        settings = {
            "tile": int(window),
            "gradient_threshold": float(config.gradient_threshold),
            "energy_percentile": float(config.response_percentile),
            "seed_occupancy": float(config.minimum_edge_occupancy),
            "seed_coherence": float(config.minimum_coherence),
            "seed_context": float(config.seed_context_coherence),
            "grow_occupancy": float(config.grow_edge_occupancy),
            "grow_coherence": float(config.grow_coherence),
            "grow_context": float(config.grow_context_coherence),
            "grow_energy_factor": float(config.grow_energy_factor),
            "orientation_bins": int(config.orientation_bins),
            "fine_tolerance_degrees": float(config.orientation_tolerance_degrees),
            "context_tolerance_degrees": float(
                config.context_orientation_tolerance_degrees
            ),
            "minimum_cells": int(config.minimum_component_cells),
            "minimum_density": float(config.minimum_component_density),
            "minimum_orientation": float(config.minimum_component_orientation),
            "minimum_polarity": float(config.minimum_polarity_balance),
            "long_padding": float(config.long_padding_tiles),
            "short_padding": float(config.short_padding_tiles),
            "profile_filter": int(config.profile_filter),
            "minimum_profile_transitions": int(
                config.minimum_profile_transitions
            ),
            "minimum_profile_contrast": float(config.minimum_profile_contrast),
            "minimum_profile_agreement": float(config.minimum_profile_agreement),
            "maximum_profile_transition_rate": float(
                config.maximum_profile_transition_rate
            ),
            "retain_low_transition_rescue": int(
                config.retain_low_transition_rescue
            ),
            "rescue_minimum_profile_transitions": int(
                config.rescue_minimum_profile_transitions
            ),
            "rescue_minimum_aspect": float(config.rescue_minimum_aspect),
            "rescue_maximum_aspect": float(config.rescue_maximum_aspect),
            "rescue_minimum_orientation": float(
                config.rescue_minimum_orientation
            ),
            "rescue_minimum_contrast": float(config.rescue_minimum_contrast),
            "rescue_minimum_agreement": float(config.rescue_minimum_agreement),
            "rescue_minimum_cells": int(config.rescue_minimum_cells),
            "rescue_maximum_cells": int(config.rescue_maximum_cells),
        }
        quads, metrics = _sttg_native.propose_context(prepared.work_gray, settings)
        raw_by_window[str(window)] = len(quads)
        for quad_work, values in zip(
            np.asarray(quads, np.float32),
            np.asarray(metrics, np.float32),
        ):
            quad_source = quad_work.copy()
            quad_source[:, 0] /= prepared.scale_x
            quad_source[:, 1] /= prepared.scale_y
            candidate = Detection(
                quad=quad_source,
                confidence=float(values[0]),
                source=f"sttg-context:w{window}",
                metrics={
                    "proposal_score": float(values[0]),
                    "proposal_cells": float(values[1]),
                    "proposal_seeds": float(values[2]),
                    "proposal_density": float(values[3]),
                    "proposal_orientation_resultant": float(values[4]),
                    "proposal_polarity_balance": float(values[5]),
                    "proposal_edge_occupancy": float(values[6]),
                    "proposal_coherence": float(values[7]),
                    "proposal_profile_transitions": float(values[8]),
                    "proposal_profile_contrast": float(values[9]),
                    "proposal_profile_agreement": float(values[10]),
                    "proposal_low_transition_rescue": (
                        float(values[11]) if len(values) > 11 else 0.0
                    ),
                    "proposal_window": float(window),
                },
            )
            long_side, short_side = candidate.long_short()
            aspect = long_side / max(short_side, EPS)
            if (
                long_side < config.minimum_source_length
                or short_side < 5.0
                or not config.minimum_candidate_aspect
                <= aspect
                <= config.maximum_candidate_aspect
            ):
                continue
            window_proposals.append(candidate)
        proposals.extend(_deduplicate(window_proposals, threshold=0.70))
    proposals.sort(key=lambda item: item.confidence, reverse=True)
    overflow = bool(
        config.maximum_candidates > 0 and len(proposals) > config.maximum_candidates
    )
    if config.maximum_candidates > 0:
        proposals = proposals[: config.maximum_candidates]
    proposal_seconds = perf_counter() - started
    return (
        proposals,
        {
            "backend": "native-context-sttg",
            "raw_candidates_by_window": raw_by_window,
            "candidate_count": len(proposals),
            "candidate_overflow": overflow,
        },
        {"native_proposal_seconds": proposal_seconds},
    )


def _verify_candidates(
    native: np.ndarray,
    proposals: Sequence[Detection],
    config: TensorConfig,
) -> tuple[list[Detection], dict[str, int]]:
    diagnostics = {
        "low_transition_rescue_proposals": sum(
            item.metrics.get("proposal_low_transition_rescue", 0.0) >= 0.5
            for item in proposals
        ),
        "periodic_parent_candidates": 0,
        "periodic_parents_split": 0,
        "periodic_children_accepted": 0,
    }
    accepted: list[Detection] = []
    for proposal in proposals:
        metrics = linear_metrics(native, proposal.quad)
        parent_accepted = _accepted_linear_metrics(metrics, config)
        retained_only_for_refinement = (
            proposal.metrics.get(
                "proposal_low_transition_rescue",
                0.0,
            )
            >= 0.5
        )
        long_side, short_side = proposal.long_short()
        parent_aspect = long_side / max(short_side, EPS)
        periodic_eligible = bool(
            config.split_periodic_bands
            and 1.20 <= parent_aspect <= 3.00
            and proposal.metrics.get(
                "proposal_orientation_resultant",
                0.0,
            )
            >= config.rescue_minimum_orientation
            and metrics.get("orientation", 0.0) >= 0.70
            and metrics.get("score", 0.0) < 0.85
        )
        if periodic_eligible:
            diagnostics["periodic_parent_candidates"] += 1
            children = _periodic_band_candidates(native, proposal, config)
            if parent_aspect <= 1.80 and not parent_accepted and children:
                # A compact oversized component normally contains one barcode
                # plus its human-readable digits. Prefer the band with the
                # strongest periodic evidence, not the visually larger text
                # band.
                child = max(
                    children,
                    key=lambda item: (
                        item.metrics.get(
                            "periodic_band_peak_transitions",
                            0.0,
                        ),
                        item.confidence,
                    ),
                )
                child.source = f"{child.source}:compact-rescue"
                accepted.append(child)
                diagnostics["periodic_parents_split"] += 1
                diagnostics["periodic_children_accepted"] += 1
                continue
            if parent_aspect > 1.80 and len(children) >= 2:
                # A broad component can contain multiple parallel barcodes.
                # Replace the merged parent only when at least two children
                # independently pass the unchanged native verifier.
                accepted.extend(children)
                diagnostics["periodic_parents_split"] += 1
                diagnostics["periodic_children_accepted"] += len(children)
                continue
        # A low-transition component is retained only to expose a possible
        # periodic sub-band. Its deliberately weak coarse profile can never
        # become a final detection by itself.
        if not parent_accepted or retained_only_for_refinement:
            continue
        proposal.metrics.update(metrics)
        proposal.confidence = float(metrics["score"])
        accepted.append(proposal)

    # Tensor components can split one long barcode at a quiet internal gap.
    # Join only independently verified, collinear fragments, then require the
    # merged native-resolution crop to pass the same physical verifier.
    changed = True
    while changed:
        changed = False
        best: tuple[
            float,
            int,
            int,
            np.ndarray,
            dict[str, float],
        ] | None = None
        for first_index, first in enumerate(accepted):
            first_points = order_quad(first.quad)
            first_edges = (
                first_points[1] - first_points[0],
                first_points[3] - first_points[0],
            )
            first_direction = max(first_edges, key=lambda value: np.linalg.norm(value))
            first_length = float(np.linalg.norm(first_direction))
            if first_length <= EPS:
                continue
            first_direction /= first_length
            first_short = first.long_short()[1]
            for second_index in range(first_index + 1, len(accepted)):
                second = accepted[second_index]
                second_points = order_quad(second.quad)
                second_edges = (
                    second_points[1] - second_points[0],
                    second_points[3] - second_points[0],
                )
                second_direction = max(
                    second_edges,
                    key=lambda value: np.linalg.norm(value),
                )
                second_length = float(np.linalg.norm(second_direction))
                if second_length <= EPS:
                    continue
                second_direction /= second_length
                if float(np.dot(first_direction, second_direction)) < 0.0:
                    second_direction = -second_direction
                alignment = float(
                    np.clip(np.dot(first_direction, second_direction), -1.0, 1.0)
                )
                if math.degrees(math.acos(alignment)) > 5.0:
                    continue
                second_short = second.long_short()[1]
                if min(first_short, second_short) / max(
                    first_short,
                    second_short,
                    EPS,
                ) < 0.60:
                    continue
                direction = first_direction + second_direction
                direction /= max(float(np.linalg.norm(direction)), EPS)
                normal = np.asarray([-direction[1], direction[0]], np.float32)
                center_delta = second.center() - first.center()
                along_delta = abs(float(np.dot(center_delta, direction)))
                across_delta = abs(float(np.dot(center_delta, normal)))
                gap = along_delta - 0.5 * (first_length + second_length)
                reference_short = max(first_short, second_short)
                if (
                    gap < -0.10 * min(first_length, second_length)
                    or gap > config.fragment_join_max_gap_short * reference_short
                    or across_delta > 0.35 * reference_short
                ):
                    continue
                if gap > 2.60 * reference_short:
                    first_along = np.asarray(first.quad, np.float32) @ direction
                    second_along = np.asarray(second.quad, np.float32) @ direction
                    if float(first.center() @ direction) <= float(
                        second.center() @ direction
                    ):
                        bridge_low = float(first_along.max())
                        bridge_high = float(second_along.min())
                    else:
                        bridge_low = float(second_along.max())
                        bridge_high = float(first_along.min())
                    first_across = np.asarray(first.quad, np.float32) @ normal
                    second_across = np.asarray(second.quad, np.float32) @ normal
                    bridge_across_low = max(
                        float(first_across.min()),
                        float(second_across.min()),
                    )
                    bridge_across_high = min(
                        float(first_across.max()),
                        float(second_across.max()),
                    )
                    if (
                        bridge_high <= bridge_low
                        or bridge_across_high <= bridge_across_low
                    ):
                        continue
                    bridge_quad = np.asarray(
                        [
                            direction * bridge_low
                            + normal * bridge_across_low,
                            direction * bridge_high
                            + normal * bridge_across_low,
                            direction * bridge_high
                            + normal * bridge_across_high,
                            direction * bridge_low
                            + normal * bridge_across_high,
                        ],
                        np.float32,
                    )
                    bridge_metrics = linear_metrics(native, bridge_quad)
                    if not (
                        bridge_metrics.get("orientation", 0.0) >= 0.50
                        and bridge_metrics.get("persistence", 0.0) >= 0.02
                        and bridge_metrics.get("transitions", 0.0) >= 6.0
                        and bridge_metrics.get("scanline_agreement", 0.0) >= 0.45
                    ):
                        continue
                points = np.vstack((first.quad, second.quad)).astype(np.float32)
                along = points @ direction
                across = points @ normal
                along_low, along_high = float(along.min()), float(along.max())
                across_low, across_high = float(across.min()), float(across.max())
                merged_quad = np.asarray(
                    [
                        direction * along_low + normal * across_low,
                        direction * along_high + normal * across_low,
                        direction * along_high + normal * across_high,
                        direction * along_low + normal * across_high,
                    ],
                    np.float32,
                )
                merged_metrics = linear_metrics(native, merged_quad)
                if not _accepted_linear_metrics(merged_metrics, config):
                    continue
                if best is None or gap < best[0]:
                    best = (
                        gap,
                        first_index,
                        second_index,
                        merged_quad,
                        merged_metrics,
                    )
        if best is None:
            break
        _, first_index, second_index, merged_quad, merged_metrics = best
        first = accepted[first_index]
        second = accepted[second_index]
        merged = Detection(
            quad=merged_quad,
            confidence=float(merged_metrics["score"]),
            source=f"{first.source}+{second.source}:fragment-join",
            metrics={
                **merged_metrics,
                "fragment_joined": 1.0,
                "fragment_count": float(
                    first.metrics.get("fragment_count", 1.0)
                    + second.metrics.get("fragment_count", 1.0)
                ),
            },
        )
        accepted = [
            item
            for index, item in enumerate(accepted)
            if index not in {first_index, second_index}
        ]
        accepted.append(merged)
        changed = True
    return _deduplicate(accepted, threshold=0.52), diagnostics


def locate_prepared(
    prepared: PreparedPage,
    config: TensorConfig = TensorConfig(),
) -> LocalizationResult:
    engine_started = perf_counter()
    if config.backend == "native-context":
        proposals, diagnostics, timings = _native_context_proposals(prepared, config)
    elif config.backend == "native":
        proposals, diagnostics, timings = _native_sttg_proposals(prepared, config)
    elif config.backend == "python":
        proposals, diagnostics, timings = _tile_tensor_proposals(prepared.work_gray, prepared, config)
        diagnostics["backend"] = "python-reference"
    else:
        raise ValueError(f"unknown tensor backend: {config.backend}")
    verification_started = perf_counter()
    accepted, verification_diagnostics = _verify_candidates(
        prepared.native_gray,
        proposals,
        config,
    )
    timings["verification_seconds"] = perf_counter() - verification_started
    timings["engine_seconds"] = perf_counter() - engine_started
    diagnostics.update(
        {
            **verification_diagnostics,
            "accepted_count": len(accepted),
            "work_width": prepared.work_gray.shape[1],
            "work_height": prepared.work_gray.shape[0],
            "work_pixels": int(prepared.work_gray.size),
            "native_pixels": int(prepared.native_gray.size),
        }
    )
    return LocalizationResult(accepted, timings, diagnostics)


def locate(
    image: np.ndarray,
    config: TensorConfig = TensorConfig(),
) -> tuple[PreparedPage, LocalizationResult]:
    prepared = prepare_page(image, config)
    return prepared, locate_prepared(prepared, config)


__all__ = [
    "Detection",
    "LocalizationResult",
    "PreparedPage",
    "TensorConfig",
    "linear_metrics",
    "locate",
    "locate_prepared",
    "prepare_page",
    "profile_config",
]
