"""Low-latency classical QR and Data Matrix extraction.

The candidate generator is intentionally permissive because it never produces
a user-visible result.  Only a checksum/error-correction-valid ZXing-C++ read
may leave this module.  The pipeline has three bounded stages:

1. one native LocalAverage page read, retaining failed matrix geometry;
2. compact candidate recovery around failed-read anchors;
3. bidirectional-gradient morphology followed by native crop decoding for
   regions not explained by either earlier stage.

Image loading and creation of the reduced work image belong to ``prepare_gray``
so queue-based callers can perform them before the measured compute stage.
Resizing time remains available as preparation diagnostics.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import zxingcpp

from barcode_detection.engines.common.classical_localizer import (
    local_qr_finder_count,
    matrix_proposal_metrics,
)


EPS = 1e-6
MATRIX_FORMATS = zxingcpp.barcode_formats_from_str("QRCode,DataMatrix")


@dataclass(frozen=True)
class Prepared2DPage:
    native_gray: np.ndarray
    work_gray: np.ndarray
    work_scale: float
    supplemental_layers: tuple[tuple[np.ndarray, float, int], ...]
    preparation_seconds: float


@dataclass
class MatrixCandidate:
    quad: np.ndarray
    source: str
    confidence: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.quad, np.float32).mean(axis=0)


@dataclass
class Classical2DResult:
    value: str
    symbology: str
    quad: np.ndarray
    source: str
    attempts: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecodeProfile:
    work_size: int = 1200
    supplemental_work_sizes: tuple[int, ...] = ()
    derive_smaller_layers_from_largest: bool = False
    morphology_sizes: tuple[int, ...] = (9, 17)
    morphology_percentiles: tuple[float, ...] = (99.55, 99.20)
    morphology_candidate_limit: int = 32
    anchor_attempt_limit: int = 6
    proposal_attempt_limit: int = 6
    scan_prepared_global: bool = False
    page_scan_workers: int = 1
    enable_global_fallback: bool = False
    global_try_downscale: bool = True
    global_try_invert: bool = True
    enable_morphology: bool = True
    enable_qr_finder_proposals: bool = False
    qr_candidate_limit: int = 12
    verify_morphology_structure: bool = True
    verify_error_anchors: bool = True
    morphology_only_without_results: bool = False


DEFAULT_PROFILE = DecodeProfile()


def prepare_gray(
    gray: np.ndarray,
    profile: DecodeProfile = DEFAULT_PROFILE,
) -> Prepared2DPage:
    """Create the reduced morphology image outside measured engine compute."""
    started = perf_counter()
    source = np.ascontiguousarray(gray, dtype=np.uint8)

    def layer(size: int) -> tuple[np.ndarray, float]:
        scale = min(1.0, float(size) / max(source.shape[:2]))
        work = (
            cv2.resize(
                source,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else source
        )
        return work, scale

    requested_sizes = (
        profile.work_size,
        *(
            size
            for size in profile.supplemental_work_sizes
            if size != profile.work_size
        ),
    )
    layers: dict[int, tuple[np.ndarray, float]] = {}
    if (
        profile.derive_smaller_layers_from_largest
        and len(requested_sizes) > 1
    ):
        largest_size = max(requested_sizes)
        largest, largest_scale = layer(largest_size)
        layers[largest_size] = (largest, largest_scale)
        for size in requested_sizes:
            if size == largest_size:
                continue
            scale = min(1.0, float(size) / max(source.shape[:2]))
            width = max(1, int(round(source.shape[1] * scale)))
            height = max(1, int(round(source.shape[0] * scale)))
            layers[size] = (
                cv2.resize(
                    largest,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                ),
                scale,
            )
    else:
        layers = {size: layer(size) for size in requested_sizes}

    work, scale = layers[profile.work_size]
    supplemental = tuple(
        (*layers[size], size)
        for size in profile.supplemental_work_sizes
        if size != profile.work_size
    )
    return Prepared2DPage(
        native_gray=source,
        work_gray=work,
        work_scale=scale,
        supplemental_layers=supplemental,
        preparation_seconds=perf_counter() - started,
    )


def _point(value: Any) -> np.ndarray:
    return np.asarray([float(value.x), float(value.y)], np.float32)


def _barcode_quad(
    barcode: Any,
    *,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    position = barcode.position
    points = np.asarray(
        [
            _point(position.top_left),
            _point(position.top_right),
            _point(position.bottom_right),
            _point(position.bottom_left),
        ],
        np.float32,
    )
    points /= scale
    points[:, 0] += offset[0]
    points[:, 1] += offset[1]
    return points


def _order_quad(points: np.ndarray) -> np.ndarray:
    source = np.asarray(points, np.float32).reshape(4, 2)
    output = np.empty((4, 2), np.float32)
    sums = source.sum(axis=1)
    differences = np.diff(source, axis=1).ravel()
    output[0] = source[np.argmin(sums)]
    output[2] = source[np.argmax(sums)]
    output[1] = source[np.argmin(differences)]
    output[3] = source[np.argmax(differences)]
    if len({tuple(point) for point in output}) != 4:
        center = source.mean(axis=0)
        cycle = source[
            np.argsort(
                np.arctan2(
                    source[:, 1] - center[1],
                    source[:, 0] - center[0],
                )
            )
        ]
        output = np.roll(cycle, -int(np.argmin(cycle.sum(axis=1))), axis=0)
    return output


def _intersection_over_smaller(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(np.asarray(first, np.float32))
    second_hull = cv2.convexHull(np.asarray(second, np.float32))
    first_area = abs(float(cv2.contourArea(first_hull)))
    second_area = abs(float(cv2.contourArea(second_hull)))
    if min(first_area, second_area) <= EPS:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    return max(0.0, float(intersection)) / min(first_area, second_area)


def _deduplicate_candidates(
    candidates: Iterable[MatrixCandidate],
) -> list[MatrixCandidate]:
    output: list[MatrixCandidate] = []
    for candidate in sorted(
        candidates,
        # When two decoder passes describe the same symbol, the tighter anchor
        # usually follows the actual finder/timing border more precisely.
        # Prefer it over a large error hull that may include neighbouring
        # symbols and make context recovery ambiguous.
        key=lambda item: (-item.confidence, abs(float(cv2.contourArea(item.quad)))),
    ):
        if any(
            _intersection_over_smaller(candidate.quad, prior.quad)
            >= (
                0.92
                if candidate.metrics.get("decoder_error_anchor", 0.0)
                and prior.metrics.get("decoder_error_anchor", 0.0)
                else 0.72
            )
            for prior in output
        ):
            continue
        output.append(candidate)
    return output


def _deduplicate_results(
    results: Iterable[Classical2DResult],
) -> list[Classical2DResult]:
    output: list[Classical2DResult] = []
    for candidate in results:
        if any(
            candidate.value == prior.value
            and _intersection_over_smaller(candidate.quad, prior.quad) >= 0.65
            for prior in output
        ):
            continue
        output.append(candidate)
    return output


def _read(
    image: np.ndarray,
    *,
    binarizer: Any,
    try_rotate: bool = True,
    try_downscale: bool = False,
    try_invert: bool = False,
    return_errors: bool = False,
    is_pure: bool = False,
) -> list[Any]:
    try:
        return list(
            zxingcpp.read_barcodes(
                image,
                formats=MATRIX_FORMATS,
                try_rotate=try_rotate,
                try_downscale=try_downscale,
                try_invert=try_invert,
                binarizer=binarizer,
                return_errors=return_errors,
                is_pure=is_pure,
            )
        )
    except Exception:
        return []


def _results_from_reads(
    reads: Sequence[Any],
    *,
    source: str,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    attempts: tuple[str, ...] = (),
) -> list[Classical2DResult]:
    output: list[Classical2DResult] = []
    for barcode in reads:
        if not barcode.valid or not barcode.text:
            continue
        quad = _barcode_quad(barcode, scale=scale, offset=offset)
        if abs(float(cv2.contourArea(quad))) <= EPS:
            continue
        output.append(
            Classical2DResult(
                value=str(barcode.text),
                symbology=str(barcode.format),
                quad=quad,
                source=source,
                attempts=attempts,
            )
        )
    return output


def _anchors_from_reads(
    reads: Sequence[Any],
    *,
    scale: float = 1.0,
    source: str = "zxing:matrix-error-anchor",
) -> list[MatrixCandidate]:
    output: list[MatrixCandidate] = []
    for barcode in reads:
        if barcode.valid and barcode.text:
            continue
        quad = _barcode_quad(barcode, scale=scale)
        if abs(float(cv2.contourArea(quad))) < 9.0:
            continue
        output.append(
            MatrixCandidate(
                quad=quad,
                source=source,
                confidence=0.98,
                metrics={
                    "decoder_error_anchor": 1.0,
                    "format_data_matrix": float(
                        str(barcode.format) == "Data Matrix"
                    ),
                    "format_qr": float(str(barcode.format) == "QR Code"),
                },
            )
        )
    return _deduplicate_candidates(output)


def _expanded_axis_crop(
    image: np.ndarray,
    quad: np.ndarray,
    expansion: float,
    *,
    minimum_side: int = 0,
) -> tuple[np.ndarray, int, int]:
    x, y, width, height = cv2.boundingRect(np.asarray(quad, np.float32))
    padding = int(round(expansion * max(width, height)))
    if minimum_side:
        padding = max(
            padding,
            int(np.ceil(max(0, minimum_side - width, minimum_side - height) / 2)),
        )
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)
    return image[y1:y2, x1:x2], x1, y1


def _rectified_view(
    image: np.ndarray,
    quad: np.ndarray,
    *,
    target_min_side: int = 180,
) -> tuple[np.ndarray, float, float]:
    ordered = _order_quad(quad)
    width = max(
        float(np.linalg.norm(ordered[1] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[3])),
    )
    height = max(
        float(np.linalg.norm(ordered[3] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[1])),
    )
    scale = max(1.0, min(4.0, target_min_side / max(1.0, min(width, height))))
    output_width = max(24, int(round(width * scale)))
    output_height = max(24, int(round(height * scale)))
    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        np.float32,
    )
    view = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(ordered, destination),
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    quiet = max(8, int(round(0.12 * min(view.shape))))
    view = cv2.copyMakeBorder(
        view,
        quiet,
        quiet,
        quiet,
        quiet,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    return view, width, height


def _decode_candidate(
    image: np.ndarray,
    candidate: MatrixCandidate,
    *,
    attempt_limit: int,
) -> tuple[Classical2DResult | None, int, tuple[str, ...]]:
    """Run a bounded evidence-selected candidate portfolio."""
    attempts: list[str] = []
    center = candidate.center

    def accept(
        reads: Sequence[Any],
        *,
        source: str,
        scale: float = 1.0,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> Classical2DResult | None:
        mapped = _results_from_reads(
            reads,
            source=source,
            scale=scale,
            offset=offset,
            attempts=tuple(attempts),
        )
        if not mapped:
            return None
        mapped = [
            item
            for item in mapped
            if _intersection_over_smaller(item.quad, candidate.quad) >= 0.30
            and (
                not candidate.metrics.get("format_data_matrix", 0.0)
                or item.symbology == "Data Matrix"
            )
            and (
                not candidate.metrics.get("format_qr", 0.0)
                or item.symbology == "QR Code"
            )
        ]
        if not mapped:
            return None
        mapped.sort(
            key=lambda item: float(
                np.linalg.norm(np.asarray(item.quad).mean(axis=0) - center)
            )
        )
        return mapped[0]

    # Decoder-error geometry is much stronger than a generic morphology
    # proposal.  Preserve the proven fixed-context recipes from Adaptive 5.1
    # first: on tiny document Data Matrix symbols, a relative crop can be too
    # tight even when it appears generously padded.
    if candidate.metrics.get("decoder_error_anchor", 0.0):
        rectangle = cv2.minAreaRect(np.asarray(candidate.quad, np.float32))
        anchor_side = max(8.0, float(max(rectangle[1])))
        relative_multiplier = 3.4
        relative_size = int(
            np.clip(
                round(anchor_side * relative_multiplier),
                120,
                min(image.shape[:2]),
            )
        )
        relative_scale = (
            3.0
            if anchor_side < 45
            else 2.0
            if anchor_side < 90
            else 1.5
            if anchor_side < 180
            else 1.0
        )
        relative_crop, relative_x, relative_y = _fixed_center_crop(
            image,
            center,
            relative_size,
        )
        relative_work = (
            relative_crop
            if relative_scale == 1.0
            else cv2.resize(
                relative_crop,
                None,
                fx=relative_scale,
                fy=relative_scale,
                interpolation=cv2.INTER_CUBIC,
            )
        )
        for binarizer, name in (
            (
                zxingcpp.Binarizer.GlobalHistogram,
                "relative34-global-pyramid",
            ),
            (
                zxingcpp.Binarizer.LocalAverage,
                "relative34-local-pyramid",
            ),
        ):
            if len(attempts) >= attempt_limit:
                break
            attempts.append(name)
            result = accept(
                _read(
                    relative_work,
                    binarizer=binarizer,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=False,
                ),
                source=f"{candidate.source}:{name}",
                scale=relative_scale,
                offset=(float(relative_x), float(relative_y)),
            )
            if result is not None:
                return result, len(attempts), tuple(attempts)

        for crop_size, scale, binarizer, name in (
            (
                500,
                1.5,
                zxingcpp.Binarizer.LocalAverage,
                "anchor500-15-local",
            ),
            (
                500,
                1.5,
                zxingcpp.Binarizer.GlobalHistogram,
                "anchor500-15-global",
            ),
            (
                300,
                2.0,
                zxingcpp.Binarizer.LocalAverage,
                "anchor300-20-local",
            ),
            (
                300,
                2.0,
                zxingcpp.Binarizer.GlobalHistogram,
                "anchor300-20-global",
            ),
        ):
            if len(attempts) >= attempt_limit:
                break
            crop, x1, y1 = _fixed_center_crop(
                image,
                center,
                crop_size,
            )
            work = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            attempts.append(name)
            result = accept(
                _read(
                    work,
                    binarizer=binarizer,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=False,
                ),
                source=f"{candidate.source}:{name}",
                scale=scale,
                offset=(float(x1), float(y1)),
            )
            if result is not None:
                return result, len(attempts), tuple(attempts)

    crop_recipes = (
        (0.70, 220, 1.0, cv2.INTER_AREA, zxingcpp.Binarizer.LocalAverage, "context-local"),
        (0.70, 220, 1.5, cv2.INTER_CUBIC, zxingcpp.Binarizer.LocalAverage, "context15-local"),
        (0.70, 220, 1.5, cv2.INTER_CUBIC, zxingcpp.Binarizer.GlobalHistogram, "context15-global"),
        (1.40, 320, 1.0, cv2.INTER_AREA, zxingcpp.Binarizer.LocalAverage, "wide-local"),
    )
    for expansion, minimum, scale, interpolation, binarizer, name in crop_recipes:
        if len(attempts) >= attempt_limit:
            break
        crop, x1, y1 = _expanded_axis_crop(
            image,
            candidate.quad,
            expansion,
            minimum_side=minimum,
        )
        if crop.size == 0:
            continue
        work = (
            crop
            if scale == 1.0
            else cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=interpolation,
            )
        )
        attempts.append(name)
        result = accept(
            _read(
                work,
                binarizer=binarizer,
                try_rotate=True,
                try_downscale=False,
                try_invert=False,
            ),
            source=f"{candidate.source}:{name}",
            scale=scale,
            offset=(float(x1), float(y1)),
        )
        if result is not None:
            return result, len(attempts), tuple(attempts)

    # Rectification is useful only after the ordinary context views fail.  The
    # pure reader skips finder detection, so it is cheap despite being late.
    for binarizer, name in (
        (zxingcpp.Binarizer.LocalAverage, "rectified-pure-local"),
        (zxingcpp.Binarizer.GlobalHistogram, "rectified-pure-global"),
    ):
        if len(attempts) >= attempt_limit:
            break
        rectified, _width, _height = _rectified_view(image, candidate.quad)
        attempts.append(name)
        reads = _read(
            rectified,
            binarizer=binarizer,
            try_rotate=False,
            try_downscale=False,
            try_invert=False,
            is_pure=True,
        )
        valid = next(
            (
                barcode
                for barcode in reads
                if barcode.valid and barcode.text
            ),
            None,
        )
        if valid is not None:
            # Pure mode reports crop coordinates.  The candidate's verified
            # source geometry remains the truthful page position.
            return (
                Classical2DResult(
                    value=str(valid.text),
                    symbology=str(valid.format),
                    quad=np.asarray(candidate.quad, np.float32),
                    source=f"{candidate.source}:{name}",
                    attempts=tuple(attempts),
                ),
                len(attempts),
                tuple(attempts),
            )
    return None, len(attempts), tuple(attempts)


def _fixed_center_crop(
    image: np.ndarray,
    center: np.ndarray,
    size: int,
) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    size = min(size, width, height)
    x1 = int(round(float(center[0]) - size / 2.0))
    y1 = int(round(float(center[1]) - size / 2.0))
    x1 = min(max(0, x1), max(0, width - size))
    y1 = min(max(0, y1), max(0, height - size))
    return image[y1 : y1 + size, x1 : x1 + size], x1, y1


def _qr_finder_count(mask: np.ndarray) -> int:
    contours, hierarchy_raw = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy_raw is None:
        return 0
    hierarchy = hierarchy_raw[0]
    count = 0
    for index in range(len(contours)):
        boxes: list[tuple[int, int, int, int]] = []
        current = index
        for _ in range(3):
            if current < 0:
                boxes = []
                break
            box = cv2.boundingRect(contours[current])
            short_side, long_side = min(box[2], box[3]), max(box[2], box[3])
            if short_side < 3 or long_side / max(1.0, short_side) > 1.35:
                boxes = []
                break
            boxes.append(box)
            current = int(hierarchy[current][2])
        if len(boxes) != 3:
            continue
        centers = [
            (x + width / 2.0, y + height / 2.0)
            for x, y, width, height in boxes
        ]
        tolerance = 0.17 * max(boxes[0][2], boxes[0][3])
        if any(
            abs(point[0] - centers[0][0]) > tolerance
            or abs(point[1] - centers[0][1]) > tolerance
            for point in centers[1:]
        ):
            continue
        count += 1
    return count


def _qr_finder_points(
    mask: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """Return deduplicated centers of nested 7x7 QR finder hypotheses."""
    contours, hierarchy_raw = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy_raw is None:
        return []
    hierarchy = hierarchy_raw[0]
    hypotheses: list[tuple[np.ndarray, float]] = []
    for index in range(len(contours)):
        boxes: list[tuple[int, int, int, int]] = []
        current = index
        for _ in range(3):
            if current < 0:
                boxes = []
                break
            box = cv2.boundingRect(contours[current])
            short_side, long_side = min(box[2], box[3]), max(box[2], box[3])
            if short_side < 4 or long_side / max(1.0, short_side) > 1.35:
                boxes = []
                break
            boxes.append(box)
            current = int(hierarchy[current][2])
        if len(boxes) != 3:
            continue
        centers = np.asarray(
            [
                (x + width / 2.0, y + height / 2.0)
                for x, y, width, height in boxes
            ],
            np.float32,
        )
        outer_side = float(max(boxes[0][2], boxes[0][3]))
        tolerance = 0.17 * outer_side
        if np.max(np.linalg.norm(centers[1:] - centers[0], axis=1)) > tolerance:
            continue
        middle_side = float(max(boxes[1][2], boxes[1][3]))
        inner_side = float(max(boxes[2][2], boxes[2][3]))
        if not (
            0.35 <= middle_side / max(1.0, outer_side) <= 0.88
            and 0.35 <= inner_side / max(1.0, middle_side) <= 0.88
        ):
            continue
        hypotheses.append((centers.mean(axis=0), outer_side))

    output: list[tuple[np.ndarray, float]] = []
    for center, side in sorted(hypotheses, key=lambda item: -item[1]):
        if any(
            float(np.linalg.norm(center - prior_center))
            <= 0.45 * max(side, prior_side)
            for prior_center, prior_side in output
        ):
            continue
        output.append((center, side))
    return output


def qr_finder_candidates(
    prepared: Prepared2DPage,
    profile: DecodeProfile = DEFAULT_PROFILE,
) -> tuple[list[MatrixCandidate], dict[str, float]]:
    """Build full QR search boxes from three nested finder patterns."""
    started = perf_counter()
    work = prepared.work_gray
    masks = (
        cv2.threshold(
            work,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1],
        cv2.adaptiveThreshold(
            work,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        ),
    )
    points: list[tuple[np.ndarray, float]] = []
    for mask in masks:
        points.extend(_qr_finder_points(mask))
    deduplicated: list[tuple[np.ndarray, float]] = []
    for center, side in sorted(points, key=lambda item: -item[1]):
        if any(
            float(np.linalg.norm(center - prior_center))
            <= 0.55 * max(side, prior_side)
            for prior_center, prior_side in deduplicated
        ):
            continue
        deduplicated.append((center, side))
    deduplicated = deduplicated[:18]

    candidates: list[MatrixCandidate] = []
    for triplet in combinations(deduplicated, 3):
        centers = [item[0] for item in triplet]
        distances = {
            (0, 1): float(np.linalg.norm(centers[0] - centers[1])),
            (0, 2): float(np.linalg.norm(centers[0] - centers[2])),
            (1, 2): float(np.linalg.norm(centers[1] - centers[2])),
        }
        hypotenuse = max(distances, key=distances.get)
        right_index = next(
            index for index in range(3) if index not in hypotenuse
        )
        leg_indices = [index for index in range(3) if index != right_index]
        origin = centers[right_index]
        first = centers[leg_indices[0]]
        second = centers[leg_indices[1]]
        first_vector = first - origin
        second_vector = second - origin
        first_length = float(np.linalg.norm(first_vector))
        second_length = float(np.linalg.norm(second_vector))
        finder_side = float(
            np.mean([item[1] for item in triplet])
        )
        if (
            min(first_length, second_length) < 2.0 * finder_side
            or max(first_length, second_length)
            / max(EPS, min(first_length, second_length))
            > 2.6
        ):
            continue
        orthogonality = abs(
            float(np.dot(first_vector, second_vector))
        ) / max(EPS, first_length * second_length)
        if orthogonality > 0.38:
            continue
        first_unit = first_vector / first_length
        second_unit = second_vector / second_length
        cross_product = float(
            first_vector[0] * second_vector[1]
            - first_vector[1] * second_vector[0]
        )
        if cross_product < 0:
            first, second = second, first
            first_vector, second_vector = second_vector, first_vector
            first_unit, second_unit = second_unit, first_unit
        margin = 0.62 * finder_side
        quad = np.asarray(
            [
                origin - margin * first_unit - margin * second_unit,
                first + margin * first_unit - margin * second_unit,
                first
                + second
                - origin
                + margin * first_unit
                + margin * second_unit,
                second - margin * first_unit + margin * second_unit,
            ],
            np.float32,
        )
        candidates.append(
            MatrixCandidate(
                quad=quad / prepared.work_scale,
                source="morphology:qr-three-finder-geometry",
                confidence=max(0.70, 0.97 - 0.35 * orthogonality),
                metrics={
                    "format_qr": 1.0,
                    "qr_finder_count": 3.0,
                    "qr_orthogonality_error": orthogonality,
                },
            )
        )

    output = _deduplicate_candidates(candidates)[
        : profile.qr_candidate_limit
    ]
    return output, {
        "seconds": perf_counter() - started,
        "finder_points": float(len(deduplicated)),
        "candidates": float(len(output)),
    }


def morphology_candidates(
    prepared: Prepared2DPage,
    profile: DecodeProfile = DEFAULT_PROFILE,
) -> tuple[list[MatrixCandidate], dict[str, float]]:
    """Generate permissive QR/Data Matrix recovery regions on a thumbnail."""
    started = perf_counter()
    work = prepared.work_gray
    gradient_x = np.abs(cv2.Scharr(work, cv2.CV_32F, 1, 0))
    gradient_y = np.abs(cv2.Scharr(work, cv2.CV_32F, 0, 1))
    candidates: list[MatrixCandidate] = []
    contour_count = 0
    response_seconds = perf_counter() - started
    threshold_started = perf_counter()

    kernel_cache: dict[int, np.ndarray] = {}
    for local_size in profile.morphology_sizes:
        energy_x = cv2.boxFilter(
            gradient_x,
            cv2.CV_32F,
            (local_size, local_size),
            normalize=True,
        )
        energy_y = cv2.boxFilter(
            gradient_y,
            cv2.CV_32F,
            (local_size, local_size),
            normalize=True,
        )
        response = np.sqrt(energy_x * energy_y)
        thresholds = np.percentile(response, profile.morphology_percentiles)
        close_size = max(5, int(round(local_size * 0.65)) | 1)
        kernel = kernel_cache.setdefault(
            close_size,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (close_size, close_size),
            ),
        )
        for threshold in np.atleast_1d(thresholds):
            mask = np.asarray(response >= threshold, np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contour_count += len(contours)
            for contour in contours:
                rectangle = cv2.minAreaRect(contour)
                width, height = (float(value) for value in rectangle[1])
                short_side, long_side = min(width, height), max(width, height)
                if (
                    short_side < 7
                    or long_side > 0.36 * max(work.shape)
                    or long_side / max(1.0, short_side) > 5.0
                ):
                    continue
                work_quad = cv2.boxPoints(rectangle).astype(np.float32)
                native_quad = work_quad / prepared.work_scale
                # Candidate-level QR evidence is diagnostic only.  One finder
                # is enough to raise ordering priority, never to accept output.
                x, y, box_width, box_height = cv2.boundingRect(
                    np.asarray(work_quad, np.int32)
                )
                pad = max(4, int(round(0.45 * max(box_width, box_height))))
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2 = min(work.shape[1], x + box_width + pad)
                y2 = min(work.shape[0], y + box_height + pad)
                roi = work[y1:y2, x1:x2]
                finder_count = 0
                if roi.size:
                    binary = cv2.threshold(
                        roi,
                        0,
                        255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                    )[1]
                    finder_count = _qr_finder_count(binary)
                score = (
                    0.55
                    + 0.08 * min(3, finder_count)
                    + 0.04 * min(3, local_size / 9.0)
                )
                candidates.append(
                    MatrixCandidate(
                        quad=native_quad,
                        source=f"morphology:bidirectional:l{local_size}",
                        confidence=score,
                        metrics={
                            "local_size": float(local_size),
                            "qr_finder_chains": float(finder_count),
                        },
                    )
                )

    output = _deduplicate_candidates(candidates)[
        : profile.morphology_candidate_limit
    ]
    return output, {
        "gradient_seconds": response_seconds,
        "response_threshold_seconds": perf_counter() - threshold_started,
        "contours": float(contour_count),
        "candidates": float(len(output)),
    }


def decode_prepared(
    prepared: Prepared2DPage,
    profile: DecodeProfile = DEFAULT_PROFILE,
) -> tuple[list[Classical2DResult], dict[str, Any]]:
    """Decode one already prepared page using the bounded classical cascade."""
    engine_started = perf_counter()
    timings: dict[str, float] = {}

    # ZXing-C++'s built-in pyramid rescans the native image before allocating
    # reduced layers.  The prepared contract owns the exact useful scales, so
    # scan those directly.  Two workers overlap the native pass with the tiny
    # prepared passes; fixed result ordering keeps output deterministic.
    work_layers = (
        (prepared.work_gray, prepared.work_scale, profile.work_size),
        *prepared.supplemental_layers,
    )
    scan_specs: list[tuple[str, np.ndarray, float, Any, str]] = [
        (
            "direct_page_seconds",
            prepared.native_gray,
            1.0,
            zxingcpp.Binarizer.LocalAverage,
            "zxing:matrix-native-local",
        )
    ]
    for layer_image, layer_scale, layer_size in work_layers:
        if layer_scale >= 0.999:
            continue
        scan_specs.append(
            (
                f"prepared_{layer_size}_seconds",
                layer_image,
                layer_scale,
                zxingcpp.Binarizer.LocalAverage,
                f"zxing:matrix-prepared-{layer_size}-local",
            )
        )
        if profile.scan_prepared_global:
            scan_specs.append(
                (
                    f"prepared_{layer_size}_global_seconds",
                    layer_image,
                    layer_scale,
                    zxingcpp.Binarizer.GlobalHistogram,
                    f"zxing:matrix-prepared-{layer_size}-global",
                )
            )

    def execute_scan(
        specification: tuple[str, np.ndarray, float, Any, str],
    ) -> tuple[list[Any], float]:
        _timing_key, image, _scale, binarizer, _source = specification
        scan_started = perf_counter()
        values = _read(
            image,
            binarizer=binarizer,
            try_rotate=True,
            try_downscale=False,
            try_invert=False,
            return_errors=True,
        )
        return values, perf_counter() - scan_started

    scan_wall_started = perf_counter()
    if profile.page_scan_workers > 1 and len(scan_specs) > 1:
        with ThreadPoolExecutor(
            max_workers=min(profile.page_scan_workers, len(scan_specs))
        ) as executor:
            scan_outputs = list(executor.map(execute_scan, scan_specs))
    else:
        scan_outputs = [execute_scan(item) for item in scan_specs]
    timings["page_scan_wall_seconds"] = (
        perf_counter() - scan_wall_started
    )

    reads: list[Any] = []
    work_reads: list[Any] = []
    results: list[Classical2DResult] = []
    anchors: list[MatrixCandidate] = []
    for index, (specification, scan_output) in enumerate(
        zip(scan_specs, scan_outputs)
    ):
        timing_key, _image, layer_scale, _binarizer, source = specification
        layer_reads, task_seconds = scan_output
        timings[timing_key] = task_seconds
        if index == 0:
            reads = layer_reads
        else:
            work_reads.extend(layer_reads)
        suffix = source.removeprefix("zxing:matrix-")
        results.extend(
            _results_from_reads(
                layer_reads,
                source=f"{source}-valid",
                scale=layer_scale,
                attempts=(f"{suffix}-valid",),
            )
        )
        anchors.extend(
            _anchors_from_reads(
                layer_reads,
                scale=layer_scale,
                source=f"{source}-error-anchor",
            )
        )
    anchors = _deduplicate_candidates(anchors)
    results = _deduplicate_results(results)

    global_reads: list[Any] = []
    if profile.enable_global_fallback:
        started = perf_counter()
        global_reads = _read(
            prepared.native_gray,
            binarizer=zxingcpp.Binarizer.GlobalHistogram,
            try_rotate=True,
            try_downscale=profile.global_try_downscale,
            try_invert=profile.global_try_invert,
            return_errors=True,
        )
        timings["global_fallback_seconds"] = perf_counter() - started
        results.extend(
            _results_from_reads(
                global_reads,
                source="zxing:matrix-global-pyramid-valid",
                attempts=("global-pyramid-valid",),
            )
        )
        anchors.extend(
            _anchors_from_reads(
                global_reads,
                source="zxing:matrix-global-pyramid-error-anchor",
            )
        )
        anchors = _deduplicate_candidates(anchors)
        results = _deduplicate_results(results)

    started = perf_counter()
    anchor_attempts = 0
    anchor_recovered = 0
    anchor_rejected = 0
    for anchor in anchors:
        if any(
            _intersection_over_smaller(anchor.quad, result.quad) >= 0.35
            for result in results
        ):
            continue
        if (
            profile.verify_error_anchors
            and anchor.metrics.get("format_qr", 0.0)
        ):
            metrics = matrix_proposal_metrics(
                prepared.native_gray,
                anchor.quad,
            )
            finder_count = local_qr_finder_count(
                prepared.native_gray,
                anchor.quad,
            )
            anchor.metrics.update(
                {
                    f"physical_{key}": float(value)
                    for key, value in metrics.items()
                }
            )
            anchor.metrics["qr_finder_count"] = float(finder_count)
            if not bool(metrics["accepted"]) and finder_count < 1:
                anchor_rejected += 1
                continue
        result, attempts, _ = _decode_candidate(
            prepared.native_gray,
            anchor,
            attempt_limit=profile.anchor_attempt_limit,
        )
        anchor_attempts += attempts
        if result is not None:
            results.append(result)
            anchor_recovered += 1
    timings["anchor_recovery_seconds"] = perf_counter() - started

    qr_candidates: list[MatrixCandidate] = []
    qr_candidate_info: dict[str, float] = {}
    qr_proposal_attempts = qr_proposal_recovered = 0
    if profile.enable_qr_finder_proposals:
        started = perf_counter()
        qr_candidates, qr_candidate_info = qr_finder_candidates(
            prepared,
            profile,
        )
        timings["qr_finder_proposal_seconds"] = (
            perf_counter() - started
        )
        started = perf_counter()
        covered = [result.quad for result in results]
        for candidate in qr_candidates:
            if any(
                _intersection_over_smaller(candidate.quad, prior) >= 0.35
                for prior in covered
            ):
                continue
            result, attempts, _ = _decode_candidate(
                prepared.native_gray,
                candidate,
                attempt_limit=profile.proposal_attempt_limit,
            )
            qr_proposal_attempts += attempts
            if result is None:
                continue
            results.append(result)
            covered.append(result.quad)
            qr_proposal_recovered += 1
        timings["qr_finder_decode_seconds"] = perf_counter() - started

    morphology: list[MatrixCandidate] = []
    morphology_info: dict[str, float] = {}
    unexplained_morphology: list[MatrixCandidate] = []
    verified_morphology: list[MatrixCandidate] = []
    proposal_attempts = proposal_recovered = 0
    if (
        profile.enable_morphology
        and (
            not profile.morphology_only_without_results
            or not results
        )
    ):
        started = perf_counter()
        morphology, morphology_info = morphology_candidates(prepared, profile)
        timings["morphology_seconds"] = perf_counter() - started
        # A failed decoder anchor is evidence, not coverage.  A tighter
        # morphology proposal nested inside a failed anchor may be the only
        # usable crop (for example, two adjacent Data Matrix symbols inside
        # one coarse error box).  Only successfully decoded geometry may
        # suppress later candidates.
        covered = [result.quad for result in results]
        unexplained_morphology = [
            candidate
            for candidate in morphology
            if not any(
                _intersection_over_smaller(candidate.quad, prior) >= 0.35
                for prior in covered
            )
        ]
        started = perf_counter()
        if profile.verify_morphology_structure:
            for candidate in unexplained_morphology:
                metrics = matrix_proposal_metrics(
                    prepared.native_gray,
                    candidate.quad,
                )
                candidate.metrics.update(
                    {
                        f"physical_{key}": float(value)
                        for key, value in metrics.items()
                    }
                )
                if bool(metrics["accepted"]):
                    candidate.confidence = max(
                        candidate.confidence,
                        float(metrics["score"]),
                    )
                    verified_morphology.append(candidate)
        else:
            verified_morphology = unexplained_morphology
        timings["morphology_verification_seconds"] = (
            perf_counter() - started
        )
        started = perf_counter()
        for candidate in verified_morphology:
            result, attempts, _ = _decode_candidate(
                prepared.native_gray,
                candidate,
                attempt_limit=profile.proposal_attempt_limit,
            )
            proposal_attempts += attempts
            if result is None:
                continue
            results.append(result)
            covered.append(result.quad)
            proposal_recovered += 1
        timings["proposal_decode_seconds"] = perf_counter() - started

    output = _deduplicate_results(results)
    timings["engine_seconds"] = perf_counter() - engine_started
    return output, {
        "timings": timings,
        "preparation_seconds": prepared.preparation_seconds,
        "direct_reads": len(reads) + len(work_reads) + len(global_reads),
        "direct_valid": sum(
            1 for barcode in reads if barcode.valid and barcode.text
        )
        + sum(
            1 for barcode in work_reads if barcode.valid and barcode.text
        )
        + sum(
            1 for barcode in global_reads if barcode.valid and barcode.text
        ),
        "error_anchors": len(anchors),
        "anchor_attempts": anchor_attempts,
        "anchor_recovered": anchor_recovered,
        "anchor_rejected": anchor_rejected,
        "qr_finder_proposals": qr_candidate_info,
        "qr_candidates": len(qr_candidates),
        "qr_proposal_attempts": qr_proposal_attempts,
        "qr_proposal_recovered": qr_proposal_recovered,
        "morphology": morphology_info,
        "morphology_candidates": len(morphology),
        "morphology_unexplained": len(unexplained_morphology),
        "morphology_verified": len(verified_morphology),
        "proposal_attempts": proposal_attempts,
        "proposal_recovered": proposal_recovered,
        "decoded": len(output),
    }
