# SPDX-License-Identifier: Apache-2.0
"""Scale-aware barcode helpers shared by the production pipeline.

The original pipeline was calibrated on large, 200-DPI document labels.  This
module keeps that proven route intact and adds conservative measurements for
small web images: scale-space proposal plans, pixels-per-module estimates, and
an evidence gate for candidates shorter than the legacy absolute length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

EPS = 1e-6


_EAN13_LEFT_ODD = (
    "0001101", "0011001", "0010011", "0111101", "0100011",
    "0110001", "0101111", "0111011", "0110111", "0001011",
)
_EAN13_LEFT_EVEN = (
    "0100111", "0110011", "0011011", "0100001", "0011101",
    "0111001", "0000101", "0010001", "0001001", "0010111",
)
_EAN13_RIGHT = (
    "1110010", "1100110", "1101100", "1000010", "1011100",
    "1001110", "1010000", "1000100", "1001000", "1110100",
)
_EAN13_PARITY = (
    "LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
    "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL",
)


@dataclass(frozen=True)
class ScalePlan:
    scale: float
    interpolation: int
    label: str


def as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ean13_checksum_valid(payload: str) -> bool:
    """Return whether *payload* is a complete checksum-valid EAN-13 value."""
    if len(payload) != 13 or not payload.isdigit():
        return False
    digits = [int(value) for value in payload]
    weighted = sum(
        value * (1 if index % 2 == 0 else 3)
        for index, value in enumerate(digits[:12])
    )
    return digits[-1] == (-weighted) % 10


def _ean13_modules(payload: str) -> np.ndarray:
    digits = [int(value) for value in payload]
    parity = _EAN13_PARITY[digits[0]]
    left = "".join(
        (_EAN13_LEFT_ODD if family == "L" else _EAN13_LEFT_EVEN)[digit]
        for digit, family in zip(digits[1:7], parity)
    )
    right = "".join(_EAN13_RIGHT[digit] for digit in digits[7:])
    return np.fromiter((int(value) for value in f"101{left}01010{right}101"), np.float32)


def _ean13_nearby_valid_modules(payload: str) -> list[np.ndarray]:
    """Build the closest checksum-valid competitors for a confidence margin."""
    digits = [int(value) for value in payload]
    output: list[np.ndarray] = []
    for index in range(12):
        for replacement in range(10):
            if replacement == digits[index]:
                continue
            candidate = digits[:]
            candidate[index] = replacement
            weighted = sum(
                value * (1 if position % 2 == 0 else 3)
                for position, value in enumerate(candidate[:12])
            )
            candidate[-1] = (-weighted) % 10
            output.append(_ean13_modules("".join(str(value) for value in candidate)))
    return output


def _normalized_profile(values: np.ndarray) -> np.ndarray | None:
    profile = np.asarray(values, dtype=np.float32).ravel()
    if profile.size < 60:
        return None
    low, high = np.percentile(profile, (5.0, 95.0))
    if high - low < 3.0:
        return None
    return np.clip((profile - low) / (high - low), 0.0, 1.0)


def _fit_binary_template(
    profile: np.ndarray,
    modules: np.ndarray,
) -> tuple[float, int, int, int]:
    """Fit a module pattern to a 1-D response without inventing pixels."""
    width = int(profile.size)
    minimum_symbol = max(60, int(round(width * 0.64)))
    best = (-1.0, 0, 0, 0)
    normalized = (profile - float(np.mean(profile))) / (float(np.std(profile)) + EPS)
    for symbol_width in range(minimum_symbol, width + 1):
        for model_index, model in enumerate(_resampled_binary_models(modules, symbol_width)):
            model = (model - float(np.mean(model))) / (float(np.std(model)) + EPS)
            correlations = np.correlate(normalized, model, mode="valid") / symbol_width
            start = int(np.argmax(correlations))
            correlation = float(correlations[start])
            if correlation > best[0]:
                best = (correlation, start, symbol_width, model_index)
    return best


def _resampled_binary_models(modules: np.ndarray, width: int) -> list[np.ndarray]:
    """Model common rasterizers and sub-pixel phases at undersampled widths."""
    source = modules.reshape(1, -1)
    high_resolution = np.repeat(modules, 16).reshape(1, -1)
    models = [
        cv2.resize(source, (width, 1), interpolation=interpolation).ravel()
        for interpolation in (
            cv2.INTER_AREA,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_NEAREST,
        )
    ]
    models.extend(
        cv2.resize(high_resolution, (width, 1), interpolation=interpolation).ravel()
        for interpolation in (cv2.INTER_AREA, cv2.INTER_CUBIC)
    )
    for phase in np.linspace(-0.5, 0.5, 5):
        positions = (np.arange(width, dtype=np.float32) + 0.5 + phase) * 95.0 / width
        models.append(modules[np.clip(positions.astype(np.int32), 0, 94)])
    return models


def repeated_ean13_template_evidence(
    image: np.ndarray,
    quad: np.ndarray,
    payload: str,
    rectify: Callable[..., np.ndarray],
) -> dict[str, float] | None:
    """Independently verify a decoded EAN-13 template in another candidate.

    Dense catalogue images often repeat one symbol using several foreground and
    background colours.  ZXing may decode one high-contrast copy while the
    others remain below one source pixel per module.  This verifier tests the
    known checksum-valid payload against the *pixels of each unresolved box*.
    It also compares the fit with every one-digit checksum-valid neighbour and
    requires agreement in multiple vertical scan bands.  It never promotes a
    candidate merely because a nearby barcode had the same format.
    """
    if not ean13_checksum_valid(payload):
        return None
    patch = rectify(image, quad, pad_long=0.0, pad_short=0.0)
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    if patch.shape[0] > patch.shape[1]:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    height, width = patch.shape[:2]
    if width < 60 or height < 12 or width / max(1.0, height) < 1.15:
        return None

    expected = _ean13_modules(payload)
    alternatives = _ean13_nearby_valid_modules(payload)
    band_results: list[tuple[float, float, int, int]] = []
    edge_width = max(2, width // 12)
    for top_fraction, bottom_fraction in ((0.03, 0.55), (0.10, 0.65), (0.20, 0.72)):
        top = int(round(height * top_fraction))
        bottom = max(top + 3, int(round(height * bottom_fraction)))
        band = patch[top:min(height, bottom)].astype(np.float32)
        if band.shape[0] < 3:
            continue
        edge_pixels = np.concatenate(
            (
                band[:, :edge_width].reshape(-1, 3),
                band[:, -edge_width:].reshape(-1, 3),
            ),
            axis=0,
        )
        background = np.median(edge_pixels, axis=0)
        # Euclidean colour distance works for black, blue, green, brown, and
        # ordinary grayscale bars while remaining invariant to their hue.
        projections = [np.quantile(np.linalg.norm(band - background, axis=2), 0.65, axis=0)]
        best_band: tuple[float, float, int, int] | None = None
        for values in projections:
            profile = _normalized_profile(values)
            if profile is None:
                continue
            correlation, start, symbol_width, model_index = _fit_binary_template(profile, expected)
            observed = profile[start : start + symbol_width]
            observed = (observed - float(np.mean(observed))) / (float(np.std(observed)) + EPS)
            alternative_correlation = -1.0
            for alternative in alternatives:
                models = _resampled_binary_models(alternative, symbol_width)
                model = models[model_index]
                model = (model - float(np.mean(model))) / (float(np.std(model)) + EPS)
                alternative_correlation = max(
                    alternative_correlation,
                    float(np.mean(observed * model)),
                )
            result = (correlation, correlation - alternative_correlation, start, symbol_width)
            if best_band is None or (result[0] + result[1]) > (best_band[0] + best_band[1]):
                best_band = result
        if best_band is not None:
            band_results.append(best_band)

    supported = [
        result
        for result in band_results
        if result[0] >= 0.68 and result[1] >= 0.012
    ]
    if len(supported) < 2:
        return None
    correlations = np.asarray([result[0] for result in supported], dtype=np.float32)
    margins = np.asarray([result[1] for result in supported], dtype=np.float32)
    starts = np.asarray([result[2] for result in supported], dtype=np.float32)
    symbol_widths = np.asarray([result[3] for result in supported], dtype=np.float32)
    if (
        float(np.median(correlations)) < 0.70
        or float(np.median(margins)) < 0.016
        or float(np.ptp(starts)) > max(3.0, width * 0.06)
        or float(np.ptp(symbol_widths)) > max(3.0, width * 0.06)
    ):
        return None
    symbol_width = float(np.median(symbol_widths))
    return {
        "ean13_template_correlation": round(float(np.median(correlations)), 4),
        "ean13_template_margin": round(float(np.median(margins)), 4),
        "ean13_template_scan_bands": float(len(supported)),
        "ean13_symbol_width_px": round(symbol_width, 2),
        "ean13_pixels_per_module": round(symbol_width / 95.0, 4),
    }


def color_linear_decode_views(
    image: np.ndarray,
    quad: np.ndarray,
    rectify: Callable[..., np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Create dark-on-light views that preserve chromatic bar contrast."""
    if image.ndim != 3 or image.shape[2] < 3:
        return []
    patch = rectify(image, quad, pad_long=0.02, pad_short=0.02)
    if patch.shape[0] > patch.shape[1]:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    height, width = patch.shape[:2]
    if width < 36 or height < 8:
        return []
    sample = patch.astype(np.float32)
    edge_width = max(2, width // 12)
    edge_pixels = np.concatenate(
        (
            sample[:, :edge_width].reshape(-1, 3),
            sample[:, -edge_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(edge_pixels, axis=0)
    distances = [
        ("bgr-distance", np.linalg.norm(sample - background, axis=2)),
        *(
            (f"channel-{channel}-distance", np.abs(sample[:, :, channel] - background[channel]))
            for channel in range(3)
        ),
    ]
    output: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for name, values in distances:
        low, high = np.percentile(values, (5.0, 95.0))
        if high - low < 18.0:
            continue
        normalized = np.clip((values - low) * 255.0 / (high - low), 0.0, 255.0)
        view = (255.0 - normalized).astype(np.uint8)
        fingerprint = cv2.resize(view, (32, 8), interpolation=cv2.INTER_AREA).tobytes()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        output.append((name, view))
    return output


def opencv_qr_geometry_proposals(
    image: np.ndarray,
    covered_quads: Sequence[np.ndarray] = (),
) -> list[tuple[np.ndarray, float]]:
    """Return high-precision QR geometry that does not require a payload read.

    OpenCV's QR detector exposes finder-pattern geometry independently from
    decoding.  Small uploaded images are tried at bounded upscales because a
    25-pixel QR finder pattern can otherwise disappear before the decoder has
    a chance to report an error position.
    """
    gray = as_gray(image)
    height, width = gray.shape[:2]
    longest = max(height, width)
    scales = [1.0]
    if longest <= 700:
        scales.extend((2.0, 3.0))
    elif longest <= 1400:
        scales.append(1.5)
    detector = cv2.QRCodeDetector()
    output: list[tuple[np.ndarray, float]] = []
    for scale in scales:
        work = (
            gray
            if scale == 1.0
            else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        )
        try:
            found, points = detector.detectMulti(work)
        except cv2.error:
            continue
        if not found or points is None:
            continue
        for raw_quad in np.asarray(points, dtype=np.float32):
            quad = raw_quad / scale
            if abs(float(cv2.contourArea(quad))) < 16.0:
                continue
            if any(_quad_containment(quad, covered) >= 0.35 for covered in covered_quads):
                continue
            if any(_quad_containment(quad, prior) >= 0.70 for prior, _prior_scale in output):
                continue
            output.append((quad.astype(np.float32), scale))
    return output


def _quad_containment(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(np.asarray(first, dtype=np.float32))
    second_hull = cv2.convexHull(np.asarray(second, dtype=np.float32))
    first_area = abs(float(cv2.contourArea(first_hull)))
    second_area = abs(float(cv2.contourArea(second_hull)))
    if min(first_area, second_area) < EPS:
        return 0.0
    intersection, _polygon = cv2.intersectConvexConvex(first_hull, second_hull)
    return float(intersection) / min(first_area, second_area)


def input_quality_diagnostics(image: np.ndarray) -> dict[str, Any]:
    """Return cheap, presentation-safe diagnostics without classifying quality."""
    gray = as_gray(image)
    height, width = gray.shape[:2]
    sample_scale = min(1.0, 720.0 / max(height, width))
    sample = (
        cv2.resize(gray, None, fx=sample_scale, fy=sample_scale, interpolation=cv2.INTER_AREA)
        if sample_scale < 1.0
        else gray
    )
    p05, p95 = np.percentile(sample, (5.0, 95.0))
    laplacian = cv2.Laplacian(sample, cv2.CV_32F)
    return {
        "width": int(width),
        "height": int(height),
        "megapixels": round(width * height / 1_000_000.0, 4),
        "p05": round(float(p05), 2),
        "p95": round(float(p95), 2),
        "contrast_span_p05_p95": round(float(p95 - p05), 2),
        "laplacian_variance": round(float(np.var(laplacian)), 2),
        "diagnostic_sample_scale": round(float(sample_scale), 4),
    }


def proposal_scale_plan(shape: Sequence[int], enabled: bool = True) -> list[ScalePlan]:
    """Choose bounded upscales only for images where modules can be undersampled."""
    plans = [ScalePlan(1.0, cv2.INTER_AREA, "native")]
    if not enabled:
        return plans
    height, width = int(shape[0]), int(shape[1])
    longest = max(height, width)
    area = height * width
    if longest <= 420 and area <= 300_000:
        plans.extend(
            [
                ScalePlan(2.0, cv2.INTER_CUBIC, "cubic2"),
                ScalePlan(3.0, cv2.INTER_CUBIC, "cubic3"),
            ]
        )
    elif longest <= 900 and area <= 900_000:
        plans.extend(
            [
                ScalePlan(1.5, cv2.INTER_CUBIC, "cubic15"),
                ScalePlan(2.0, cv2.INTER_CUBIC, "cubic2"),
            ]
        )
    elif longest <= 1600 and area <= 2_500_000:
        plans.append(ScalePlan(1.5, cv2.INTER_CUBIC, "cubic15"))
    return plans


def _binary_profile(profile: np.ndarray) -> np.ndarray:
    values = np.clip(profile, 0, 255).astype(np.uint8).reshape(1, -1)
    if values.shape[1] >= 3:
        values = cv2.medianBlur(values, 3)
    thresholded = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresholded.ravel() > 0


def _runs(binary: np.ndarray) -> list[int]:
    if binary.size == 0:
        return []
    changes = np.flatnonzero(np.diff(binary.astype(np.int8)) != 0) + 1
    boundaries = np.concatenate(([0], changes, [len(binary)]))
    return [int(value) for value in np.diff(boundaries) if value > 0]


def estimate_linear_module_metrics(
    gray: np.ndarray,
    quad: np.ndarray,
    rectify: Callable[..., np.ndarray],
) -> dict[str, float]:
    """Estimate narrow-module sampling and run periodicity after rectification.

    This is not a decoder and does not infer a payload.  It measures whether
    repeated black/white runs can plausibly be expressed as integer multiples
    of a shared narrow element, which is stronger evidence than box length.
    """
    patch = as_gray(rectify(gray, quad, pad_long=0.02, pad_short=0.02))
    height, width = patch.shape[:2]
    if width < 16 or height < 4:
        return {
            "module_width_px": 0.0,
            "module_periodicity": 0.0,
            "module_run_count": 0.0,
            "module_sampling_quality": 0.0,
        }
    top = max(0, int(round(height * 0.12)))
    bottom = min(height, max(top + 3, int(round(height * 0.78))))
    profile = np.median(patch[top:bottom], axis=0)
    run_lengths = _runs(_binary_profile(profile))
    # Quiet zones and crop borders are intentionally excluded. They are much
    # wider than symbol modules and otherwise bias the pitch estimate upward.
    if len(run_lengths) > 4:
        run_lengths = run_lengths[1:-1]
    if len(run_lengths) < 6:
        return {
            "module_width_px": 0.0,
            "module_periodicity": 0.0,
            "module_run_count": float(len(run_lengths)),
            "module_sampling_quality": 0.0,
        }
    values = np.asarray(run_lengths, dtype=np.float32)
    cutoff = float(np.percentile(values, 55.0))
    narrow = values[values <= max(1.0, cutoff)]
    pitch = float(np.median(narrow)) if narrow.size else float(np.min(values))
    pitch = max(0.5, pitch)
    ratios = values / pitch
    integer_error = np.abs(ratios - np.clip(np.round(ratios), 1.0, 8.0))
    periodicity = float(np.mean(np.exp(-2.8 * integer_error)))
    sampling_quality = float(np.clip((pitch - 0.65) / 2.35, 0.0, 1.0))
    return {
        "module_width_px": round(pitch, 4),
        "module_periodicity": round(periodicity, 4),
        "module_run_count": float(len(values)),
        "module_sampling_quality": round(sampling_quality, 4),
    }


def adaptive_short_linear_accept(
    detection: Any,
    minimum_score: float,
    minimum_length: float,
) -> bool:
    """Conservatively accept a short candidate using structure, not length alone."""
    metrics = detection.metrics
    long_side, short_side = detection.long_short()
    aspect = long_side / max(EPS, short_side)
    if long_side >= minimum_length:
        return False
    source_scales = {
        source.rsplit(":scale-", 1)[-1]
        for source in detection.sources
        if ":scale-" in source
    }
    multi_scale = len(source_scales) >= 2
    return bool(
        long_side >= 36.0
        and aspect >= 1.70
        and metrics.get("score", 0.0) >= max(0.50, minimum_score - 0.02)
        and metrics.get("orientation", 0.0) >= 0.70
        and metrics.get("persistence", 0.0) >= 0.065
        and metrics.get("transitions", 0.0) >= 10.0
        and metrics.get("scanline_agreement", 0.0) >= 0.58
        and 0.025 <= metrics.get("transition_rate", 0.0) <= 0.72
        and metrics.get("module_width_px", 0.0) >= 0.75
        and metrics.get("module_periodicity", 0.0) >= 0.55
        and (multi_scale or metrics.get("module_periodicity", 0.0) >= 0.68)
    )


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    values = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = values.sum(axis=1)
    differences = np.diff(values, axis=1).ravel()
    ordered[0] = values[np.argmin(sums)]
    ordered[2] = values[np.argmax(sums)]
    ordered[1] = values[np.argmin(differences)]
    ordered[3] = values[np.argmax(differences)]
    return ordered


def split_dense_linear_quad(
    image: np.ndarray,
    quad: np.ndarray,
) -> list[np.ndarray]:
    """Split a proposal that bridges adjacent linear symbols.

    OpenCV's coherence grouping is intentionally generous. On a dense barcode
    catalogue, quiet zones between neighboring symbols can be narrower than
    the grouping neighborhood, producing one very wide rectangle around two
    or three independent codes. This function rectifies that rectangle, finds
    long-axis intervals with persistent vertical-edge evidence, and maps the
    separated intervals back to the original image.

    Returning an empty list means that the proposal is not safely splittable.
    The function never creates a final detection by itself; every child still
    passes the normal structural verifier and decoder cascade.
    """
    gray = as_gray(image)
    source = _order_quad(np.asarray(quad, dtype=np.float32))
    width = max(
        float(np.linalg.norm(source[1] - source[0])),
        float(np.linalg.norm(source[2] - source[3])),
    )
    height = max(
        float(np.linalg.norm(source[3] - source[0])),
        float(np.linalg.norm(source[2] - source[1])),
    )
    # Rotate a vertical proposal into the same horizontal analysis convention.
    if height > width:
        source = source[[3, 0, 1, 2]]
        width, height = height, width
    output_width = max(2, int(round(width)))
    output_height = max(2, int(round(height)))
    if output_width / max(EPS, output_height) < 5.0 or output_width < 70:
        return []

    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    page_to_patch = cv2.getPerspectiveTransform(source, destination)
    patch = cv2.warpPerspective(
        gray,
        page_to_patch,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    gradient = np.abs(cv2.Scharr(patch, cv2.CV_32F, 1, 0))
    edge_threshold = max(64.0, 0.15 * float(np.percentile(gradient, 70.0)))
    column_support = np.mean(gradient > edge_threshold, axis=0)
    active = (column_support >= 0.12).astype(np.uint8).reshape(1, -1)
    join_width = int(np.clip(round(output_height * 0.18), 5, 15))
    if join_width % 2 == 0:
        join_width += 1
    active = cv2.morphologyEx(
        active,
        cv2.MORPH_CLOSE,
        np.ones((1, join_width), dtype=np.uint8),
    ).ravel()

    changes = np.flatnonzero(np.diff(np.concatenate(([0], active, [0]))))
    minimum_span = max(12.0, output_height * 0.40)
    spans = [
        (int(start), int(end))
        for start, end in zip(changes[::2], changes[1::2])
        if end - start >= minimum_span
    ]
    if len(spans) < 2:
        return []

    minimum_gap = max(5.0, output_height * 0.12)
    groups: list[list[int]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        if start - groups[-1][1] < minimum_gap:
            groups[-1][1] = end
        else:
            groups.append([start, end])
    if len(groups) < 2:
        return []

    inverse = np.linalg.inv(page_to_patch)
    image_height, image_width = gray.shape[:2]
    children: list[np.ndarray] = []
    for index, (start, end) in enumerate(groups):
        left_boundary = 0 if index == 0 else (groups[index - 1][1] + start) / 2.0
        right_boundary = output_width - 1 if index == len(groups) - 1 else (end + groups[index + 1][0]) / 2.0
        quiet_margin = max(2.0, output_height * 0.10)
        left = max(left_boundary, start - quiet_margin)
        right = min(right_boundary, end + quiet_margin)
        if right - left < minimum_span:
            continue
        local_quad = np.asarray(
            [[[left, 0], [right, 0], [right, output_height - 1], [left, output_height - 1]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(local_quad, inverse.astype(np.float64))[0]
        mapped[:, 0] = np.clip(mapped[:, 0], 0, image_width - 1)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, image_height - 1)
        if abs(float(cv2.contourArea(mapped))) >= 20.0:
            children.append(mapped.astype(np.float32))
    return children if len(children) >= 2 else []


def dense_layout_short_review_allowed(
    proposals: Sequence[Any],
    accepted_count: int,
) -> bool:
    """Allow strong short review boxes only on an established barcode sheet."""
    return bool(accepted_count >= 8 and len(proposals) >= 12)


def dense_layout_candidate_review_allowed(detection: Any) -> bool:
    """Validate an unresolved child using persistent edges on a barcode sheet.

    At approximately one pixel per module, intensity thresholding can collapse
    legitimate one-pixel bars. Persistent Scharr edge columns remain observable
    and are a safer fallback, but only after the page has independently
    established a dense barcode layout.
    """
    metrics = detection.metrics
    long_side, short_side = detection.long_short()
    persistent_edge_columns = metrics.get("persistence", 0.0) * long_side
    return bool(
        long_side >= 32.0
        and long_side / max(EPS, short_side) >= 1.15
        and metrics.get("score", 0.0) >= 0.72
        and metrics.get("orientation", 0.0) >= 0.82
        and metrics.get("persistence", 0.0) >= 0.08
        and metrics.get("support_strength", 0.0) >= 0.70
        and metrics.get("scanline_agreement", 0.0) >= 0.50
        and (
            metrics.get("transitions", 0.0) >= 8.0
            or persistent_edge_columns >= 14.0
        )
    )


def relative_matrix_contexts(
    image_shape: Sequence[int],
    quad: np.ndarray,
) -> list[tuple[int, int, float, float, float, str]]:
    """Build recovery windows from candidate geometry rather than page DPI."""
    rectangle = cv2.minAreaRect(np.asarray(quad, dtype=np.float32))
    candidate_side = max(8.0, float(max(rectangle[1])))
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    maximum = min(image_height, image_width)
    contexts: list[tuple[int, int, float, float, float, str]] = []
    for multiplier, fraction_x, fraction_y in (
        (3.5, 0.50, 0.50),
        (5.5, 0.65, 0.60),
        (8.0, 0.80, 0.70),
    ):
        size = int(np.clip(round(candidate_side * multiplier), 120, maximum))
        if candidate_side < 45:
            scale = 3.0
        elif candidate_side < 90:
            scale = 2.0
        elif candidate_side < 180:
            scale = 1.5
        else:
            scale = 1.0
        contexts.append((size, size, fraction_x, fraction_y, scale, f"relative-{multiplier:g}x-s{scale:g}"))
    # Preserve the page-21 recovery geometry that is already validated.
    for recipe in (
        (500, 500, 0.80, 0.75, 1.0, "legacy-offset500-native"),
        (500, 500, 0.65, 0.60, 1.5, "legacy-offset500-cubic15"),
        (700, 700, 0.80, 0.60, 1.5, "legacy-offset700-cubic15"),
    ):
        width = min(image_width, recipe[0])
        height = min(image_height, recipe[1])
        candidate = (width, height, *recipe[2:])
        if candidate not in contexts:
            contexts.append(candidate)
    return contexts


def unique_scales(items: Iterable[float]) -> list[float]:
    return sorted({round(float(item), 4) for item in items})
