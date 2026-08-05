"""Conservative recovery for repeated, undersampled EAN-13 catalogue symbols."""

from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np


EPS = 1e-6
L = ("0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011")
G = ("0100111", "0110011", "0011011", "0100001", "0011101", "0111001", "0000101", "0010001", "0001001", "0010111")
R = ("1110010", "1100110", "1101100", "1000010", "1011100", "1001110", "1010000", "1000100", "1001000", "1110100")
PARITY = ("LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL")


def checksum_valid(payload: str) -> bool:
    if len(payload) != 13 or not payload.isdigit():
        return False
    digits = [int(value) for value in payload]
    total = sum(value * (1 if index % 2 == 0 else 3) for index, value in enumerate(digits[:12]))
    return digits[-1] == (-total) % 10


def modules(payload: str) -> np.ndarray:
    digits = [int(value) for value in payload]
    left = "".join((L if family == "L" else G)[digit] for digit, family in zip(digits[1:7], PARITY[digits[0]]))
    right = "".join(R[digit] for digit in digits[7:])
    return np.fromiter((int(value) for value in f"101{left}01010{right}101"), np.float32)


def neighbours(payload: str) -> list[np.ndarray]:
    digits = [int(value) for value in payload]
    output: list[np.ndarray] = []
    for index in range(12):
        for replacement in range(10):
            if replacement == digits[index]:
                continue
            candidate = digits[:]
            candidate[index] = replacement
            total = sum(value * (1 if position % 2 == 0 else 3) for position, value in enumerate(candidate[:12]))
            candidate[-1] = (-total) % 10
            output.append(modules("".join(str(value) for value in candidate)))
    return output


def raster_models(pattern: np.ndarray, width: int) -> list[np.ndarray]:
    source = pattern.reshape(1, -1)
    high = np.repeat(pattern, 16).reshape(1, -1)
    output = [cv2.resize(source, (width, 1), interpolation=method).ravel() for method in (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST)]
    output.extend(cv2.resize(high, (width, 1), interpolation=method).ravel() for method in (cv2.INTER_AREA, cv2.INTER_CUBIC))
    for phase in np.linspace(-0.5, 0.5, 5):
        positions = (np.arange(width, dtype=np.float32) + 0.5 + phase) * 95.0 / width
        output.append(pattern[np.clip(positions.astype(np.int32), 0, 94)])
    return output


def fit(profile: np.ndarray, pattern: np.ndarray) -> tuple[float, int, int, int]:
    normalized = (profile - float(np.mean(profile))) / (float(np.std(profile)) + EPS)
    width = profile.size
    best = (-1.0, 0, 0, 0)
    for symbol_width in range(max(60, int(round(width * 0.64))), width + 1):
        for model_index, model in enumerate(raster_models(pattern, symbol_width)):
            model = (model - float(np.mean(model))) / (float(np.std(model)) + EPS)
            correlations = np.correlate(normalized, model, mode="valid") / symbol_width
            start = int(np.argmax(correlations))
            if float(correlations[start]) > best[0]:
                best = (float(correlations[start]), start, symbol_width, model_index)
    return best


def template_evidence(
    image: np.ndarray,
    quad: np.ndarray,
    payload: str,
    rectify: Callable[..., np.ndarray],
) -> dict[str, float] | None:
    """Require checksum, neighbour margin, and multi-scan-band agreement."""
    if not checksum_valid(payload):
        return None
    patch = rectify(image, quad, pad_long=0.0, pad_short=0.0)
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    if patch.shape[0] > patch.shape[1]:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    height, width = patch.shape[:2]
    if width < 60 or height < 12 or width / max(1.0, height) < 1.15:
        return None
    expected = modules(payload)
    alternatives = neighbours(payload)
    accepted: list[tuple[float, float, int, int]] = []
    for top_fraction, bottom_fraction in ((0.03, 0.55), (0.10, 0.65), (0.20, 0.72)):
        top = int(round(height * top_fraction))
        bottom = max(top + 3, int(round(height * bottom_fraction)))
        band = patch[top:min(height, bottom)].astype(np.float32)
        edge_width = max(2, width // 12)
        edge_pixels = np.concatenate((band[:, :edge_width].reshape(-1, 3), band[:, -edge_width:].reshape(-1, 3)))
        background = np.median(edge_pixels, axis=0)
        profile = np.quantile(np.linalg.norm(band - background, axis=2), 0.65, axis=0)
        low, high = np.percentile(profile, (5.0, 95.0))
        if high - low < 3.0:
            continue
        profile = np.clip((profile - low) / (high - low), 0.0, 1.0)
        correlation, start, symbol_width, model_index = fit(profile, expected)
        observed = profile[start:start + symbol_width]
        observed = (observed - float(np.mean(observed))) / (float(np.std(observed)) + EPS)
        alternative_correlation = -1.0
        for alternative in alternatives:
            model = raster_models(alternative, symbol_width)[model_index]
            model = (model - float(np.mean(model))) / (float(np.std(model)) + EPS)
            alternative_correlation = max(alternative_correlation, float(np.mean(observed * model)))
        margin = correlation - alternative_correlation
        if correlation >= 0.68 and margin >= 0.012:
            accepted.append((correlation, margin, start, symbol_width))
    if len(accepted) < 2:
        return None
    correlations = np.asarray([value[0] for value in accepted])
    margins = np.asarray([value[1] for value in accepted])
    starts = np.asarray([value[2] for value in accepted])
    widths = np.asarray([value[3] for value in accepted])
    if np.median(correlations) < 0.70 or np.median(margins) < 0.016 or np.ptp(starts) > max(3.0, width * 0.06) or np.ptp(widths) > max(3.0, width * 0.06):
        return None
    symbol_width = float(np.median(widths))
    return {
        "ean13_template_correlation": round(float(np.median(correlations)), 4),
        "ean13_template_margin": round(float(np.median(margins)), 4),
        "ean13_template_scan_bands": float(len(accepted)),
        "ean13_symbol_width_px": round(symbol_width, 2),
        "ean13_pixels_per_module": round(symbol_width / 95.0, 4),
    }


def _containment(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(np.asarray(first, np.float32))
    second_hull = cv2.convexHull(np.asarray(second, np.float32))
    areas = (abs(float(cv2.contourArea(first_hull))), abs(float(cv2.contourArea(second_hull))))
    if min(areas) < EPS:
        return 0.0
    intersection, _polygon = cv2.intersectConvexConvex(first_hull, second_hull)
    return float(intersection) / min(areas)


def qr_geometry_proposals(
    image: np.ndarray,
    covered: Sequence[np.ndarray] = (),
) -> list[tuple[np.ndarray, float]]:
    """High-precision OpenCV QR finder geometry, independent of decoding."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    longest = max(gray.shape[:2])
    scales = [1.0, 2.0, 3.0] if longest <= 700 else [1.0, 1.5] if longest <= 1400 else [1.0]
    detector = cv2.QRCodeDetector()
    output: list[tuple[np.ndarray, float]] = []
    for scale in scales:
        work = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            found, points = detector.detectMulti(work)
        except cv2.error:
            continue
        if not found or points is None:
            continue
        for raw_quad in np.asarray(points, np.float32):
            quad = raw_quad / scale
            if abs(float(cv2.contourArea(quad))) < 16.0:
                continue
            if any(_containment(quad, prior) >= 0.35 for prior in covered):
                continue
            if any(_containment(quad, prior) >= 0.70 for prior, _scale in output):
                continue
            output.append((quad.astype(np.float32), scale))
    return output
