"""Scale-aware measurements for the 5.1 generalization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ScalePlan:
    scale: float
    interpolation: int
    label: str


def as_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def scale_plan(shape: Sequence[int], enabled: bool = True) -> list[ScalePlan]:
    plans = [ScalePlan(1.0, cv2.INTER_AREA, "native")]
    if not enabled:
        return plans
    height, width = int(shape[0]), int(shape[1])
    longest, area = max(height, width), height * width
    if longest <= 420 and area <= 300_000:
        plans += [ScalePlan(2.0, cv2.INTER_CUBIC, "cubic2"), ScalePlan(3.0, cv2.INTER_CUBIC, "cubic3")]
    elif longest <= 900 and area <= 900_000:
        plans += [ScalePlan(1.5, cv2.INTER_CUBIC, "cubic15"), ScalePlan(2.0, cv2.INTER_CUBIC, "cubic2")]
    elif longest <= 1600 and area <= 2_500_000:
        plans += [ScalePlan(1.5, cv2.INTER_CUBIC, "cubic15")]
    return plans


def input_diagnostics(image: np.ndarray) -> dict[str, Any]:
    gray = as_gray(image)
    height, width = gray.shape[:2]
    scale = min(1.0, 720.0 / max(height, width))
    sample = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    p05, p95 = np.percentile(sample, (5.0, 95.0))
    return {
        "width": width,
        "height": height,
        "megapixels": round(width * height / 1_000_000.0, 4),
        "contrast_span_p05_p95": round(float(p95 - p05), 2),
        "laplacian_variance": round(float(np.var(cv2.Laplacian(sample, cv2.CV_32F))), 2),
    }


def module_metrics(gray: np.ndarray, quad: np.ndarray, rectify: Callable[..., np.ndarray]) -> dict[str, float]:
    patch = as_gray(rectify(gray, quad, pad_long=0.02, pad_short=0.02))
    height, width = patch.shape[:2]
    if width < 16 or height < 4:
        return {"module_width_px": 0.0, "module_periodicity": 0.0, "module_run_count": 0.0}
    strip = patch[int(height * 0.12) : max(int(height * 0.78), int(height * 0.12) + 3)]
    profile = np.median(strip, axis=0).astype(np.uint8).reshape(1, -1)
    if profile.shape[1] >= 3:
        profile = cv2.medianBlur(profile, 3)
    binary = cv2.threshold(profile, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1].ravel() > 0
    changes = np.flatnonzero(np.diff(binary.astype(np.int8)) != 0) + 1
    boundaries = np.concatenate(([0], changes, [len(binary)]))
    runs = np.diff(boundaries).astype(np.float32)
    if len(runs) > 4:
        runs = runs[1:-1]
    if len(runs) < 6:
        return {"module_width_px": 0.0, "module_periodicity": 0.0, "module_run_count": float(len(runs))}
    narrow = runs[runs <= max(1.0, float(np.percentile(runs, 55.0)))]
    pitch = max(0.5, float(np.median(narrow)) if narrow.size else float(np.min(runs)))
    ratios = runs / pitch
    error = np.abs(ratios - np.clip(np.round(ratios), 1.0, 8.0))
    return {
        "module_width_px": round(pitch, 4),
        "module_periodicity": round(float(np.mean(np.exp(-2.8 * error))), 4),
        "module_run_count": float(len(runs)),
    }


def accept_short(detection: Any, minimum_score: float, minimum_length: float) -> bool:
    metrics = detection.metrics
    long_side, short_side = detection.long_short()
    if long_side >= minimum_length:
        return False
    sources = {source.rsplit(":scale-", 1)[-1] for source in detection.sources if ":scale-" in source}
    return bool(
        long_side >= 36.0
        and long_side / max(1e-6, short_side) >= 1.70
        and metrics.get("score", 0.0) >= max(0.50, minimum_score - 0.02)
        and metrics.get("orientation", 0.0) >= 0.70
        and metrics.get("persistence", 0.0) >= 0.065
        and metrics.get("transitions", 0.0) >= 10.0
        and metrics.get("scanline_agreement", 0.0) >= 0.58
        and 0.025 <= metrics.get("transition_rate", 0.0) <= 0.72
        and metrics.get("module_width_px", 0.0) >= 0.75
        and metrics.get("module_periodicity", 0.0) >= 0.55
        and (len(sources) >= 2 or metrics.get("module_periodicity", 0.0) >= 0.68)
    )


def _order_quad(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = values.sum(axis=1)
    differences = np.diff(values, axis=1).ravel()
    ordered[0] = values[np.argmin(sums)]
    ordered[2] = values[np.argmax(sums)]
    ordered[1] = values[np.argmin(differences)]
    ordered[3] = values[np.argmax(differences)]
    return ordered


def split_dense_linear_quad(image: np.ndarray, quad: np.ndarray) -> list[np.ndarray]:
    """Split an over-wide OpenCV proposal at genuine inter-symbol gaps."""
    gray = as_gray(image)
    source = _order_quad(np.asarray(quad, dtype=np.float32))
    width = max(float(np.linalg.norm(source[1] - source[0])), float(np.linalg.norm(source[2] - source[3])))
    height = max(float(np.linalg.norm(source[3] - source[0])), float(np.linalg.norm(source[2] - source[1])))
    if height > width:
        source = source[[3, 0, 1, 2]]
        width, height = height, width
    output_width, output_height = max(2, int(round(width))), max(2, int(round(height)))
    if output_width / max(1e-6, output_height) < 5.0 or output_width < 70:
        return []
    destination = np.asarray(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        np.float32,
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
    active = (np.mean(gradient > edge_threshold, axis=0) >= 0.12).astype(np.uint8).reshape(1, -1)
    join_width = int(np.clip(round(output_height * 0.18), 5, 15))
    if join_width % 2 == 0:
        join_width += 1
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((1, join_width), np.uint8)).ravel()
    changes = np.flatnonzero(np.diff(np.concatenate(([0], active, [0]))))
    minimum_span = max(12.0, output_height * 0.40)
    spans = [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2]) if b - a >= minimum_span]
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
        margin = max(2.0, output_height * 0.10)
        left, right = max(left_boundary, start - margin), min(right_boundary, end + margin)
        if right - left < minimum_span:
            continue
        local = np.asarray([[[left, 0], [right, 0], [right, output_height - 1], [left, output_height - 1]]], np.float32)
        mapped = cv2.perspectiveTransform(local, inverse.astype(np.float64))[0]
        mapped[:, 0] = np.clip(mapped[:, 0], 0, image_width - 1)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, image_height - 1)
        if abs(float(cv2.contourArea(mapped))) >= 20.0:
            children.append(mapped.astype(np.float32))
    return children if len(children) >= 2 else []


def dense_layout_enabled(proposals: Sequence[Any], accepted_count: int) -> bool:
    return bool(accepted_count >= 8 and len(proposals) >= 12)


def dense_layout_candidate_allowed(detection: Any) -> bool:
    metrics = detection.metrics
    long_side, short_side = detection.long_short()
    persistent_columns = metrics.get("persistence", 0.0) * long_side
    return bool(
        long_side >= 32.0
        and long_side / max(1e-6, short_side) >= 1.15
        and metrics.get("score", 0.0) >= 0.72
        and metrics.get("orientation", 0.0) >= 0.82
        and metrics.get("persistence", 0.0) >= 0.08
        and metrics.get("support_strength", 0.0) >= 0.70
        and metrics.get("scanline_agreement", 0.0) >= 0.50
        and (metrics.get("transitions", 0.0) >= 8.0 or persistent_columns >= 14.0)
    )


def relative_contexts(shape: Sequence[int], quad: np.ndarray) -> list[tuple[int, int, float, float, float, str]]:
    side = max(8.0, float(max(cv2.minAreaRect(np.asarray(quad, np.float32))[1])))
    maximum = min(int(shape[0]), int(shape[1]))
    output: list[tuple[int, int, float, float, float, str]] = []
    for multiplier, fx, fy in ((3.5, 0.5, 0.5), (5.5, 0.65, 0.60), (8.0, 0.80, 0.70)):
        size = int(np.clip(round(side * multiplier), 120, maximum))
        scale = 3.0 if side < 45 else 2.0 if side < 90 else 1.5 if side < 180 else 1.0
        output.append((size, size, fx, fy, scale, f"relative-{multiplier:g}x-s{scale:g}"))
    return output
