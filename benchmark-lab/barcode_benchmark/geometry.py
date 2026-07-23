from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np


def polygon(value: Any, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError(f"{label}: polygon must contain at least three [x, y] points")
    if not np.isfinite(points).all():
        raise ValueError(f"{label}: polygon contains a non-finite coordinate")
    hull = cv2.convexHull(points).reshape(-1, 2)
    if abs(float(cv2.contourArea(hull))) <= 1e-6:
        raise ValueError(f"{label}: polygon has zero area")
    return hull


def similarity(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    area_first = abs(float(cv2.contourArea(first)))
    area_second = abs(float(cv2.contourArea(second)))
    intersection, _ = cv2.intersectConvexConvex(
        first.astype(np.float32),
        second.astype(np.float32),
    )
    intersection = max(0.0, float(intersection))
    union = area_first + area_second - intersection
    iou = intersection / max(1e-9, union)
    intersection_over_smaller = intersection / max(1e-9, min(area_first, area_second))
    return iou, intersection_over_smaller


def polygon_long_side(points: np.ndarray) -> float:
    rectangle = cv2.minAreaRect(points.astype(np.float32))
    return float(max(rectangle[1]))


def size_bin(long_side: float) -> str:
    if long_side < 32:
        return "<32"
    if long_side < 60:
        return "32-59"
    if long_side < 100:
        return "60-99"
    if long_side < 200:
        return "100-199"
    if long_side < 400:
        return "200-399"
    return ">=400"


def maximum_cardinality_matches(
    scores: np.ndarray,
    eligible: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Return a one-to-one maximum-cardinality matching.

    Neighbor and truth ordering prefers high-overlap and constrained matches.
    Cardinality is exact; score is a deterministic tie preference rather than a
    claim of globally maximum total overlap.
    """
    truth_count, prediction_count = scores.shape
    neighbors: list[list[int]] = []
    for truth_index in range(truth_count):
        candidates = np.flatnonzero(eligible[truth_index]).tolist()
        candidates.sort(key=lambda item: (-float(scores[truth_index, item]), item))
        neighbors.append(candidates)

    prediction_to_truth = [-1] * prediction_count

    def augment(truth_index: int, visited: set[int]) -> bool:
        for prediction_index in neighbors[truth_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            incumbent = prediction_to_truth[prediction_index]
            if incumbent < 0 or augment(incumbent, visited):
                prediction_to_truth[prediction_index] = truth_index
                return True
        return False

    truth_order: Iterable[int] = sorted(
        range(truth_count),
        key=lambda item: (len(neighbors[item]), -max((scores[item, p] for p in neighbors[item]), default=0.0), item),
    )
    for truth_index in truth_order:
        augment(truth_index, set())

    matches = [
        (truth_index, prediction_index, float(scores[truth_index, prediction_index]))
        for prediction_index, truth_index in enumerate(prediction_to_truth)
        if truth_index >= 0
    ]
    return sorted(matches)

