from __future__ import annotations

from typing import Any

from .fusion import fuse_predictions


Prediction = dict[str, Any]


def relative_box_area(prediction: Prediction, width: int, height: int) -> float:
    points = prediction.get("polygon", [])
    if not points or width <= 0 or height <= 0:
        return 0.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
    return area / float(width * height)


def route_small_evidence(
    predictions: list[Prediction],
    *,
    width: int,
    height: int,
    relative_area_threshold: float,
) -> bool:
    """Route empty or small-region evidence to the more expensive detector."""
    if not predictions:
        return True
    return min(
        relative_box_area(prediction, width, height)
        for prediction in predictions
    ) < relative_area_threshold


def conditional_fusion(
    fast_predictions: list[Prediction],
    fallback_predictions: list[Prediction],
    *,
    routed: bool,
    overlap_threshold: float = 0.45,
) -> list[Prediction]:
    if not routed:
        return fast_predictions
    return fuse_predictions(
        fast_predictions,
        [fallback_predictions],
        overlap_threshold=overlap_threshold,
    )
