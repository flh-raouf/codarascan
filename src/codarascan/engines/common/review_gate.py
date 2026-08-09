# SPDX-License-Identifier: Apache-2.0
"""Additional, opt-in evidence gates for the guarded v3 extractor.

v3 is deliberately layered on top of v2 instead of changing v2's thresholds.  It
adds two narrowly scoped certificates:

* a chromatic certificate for real bars whose luminance occupancy is misleading;
* a degraded dense-layout certificate for a page that contains several aligned
  barcode panels, where blur lowers the one-dimensional module score.

The Base decoder and the v2 gate remain the first authorities.  These certificates
only promote a candidate that v2 already rejected, and every decision is recorded
in the region evidence for later audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from codarascan.engines.common.linear_recovery import (
    ReviewDecision,
    ReviewGateV2Config,
    _edge_tracks,
    _occupancy_features,
    _pitch_features,
    _row_masks,
    _runs,
    assess_linear_review_v2,
)
from codarascan.engines.common.linear_recovery import (
    _metric as _v2_metric,
)
from codarascan.engines.common.recovery.adaptive import as_gray
from codarascan.engines.common.recovery.locator import rectify_quad


@dataclass(frozen=True)
class ReviewGateV3Config(ReviewGateV2Config):
    """v2 settings plus bounded rescue thresholds for v3-only evidence."""

    # The chromatic rescue is intentionally not allowed to promote a strip that
    # is only a handful of source pixels tall.  In the corpus, that gap is
    # decisive: genuine thin barcodes start at 24px, while the two text rescues
    # are 16-17px high and become barcode-like only after colour-distance
    # normalization.
    minimum_chromatic_source_height: float = 20.0
    minimum_chromatic_gain: float = 0.12
    minimum_chromatic_contrast: float = 18.0
    minimum_chromatic_transitions: int = 18
    minimum_chromatic_periodicity: float = 0.92
    minimum_chromatic_module_stability: float = 0.84
    minimum_chromatic_edge_support: float = 0.94
    minimum_chromatic_verticality: float = 0.94
    minimum_chromatic_threshold_stability: float = 0.84
    minimum_chromatic_occupancy: float = 0.42
    minimum_chromatic_row_overlap: float = 0.94
    minimum_chromatic_diversity: float = 0.90
    minimum_dense_base_score: float = 0.94
    minimum_dense_transitions: int = 36
    minimum_dense_periodicity: float = 0.56
    minimum_dense_module_stability: float = 0.88
    minimum_dense_edge_support: float = 0.94
    minimum_dense_verticality: float = 0.94
    minimum_dense_threshold_stability: float = 0.80
    minimum_dense_occupancy: float = 0.30
    minimum_dense_row_overlap: float = 0.94
    minimum_dense_diversity: float = 0.90
    minimum_dense_decoded_anchors: int = 2
    minimum_dense_layout_regions: int = 4


def _metric(evidence: dict[str, Any], name: str) -> float:
    return _v2_metric(evidence, name)


def _profile_metrics(profile: np.ndarray) -> dict[str, float]:
    values = np.asarray(profile, dtype=np.uint8).reshape(1, -1)
    if values.shape[1] < 12:
        return {
            "transitions": 0.0,
            "dark_fraction": 0.0,
            "threshold_stability": 0.0,
            "module_periodicity": 0.0,
            "module_stability": 0.0,
            "edge_track_support": 0.0,
            "verticality": 0.0,
            "high_occupancy_fraction": 0.0,
            "row_mask_overlap": 0.0,
            "run_width_diversity": 0.0,
            "contrast": 0.0,
        }
    rows, threshold_stability = _row_masks(values)
    if rows.size == 0:
        return {
            "transitions": 0.0,
            "dark_fraction": 0.0,
            "threshold_stability": 0.0,
            "module_periodicity": 0.0,
            "module_stability": 0.0,
            "edge_track_support": 0.0,
            "verticality": 0.0,
            "high_occupancy_fraction": 0.0,
            "row_mask_overlap": 0.0,
            "run_width_diversity": 0.0,
            "contrast": 0.0,
        }
    consensus = (np.mean(rows, axis=0) >= 0.5).astype(np.uint8)
    runs = _runs(consensus)
    pitch = _pitch_features(runs)
    edges = _edge_tracks(rows, consensus, pitch["module_width_px"])
    occupancy = _occupancy_features(rows, consensus)
    pitch["module_stability"] = float(
        np.clip(
            0.55 * threshold_stability
            + 0.45 * edges["transition_count_stability"],
            0.0,
            1.0,
        )
    )
    return {
        "transitions": float(max(0, len(runs) - 1)),
        "dark_fraction": float(np.mean(consensus)),
        "threshold_stability": float(threshold_stability),
        "module_periodicity": float(pitch["module_periodicity"]),
        "module_stability": float(pitch["module_stability"]),
        "edge_track_support": float(edges["edge_track_support"]),
        "verticality": float(edges["verticality"]),
        "high_occupancy_fraction": float(occupancy["high_occupancy_fraction"]),
        "row_mask_overlap": float(occupancy["row_mask_overlap"]),
        "run_width_diversity": float(pitch["run_width_diversity"]),
        "contrast": float(np.percentile(profile, 95.0) - np.percentile(profile, 5.0)),
    }


def _chromatic_metrics(
    image: np.ndarray,
    quad: np.ndarray | list[list[float]] | list[float],
) -> dict[str, Any]:
    """Compare luminance with each original channel after the same rectification."""

    try:
        points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        if image.ndim != 3 or image.shape[2] < 3:
            return {"available": False}
        patch = rectify_quad(image, points, pad_long=0.04, pad_short=0.12)
    except (ValueError, TypeError, cv2.error):
        return {"available": False}

    height, width = patch.shape[:2]
    if width < 36 or height < 8:
        return {"available": False}
    top = max(0, int(round(height * 0.10)))
    bottom = min(height, max(top + 3, int(round(height * 0.84))))
    # _profile_metrics takes one representative horizontal scanline.  Using the
    # median across the barcode height keeps the test insensitive to isolated
    # specks and is equivalent to the v2 multi-row consensus in spirit.
    gray = as_gray(patch)
    views: list[tuple[str, np.ndarray]] = [("luminance", gray)]
    for channel, name in enumerate(("blue", "green", "red")):
        views.append((name, patch[:, :, channel]))

    background = np.median(
        np.concatenate((patch[:, : max(2, width // 12)], patch[:, -max(2, width // 12) :]), axis=1),
        axis=(0, 1),
    )
    distance = np.linalg.norm(patch.astype(np.float32) - background, axis=2)
    low, high = np.percentile(distance, (5.0, 95.0))
    if high - low >= 18.0:
        distance_view = np.clip((distance - low) * 255.0 / (high - low), 0.0, 255.0)
        # Bars are farther from the page's edge/background colour. Invert that
        # distance so the existing dark-bar profile evaluator can be reused.
        views.append(("chromatic-distance", (255.0 - distance_view).astype(np.uint8)))

    measured: dict[str, dict[str, float]] = {}
    for name, view in views:
        profile = np.median(view[top:bottom], axis=0).astype(np.uint8)
        measured[name] = _profile_metrics(profile)

    baseline = measured["luminance"]
    candidates = [
        (name, metrics)
        for name, metrics in measured.items()
        if name != "luminance"
    ]
    if not candidates:
        return {"available": False, "luminance": baseline}
    best_name, best = max(
        candidates,
        key=lambda item: (
            item[1]["high_occupancy_fraction"] - baseline["high_occupancy_fraction"],
            item[1]["module_periodicity"],
            item[1]["edge_track_support"],
        ),
    )
    return {
        "available": True,
        "selected_view": best_name,
        "luminance": baseline,
        "selected": best,
        "occupancy_gain": best["high_occupancy_fraction"] - baseline["high_occupancy_fraction"],
        "channel_contrast": best["contrast"],
    }


def _layout_certificate(
    layout_context: Iterable[Any],
    candidate_quad: np.ndarray | list[list[float]] | list[float],
    config: ReviewGateV3Config,
) -> dict[str, Any]:
    """Prove that this candidate belongs to a multi-row barcode layout."""

    try:
        candidate = np.asarray(candidate_quad, dtype=np.float32).reshape(4, 2)
    except (ValueError, TypeError):
        return {"accepted": False, "reason": "invalid-candidate-geometry"}
    entries: list[dict[str, Any]] = []
    for item in layout_context:
        quad = getattr(item, "quad", None)
        status = getattr(getattr(item, "status", None), "value", None)
        decoded = bool(getattr(item, "decoded", False))
        if isinstance(item, dict):
            quad = item.get("quad")
            status = item.get("status", status)
            decoded = bool(item.get("decoded", status == "decoded"))
        if quad is None:
            continue
        try:
            points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        except (ValueError, TypeError):
            continue
        ordered = points
        width = max(
            float(np.linalg.norm(ordered[1] - ordered[0])),
            float(np.linalg.norm(ordered[2] - ordered[3])),
        )
        height = max(
            float(np.linalg.norm(ordered[3] - ordered[0])),
            float(np.linalg.norm(ordered[2] - ordered[1])),
        )
        entries.append(
            {
                "quad": points,
                "center": points.mean(axis=0),
                "width": width,
                "height": height,
                "decoded": decoded or status == "decoded",
            }
        )
    if len(entries) < config.minimum_dense_layout_regions:
        return {"accepted": False, "region_count": len(entries), "row_count": 0}

    median_height = float(np.median([max(1.0, item["height"]) for item in entries]))
    row_tolerance = max(28.0, 0.75 * median_height)
    rows: list[list[dict[str, Any]]] = []
    for entry in sorted(entries, key=lambda item: float(item["center"][1])):
        matching = next(
            (row for row in rows if abs(float(entry["center"][1] - np.mean([i["center"][1] for i in row]))) <= row_tolerance),
            None,
        )
        if matching is None:
            rows.append([entry])
        else:
            matching.append(entry)
    usable_rows = [row for row in rows if len(row) >= 2]
    candidate_center = candidate.mean(axis=0)
    candidate_row = next(
        (
            row
            for row in usable_rows
            if abs(float(candidate_center[1] - np.mean([item["center"][1] for item in row]))) <= row_tolerance
        ),
        [],
    )
    decoded_anchors = sum(1 for item in entries if item["decoded"])
    accepted = bool(
        len(usable_rows) >= 2
        and len(candidate_row) >= 2
        and decoded_anchors >= config.minimum_dense_decoded_anchors
    )
    return {
        "accepted": accepted,
        "region_count": len(entries),
        "row_count": len(usable_rows),
        "candidate_row_size": len(candidate_row),
        "decoded_anchors": decoded_anchors,
        "row_tolerance": round(row_tolerance, 3),
    }


def assess_linear_review_v3(
    image: np.ndarray,
    quad: np.ndarray | list[list[float]] | list[float],
    *,
    color_image: np.ndarray | None = None,
    base_metrics: dict[str, Any] | None = None,
    sources: Iterable[str] = (),
    layout_context: Iterable[Any] = (),
    config: ReviewGateV3Config | None = None,
) -> ReviewDecision:
    """Apply v2, then the two v3-only physical certificates."""

    config = config or ReviewGateV3Config()
    base = assess_linear_review_v2(
        image,
        quad,
        base_metrics=base_metrics,
        sources=sources,
        config=config,
    )
    evidence = dict(base.evidence)
    color = _chromatic_metrics(color_image, quad) if color_image is not None else {"available": False}
    layout = _layout_certificate(layout_context, quad, config)

    selected = color.get("selected", {}) if isinstance(color, dict) else {}
    try:
        points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        source_short_side = min(
            (float(value) for value in cv2.minAreaRect(points)[1]),
            default=0.0,
        )
    except (cv2.error, TypeError, ValueError):
        source_short_side = 0.0
    chromatic_certificate = bool(
        not base.accepted
        and bool(color.get("available"))
        and source_short_side >= config.minimum_chromatic_source_height
        and _metric(color, "occupancy_gain") >= config.minimum_chromatic_gain
        and _metric(color, "channel_contrast") >= config.minimum_chromatic_contrast
        and _metric(selected, "transitions") >= config.minimum_chromatic_transitions
        and _metric(selected, "module_periodicity") >= config.minimum_chromatic_periodicity
        and _metric(selected, "module_stability") >= config.minimum_chromatic_module_stability
        and _metric(selected, "edge_track_support") >= config.minimum_chromatic_edge_support
        and _metric(selected, "verticality") >= config.minimum_chromatic_verticality
        and _metric(selected, "threshold_stability") >= config.minimum_chromatic_threshold_stability
        and _metric(selected, "high_occupancy_fraction") >= config.minimum_chromatic_occupancy
        and _metric(selected, "row_mask_overlap") >= config.minimum_chromatic_row_overlap
        and _metric(selected, "run_width_diversity") >= config.minimum_chromatic_diversity
        and not bool(evidence.get("uniform_veto"))
        and not bool(evidence.get("hatch_veto"))
    )
    dense_certificate = bool(
        not base.accepted
        and bool(layout.get("accepted"))
        and _metric(evidence, "base_score") >= config.minimum_dense_base_score
        and _metric(evidence, "transitions") >= config.minimum_dense_transitions
        and _metric(evidence, "module_periodicity") >= config.minimum_dense_periodicity
        and _metric(evidence, "module_stability") >= config.minimum_dense_module_stability
        and _metric(evidence, "edge_track_support") >= config.minimum_dense_edge_support
        and _metric(evidence, "verticality") >= config.minimum_dense_verticality
        and _metric(evidence, "threshold_stability") >= config.minimum_dense_threshold_stability
        and _metric(evidence, "high_occupancy_fraction") >= config.minimum_dense_occupancy
        and _metric(evidence, "row_mask_overlap") >= config.minimum_dense_row_overlap
        and _metric(evidence, "run_width_diversity") >= config.minimum_dense_diversity
        and not bool(evidence.get("uniform_veto"))
        and not bool(evidence.get("hatch_veto"))
    )

    accepted = bool(base.accepted or chromatic_certificate or dense_certificate)
    if base.accepted:
        reason = base.reason
    elif chromatic_certificate:
        reason = "chromatic-contrast-physical-review-certificate"
    elif dense_certificate:
        reason = "degraded-dense-layout-physical-review-certificate"
    else:
        reason = base.reason

    def rounded(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, (np.floating, float)):
            return round(float(value), 4)
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value

    evidence.update(
        {
            "validation": "review-gate-v3",
            "v2_gate_accepted": bool(base.accepted),
            "v2_gate_reason": base.reason,
            "chromatic": rounded(color),
            "layout": rounded(layout),
            "chromatic_certificate": chromatic_certificate,
            "chromatic_source_short_side": round(source_short_side, 4),
            "chromatic_source_height_veto": bool(
                source_short_side < config.minimum_chromatic_source_height
            ),
            "degraded_dense_layout_certificate": dense_certificate,
            "accepted": accepted,
            "reason": reason,
        }
    )
    return ReviewDecision(accepted, max(base.score, 0.90 if (chromatic_certificate or dense_certificate) else base.score), reason, evidence)


__all__ = [
    "ReviewGateV3Config",
    "ReviewDecision",
    "assess_linear_review_v3",
]
