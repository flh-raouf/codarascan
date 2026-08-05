"""Self-contained physical-evidence gate for the guarded v2 Base extractor.

The Base pipeline deliberately keeps a high-recall proposal and decoder path. This module decides whether an undecoded linear proposal is strong enough to expose as an operator-facing review result. It never runs before decoding and it never changes a decoded result.

The gate is intentionally family-based. A high edge score cannot compensate for a failed module/grammar or stability family, which is the failure mode that lets form slots and repeated typography look like barcodes in the Base score. The v2 wrapper below adds the narrow compressed-long rescue and sparse-hatch veto.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from pipeline.adaptive_generalization import as_gray
from pipeline.deterministic_barcode_locator.barcode_locator import (
    EPS,
    rectify_quad,
)


@dataclass(frozen=True)
class ReviewGateConfig:
    """Conservative thresholds for user-facing unresolved linear results."""

    minimum_base_score: float = 0.62
    minimum_transitions: int = 18
    minimum_module_periodicity: float = 0.78
    minimum_module_stability: float = 0.55
    minimum_edge_support: float = 0.58
    minimum_verticality: float = 0.72
    minimum_high_occupancy: float = 0.36
    minimum_threshold_stability: float = 0.58
    minimum_family_count: int = 3
    uniform_diversity_veto: float = 0.12
    uniform_entropy_veto: float = 0.42
    hatch_verticality_veto: float = 0.56
    minimum_long_linear_base_score: float = 0.96
    minimum_long_linear_aspect: float = 18.0
    minimum_long_linear_height: int = 36
    minimum_long_linear_transitions: int = 100
    minimum_long_linear_dark_fraction: float = 0.30
    minimum_long_linear_occupancy: float = 0.30
    minimum_long_linear_row_overlap: float = 0.96
    maximum_long_linear_outside_activity: float = 0.16


@dataclass(frozen=True)
class ReviewDecision:
    accepted: bool
    score: float
    reason: str
    evidence: dict[str, Any]


def _runs(binary: np.ndarray) -> list[int]:
    values = np.asarray(binary, dtype=np.uint8).ravel()
    if values.size == 0:
        return []
    changes = np.flatnonzero(np.diff(values.astype(np.int8)) != 0) + 1
    boundaries = np.concatenate(([0], changes, [values.size]))
    return [int(length) for length in np.diff(boundaries) if length > 0]


def _transition_positions(binary: np.ndarray) -> np.ndarray:
    values = np.asarray(binary, dtype=np.uint8).ravel()
    return np.flatnonzero(np.diff(values.astype(np.int8)) != 0).astype(np.float32) + 0.5


def _thresholds(profile: np.ndarray) -> tuple[float, ...]:
    values = np.asarray(profile, dtype=np.uint8).reshape(1, -1)
    low, high = np.percentile(values, (5.0, 95.0))
    span = max(1.0, float(high - low))
    otsu = float(cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    return tuple(
        float(np.clip(otsu + delta * span, low + 0.08 * span, high - 0.08 * span))
        for delta in (-0.16, 0.0, 0.16)
    )


def _binarise(profile: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(profile, dtype=np.float32).ravel() <= threshold).astype(np.uint8)


def _selected_rows(strip: np.ndarray, count: int = 15) -> np.ndarray:
    if strip.shape[0] <= count:
        return strip
    indices = np.linspace(0, strip.shape[0] - 1, count).round().astype(np.int32)
    return strip[indices]


def _row_masks(strip: np.ndarray) -> tuple[np.ndarray, float]:
    rows: list[np.ndarray] = []
    stability: list[float] = []
    for profile in _selected_rows(strip):
        thresholds = _thresholds(profile)
        masks = [_binarise(profile, threshold) for threshold in thresholds]
        rows.append(masks[1])
        reference = masks[1]
        stability.append(float(np.median([np.mean(mask == reference) for mask in masks])))
    if not rows:
        return np.zeros((0, strip.shape[1]), dtype=np.uint8), 0.0
    return np.stack(rows), float(np.median(stability))


def _pitch_features(runs: Iterable[int]) -> dict[str, float]:
    values = np.asarray([float(value) for value in runs if value > 0], dtype=np.float32)
    if values.size < 6:
        return {
            "module_width_px": 0.0,
            "module_periodicity": 0.0,
            "module_stability": 0.0,
            "run_width_cv": 0.0,
            "run_width_entropy": 0.0,
            "run_width_diversity": 0.0,
        }
    if values.size > 4:
        values = values[1:-1]
    narrow_limit = float(np.percentile(values, 55.0))
    narrow = values[values <= max(1.0, narrow_limit)]
    pitch = float(np.median(narrow)) if narrow.size else float(np.min(values))
    pitch = max(0.5, pitch)
    ratios = values / pitch
    integer_error = np.abs(ratios - np.clip(np.round(ratios), 1.0, 8.0))
    periodicity = float(np.mean(np.exp(-2.8 * integer_error)))
    normalized = np.clip(np.round(ratios * 2.0) / 2.0, 0.5, 8.0)
    unique = np.unique(normalized)
    diversity = float(np.clip((len(unique) - 1.0) / 4.0, 0.0, 1.0))
    counts = np.bincount(np.clip(np.round(normalized * 2.0).astype(np.int32), 1, 16), minlength=17)[1:]
    probabilities = counts[counts > 0].astype(np.float32)
    probabilities /= max(EPS, float(np.sum(probabilities)))
    entropy = float(-np.sum(probabilities * np.log2(probabilities)) / max(1.0, np.log2(16.0)))
    return {
        "module_width_px": pitch,
        "module_periodicity": periodicity,
        "module_stability": 0.0,
        "run_width_cv": float(np.std(values) / max(EPS, np.mean(values))),
        "run_width_entropy": entropy,
        "run_width_diversity": diversity,
    }


def _edge_tracks(rows: np.ndarray, consensus: np.ndarray, pitch: float) -> dict[str, float]:
    reference = _transition_positions(consensus)
    if rows.size == 0 or reference.size == 0:
        return {
            "edge_track_support": 0.0,
            "transition_count_stability": 0.0,
            "verticality": 0.0,
            "edge_drift_px": float("inf"),
        }
    tolerance = max(1.5, min(5.0, 0.90 * max(0.75, pitch)))
    supports: list[float] = []
    drifts: list[float] = []
    counts: list[float] = []
    row_positions: list[np.ndarray] = []
    for row in rows:
        positions = _transition_positions(row)
        row_positions.append(positions)
        counts.append(float(len(positions)))
        if positions.size == 0:
            supports.append(0.0)
            drifts.append(float("inf"))
            continue
        distances = np.min(np.abs(positions[:, None] - reference[None, :]), axis=1)
        matched = distances <= tolerance
        supports.append(float(np.mean(matched)))
        drifts.append(float(np.median(distances[matched])) if np.any(matched) else float("inf"))
    median_count = float(np.median(counts))
    count_stability = float(np.exp(-abs(float(np.std(counts)) / max(1.0, median_count))))
    finite_drifts = [value for value in drifts if np.isfinite(value)]
    drift = float(np.median(finite_drifts)) if finite_drifts else float("inf")
    verticality = float(np.exp(-drift / max(1.0, pitch))) if np.isfinite(drift) else 0.0
    return {
        "edge_track_support": float(np.median(supports)),
        "transition_count_stability": count_stability,
        "verticality": verticality,
        "edge_drift_px": drift,
    }


def _occupancy_features(rows: np.ndarray, consensus: np.ndarray) -> dict[str, float]:
    if rows.size == 0:
        return {
            "high_occupancy_fraction": 0.0,
            "row_mask_overlap": 0.0,
            "vertical_persistence": 0.0,
        }
    occupancy = np.mean(rows, axis=0)
    masks = occupancy >= 0.55
    high_fraction = float(np.mean(masks))
    row_overlap = float(np.median([np.mean(row == consensus) for row in rows]))
    vertical_persistence = float(np.percentile(occupancy, 75.0))
    return {
        "high_occupancy_fraction": high_fraction,
        "row_mask_overlap": row_overlap,
        "vertical_persistence": vertical_persistence,
    }


def _outside_activity(patch: np.ndarray, strip: np.ndarray) -> float:
    if patch.size == 0 or strip.size == 0:
        return 1.0
    border = max(2, min(strip.shape[1] // 12, patch.shape[1] // 8))
    margins = np.concatenate((patch[:, :border].ravel(), patch[:, -border:].ravel()))
    interior = strip.ravel()
    border_dark = float(np.mean(margins < np.median(interior)))
    return border_dark


def _assess_linear_review_base(
    image: np.ndarray,
    quad: np.ndarray | list[list[float]] | list[float],
    *,
    base_metrics: dict[str, Any] | None = None,
    sources: Iterable[str] = (),
    config: ReviewGateConfig | None = None,
) -> ReviewDecision:
    """Return whether an undecoded linear candidate is safe to surface for review."""

    config = config or ReviewGateConfig()
    metrics = base_metrics if isinstance(base_metrics, dict) else {}
    try:
        points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        gray = as_gray(np.asarray(image))
        patch = as_gray(rectify_quad(gray, points, pad_long=0.04, pad_short=0.12))
    except (ValueError, TypeError, cv2.error):
        return ReviewDecision(False, 0.0, "invalid-geometry", {"validation": "review-gate-base"})

    height, width = patch.shape[:2]
    if width < 36 or height < 6 or width / max(1.0, height) < 1.45:
        return ReviewDecision(
            False,
            0.0,
            "candidate-too-small-or-square",
            {"validation": "review-gate-base", "rectified_width": width, "rectified_height": height},
        )

    top = max(0, int(round(height * 0.10)))
    bottom = min(height, max(top + 3, int(round(height * 0.84))))
    strip = patch[top:bottom]
    rows, threshold_stability = _row_masks(strip)
    if rows.size == 0:
        return ReviewDecision(False, 0.0, "no-scanlines", {"validation": "review-gate-base"})
    consensus = (np.mean(rows, axis=0) >= 0.5).astype(np.uint8)
    run_lengths = _runs(consensus)
    transitions = max(0, len(run_lengths) - 1)
    dark_fraction = float(np.mean(consensus))
    pitch_features = _pitch_features(run_lengths)
    edge_features = _edge_tracks(rows, consensus, pitch_features["module_width_px"])
    occupancy = _occupancy_features(rows, consensus)
    source_values = tuple(str(source) for source in sources)
    source_families = {
        value.split(":", 1)[0]
        for value in source_values
        if value and not value.startswith("deterministic:")
    }
    source_count = len(source_families)

    base_score = _number(metrics.get("score"))
    base_aspect = _number(metrics.get("aspect"))
    orientation = _number(metrics.get("orientation"))
    persistence = _number(metrics.get("persistence"))
    scanline_agreement = _number(metrics.get("scanline_agreement"))
    module_periodicity = pitch_features["module_periodicity"]
    module_stability = float(
        np.clip(
            0.55 * threshold_stability
            + 0.45 * edge_features["transition_count_stability"],
            0.0,
            1.0,
        )
    )
    pitch_features["module_stability"] = module_stability
    uniform_veto = bool(
        transitions >= 10
        and pitch_features["run_width_diversity"] <= config.uniform_diversity_veto
        and pitch_features["run_width_entropy"] <= config.uniform_entropy_veto
        and edge_features["verticality"] >= 0.62
    )
    hatch_veto = bool(
        transitions >= 8
        and edge_features["verticality"] < config.hatch_verticality_veto
        and edge_features["edge_track_support"] < 0.55
    )
    document_regular_veto = bool(
        transitions < config.minimum_transitions
        and module_periodicity < config.minimum_module_periodicity
        and source_count < 2
        and base_score >= 0.78
    )
    # A genuine short/tall label can be under-sampled enough that its run grammar is
    # weaker than the long document-code profile.  Keep this rescue narrow: it needs
    # the high Base score, enough height to observe real bars, stable vertical tracks,
    # and a barcode-like dark occupancy.  The page-17 visual-only Code 128 is the
    # regression target; the long page-48 form-slot candidates do not satisfy it.
    short_tall_certificate = bool(
        1.65 <= base_aspect <= 3.0
        and height >= 40
        and base_score >= 0.88
        and transitions >= config.minimum_transitions
        and dark_fraction >= 0.48
        and edge_features["edge_track_support"] >= 0.72
        and edge_features["verticality"] >= 0.75
        and scanline_agreement >= 0.82
        and not uniform_veto
        and not hatch_veto
    )
    # Some genuine printed symbols use several narrow/wide elements but rasterise
    # poorly enough that a single consensus profile underestimates integer-module
    # periodicity.  High run-width diversity plus strong multi-row continuity is a
    # safer fallback than lowering the periodicity threshold for every candidate.
    varied_rhythm_certificate = bool(
        base_score >= 0.90
        and transitions >= 30
        and pitch_features["run_width_diversity"] >= 0.75
        and threshold_stability >= 0.75
        and module_stability >= 0.75
        and edge_features["edge_track_support"] >= 0.72
        and edge_features["verticality"] >= 0.75
        and occupancy["high_occupancy_fraction"] >= 0.38
        and occupancy["row_mask_overlap"] >= 0.84
        and not uniform_veto
        and not hatch_veto
    )
    # Residence certificates and related administrative forms contain a genuine
    # long barcode whose narrow rasterised bars can occupy only about one third of
    # the scanline.  Do not lower the occupancy floor globally: this certificate
    # requires a very long candidate, near-perfect row persistence, strong edge
    # tracks, and a high Base score.  It is deliberately evaluated after the
    # general varied-rhythm rescue so the established page-92 reason remains stable.
    long_linear_certificate = bool(
        not varied_rhythm_certificate
        and base_score >= config.minimum_long_linear_base_score
        and base_aspect >= config.minimum_long_linear_aspect
        and height >= config.minimum_long_linear_height
        and transitions >= config.minimum_long_linear_transitions
        and dark_fraction >= config.minimum_long_linear_dark_fraction
        and pitch_features["run_width_diversity"] >= 0.75
        and threshold_stability >= 0.90
        and module_stability >= 0.86
        and edge_features["edge_track_support"] >= 0.90
        and edge_features["verticality"] >= 0.95
        and occupancy["high_occupancy_fraction"] >= config.minimum_long_linear_occupancy
        and occupancy["row_mask_overlap"] >= config.minimum_long_linear_row_overlap
        and occupancy["vertical_persistence"] >= 0.85
        and _outside_activity(patch, strip) <= config.maximum_long_linear_outside_activity
        and not uniform_veto
        and not hatch_veto
    )

    family_geometry = float(
        np.clip(
            0.35 * edge_features["edge_track_support"]
            + 0.30 * edge_features["verticality"]
            + 0.20 * scanline_agreement
            + 0.15 * orientation,
            0.0,
            1.0,
        )
    )
    family_grammar = float(
        np.clip(
            0.62 * module_periodicity
            + 0.23 * module_stability
            + 0.15 * pitch_features["run_width_diversity"],
            0.0,
            1.0,
        )
    )
    family_continuity = float(
        np.clip(
            0.40 * occupancy["high_occupancy_fraction"] / 0.55
            + 0.35 * occupancy["row_mask_overlap"]
            + 0.25 * occupancy["vertical_persistence"],
            0.0,
            1.0,
        )
    )
    family_stability = float(
        np.clip(
            0.60 * threshold_stability
            + 0.40 * edge_features["transition_count_stability"],
            0.0,
            1.0,
        )
    )
    family_count = sum(
        value
        for value in (
            family_geometry >= 0.64,
            family_grammar >= 0.66,
            family_continuity >= 0.52,
            family_stability >= config.minimum_threshold_stability,
        )
    )
    score = float(
        np.clip(
            0.30 * min(family_geometry, family_grammar, family_stability)
            + 0.25 * family_continuity
            + 0.20 * module_periodicity
            + 0.15 * edge_features["verticality"]
            + 0.10 * min(1.0, source_count / 2.0),
            0.0,
            1.0,
        )
    )
    standard_certificate = bool(
        base_score >= config.minimum_base_score
        and transitions >= config.minimum_transitions
        and module_periodicity >= config.minimum_module_periodicity
        and module_stability >= config.minimum_module_stability
        and edge_features["edge_track_support"] >= config.minimum_edge_support
        and edge_features["verticality"] >= config.minimum_verticality
        and occupancy["high_occupancy_fraction"] >= config.minimum_high_occupancy
        and threshold_stability >= config.minimum_threshold_stability
        and family_count >= config.minimum_family_count
        and not uniform_veto
        and not hatch_veto
        and not document_regular_veto
    )
    accepted = bool(
        standard_certificate
        or short_tall_certificate
        or varied_rhythm_certificate
        or long_linear_certificate
    )
    if short_tall_certificate:
        reason = "short-tall-physical-review-certificate"
    elif varied_rhythm_certificate:
        reason = "varied-rhythm-physical-review-certificate"
    elif long_linear_certificate:
        reason = "long-linear-physical-review-certificate"
    elif accepted:
        reason = "physical-review-certificate"
    elif uniform_veto:
        reason = "uniform-run-pattern"
    elif hatch_veto:
        reason = "diagonal-or-unstable-edge-tracks"
    elif document_regular_veto:
        reason = "document-regularity-without-barcode-grammar"
    elif family_count < config.minimum_family_count:
        reason = "insufficient-independent-evidence-families"
    else:
        reason = "physical-review-gate-failed"
    evidence = {
        "validation": "review-gate-base",
        "accepted": accepted,
        "reason": reason,
        "score": round(score, 4),
        "rectified_width": int(width),
        "rectified_height": int(height),
        "base_score": round(base_score, 4),
        "base_aspect": round(base_aspect, 4),
        "transitions": int(transitions),
        "dark_fraction": round(dark_fraction, 4),
        "threshold_stability": round(threshold_stability, 4),
        "module_width_px": round(pitch_features["module_width_px"], 4),
        "module_periodicity": round(module_periodicity, 4),
        "module_stability": round(module_stability, 4),
        "run_width_cv": round(pitch_features["run_width_cv"], 4),
        "run_width_entropy": round(pitch_features["run_width_entropy"], 4),
        "run_width_diversity": round(pitch_features["run_width_diversity"], 4),
        "edge_track_support": round(edge_features["edge_track_support"], 4),
        "transition_count_stability": round(edge_features["transition_count_stability"], 4),
        "verticality": round(edge_features["verticality"], 4),
        "edge_drift_px": round(edge_features["edge_drift_px"], 4)
        if np.isfinite(edge_features["edge_drift_px"])
        else None,
        **{key: round(value, 4) for key, value in occupancy.items()},
        "family_geometry": round(family_geometry, 4),
        "family_grammar": round(family_grammar, 4),
        "family_continuity": round(family_continuity, 4),
        "family_stability": round(family_stability, 4),
        "family_count": int(family_count),
        "source_families": sorted(source_families),
        "source_family_count": int(source_count),
        "uniform_veto": uniform_veto,
        "hatch_veto": hatch_veto,
        "document_regular_veto": document_regular_veto,
        "short_tall_certificate": short_tall_certificate,
        "varied_rhythm_certificate": varied_rhythm_certificate,
        "long_linear_certificate": long_linear_certificate,
        "standard_certificate": standard_certificate,
        "outside_activity": round(_outside_activity(patch, strip), 4),
    }
    return ReviewDecision(accepted, score, reason, evidence)


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0

@dataclass(frozen=True)
class ReviewGateV2Config(ReviewGateConfig):
    """Base-gate settings plus the narrow compressed-long certificate."""

    minimum_compressed_long_base_score: float = 0.97
    minimum_compressed_long_aspect: float = 24.0
    minimum_compressed_long_height: int = 26
    minimum_compressed_long_transitions: int = 170
    minimum_compressed_long_dark_fraction: float = 0.28
    minimum_compressed_long_occupancy: float = 0.28
    minimum_compressed_long_row_overlap: float = 0.985
    minimum_compressed_long_threshold_stability: float = 0.94
    minimum_compressed_long_module_stability: float = 0.94
    minimum_compressed_long_edge_support: float = 0.97
    minimum_compressed_long_verticality: float = 0.98
    minimum_compressed_long_persistence: float = 0.93
    minimum_compressed_long_diversity: float = 0.90
    maximum_compressed_long_outside_activity: float = 0.10


def _metric(evidence: dict[str, Any], name: str) -> float:
    try:
        value = float(evidence.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def assess_linear_review_v2(
    image: np.ndarray,
    quad: np.ndarray | list[list[float]] | list[float],
    *,
    base_metrics: dict[str, Any] | None = None,
    sources: Iterable[str] = (),
    config: ReviewGateV2Config | None = None,
) -> ReviewDecision:
    """Apply the shared base gate and then the v2 physical certificates."""

    config = config or ReviewGateV2Config()
    base = _assess_linear_review_base(
        image,
        quad,
        base_metrics=base_metrics,
        sources=sources,
        config=config,
    )
    evidence = dict(base.evidence)

    # This is intentionally a veto for a very specific negative shape rather than
    # a new global minimum.  The page-519 field contains roughly twenty diagonal
    # hatch strokes: after rectification they have excellent apparent verticality,
    # but not the density or length grammar of a barcode.
    sparse_hatch_veto = bool(
        _metric(evidence, "base_aspect") < 12.0
        and _metric(evidence, "rectified_height") <= 30.0
        and _metric(evidence, "transitions") <= 32.0
        and _metric(evidence, "dark_fraction") <= 0.20
        and _metric(evidence, "high_occupancy_fraction") <= 0.20
        and _metric(evidence, "outside_activity") >= 0.18
    )

    # All thresholds are conjunctive.  The certificate is therefore allowed to
    # recover the compressed residence-certificate bars without turning any one
    # attractive metric (score, length, or verticality) into a barcode verdict.
    compressed_long_certificate = bool(
        not base.accepted
        and not sparse_hatch_veto
        and _metric(evidence, "base_score")
        >= config.minimum_compressed_long_base_score
        and _metric(evidence, "base_aspect")
        >= config.minimum_compressed_long_aspect
        and _metric(evidence, "rectified_height")
        >= config.minimum_compressed_long_height
        and _metric(evidence, "transitions")
        >= config.minimum_compressed_long_transitions
        and _metric(evidence, "dark_fraction")
        >= config.minimum_compressed_long_dark_fraction
        and _metric(evidence, "high_occupancy_fraction")
        >= config.minimum_compressed_long_occupancy
        and _metric(evidence, "row_mask_overlap")
        >= config.minimum_compressed_long_row_overlap
        and _metric(evidence, "threshold_stability")
        >= config.minimum_compressed_long_threshold_stability
        and _metric(evidence, "module_stability")
        >= config.minimum_compressed_long_module_stability
        and _metric(evidence, "edge_track_support")
        >= config.minimum_compressed_long_edge_support
        and _metric(evidence, "verticality")
        >= config.minimum_compressed_long_verticality
        and _metric(evidence, "vertical_persistence")
        >= config.minimum_compressed_long_persistence
        and _metric(evidence, "run_width_diversity")
        >= config.minimum_compressed_long_diversity
        and _metric(evidence, "outside_activity")
        <= config.maximum_compressed_long_outside_activity
    )

    accepted = bool((base.accepted or compressed_long_certificate) and not sparse_hatch_veto)
    if sparse_hatch_veto:
        reason = "sparse-diagonal-hatch-pattern"
    elif compressed_long_certificate:
        reason = "compressed-long-physical-review-certificate"
    else:
        reason = base.reason

    evidence.update(
        {
            "validation": "review-gate-v2",
            "base_gate_accepted": bool(base.accepted),
            "base_gate_reason": base.reason,
            "sparse_hatch_veto": sparse_hatch_veto,
            "compressed_long_certificate": compressed_long_certificate,
            "accepted": accepted,
            "reason": reason,
        }
    )
    return ReviewDecision(accepted, base.score, reason, evidence)


__all__ = [
    "ReviewGateConfig",
    "ReviewGateV2Config",
    "ReviewDecision",
    "assess_linear_review_v2",
]
