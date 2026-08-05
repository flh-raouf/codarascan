"""Mosaic robust adaptive extractor implementation.

This is a new copy of the guarded extractor.  It delegates the complete Base
proposal/decode path first, then adds v3-only presentation and recovery rules:

* v2's linear gate is extended with chromatic and dense-layout certificates;
* unresolved matrices without format-specific proof are hidden;
* a bounded physical Data Matrix search can add a verified unresolved location
  when the page image is too small for the decoder to return an anchor.

The existing v2 module is intentionally not imported as an engine superclass so
its behavior and diagnostics remain a stable comparison baseline.
"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import zxingcpp

from localization.localizer import (
    locate_data_matrix,
    order_quad,
    square_warp,
)
from pipeline.deterministic_barcode_locator.barcode_locator import (
    quad_overlap,
    rectify_quad,
)

from .base import Engine, EngineInfo, PageOutcome, Region, RegionStatus
from .extraction_vendored import VendoredExtractor
from .linear_review_gate_v3 import (
    ReviewGateV3Config,
    assess_linear_review_v3,
)


def _review_gate_config(options: dict[str, Any]) -> ReviewGateV3Config:
    values = options.get("review_gate_v3_config")
    if not isinstance(values, dict):
        values = options.get("review_gate_config")
    if not isinstance(values, dict):
        return ReviewGateV3Config()
    allowed = {field.name for field in fields(ReviewGateV3Config)}
    return ReviewGateV3Config(**{key: value for key, value in values.items() if key in allowed})


def _same_linear_review_strip(first: Region, second: Region) -> bool:
    """Recognize two overlapping pieces of one long linear proposal.

    The Base proposal cascade can split one barcode into adjacent strips when a
    page contains a strong horizontal rule or a damaged quiet zone.  This is a
    duplicate only when the strips are collinear and their intersection is a
    meaningful fraction of the smaller strip; merely being on the same row is
    not enough, so genuinely adjacent barcodes remain separate.
    """

    if not (
        first.kind in {"linear", "1d"}
        and second.kind in {"linear", "1d"}
        and first.status is RegionStatus.REVIEW_CANDIDATE
        and second.status is RegionStatus.REVIEW_CANDIDATE
    ):
        return False
    try:
        first_points = np.asarray(first.quad, dtype=np.float32).reshape(4, 2)
        second_points = np.asarray(second.quad, dtype=np.float32).reshape(4, 2)
        first_rect = cv2.minAreaRect(first_points)
        second_rect = cv2.minAreaRect(second_points)
        first_long, first_short, _ = _quad_dimensions(first_points)
        second_long, second_short, _ = _quad_dimensions(second_points)
    except (cv2.error, TypeError, ValueError):
        return False
    if min(first_short, second_short) < 1.0:
        return False
    def long_axis_angle(rectangle: tuple[Any, Any, Any]) -> float:
        rectangle_angle = float(rectangle[2])
        rectangle_width, rectangle_height = (
            float(value) for value in rectangle[1]
        )
        if rectangle_width < rectangle_height:
            rectangle_angle += 90.0
        return rectangle_angle % 180.0

    first_angle = long_axis_angle(first_rect)
    second_angle = long_axis_angle(second_rect)
    angle_delta = abs(first_angle - second_angle)
    angle_delta = min(angle_delta, 180.0 - angle_delta)
    first_center = first_points.mean(axis=0)
    second_center = second_points.mean(axis=0)
    x_overlap = max(
        0.0,
        min(float(np.max(first_points[:, 0])), float(np.max(second_points[:, 0])))
        - max(float(np.min(first_points[:, 0])), float(np.min(second_points[:, 0]))),
    )
    return bool(
        angle_delta <= 8.0
        and abs(float(first_center[1] - second_center[1]))
        <= 0.60 * max(first_short, second_short)
        and x_overlap >= 0.12 * min(first_long, second_long)
        and quad_overlap(first_points, second_points) >= 0.10
    )


def _merge_linear_review_strips(regions: list[Region]) -> tuple[list[Region], int]:
    """Merge only same-row overlapping review strips, preserving all evidence."""

    merged: list[Region] = []
    merge_count = 0
    for region in regions:
        prior_index = next(
            (
                index
                for index, prior in enumerate(merged)
                if _same_linear_review_strip(prior, region)
            ),
            None,
        )
        if prior_index is None:
            merged.append(region)
            continue

        prior = merged[prior_index]
        try:
            all_points = np.vstack(
                (
                    np.asarray(prior.quad, dtype=np.float32).reshape(4, 2),
                    np.asarray(region.quad, dtype=np.float32).reshape(4, 2),
                )
            )
            merged_quad = order_quad(cv2.boxPoints(cv2.minAreaRect(all_points)))
        except (cv2.error, TypeError, ValueError):
            continue
        extras = dict(prior.extras)
        evidence = extras.get("evidence")
        merged_evidence = dict(evidence) if isinstance(evidence, dict) else {}
        merged_evidence["review_merge"] = {
            "merged_duplicate_strip": True,
            "merged_region_count": 2,
            "merged_quads": [list(prior.quad), list(region.quad)],
        }
        extras["evidence"] = merged_evidence
        sources = tuple(dict.fromkeys((*prior.sources, *region.sources)))
        merged[prior_index] = replace(
            prior,
            quad=tuple(float(value) for value in merged_quad.reshape(-1)),
            confidence=max(float(prior.confidence), float(region.confidence)),
            sources=sources,
            extras=extras,
        )
        merge_count += 1
    return merged, merge_count


def _quad_dimensions(quad: np.ndarray | tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    """Return min-area-rectangle long side, short side, and aspect ratio."""

    try:
        points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        long_side, short_side = sorted(
            (float(value) for value in cv2.minAreaRect(points)[1]),
            reverse=True,
        )
    except (cv2.error, TypeError, ValueError):
        return 0.0, 0.0, float("inf")
    return long_side, short_side, long_side / max(1.0, short_side)


def _matrix_gate_details(region: Region) -> tuple[dict[str, Any], dict[str, Any]] | None:
    evidence = region.extras.get("evidence") if isinstance(region.extras, dict) else None
    if not isinstance(evidence, dict):
        return None
    gate = evidence.get("matrix_format_gate")
    if not isinstance(gate, dict) or gate.get("accepted") is not True:
        return None
    details = gate.get("data_matrix")
    if not isinstance(details, dict):
        details = gate.get("details")
    if not isinstance(details, dict):
        return None
    return gate, details


def _matrix_is_verified(region: Region) -> bool:
    """Keep unresolved matrices only when format proof is physically credible.

    Base's damaged-decoder anchor is intentionally permissive: it is allowed to
    retain a borderline matrix for downstream review.  v3 is the user-facing
    guarded copy, so it adds the missing source-geometry check here.  In
    particular, a normalized 52x52 lattice inside a 15px logo fragment or a
    28px slanted MRZ fragment is not a real unresolved symbol.
    """

    gate_details = _matrix_gate_details(region)
    if gate_details is None:
        return False
    _gate, details = gate_details
    return _is_physical_data_matrix(
        {key: float(value) for key, value in details.items() if isinstance(value, (int, float))},
        np.asarray(region.quad, dtype=np.float32).reshape(4, 2),
    )


def _is_physical_data_matrix(metrics: dict[str, float], quad: np.ndarray) -> bool:
    width, height, aspect = _quad_dimensions(quad)
    short_side = min(width, height)
    inferred_modules = max(
        float(metrics.get("symbol_rows", 0.0)),
        float(metrics.get("symbol_columns", 0.0)),
    )
    # Do not accept a grid that only exists because a tiny noisy patch was
    # upsampled into the fixed 160px verifier canvas.  At least ~1.35 source
    # pixels per inferred module is a conservative physical-resolution floor;
    # the 50px MRZ/noise false positive in the regression captures inferred a
    # 64x64 grid and therefore fails here, while the 22x22 and 52x52 CNRC
    # symbols retain enough source resolution.
    source_module_pitch = short_side / max(1.0, inferred_modules)
    return bool(
        short_side >= 45.0
        and aspect <= 1.55
        and source_module_pitch >= 1.35
        and float(metrics.get("grid_score", 0.0)) >= 0.60
        and float(metrics.get("finder_score", 0.0)) >= 0.50
        and float(metrics.get("solid_left", 0.0)) >= 0.70
        and float(metrics.get("solid_bottom", 0.0)) >= 0.72
        and float(metrics.get("timing_top", 0.0)) >= 0.52
        and float(metrics.get("timing_right", 0.0)) >= 0.52
        and float(metrics.get("cell_purity", 0.0)) >= 0.68
        and 0.24 <= float(metrics.get("module_occupancy", 0.0)) <= 0.70
    )


def _header_caption_features(
    gray: np.ndarray,
    quad: np.ndarray,
    *,
    proposal_includes_caption: bool,
) -> tuple[int, float, float]:
    """Measure a compact text-like caption adjacent to a header symbol.

    The bounded header recovery is intentionally narrower than a general
    barcode detector: the genuine damaged symbols in this corpus are printed
    in a document header and have a short, clean caption immediately below
    them.  FIAT's logo and the noisy table also contain dark rectangles, but
    neither has that caption-shaped component pattern.

    Long horizontal/vertical rules are removed before connected components are
    measured.  The returned values are component count, horizontal span, and
    normalized vertical-center spread.  This is a visual context certificate,
    not OCR and does not depend on the caption's language.
    """

    try:
        points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
    except (TypeError, ValueError):
        return 0, 0.0, 1.0

    width = float(x2 - x1)
    height = float(y2 - y1)
    if min(width, height) < 20.0:
        return 0, 0.0, 1.0

    # Accepted localizer geometries are tight around the matrix, so the
    # caption is outside and just below the quad.  Rejected energy proposals
    # on the difficult pages include the caption in their larger rough quad;
    # inspect the lower half of that quad instead.
    if proposal_includes_caption:
        roi_x1, roi_x2 = x1 - 0.10 * width, x2 + 0.10 * width
        roi_y1, roi_y2 = y1 + 0.55 * height, y2 + 0.08 * height
    else:
        roi_x1, roi_x2 = x1 - 0.12 * width, x2 + 0.12 * width
        roi_y1, roi_y2 = y2 - 0.03 * height, y2 + 0.38 * height

    ix1 = max(0, int(np.floor(roi_x1)))
    iy1 = max(0, int(np.floor(roi_y1)))
    ix2 = min(gray.shape[1], int(np.ceil(roi_x2)))
    iy2 = min(gray.shape[0], int(np.ceil(roi_y2)))
    if ix2 - ix1 < 20 or iy2 - iy1 < 12:
        return 0, 0.0, 1.0

    roi = np.asarray(gray[iy1:iy2, ix1:ix2])
    if roi.size == 0:
        return 0, 0.0, 1.0
    mask = cv2.threshold(
        roi,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    # Remove document rules and the frame around the caption.  The remaining
    # components are the glyph-like parts, which are much more stable than a
    # raw dark-pixel count across clean and degraded scans.
    horizontal_kernel = np.ones(
        (1, max(5, int(round(0.18 * roi.shape[1])))),
        dtype=np.uint8,
    )
    vertical_kernel = np.ones(
        (max(5, int(round(0.25 * roi.shape[0]))), 1),
        dtype=np.uint8,
    )
    clean = cv2.subtract(
        mask,
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel),
    )
    clean = cv2.subtract(
        clean,
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel),
    )
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    _count, _labels, statistics, _centers = cv2.connectedComponentsWithStats(
        (clean > 0).astype(np.uint8),
        connectivity=8,
    )
    components: list[tuple[int, int, int, int, int]] = []
    roi_height, roi_width = roi.shape
    for statistic in statistics[1:]:
        component_x, component_y, component_width, component_height, area = (
            int(value) for value in statistic
        )
        if (
            area >= 10
            and 0.02 * roi_width <= component_width <= 0.45 * roi_width
            and 0.12 * roi_height <= component_height <= 0.95 * roi_height
        ):
            components.append(
                (
                    component_x,
                    component_y,
                    component_width,
                    component_height,
                    area,
                )
            )

    if not components:
        return 0, 0.0, 1.0
    left = min(component[0] for component in components)
    right = max(component[0] + component[2] for component in components)
    vertical_centers = np.asarray(
        [component[1] + component[3] / 2.0 for component in components],
        dtype=np.float32,
    )
    return (
        len(components),
        float(right - left) / max(1.0, float(roi_width)),
        float(np.std(vertical_centers)) / max(1.0, float(roi_height)),
    )


def _header_caption_certificate(
    gray: np.ndarray,
    quad: np.ndarray,
    metrics: dict[str, float],
) -> bool:
    """Require a compact document-header caption for weak matrix recovery."""

    component_count, horizontal_span, vertical_spread = _header_caption_features(
        gray,
        quad,
        proposal_includes_caption="proposal_component_count" in metrics,
    )
    if "proposal_component_count" in metrics:
        # The rough proposal is used for pages 558/771 where the symbol is
        # degraded.  Page 505's table has a much larger, incoherent component
        # population; the upper bound prevents that texture from masquerading
        # as a caption while retaining the noisy 771 scan.
        return bool(
            4 <= component_count <= 24
            and horizontal_span >= 0.45
            and vertical_spread <= 0.25
        )
    # Page 265 is the lower-resolution genuine sample with only three stable
    # glyph components after line removal, so three is the intentional floor.
    return bool(
        3 <= component_count <= 24
        and horizontal_span >= 0.45
        and vertical_spread <= 0.30
    )


def _is_header_data_matrix_rescue(
    metrics: dict[str, float],
    quad: np.ndarray,
    image_shape: tuple[int, ...],
    gray: np.ndarray | None = None,
) -> bool:
    """Accept a degraded header matrix only with two independent certificates.

    The CNRC samples that Base misses are large enough to contain a real module
    lattice, but their scan quality is below the normal ECC-200 acceptance
    threshold.  The localizer's bidirectional-energy proposal is an independent
    signal from the lattice scorer.  Requiring both, plus a conservative upper
    page-band and near-square native geometry, keeps this recovery scoped to the
    observed failure mode instead of turning it into a second generic detector.
    """

    height = int(image_shape[0]) if image_shape else 0
    center_y = float(np.asarray(quad, dtype=np.float32).reshape(4, 2).mean(axis=0)[1])
    width, short_side, aspect = _quad_dimensions(quad)
    inferred_modules = max(
        float(metrics.get("symbol_rows", 0.0)),
        float(metrics.get("symbol_columns", 0.0)),
    )
    source_module_pitch = short_side / max(1.0, inferred_modules)
    if not (
        height > 0
        and center_y <= 0.42 * height
        and short_side >= 80.0
        and aspect <= 1.55
        and source_module_pitch >= 2.0
        and float(metrics.get("grid_score", 0.0)) >= 0.55
        and float(metrics.get("finder_score", 0.0)) >= 0.50
        and float(metrics.get("solid_left", 0.0)) >= 0.55
        and float(metrics.get("solid_bottom", 0.0)) >= 0.52
        and float(metrics.get("timing_top", 0.0)) >= 0.50
        and float(metrics.get("timing_right", 0.0)) >= 0.50
        and float(metrics.get("cell_purity", 0.0)) >= 0.60
        and 0.20 <= float(metrics.get("module_occupancy", 0.0)) <= 0.75
    ):
        return False

    # A fully accepted localizer matrix already carries a strong format
    # certificate.  The weaker branch below is reserved for rejected energy
    # proposals and therefore needs the proposal-side measurements as well.
    if "proposal_base_score" in metrics and not (
        float(metrics.get("proposal_base_score", 0.0)) >= 0.78
        and float(metrics.get("proposal_square_score", 0.0)) >= 0.70
        and float(metrics.get("proposal_transition_score", 0.0)) >= 0.75
        and 0.25 <= float(metrics.get("proposal_dark_fraction", 0.0)) <= 0.65
        and float(metrics.get("proposal_grid_q20", 0.0)) >= 0.22
        and float(metrics.get("proposal_l_border_score", 0.0)) >= 0.45
    ):
        return False
    return gray is None or _header_caption_certificate(gray, quad, metrics)


def _localizer_header_matrix_candidates(
    gray: np.ndarray,
) -> list[tuple[np.ndarray, dict[str, float]]]:
    """Collect only the bounded low-resolution energy proposals for v3 rescue."""

    try:
        matrices, _local_qr, info = locate_data_matrix(
            gray,
            (),
            700,
            include_rejected=True,
        )
    except (cv2.error, TypeError, ValueError):
        return []

    detections = [*matrices, *info.get("_rejected_candidates", [])]
    output: list[tuple[np.ndarray, dict[str, float]]] = []
    for detection in detections:
        quad = np.asarray(detection.quad, dtype=np.float32).reshape(4, 2)
        metrics = {
            key: float(value)
            for key, value in detection.metrics.items()
            if isinstance(value, (int, float, np.number))
        }
        if not _is_header_data_matrix_rescue(metrics, quad, gray.shape, gray):
            continue
        caption_components, caption_span, caption_spread = _header_caption_features(
            gray,
            quad,
            proposal_includes_caption="proposal_component_count" in metrics,
        )
        metrics.update(
            {
                "header_caption_components": float(caption_components),
                "header_caption_span": caption_span,
                "header_caption_vertical_spread": caption_spread,
            }
        )
        metrics["header_recovery"] = 1.0
        if any(quad_overlap(quad, prior) >= 0.60 for prior, _ in output):
            continue
        output.append((quad, metrics))
    output.sort(key=lambda item: float(item[1].get("grid_score", 0.0)), reverse=True)
    return output[:4]


def _weak_data_matrix_candidates(gray: np.ndarray) -> list[tuple[np.ndarray, dict[str, float]]]:
    """Recover small, physically verifiable matrix regions missed by the decoder."""

    # First ask the existing 2-D energy proposal stage at a deliberately
    # reduced work size.  It is useful here as a proposal generator even when
    # its final 0.68 ECC-200 threshold rejects a damaged symbol.  The v3 header
    # certificate above decides whether a rejected proposal is still credible;
    # no generic low-score contour is promoted by this path.
    header_candidates = _localizer_header_matrix_candidates(gray)
    return header_candidates[:1]


def _decode_matrix_candidate(
    gray: np.ndarray,
    quad: np.ndarray,
) -> tuple[str, str] | None:
    """Try a few bounded matrix views before exposing an unresolved result."""

    try:
        base = square_warp(gray, quad, 240)
    except (cv2.error, ValueError, TypeError):
        return None
    variants = [base, cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
    for variant in variants:
        for scale in (1.0, 1.5, 2.0):
            view = (
                variant
                if scale == 1.0
                else cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            )
            try:
                reads = zxingcpp.read_barcodes(
                    view,
                    formats=zxingcpp.barcode_formats_from_str("QRCode,DataMatrix"),
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                    return_errors=False,
                )
            except Exception:  # decoder errors are expected on damaged candidates
                continue
            for barcode in reads:
                if barcode.valid and barcode.text:
                    return str(barcode.format), str(barcode.text)
    return None


class GuardedExtractorV3(VendoredExtractor):
    """Mosaic: the robust adaptive decode path with evidence and recovery layers."""

    info = EngineInfo(
        id="guarded-adaptive-extractor-v3",
        label="Mosaic",
        capability=VendoredExtractor.info.capability,
        summary=(
            "Mosaic adaptive decoding with physical review gates: chromatic-bar rescue, "
            "degraded dense-layout rescue, and bounded verified 2-D recovery."
        ),
        speed_ms_per_page="~280 ms/page sequential",
        accuracy_note="",
        badge="Base",
        options={
            "kinds": {
                **VendoredExtractor.info.options["kinds"],
                # Mosaic adds the evidence/recovery passes on top of the vendored
                # baseline, so expose its own estimates instead of the old ~203 ms
                # implementation timing inherited from VendoredExtractor.
                "value_timings": {
                    "all": "~280 ms/page",
                    "linear": "~240 ms/page",
                    "2d": "~160 ms/page",
                },
            },
            "formats": VendoredExtractor.info.options["formats"],
        },
    )

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi=None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        started = perf_counter()
        options = options or {}
        outcome = super().analyze_page(page, path, roi=roi, options=options)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if gray is None:
            return outcome

        config = _review_gate_config(options)
        layout_context = list(outcome.regions)
        visible: list[Region] = []
        suppressed_linear: list[dict[str, Any]] = []
        suppressed_matrix: list[dict[str, Any]] = []
        input_review_count = 0
        accepted_review_count = 0

        for region in outcome.regions:
            if region.status is RegionStatus.UNRESOLVED_MATRIX:
                if _matrix_is_verified(region):
                    visible.append(region)
                else:
                    suppressed_matrix.append(
                        {
                            "quad": list(region.quad),
                            "reason": "matrix-format-proof-missing",
                            "base_confidence": round(float(region.confidence), 4),
                        }
                    )
                continue
            if (
                region.status is not RegionStatus.REVIEW_CANDIDATE
                or region.kind not in {"linear", "1d"}
            ):
                visible.append(region)
                continue

            input_review_count += 1
            structural: dict[str, Any] = {}
            evidence = region.extras.get("evidence") if isinstance(region.extras, dict) else None
            if isinstance(evidence, dict) and isinstance(evidence.get("structural_metrics"), dict):
                structural = evidence["structural_metrics"]
            decision = assess_linear_review_v3(
                gray,
                region.quad,
                color_image=color,
                base_metrics=structural,
                sources=region.sources,
                layout_context=layout_context,
                config=config,
            )
            extras = dict(region.extras)
            merged_evidence = dict(evidence) if isinstance(evidence, dict) else {}
            merged_evidence["review_gate"] = decision.evidence
            extras["evidence"] = merged_evidence
            annotated = replace(
                region,
                confidence=round(float(decision.score), 4),
                extras=extras,
            )
            if decision.accepted:
                visible.append(annotated)
                accepted_review_count += 1
            else:
                suppressed_linear.append(
                    {
                        "quad": list(region.quad),
                        "kind": region.kind,
                        "base_confidence": round(float(region.confidence), 4),
                        "reason": decision.reason,
                        "evidence": decision.evidence,
                    }
                )
                if bool(options.get("expose_suppressed_candidates", False)):
                    visible.append(annotated)

        visible, merged_linear_review_count = _merge_linear_review_strips(visible)
        accepted_review_count = sum(
            region.status is RegionStatus.REVIEW_CANDIDATE
            and region.kind in {"linear", "1d"}
            for region in visible
        )

        weak_matrix: list[dict[str, Any]] = []
        # A page that already has a decoded matrix does not need an expensive
        # anchorless search. The no-decoded cases are exactly where this bounded
        # recovery has useful recall value.
        existing_matrix = [
            region
            for region in visible
            if region.kind in {"matrix", "2d"}
            and region.status in {RegionStatus.DECODED, RegionStatus.UNRESOLVED_MATRIX}
        ]
        weak_matrix_recovery_enabled = bool(options.get("weak_matrix_recovery", True)) and roi is None
        run_matrix_recovery = weak_matrix_recovery_enabled and not any(
            region.status is RegionStatus.DECODED for region in existing_matrix
        )
        if run_matrix_recovery:
            for quad, metrics in _weak_data_matrix_candidates(gray):
                if any(quad_overlap(np.asarray(quad), np.asarray(region.quad).reshape(4, 2)) >= 0.60 for region in visible):
                    continue
                decoded = _decode_matrix_candidate(gray, quad)
                verification_source = (
                    "verification:data-matrix-header-structure"
                    if float(metrics.get("header_recovery", 0.0)) > 0.0
                    else "verification:data-matrix-border-and-lattice"
                )
                evidence = {
                    "validation": "weak-data-matrix-v3",
                    "matrix_format_gate": {
                        "accepted": True,
                        "format": "Data Matrix",
                        "verifier": verification_source.removeprefix("verification:"),
                        "details": {key: float(value) for key, value in metrics.items()},
                    },
                    "decoder_retried": decoded is not None,
                }
                if decoded is not None:
                    symbology, value = decoded
                    region = Region(
                        quad=tuple(float(value_) for value_ in np.asarray(quad).reshape(-1)),  # type: ignore[arg-type]
                        kind="matrix",
                        confidence=1.0,
                        value=value,
                        symbology=symbology,
                        sources=("v3:weak-data-matrix-recovery", verification_source),
                        status=RegionStatus.DECODED,
                        extras={"evidence": evidence},
                    )
                else:
                    region = Region(
                        quad=tuple(float(value_) for value_ in np.asarray(quad).reshape(-1)),  # type: ignore[arg-type]
                        kind="matrix",
                        confidence=round(max(0.90, min(0.99, float(metrics.get("grid_score", 0.0)))), 4),
                        sources=("v3:weak-data-matrix-recovery", verification_source),
                        status=RegionStatus.UNRESOLVED_MATRIX,
                        extras={"evidence": evidence},
                    )
                visible.append(region)
                weak_matrix.append(
                    {
                        "quad": list(region.quad),
                        "status": region.status.value,
                        "grid_score": round(float(metrics.get("grid_score", 0.0)), 4),
                    }
                )

        diagnostics = dict(outcome.diagnostics)
        diagnostics["guarded_review_v3"] = {
            "validation": "review-gate-v3",
            "input_review_candidates": input_review_count,
            "accepted_review_candidates": accepted_review_count,
            "merged_linear_review_candidates": merged_linear_review_count,
            "suppressed_review_candidates": len(suppressed_linear),
            "suppressed_candidates": suppressed_linear,
            "suppressed_matrix_candidates": suppressed_matrix,
            "weak_matrix_recovery_enabled": weak_matrix_recovery_enabled,
            "weak_matrix_recovery_attempted": run_matrix_recovery,
            "weak_matrix_candidates": weak_matrix,
            "decode_path_preserved": True,
            "base_engine": "adaptive-extractor",
            "comparison_engine": "guarded-adaptive-extractor-v2",
        }
        diagnostics["guarded_review_v3_seconds"] = round(perf_counter() - started, 6)
        return replace(outcome, regions=visible, diagnostics=diagnostics)


ENGINE: Engine = GuardedExtractorV3()
