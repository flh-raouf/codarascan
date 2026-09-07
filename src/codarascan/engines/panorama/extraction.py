# SPDX-License-Identifier: Apache-2.0
"""Panorama high-recall extraction engine.

Panorama combines Tessera, a whole-page ZXing pass, a conservative split
retry, and a failure-tolerant Mosaic fallback. The resulting union is
normalized and reconciled before it crosses the public package boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import zxingcpp

from codarascan import __version__ as CODARASCAN_VERSION
from codarascan.core.contracts import (
    Capability,
    EngineInfo,
    PageOutcome,
    Region,
    RegionStatus,
    Roi,
)
from codarascan.engines.common.native import sttg_localizer
from codarascan.engines.common.tensor_decoder import decode_tensor_candidate
from codarascan.engines.common.vendored_decoder import normalize_engine_options
from codarascan.engines.mosaic.extraction import ENGINE as _MOSAIC_EXTRACTOR
from codarascan.engines.panorama.reconciliation import reconcile_regions
from codarascan.engines.tessera.extraction import ENGINE as _TESSERA_EXTRACTOR
from codarascan.formats import resolve_format

_SPLIT_GAP_SHORT = 1.5


@dataclass(frozen=True)
class PreparedPanoramaPage:
    path: Path
    gray: np.ndarray | None
    width: int
    height: int
    roi: Roi | None
    configured: dict[str, Any]
    engine_options: dict[str, Any]
    normalized_options: dict[str, Any]
    tessera_prepared: Any | None
    tessera_preparation_error: str | None
    preparation_timings: dict[str, float]
    skipped: str | None = None


def _boolean_option(options: dict[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _positive_crosses(points: np.ndarray) -> bool:
    values: list[float] = []
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        third = points[(index + 2) % 4]
        values.append(
            float(
                (second[0] - first[0]) * (third[1] - second[1])
                - (second[1] - first[1]) * (third[0] - second[0])
            )
        )
    return all(value > 1e-6 for value in values)


def _cycle_quad(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    if not _positive_crosses(ordered):
        ordered = ordered[[0, 3, 2, 1]]
    return np.asarray(ordered, dtype=np.float32)


def _canonical_quad(
    quad: Iterable[float], width: int, height: int
) -> tuple[tuple[float, float, float, float, float, float, float, float], bool]:
    """Return a finite, clockwise, non-degenerate page quad.

    Very thin ZXing endpoint quads can make sum/difference ordering ambiguous.
    A minimum-area rectangle preserves their position and orientation; the
    axis-aligned fallback is used only when the engine returned a truly
    degenerate set of points.
    """

    raw = np.asarray(tuple(quad), dtype=np.float32)
    if raw.size != 8 or not np.all(np.isfinite(raw)):
        raise ValueError("engine returned a non-finite or incomplete quadrilateral")
    points = raw.reshape(4, 2)
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height))
    original = points.copy()
    ordered = _cycle_quad(points)
    if len({tuple(point) for point in ordered}) != 4 or not _positive_crosses(ordered):
        rectangle = cv2.minAreaRect(points)
        (center_x, center_y), (rect_width, rect_height), angle = rectangle
        rectangle = (
            (center_x, center_y),
            (max(1.0, float(rect_width)), max(1.0, float(rect_height))),
            angle,
        )
        ordered = _cycle_quad(cv2.boxPoints(rectangle))
    if not _positive_crosses(ordered):
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        if x2 - x1 < 1.0:
            midpoint = 0.5 * (x1 + x2)
            x1, x2 = max(0.0, midpoint - 0.5), min(float(width), midpoint + 0.5)
        if y2 - y1 < 1.0:
            midpoint = 0.5 * (y1 + y2)
            y1, y2 = max(0.0, midpoint - 0.5), min(float(height), midpoint + 0.5)
        ordered = np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)
    if not _positive_crosses(ordered):
        raise ValueError("engine quadrilateral could not be normalized")
    changed = not np.allclose(original, ordered, atol=1e-3)
    return tuple(float(value) for value in ordered.reshape(-1)), changed  # type: ignore[return-value]


def _format_label(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return resolve_format(str(value)).label
    except Exception:  # noqa: BLE001 - retain an engine's future format label
        return str(value)


def _convert_package_region(
    raw: Any,
    *,
    stage: str,
    width: int,
    height: int,
) -> Region:
    quad, normalized = _canonical_quad(raw.quad, width, height)
    extras = dict(getattr(raw, "extras", {}) or {})
    extras["panorama"] = {
        "stage": stage,
        "geometry_normalized": normalized,
    }
    raw_status = getattr(raw.status, "value", str(raw.status))
    return Region(
        quad=quad,
        kind="linear" if str(raw.kind).lower() in {"linear", "1d"} else "matrix",
        confidence=max(0.0, min(1.0, float(raw.confidence))),
        value=str(raw.value) if raw.value not in (None, "") else None,
        raw_bytes=(
            bytes(raw.raw_bytes)
            if getattr(raw, "raw_bytes", None) is not None
            else None
        ),
        symbology=_format_label(raw.symbology),
        sources=tuple(
            dict.fromkeys(
                (f"panorama:{stage}", *(getattr(raw, "sources", ()) or ()))
            )
        ),
        status=RegionStatus(raw_status),
        extras=extras,
    )


def _convert_stage(
    regions: Iterable[Any],
    *,
    stage: str,
    width: int,
    height: int,
) -> tuple[list[Region], list[str]]:
    converted: list[Region] = []
    errors: list[str] = []
    for index, raw in enumerate(regions):
        try:
            converted.append(
                _convert_package_region(raw, stage=stage, width=width, height=height)
            )
        except Exception as exc:  # noqa: BLE001 - one candidate must not erase a page
            errors.append(f"region {index}: {type(exc).__name__}: {exc}")
    return converted, errors


def _direct_linear_regions(gray: np.ndarray, linear_formats: Any) -> list[Region]:
    if not linear_formats:
        return []
    reads = zxingcpp.read_barcodes(
        gray,
        formats=linear_formats,
        try_rotate=True,
        try_downscale=False,
        try_invert=True,
        return_errors=False,
        binarizer=zxingcpp.Binarizer.LocalAverage,
    )
    height, width = gray.shape[:2]
    regions: list[Region] = []
    for barcode in reads:
        if not barcode.valid or not barcode.text:
            continue
        position = barcode.position
        points = (
            position.top_left,
            position.top_right,
            position.bottom_right,
            position.bottom_left,
        )
        quad, normalized = _canonical_quad(
            (coordinate for point in points for coordinate in (point.x, point.y)),
            width,
            height,
        )
        regions.append(
            Region(
                quad=quad,
                kind="linear",
                confidence=1.0,
                value=str(barcode.text),
                raw_bytes=bytes(barcode.bytes),
                symbology=_format_label(barcode.format),
                sources=("panorama:whole-page-zxing",),
                status=RegionStatus.DECODED,
                extras={
                    "panorama": {
                        "stage": "whole-page-zxing",
                        "geometry_normalized": normalized,
                    }
                },
            )
        )
    return regions


def _has_fragment_join(regions: Iterable[Any]) -> bool:
    return any(
        "fragment-join" in " ".join(str(source) for source in (raw.sources or ()))
        for raw in regions
    )


def _split_joined_regions(gray: np.ndarray, linear_formats: Any) -> list[Region]:
    if not linear_formats:
        return []
    config = replace(
        sttg_localizer.profile_config("document"),
        fragment_join_max_gap_short=_SPLIT_GAP_SHORT,
    )
    prepared = sttg_localizer.prepare_page(gray, config)
    localized = sttg_localizer.locate_prepared(prepared, config)
    height, width = gray.shape[:2]
    regions: list[Region] = []
    for detection in localized.detections:
        decoded, attempts = decode_tensor_candidate(gray, detection, linear_formats)
        if decoded is None:
            continue
        quad, normalized = _canonical_quad(detection.quad.reshape(-1), width, height)
        regions.append(
            Region(
                quad=quad,
                kind="linear",
                confidence=1.0,
                value=str(decoded.text),
                raw_bytes=bytes(decoded.raw_bytes),
                symbology=_format_label(decoded.symbology),
                sources=tuple(
                    dict.fromkeys(
                        (
                            "panorama:split-guard",
                            str(detection.source),
                            str(decoded.route),
                        )
                    )
                ),
                status=RegionStatus.DECODED,
                extras={
                    "attempts": list(attempts),
                    "evidence": {
                        key: round(float(value), 6)
                        for key, value in detection.metrics.items()
                    },
                    "panorama": {
                        "stage": "split-guard",
                        "fragment_join_max_gap_short": _SPLIT_GAP_SHORT,
                        "geometry_normalized": normalized,
                    },
                },
            )
        )
    return regions


def _faded_review_candidate(
    raw: dict[str, Any], *, width: int, height: int
) -> Region | None:
    evidence = raw.get("evidence")
    if not isinstance(evidence, dict):
        return None
    accepted = bool(
        float(evidence.get("base_score", 0.0)) >= 0.96
        and float(evidence.get("transitions", 0.0)) >= 60.0
        and float(evidence.get("module_periodicity", 0.0)) >= 0.65
        and float(evidence.get("module_stability", 0.0)) >= 0.94
        and float(evidence.get("edge_track_support", 0.0)) >= 0.95
        and float(evidence.get("verticality", 0.0)) >= 0.95
        and float(evidence.get("high_occupancy_fraction", 0.0)) >= 0.32
        and float(evidence.get("row_mask_overlap", 0.0)) >= 0.95
        and float(evidence.get("threshold_stability", 0.0)) >= 0.90
        and float(evidence.get("outside_activity", 1.0)) <= 0.10
        and not bool(evidence.get("uniform_veto", False))
        and not bool(evidence.get("hatch_veto", False))
        and not bool(evidence.get("document_regular_veto", False))
    )
    if not accepted:
        return None
    quad, normalized = _canonical_quad(raw.get("quad", ()), width, height)
    return Region(
        quad=quad,
        kind="linear",
        confidence=max(
            0.0, min(1.0, float(evidence.get("score", raw.get("base_confidence", 0.0))))
        ),
        sources=("panorama:mosaic-faded-certificate",),
        status=RegionStatus.REVIEW_CANDIDATE,
        extras={
            "evidence": evidence,
            "panorama": {
                "stage": "mosaic-faded-certificate",
                "original_suppression_reason": raw.get("reason"),
                "geometry_normalized": normalized,
            },
        },
    )


def _promoted_faded_reviews(
    diagnostics: dict[str, Any], *, width: int, height: int
) -> list[Region]:
    guarded = diagnostics.get("guarded_review_v3")
    if not isinstance(guarded, dict):
        return []
    suppressed = guarded.get("suppressed_candidates")
    if not isinstance(suppressed, list):
        return []
    promoted: list[Region] = []
    for raw in suppressed:
        if not isinstance(raw, dict):
            continue
        try:
            candidate = _faded_review_candidate(raw, width=width, height=height)
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None:
            promoted.append(candidate)
    return promoted


def _union_regions(
    regions: Iterable[Region],
    *,
    gray: np.ndarray | None = None,
) -> tuple[list[Region], dict[str, Any]]:
    return reconcile_regions(regions, gray=gray)


def _inside_roi(region: Region, roi: Roi, width: int, height: int) -> bool:
    xs = region.quad[0::2]
    ys = region.quad[1::2]
    return roi.contains_point(sum(xs) / 4.0, sum(ys) / 4.0, width, height)


class PanoramaExtractor:
    """Failure-tolerant, high-recall union of complementary scan stages."""

    info = EngineInfo(
        id="panorama-extractor",
        label="Panorama Extractor",
        capability=Capability.DECODE,
        summary=(
            "High-recall union: Tessera reviews, whole-page linear decode, "
            "split-safe joined proposals, and a failure-tolerant Mosaic fallback."
        ),
        speed_ms_per_page="~1–7 s/page sequential",
        accuracy_note=(
            f"CodaraScan {CODARASCAN_VERSION} Panorama engine; prioritizes review and "
            "decoded recall over latency. Validate before deployment."
        ),
        badge="High Recall",
        options={
            "kinds": {
                "type": "enum",
                "values": ["all", "linear", "2d"],
                "default": "all",
                "label": "Symbol types",
                "value_timings": {
                    "all": "~1–7 s/page; difficult recovery can be higher",
                    "linear": "~1–6 s/page with Mosaic fallback",
                    "2d": "~1–4 s/page with Mosaic fallback",
                },
            },
            "formats": {
                "type": "format-list",
                "default": [],
                "label": "Barcode formats",
            },
            "include_mosaic": {
                "type": "bool",
                "default": True,
                "label": "Include Mosaic fallback",
                "help": "Adds complementary Mosaic results; failures remain page-local and non-fatal.",
            },
            "split_joined_regions": {
                "type": "bool",
                "default": True,
                "label": "Split joined Tessera regions",
                "help": "Retries only pages where Tessera reports a fragment join.",
            },
            "promote_faded_reviews": {
                "type": "bool",
                "default": True,
                "label": "Keep strongly structured faded candidates",
                "help": "Promotes only the narrow, audit-tested Mosaic faded-barcode certificate.",
            },
        },
    )
    requires_decode_options = True

    def __init__(self) -> None:
        self._tessera = _TESSERA_EXTRACTOR
        self._mosaic = _MOSAIC_EXTRACTOR

    def warm(self) -> None:
        self._tessera.warm()
        self._mosaic.warm()

    def prepare_page(
        self,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedPanoramaPage:
        configured = dict(options or {})
        normalized_options = normalize_engine_options(configured)
        engine_options = normalized_options

        started = perf_counter()
        load_started = perf_counter()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        load_seconds = perf_counter() - load_started
        if gray is None:
            raise RuntimeError(f"unable to read page image: {path}")
        height, width = gray.shape[:2]

        tessera_started = perf_counter()
        tessera_prepared: Any | None = None
        tessera_error: str | None = None
        try:
            prepare = self._tessera.prepare_page
            # Full-page preparation is intentional. ROI is applied to the
            # final union so localization remains scale/tile stable.
            tessera_prepared = prepare(path, roi=None, options=engine_options)
        except Exception as exc:  # noqa: BLE001 - direct/Mosaic stages can still succeed
            tessera_error = f"{type(exc).__name__}: {exc}"
        tessera_seconds = perf_counter() - tessera_started
        return PreparedPanoramaPage(
            path=path,
            gray=gray,
            width=width,
            height=height,
            roi=roi,
            configured=configured,
            engine_options=engine_options,
            normalized_options=normalized_options,
            tessera_prepared=tessera_prepared,
            tessera_preparation_error=tessera_error,
            preparation_timings={
                "image_load_seconds": round(load_seconds, 6),
                "tessera_preparation_seconds": round(tessera_seconds, 6),
                "total_seconds": round(perf_counter() - started, 6),
            },
        )

    def analyze_prepared(
        self, page: int, prepared: PreparedPanoramaPage
    ) -> PageOutcome:
        if not isinstance(prepared, PreparedPanoramaPage):
            raise TypeError("Panorama requires PreparedPanoramaPage input")
        if prepared.skipped is not None:
            return PageOutcome(
                page=page,
                diagnostics={
                    "codarascan": {
                        "package_version": CODARASCAN_VERSION,
                        "mode": "panorama",
                        "skipped": prepared.skipped,
                    },
                    "preparation_timings": prepared.preparation_timings,
                },
            )
        assert prepared.gray is not None

        started = perf_counter()
        all_regions: list[Region] = []
        stages: dict[str, Any] = {}
        linear_formats = prepared.normalized_options.get("linear_formats")

        direct_started = perf_counter()
        try:
            direct = _direct_linear_regions(prepared.gray, linear_formats)
            all_regions.extend(direct)
            stages["whole_page_zxing"] = {
                "status": "ok",
                "regions": len(direct),
                "elapsed_ms": round((perf_counter() - direct_started) * 1000.0, 2),
            }
        except Exception as exc:  # noqa: BLE001 - stage-local fallback
            stages["whole_page_zxing"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        tessera_raw: list[Any] = []
        tessera_diagnostics: dict[str, Any] = {}
        if prepared.tessera_prepared is None:
            stages["tessera"] = {
                "status": "error",
                "error": prepared.tessera_preparation_error
                or "preparation unavailable",
            }
        else:
            tessera_started = perf_counter()
            try:
                outcome = self._tessera.analyze_prepared(
                    page, prepared.tessera_prepared
                )
                tessera_raw = list(outcome.regions)
                tessera_diagnostics = dict(outcome.diagnostics)
                converted, conversion_errors = _convert_stage(
                    tessera_raw,
                    stage="tessera",
                    width=prepared.width,
                    height=prepared.height,
                )
                all_regions.extend(converted)
                stages["tessera"] = {
                    "status": "ok",
                    "raw_regions": len(tessera_raw),
                    "regions": len(converted),
                    "conversion_errors": conversion_errors,
                    "elapsed_ms": round((perf_counter() - tessera_started) * 1000.0, 2),
                }
            except Exception as exc:  # noqa: BLE001 - direct/Mosaic results remain valid
                stages["tessera"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        split_enabled = _boolean_option(
            prepared.configured, "split_joined_regions", True
        )
        if split_enabled and _has_fragment_join(tessera_raw):
            split_started = perf_counter()
            try:
                split = _split_joined_regions(prepared.gray, linear_formats)
                all_regions.extend(split)
                stages["split_guard"] = {
                    "status": "ok",
                    "triggered": True,
                    "regions": len(split),
                    "fragment_join_max_gap_short": _SPLIT_GAP_SHORT,
                    "elapsed_ms": round((perf_counter() - split_started) * 1000.0, 2),
                }
            except Exception as exc:  # noqa: BLE001 - joined parent remains available
                stages["split_guard"] = {
                    "status": "error",
                    "triggered": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            stages["split_guard"] = {
                "status": "skipped",
                "triggered": False,
                "reason": "disabled or no Tessera fragment join",
            }

        include_mosaic = _boolean_option(prepared.configured, "include_mosaic", True)
        if include_mosaic:
            mosaic_started = perf_counter()
            try:
                # Raw engine output is intentional so one malformed candidate cannot
                # erase otherwise valid page results.
                mosaic_outcome = self._mosaic.analyze_page(
                    page,
                    prepared.path,
                    roi=None,
                    options=prepared.engine_options,
                )
                converted, conversion_errors = _convert_stage(
                    mosaic_outcome.regions,
                    stage="mosaic",
                    width=prepared.width,
                    height=prepared.height,
                )
                all_regions.extend(converted)
                promoted: list[Region] = []
                if _boolean_option(prepared.configured, "promote_faded_reviews", True):
                    promoted = _promoted_faded_reviews(
                        mosaic_outcome.diagnostics,
                        width=prepared.width,
                        height=prepared.height,
                    )
                    all_regions.extend(promoted)
                stages["mosaic"] = {
                    "status": "ok",
                    "raw_regions": len(mosaic_outcome.regions),
                    "regions": len(converted),
                    "promoted_faded_reviews": len(promoted),
                    "conversion_errors": conversion_errors,
                    "elapsed_ms": round((perf_counter() - mosaic_started) * 1000.0, 2),
                    "diagnostics": mosaic_outcome.diagnostics,
                }
            except Exception as exc:  # noqa: BLE001 - the page's earlier results survive
                stages["mosaic"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            stages["mosaic"] = {"status": "skipped", "reason": "disabled by option"}

        union, union_diagnostics = _union_regions(all_regions, gray=prepared.gray)
        roi_dropped = 0
        if prepared.roi is not None:
            before = len(union)
            union = [
                region
                for region in union
                if _inside_roi(region, prepared.roi, prepared.width, prepared.height)
            ]
            roi_dropped = before - len(union)

        elapsed_ms = (perf_counter() - started) * 1000.0
        return PageOutcome(
            page=page,
            regions=union,
            elapsed_ms=elapsed_ms,
            diagnostics={
                "codarascan": {
                    "package_version": CODARASCAN_VERSION,
                    "mode": "panorama",
                    "component_modes": ["fast", "robust"]
                    if include_mosaic
                    else ["fast"],
                },
                "mode": "panorama-union",
                "preparation_timings": prepared.preparation_timings,
                "linear": tessera_diagnostics.get("linear", {}),
                "panorama": {
                    "stages": stages,
                    "union": union_diagnostics,
                    "stable_full_page_roi": prepared.roi is not None,
                    "dropped_outside_roi": roi_dropped,
                },
            },
        )

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        return self.analyze_prepared(
            page,
            self.prepare_page(path, roi=roi, options=options),
        )


ENGINE = PanoramaExtractor()


__all__ = [
    "ENGINE",
    "PanoramaExtractor",
    "PreparedPanoramaPage",
]
