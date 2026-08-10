# SPDX-License-Identifier: Apache-2.0
"""Low-latency classical 2-D barcode extraction.

The engine is deliberately separate from Adaptive extraction.  It searches
only QR Code and Data Matrix, accepts only error-correction-valid ZXing-C++
payloads, and never promotes morphology or decoder-error geometry as a
user-visible result.

Two modes share the same native-pixel verification:

* ``robust`` scans the empirically complementary 700/925/1200 prepared scales
  and enables a physically verified morphology rescue for an anchorless code;
* ``fast`` uses only native and 1200-pixel LocalAverage evidence.

Image loading and prepared-layer creation implement the ``PreparedEngine``
contract, so queue preparation is reported outside engine compute.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import zxingcpp

from codarascan.core.contracts import (
    CONTEXT_MARGIN_PX,
    Capability,
    Engine,
    EngineInfo,
    PageOutcome,
    Region,
    RegionStatus,
    Roi,
    region_center,
)
from codarascan.engines.common.classical_2d import (
    DecodeProfile,
    Prepared2DPage,
    decode_prepared,
    prepare_gray,
)

_WARM_LOCK = threading.Lock()
_FAST_MODE = "fast"
_ROBUST_MODE = "robust"


def profile_for_mode(mode: str, stage_parallel: bool) -> DecodeProfile:
    workers = 2 if stage_parallel else 1
    if mode == _FAST_MODE:
        return DecodeProfile(
            work_size=1200,
            page_scan_workers=workers,
            enable_morphology=False,
        )
    if mode == _ROBUST_MODE:
        return DecodeProfile(
            work_size=700,
            supplemental_work_sizes=(925, 1200),
            scan_prepared_global=True,
            page_scan_workers=workers,
            morphology_sizes=(9,),
            morphology_percentiles=(99.55, 99.20),
            morphology_candidate_limit=24,
            verify_morphology_structure=True,
            morphology_only_without_results=True,
        )
    raise ValueError(
        f"mode must be {_FAST_MODE!r} or {_ROBUST_MODE!r}, got {mode!r}"
    )


def allowed_matrix_formats(options: dict[str, Any]) -> frozenset[str] | None:
    selected = options.get("matrix_formats")
    if selected is None:
        return None
    names = {str(value) for value in selected}
    if not names:
        return frozenset()
    broad = {"All Matrix", "All Readable", "All"}
    if names & broad:
        return None
    return frozenset(names)


@dataclass(frozen=True)
class PreparedClassical2DPage:
    page_input: Prepared2DPage
    profile: DecodeProfile
    mode: str
    offset_x: int
    offset_y: int
    page_width: int
    page_height: int
    roi: Roi | None
    allowed_formats: frozenset[str] | None
    matrix_formats: Any
    load_seconds: float
    preparation_seconds: float


def prepare_loaded_gray(
    gray: np.ndarray,
    *,
    options: dict[str, Any],
    offset_x: int,
    offset_y: int,
    page_width: int,
    page_height: int,
    roi: Roi | None,
    load_seconds: float = 0.0,
) -> PreparedClassical2DPage:
    """Prepare one already-loaded, already-cropped grayscale page.

    Composite engines use this to share one image load and ROI crop between
    their 1-D and 2-D branches.  It preserves the same preparation and timing
    contract as :meth:`Classical2DExtractor.prepare_page`.
    """
    mode = str(options.get("mode") or _ROBUST_MODE).strip().lower()
    profile = profile_for_mode(
        mode,
        bool(options.get("stage_parallel", False)),
    )
    if options.get("enable_qr_finder_proposals"):
        profile = replace(profile, enable_qr_finder_proposals=True)
    prepared = prepare_gray(gray, profile)
    return PreparedClassical2DPage(
        page_input=prepared,
        profile=profile,
        mode=mode,
        offset_x=offset_x,
        offset_y=offset_y,
        page_width=page_width,
        page_height=page_height,
        roi=roi,
        allowed_formats=allowed_matrix_formats(options),
        matrix_formats=options.get("matrix_formats") or zxingcpp.BarcodeFormat.AllMatrix,
        load_seconds=load_seconds,
        preparation_seconds=prepared.preparation_seconds,
    )


class Classical2DExtractor:
    """Checksum/error-correction-valid classical 2-D reader."""

    info = EngineInfo(
        id="classical-2d-extractor",
        label="Fast classical 2D extractor",
        capability=Capability.DECODE,
        summary=(
            "CPU-only 2-D extraction using complementary "
            "prepared scales, decoder-error geometry, and physically verified "
            "morphology. Never returns an undecoded morphology box."
        ),
        speed_ms_per_page=(
            "~9–18 ms/page typical engine compute; hard recoveries are higher"
        ),
        accuracy_note=(
            "Robust mode: 147/147 exact on the normal 2D suites; "
            "fast mode: 143/147 for lower latency"
        ),
        badge="Latest fast · 2D",
        options={
            "mode": {
                "type": "enum",
                "values": [_ROBUST_MODE, _FAST_MODE],
                "default": _ROBUST_MODE,
                "label": "2D processing mode",
                "value_timings": {
                    _ROBUST_MODE: (
                        "~9–18 ms/page typical; hard recovery pages are higher"
                    ),
                    _FAST_MODE: "~8–12 ms/page typical engine compute",
                },
            },
            "kinds": {
                "type": "enum",
                "values": ["2d"],
                "default": "2d",
                "label": "Symbol types",
            },
            "formats": {
                "type": "format-list",
                "default": [],
                "label": "Barcode formats",
                "help": (
                    "Every selected ZXing-C++ matrix format is routed. QR Code and "
                    "Data Matrix additionally retain specialized physical recovery."
                ),
            },
        },
    )

    def __init__(self) -> None:
        self._warmed = False

    def warm(self) -> None:
        with _WARM_LOCK:
            if self._warmed:
                return
            zxingcpp.barcode_formats_from_str("AllMatrix")
            self._warmed = True

    def prepare_page(
        self,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedClassical2DPage:
        options = options or {}
        load_started = perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        load_seconds = perf_counter() - load_started
        if image is None:
            raise RuntimeError(f"unable to read page image: {path}")
        page_height, page_width = image.shape[:2]
        target = image
        offset_x = offset_y = 0
        if roi is not None:
            target, offset_x, offset_y = roi.crop(
                image,
                margin=CONTEXT_MARGIN_PX,
            )

        return prepare_loaded_gray(
            target,
            options=options,
            offset_x=offset_x,
            offset_y=offset_y,
            page_width=page_width,
            page_height=page_height,
            roi=roi,
            load_seconds=load_seconds,
        )

    def analyze_prepared(
        self,
        page: int,
        prepared: PreparedClassical2DPage,
    ) -> PageOutcome:
        results, info = decode_prepared(
            prepared.page_input,
            prepared.profile,
            prepared.matrix_formats,
        )
        regions: list[Region] = []
        for result in results:
            # ZXing applies the exact selected bitmask during every read. Some
            # subtype selectors intentionally report their public family name
            # after a successful decode (for example Compact PDF417 reports
            # PDF417), so a second string-label filter would discard valid work.
            points = np.asarray(result.quad, np.float32).reshape(4, 2)
            points[:, 0] += prepared.offset_x
            points[:, 1] += prepared.offset_y
            region = Region(
                quad=tuple(float(value) for value in points.reshape(-1)),  # type: ignore[arg-type]
                kind="2d",
                confidence=1.0,
                value=result.value,
                raw_bytes=result.raw_bytes,
                symbology=result.symbology,
                sources=(result.source,),
                status=RegionStatus.DECODED,
                extras={
                    "attempts": list(result.attempts),
                    "classical_2d_mode": prepared.mode,
                },
            )
            if prepared.roi is None or prepared.roi.contains_point(
                *region_center(region.quad),
                prepared.page_width,
                prepared.page_height,
            ):
                regions.append(region)

        engine_seconds = float(info["timings"]["engine_seconds"])
        diagnostics = {
            "mode": prepared.mode,
            "formats": (
                "all-matrix"
                if prepared.allowed_formats is None
                else sorted(prepared.allowed_formats)
            ),
            "valid_only": True,
            "unresolved_geometry_exposed": False,
            "page_size": {
                "width": prepared.page_width,
                "height": prepared.page_height,
            },
            "roi_applied": prepared.roi is not None,
            "preparation_timings": {
                "load_seconds": round(prepared.load_seconds, 6),
                "resize_seconds": round(
                    prepared.preparation_seconds,
                    6,
                ),
                "total_seconds": round(
                    prepared.load_seconds + prepared.preparation_seconds,
                    6,
                ),
            },
            "engine_timings": {
                key: round(float(value), 6)
                for key, value in info["timings"].items()
            },
            "cascade": {
                key: value
                for key, value in info.items()
                if key not in {"timings", "preparation_seconds"}
            },
        }
        return PageOutcome(
            page=page,
            regions=regions,
            elapsed_ms=1000.0 * engine_seconds,
            diagnostics=diagnostics,
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


ENGINE: Engine = Classical2DExtractor()
