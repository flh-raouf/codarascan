# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Panorama high-recall engine."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from codarascan.core.contracts import Region, RegionStatus, Roi
from codarascan.engines.panorama.extraction import (
    PanoramaExtractor,
    _canonical_quad,
    _faded_review_candidate,
    _positive_crosses,
    _union_regions,
)
from codarascan.scanner import Scanner


def _barcode(payload: str, scale: int = 2) -> np.ndarray:
    return np.asarray(
        zxingcpp.write_barcode_to_image(
            zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.Code128),
            scale=scale,
            add_quiet_zones=True,
        )
    ).copy()


def _write_adjacent(path: Path, *, include_outside: bool = False) -> Path:
    values = ["BRYEKNFJ1T5702729", "90682"]
    symbols = [_barcode(value) for value in values]
    width = sum(symbol.shape[1] for symbol in symbols) + 180
    height = max(symbol.shape[0] for symbol in symbols) + (
        520 if include_outside else 100
    )
    canvas = np.full((height, width), 255, np.uint8)
    left = 40
    for symbol in symbols:
        canvas[40 : 40 + symbol.shape[0], left : left + symbol.shape[1]] = symbol
        left += symbol.shape[1] + 40
    if include_outside:
        outside = _barcode("OUTSIDE-ROI")
        canvas[-outside.shape[0] - 30 : -30, 40 : 40 + outside.shape[1]] = outside
    assert cv2.imwrite(str(path), canvas)
    return path


def _region(
    value: str | None,
    quad: tuple[float, float, float, float, float, float, float, float],
    *,
    sources: tuple[str, ...] = (),
    status: RegionStatus = RegionStatus.DECODED,
) -> Region:
    return Region(
        quad=quad,
        kind="linear",
        confidence=1.0 if value else 0.9,
        value=value,
        symbology="Code 128" if value else None,
        sources=sources,
        status=status,
    )


def test_canonical_quad_repairs_reverse_winding() -> None:
    raw = (10.0, 20.0, 10.0, 80.0, 210.0, 80.0, 210.0, 20.0)

    quad, changed = _canonical_quad(raw, 300, 200)

    assert changed is True
    assert _positive_crosses(np.asarray(quad, dtype=np.float32).reshape(4, 2))


def test_union_replaces_a_joined_parent_with_two_independent_decodes() -> None:
    joined = _region(
        "LEFT",
        (0, 0, 220, 0, 220, 60, 0, 60),
        sources=("panorama:tessera", "sttg:fragment-join"),
    )
    left = _region(
        "LEFT",
        (0, 0, 100, 0, 100, 60, 0, 60),
        sources=("panorama:whole-page-zxing",),
    )
    right = _region(
        "RIGHT",
        (120, 0, 220, 0, 220, 60, 120, 60),
        sources=("panorama:whole-page-zxing",),
    )
    spanning_review = _region(
        None,
        (0, 0, 220, 0, 220, 60, 0, 60),
        status=RegionStatus.REVIEW_CANDIDATE,
    )

    union, diagnostics = _union_regions([joined, left, right, spanning_review])

    assert [(item.value, item.quad) for item in union] == [
        ("LEFT", left.quad),
        ("RIGHT", right.quad),
    ]
    assert diagnostics["deduplicated_or_covered"] == 2


def test_union_preserves_repeated_payloads_at_distinct_locations() -> None:
    first = _region("SAME", (0, 0, 100, 0, 100, 40, 0, 40))
    second = _region("SAME", (200, 0, 300, 0, 300, 40, 200, 40))

    union, _ = _union_regions([first, second])

    assert len(union) == 2


def test_faded_certificate_is_narrow() -> None:
    evidence = {
        "base_score": 0.99,
        "transitions": 96,
        "module_periodicity": 0.69,
        "module_stability": 0.98,
        "edge_track_support": 1.0,
        "verticality": 1.0,
        "high_occupancy_fraction": 0.38,
        "row_mask_overlap": 0.99,
        "threshold_stability": 0.97,
        "outside_activity": 0.0,
        "score": 0.76,
        "uniform_veto": False,
        "hatch_veto": False,
        "document_regular_veto": False,
    }
    raw = {
        "quad": [10, 10, 210, 10, 210, 70, 10, 70],
        "evidence": evidence,
        "reason": "physical-review-gate-failed",
    }

    assert _faded_review_candidate(raw, width=300, height=200) is not None

    evidence["module_periodicity"] = 0.4
    assert _faded_review_candidate(raw, width=300, height=200) is None


def test_panorama_direct_pass_keeps_both_adjacent_codes(tmp_path: Path) -> None:
    source = _write_adjacent(tmp_path / "adjacent.png")
    engine = PanoramaExtractor()

    outcome = engine.analyze_page(
        1,
        source,
        options={
            "kinds": "linear",
            "formats": ("code-128",),
            "include_mosaic": False,
            "split_joined_regions": True,
        },
    )

    assert {item.value for item in outcome.regions if item.decoded} >= {
        "BRYEKNFJ1T5702729",
        "90682",
    }
    assert outcome.diagnostics["codarascan"]["mode"] == "panorama"


def test_public_panorama_mode_returns_canonical_metadata(tmp_path: Path) -> None:
    source = _write_adjacent(tmp_path / "public-panorama.png")

    result = Scanner(
        mode="panorama", symbols="linear", formats=("code-128",)
    ).scan_image(source, diagnostics=True)

    assert {getattr(symbol, "text", None) for symbol in result.symbols} >= {
        "BRYEKNFJ1T5702729",
        "90682",
    }
    assert result.metadata.mode == "panorama"
    assert result.metadata.engine == "panorama-extractor"
    assert result.metadata.package_version == "0.2.0"


def test_panorama_applies_roi_after_full_page_localization(tmp_path: Path) -> None:
    source = _write_adjacent(tmp_path / "roi.png", include_outside=True)
    engine = PanoramaExtractor()

    outcome = engine.analyze_page(
        1,
        source,
        roi=Roi(x=0.0, y=0.0, w=1.0, h=0.45),
        options={
            "kinds": "linear",
            "formats": ("code-128",),
            "include_mosaic": False,
        },
    )

    values = {item.value for item in outcome.regions if item.decoded}
    assert {"BRYEKNFJ1T5702729", "90682"} <= values
    assert "OUTSIDE-ROI" not in values
    assert outcome.diagnostics["panorama"]["stable_full_page_roi"] is True
    assert outcome.diagnostics["panorama"]["dropped_outside_roi"] >= 1


def test_mosaic_failure_does_not_erase_direct_or_tessera_results(
    tmp_path: Path,
) -> None:
    source = _write_adjacent(tmp_path / "mosaic-failure.png")
    engine = PanoramaExtractor()

    class BrokenMosaic:
        def analyze_page(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("deliberate Mosaic failure")

    engine._mosaic = BrokenMosaic()  # type: ignore[assignment]

    outcome = engine.analyze_page(
        1,
        source,
        options={"kinds": "linear", "formats": ("code-128",), "include_mosaic": True},
    )

    assert {item.value for item in outcome.regions if item.decoded} >= {
        "BRYEKNFJ1T5702729",
        "90682",
    }
    mosaic = outcome.diagnostics["panorama"]["stages"]["mosaic"]
    assert mosaic["status"] == "error"
    assert "deliberate Mosaic failure" in mosaic["error"]
