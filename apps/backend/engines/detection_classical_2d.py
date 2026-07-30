"""Geometry-only adapter for the fast classical QR/Data Matrix cascade.

The underlying cascade uses ZXing-C++ as its final physical/error-correction
validation gate.  Separation deliberately discards the decoded payload and
returns only geometry.  Keeping that gate is what prevents tables, stamps, ID
cards, and generic square texture from becoming visible false positives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    Capability,
    Engine,
    EngineInfo,
    PageOutcome,
    Region,
    RegionStatus,
    Roi,
)
from .extraction_classical_2d import (
    ENGINE as CLASSICAL_2D_EXTRACTOR,
)
from .extraction_classical_2d import PreparedClassical2DPage


class Classical2DDetector:
    """Locate only QR/Data Matrix symbols that pass valid decode evidence."""

    info = EngineInfo(
        id="classical-2d-localizer",
        label="Fast classical 2D localizer",
        capability=Capability.DETECT,
        summary=(
            "Geometry-only QR and Data Matrix route backed by the latest "
            "classical 2-D cascade. It internally validates candidates with "
            "ZXing but never exposes payloads in Separation."
        ),
        speed_ms_per_page=(
            "~9–18 ms/page typical engine compute; hard recoveries are higher"
        ),
        accuracy_note=(
            "147/147 normal 2-D symbols · 67/101 severe stress · "
            "valid-decode geometry only"
        ),
        badge="Latest fast · 2D",
        options={
            "mode": {
                "type": "enum",
                "values": ["robust", "fast"],
                "default": "robust",
                "label": "2D localization mode",
                "value_timings": {
                    "robust": (
                        "~9–18 ms/page typical; hard recovery pages are higher"
                    ),
                    "fast": "~8–12 ms/page typical engine compute",
                },
            },
            "kinds": {
                "type": "enum",
                "values": ["2d"],
                "default": "2d",
                "label": "Symbol types",
            },
        },
    )

    def warm(self) -> None:
        CLASSICAL_2D_EXTRACTOR.warm()

    def prepare_page(
        self,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedClassical2DPage:
        return CLASSICAL_2D_EXTRACTOR.prepare_page(
            path,
            roi=roi,
            options=options,
        )

    def analyze_prepared(
        self,
        page: int,
        prepared: PreparedClassical2DPage,
    ) -> PageOutcome:
        extracted = CLASSICAL_2D_EXTRACTOR.analyze_prepared(page, prepared)
        regions = [
            Region(
                quad=region.quad,
                kind="2d",
                confidence=region.confidence,
                sources=(
                    *region.sources,
                    "separation:payload-discarded",
                ),
                status=RegionStatus.LOCALIZED,
                extras={
                    "validated_symbology": region.symbology,
                    "validation": "error-correction-valid decode",
                    **region.extras,
                },
            )
            for region in extracted.regions
        ]
        diagnostics = {
            **extracted.diagnostics,
            "mode": "classical-2d-validated-geometry",
            "payloads_discarded": len(regions),
            "payload_exposed": False,
            "uses_internal_zxing_validation": True,
        }
        return PageOutcome(
            page=page,
            regions=regions,
            elapsed_ms=extracted.elapsed_ms,
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


ENGINE: Engine = Classical2DDetector()
