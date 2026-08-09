# SPDX-License-Identifier: Apache-2.0
"""Detection-only product adapter for the Mosaic robust extractor.

Mosaic is primarily a decoding engine, but its candidate pipeline is also
useful when separation only needs geometry. This adapter runs that same pipeline and
deliberately strips payloads before returning the outcome, preserving the detection
contract while allowing it to be compared with the Tessera localizer.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from codarascan.core.contracts import (
    Capability,
    Engine,
    EngineInfo,
    PageOutcome,
    Region,
    RegionStatus,
    Roi,
)
from codarascan.engines.mosaic.extraction import ENGINE as MOSAIC_EXTRACTOR


def _as_detection_outcome(outcome: PageOutcome) -> PageOutcome:
    """Convert Mosaic's located/decoded regions into geometry-only results."""

    regions: list[Region] = []
    for region in outcome.regions:
        extras = dict(region.extras)
        # A detection result must not carry a payload, even through an engine-specific
        # diagnostic field. Geometry and review evidence remain useful to callers.
        extras.pop("decoded", None)
        sources = tuple(dict.fromkeys((*region.sources, "guarded-v3:detection-only")))
        regions.append(
            replace(
                region,
                value=None,
                symbology=None,
                sources=sources,
                status=RegionStatus.LOCALIZED,
                extras=extras,
            )
        )
    return replace(outcome, regions=regions)


class MosaicLocalizer:
    """Run Mosaic's physical gates while exposing only localization geometry."""

    # The detection route normally prepares options only for DETECT engines. Mosaic
    # reuses the adaptive decoder internally, so the job service supplies its normal
    # extraction preparation options as well.
    requires_decode_options = True

    info = EngineInfo(
        id="mosaic-localizer",
        label="Mosaic Localizer",
        capability=Capability.DETECT,
        summary=(
            "Mosaic candidate localization with physical review gates and verified "
            "2-D recovery. Payloads are intentionally discarded for detection-only runs."
        ),
        speed_ms_per_page="~280 ms/page sequential",
        accuracy_note=(
            "Robust fallback for small, noisy, damaged, or unresolved symbols; "
            "only physically verified geometry is exposed."
        ),
        badge="Base",
        options=MOSAIC_EXTRACTOR.info.options,
    )

    def warm(self) -> None:
        MOSAIC_EXTRACTOR.warm()

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        outcome = MOSAIC_EXTRACTOR.analyze_page(
            page,
            path,
            roi=roi,
            options=options,
        )
        return _as_detection_outcome(outcome)


ENGINE: Engine = MosaicLocalizer()
