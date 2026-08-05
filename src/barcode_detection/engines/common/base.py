"""Stable contract imports shared by the active engine implementations."""

from barcode_detection.core.contracts import (
    CONTEXT_MARGIN_PX,
    Capability,
    Engine,
    EngineInfo,
    PageOutcome,
    PreparedEngine,
    Region,
    RegionStatus,
    Roi,
    region_center,
)

__all__ = [
    "CONTEXT_MARGIN_PX",
    "Capability",
    "Engine",
    "EngineInfo",
    "PageOutcome",
    "PreparedEngine",
    "Region",
    "RegionStatus",
    "Roi",
    "region_center",
]

