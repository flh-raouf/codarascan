"""Shared contracts used by barcode engines and application adapters."""

from .contracts import (
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
