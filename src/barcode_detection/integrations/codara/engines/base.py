"""Compatibility bridge for the historical Codara engine imports.

The shared contract now lives in :mod:`barcode_detection.core.contracts`.
The copied adapters continue to use ``.base`` so their source remains easy to
compare with the Codara application while the contract is promoted to the
repository's stable core.
"""

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
