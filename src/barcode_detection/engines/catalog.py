"""Repository-local catalog of the current engine lineages.

This module is intentionally dependency-free. It can be imported by tooling
without loading OpenCV, ZXing-C++, or the local engine implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CapabilityName = Literal["detect", "decode"]
Role = Literal["recommended", "base"]


@dataclass(frozen=True)
class EngineSpec:
    """Static metadata for one user-facing engine capability."""

    name: str
    capability: CapabilityName
    role: Role
    source: str
    runtime: str = "barcode_detection"


ENGINE_CATALOG: tuple[EngineSpec, ...] = (
    EngineSpec(
        name="Tessera",
        capability="detect",
        role="recommended",
        source="src/barcode_detection/engines/tessera/detection.py",
    ),
    EngineSpec(
        name="Tessera",
        capability="decode",
        role="recommended",
        source="src/barcode_detection/engines/tessera/extraction.py",
    ),
    EngineSpec(
        name="Mosaic",
        capability="detect",
        role="base",
        source="src/barcode_detection/engines/mosaic/detection.py",
    ),
    EngineSpec(
        name="Mosaic",
        capability="decode",
        role="base",
        source="src/barcode_detection/engines/mosaic/extraction.py",
    ),
)
