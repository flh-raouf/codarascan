# SPDX-License-Identifier: Apache-2.0
"""Repository-local catalog of the current engine lineages.

This module is intentionally dependency-free. It can be imported by tooling
without loading OpenCV, ZXing-C++, or the local engine implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CapabilityName = Literal["detect", "decode"]
Role = Literal["recommended", "base", "high-recall"]


@dataclass(frozen=True)
class EngineSpec:
    """Static metadata for one user-facing engine capability."""

    name: str
    capability: CapabilityName
    role: Role
    source: str
    runtime: str = "codarascan"


ENGINE_CATALOG: tuple[EngineSpec, ...] = (
    EngineSpec(
        name="Tessera",
        capability="detect",
        role="recommended",
        source="src/codarascan/engines/tessera/detection.py",
    ),
    EngineSpec(
        name="Tessera",
        capability="decode",
        role="recommended",
        source="src/codarascan/engines/tessera/extraction.py",
    ),
    EngineSpec(
        name="Mosaic",
        capability="detect",
        role="base",
        source="src/codarascan/engines/mosaic/detection.py",
    ),
    EngineSpec(
        name="Mosaic",
        capability="decode",
        role="base",
        source="src/codarascan/engines/mosaic/extraction.py",
    ),
    EngineSpec(
        name="Panorama",
        capability="decode",
        role="high-recall",
        source="src/codarascan/engines/panorama/extraction.py",
    ),
)
