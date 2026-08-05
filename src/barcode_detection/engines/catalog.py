"""Repository-local catalog of the current engine lineages.

This module is intentionally dependency-free. It can be imported by tooling
without importing the Codara runtime, whose application modules are not present
in this checkout.
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
    runtime: str = "Codara"


ENGINE_CATALOG: tuple[EngineSpec, ...] = (
    EngineSpec(
        name="Tessera",
        capability="detect",
        role="recommended",
        source="src/barcode_detection/integrations/codara/engines/detection_tensor_p7.py",
    ),
    EngineSpec(
        name="Tessera",
        capability="decode",
        role="recommended",
        source="src/barcode_detection/integrations/codara/engines/extraction_tensor_adaptive.py",
    ),
    EngineSpec(
        name="Mosaic",
        capability="detect",
        role="base",
        source="src/barcode_detection/integrations/codara/engines/detection_guarded_v3.py",
    ),
    EngineSpec(
        name="Mosaic",
        capability="decode",
        role="base",
        source="src/barcode_detection/integrations/codara/engines/extraction_guarded_v3.py",
    ),
)
