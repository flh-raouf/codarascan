# SPDX-License-Identifier: Apache-2.0
"""Stable engine catalog with lazy access to the local runtime registry."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from codarascan.core.contracts import (
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

from .catalog import ENGINE_CATALOG, EngineSpec

_REGISTRY_EXPORTS = frozenset(
    {
        "DEFAULT_BY_CAPABILITY",
        "default_engine",
        "get_engine",
        "list_engines",
        "warm_all",
    }
)


def load_registry() -> Any:
    """Load the local production registry after engine dependencies are installed."""

    try:
        return import_module(f"{__name__}.registry")
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown module"
        engine_modules = {
            "zxingcpp",
            "cv2",
            "numpy",
        }
        if missing in engine_modules or any(
            missing.startswith(f"{module}.") for module in engine_modules
        ):
            raise RuntimeError(
                "The Tessera/Mosaic runtime is unavailable because "
                f"{missing!r} is missing. Install the engine dependencies with "
                "`pip install -e .` before loading the registry."
            ) from exc
        raise


def __getattr__(name: str) -> Any:
    if name in _REGISTRY_EXPORTS:
        return getattr(load_registry(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "ENGINE_CATALOG",
    "EngineSpec",
    "DEFAULT_BY_CAPABILITY",
    "default_engine",
    "get_engine",
    "list_engines",
    "load_registry",
    "warm_all",
]
