"""Lazy boundary for the copied Codara engine registry.

Importing the package is safe in this repository. The implementation modules
are loaded only when a registry export is requested, because their native
``pipeline`` and ``localization`` dependencies are owned by Codara.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import Capability, Engine, EngineInfo, PageOutcome, Region, Roi
from ..runtime import codara_backend_root

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
    """Load the Codara-backed registry in a configured runtime environment."""
    codara_backend_root()
    try:
        return import_module(f"{__name__}.registry")
    except ModuleNotFoundError as exc:
        if exc.name in {"localization", "pipeline", "classical_2d"}:
            raise RuntimeError(
                "The Codara runtime is not configured; install or expose its "
                "pipeline and localization packages before loading this registry."
            ) from exc
        raise


def __getattr__(name: str) -> Any:
    if name in _REGISTRY_EXPORTS:
        return getattr(load_registry(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Capability",
    "Engine",
    "EngineInfo",
    "PageOutcome",
    "Region",
    "Roi",
    "DEFAULT_BY_CAPABILITY",
    "default_engine",
    "get_engine",
    "list_engines",
    "load_registry",
    "warm_all",
]
