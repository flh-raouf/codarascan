"""Production Tessera and Mosaic engine registry.

The registry is the only runtime entry point for product engines. The engines
and their shared recovery pipeline are repository-owned; callers do not need an
application-specific import path.
"""

from __future__ import annotations

from barcode_detection.core.contracts import (
    Capability,
    Engine,
    EngineInfo,
    PageOutcome,
    Region,
    Roi,
)

from .mosaic.detection import ENGINE as _MOSAIC_LOCALIZER
from .mosaic.extraction import ENGINE as _MOSAIC_EXTRACTOR
from .tessera.detection import ENGINE as _TESSERA_LOCALIZER
from .tessera.extraction import ENGINE as _TESSERA_EXTRACTOR


_REGISTRY: dict[str, Engine] = {
    _TESSERA_EXTRACTOR.info.id: _TESSERA_EXTRACTOR,
    _MOSAIC_EXTRACTOR.info.id: _MOSAIC_EXTRACTOR,
    _TESSERA_LOCALIZER.info.id: _TESSERA_LOCALIZER,
    _MOSAIC_LOCALIZER.info.id: _MOSAIC_LOCALIZER,
}

# Older clients may still send these values. They are input aliases only;
# listings and response descriptors expose the canonical IDs.
_LEGACY_ENGINE_IDS: dict[str, str] = {
    "tensor-adaptive-extractor": "tessera-extractor",
    "tensor-p7-localizer": "tessera-localizer",
    "guarded-adaptive-extractor-v3": "mosaic-extractor",
    "guarded-adaptive-localizer-v3": "mosaic-localizer",
}

DEFAULT_BY_CAPABILITY: dict[Capability, str] = {
    Capability.DECODE: _TESSERA_EXTRACTOR.info.id,
    Capability.DETECT: _TESSERA_LOCALIZER.info.id,
}

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
    "warm_all",
]


def get_engine(engine_id: str | None, capability: Capability) -> Engine:
    """Resolve a canonical or legacy engine ID for one capability."""

    if not engine_id:
        return default_engine(capability)
    canonical_id = _LEGACY_ENGINE_IDS.get(engine_id, engine_id)
    engine = _REGISTRY.get(canonical_id)
    if engine is None:
        raise KeyError(f"unknown engine {engine_id!r}")
    if engine.info.capability is not capability:
        raise ValueError(
            f"engine {engine_id!r} provides {engine.info.capability.value}, "
            f"but {capability.value} was requested"
        )
    if not engine.info.available:
        raise RuntimeError(
            engine.info.unavailable_reason or f"engine {engine_id!r} is unavailable"
        )
    return engine


def default_engine(capability: Capability) -> Engine:
    return _REGISTRY[DEFAULT_BY_CAPABILITY[capability]]


def list_engines(capability: Capability | None = None) -> list[EngineInfo]:
    """Return the production engines available to callers."""

    infos = [engine.info for engine in _REGISTRY.values()]
    if capability is not None:
        infos = [info for info in infos if info.capability is capability]
    return infos


def warm_all() -> dict[str, str]:
    """Warm all available engines without making startup fail on one bad engine."""

    status: dict[str, str] = {}
    for engine_id, engine in _REGISTRY.items():
        if not engine.info.available:
            status[engine_id] = "unavailable"
            continue
        try:
            engine.warm()
            status[engine_id] = "ready"
        except Exception as exc:  # noqa: BLE001 - status is surfaced to callers
            status[engine_id] = f"warm failed: {exc}"
    return status
