"""Production engine registry.

Research and benchmark adapters live outside the product registry. The product
surface exposes the Recommended and Base choices for extraction and separation.
Older implementations remain in their original modules and can be restored by
re-enabling their registry lines when needed.
"""
from __future__ import annotations

from .base import Capability, Engine, EngineInfo, PageOutcome, Region, Roi
from .detection_guarded_v3 import ENGINE as _GUARDED_LOCALIZER_V3
from .detection_tensor_p7 import ENGINE as _TENSOR_P7
from .extraction_guarded_v3 import ENGINE as _GUARDED_EXTRACTOR_V3
from .extraction_tensor_adaptive import ENGINE as _TENSOR_ADAPTIVE

# Retained but intentionally hidden from the production/frontend surface. These
# imports are commented rather than deleted so the implementations remain
# available for future benchmarks or a deliberate reactivation:
# from .extraction_guarded_v2 import ENGINE as _GUARDED_EXTRACTOR_V2
# from .extraction_vendored import ENGINE as _EXTRACTOR
# from .detection_p7 import ENGINE as _P7
# from .detection_sttg import ENGINE as _STTG_LINEAR_FAST
# from .detection_classical_2d import ENGINE as _CLASSICAL_2D_LOCALIZER
# from .detection_p7_linear import ENGINE as _P7_LINEAR_FAST
# from .extraction_classical_2d import ENGINE as _CLASSICAL_2D
# from .extraction_fast import ENGINE as _FAST_EXTRACTOR

_REGISTRY: dict[str, Engine] = {
    # Extraction: Tessera (Recommended), then Mosaic (product Base).
    _TENSOR_ADAPTIVE.info.id: _TENSOR_ADAPTIVE,
    _GUARDED_EXTRACTOR_V3.info.id: _GUARDED_EXTRACTOR_V3,

    # Retained extraction implementations, intentionally disabled from the
    # product registry. Mosaic keeps the vendored extractor as an internal
    # implementation dependency, but the standalone Base entry is retired.
    # _EXTRACTOR.info.id: _EXTRACTOR,
    # _GUARDED_EXTRACTOR_V2.info.id: _GUARDED_EXTRACTOR_V2,

    # Separation: Tessera (Recommended), then Mosaic (product Base).
    _TENSOR_P7.info.id: _TENSOR_P7,
    _GUARDED_LOCALIZER_V3.info.id: _GUARDED_LOCALIZER_V3,

    # Retained separation implementations, intentionally disabled from the
    # product registry. The coarse-to-fine localizer remains available for
    # direct benchmarks or a deliberate reactivation.
    # _P7.info.id: _P7,
    # Retained fast engines, intentionally disabled from the product registry:
    # _CLASSICAL_2D.info.id: _CLASSICAL_2D,
    # _FAST_EXTRACTOR.info.id: _FAST_EXTRACTOR,
    # _STTG_LINEAR_FAST.info.id: _STTG_LINEAR_FAST,
    # _CLASSICAL_2D_LOCALIZER.info.id: _CLASSICAL_2D_LOCALIZER,
    # _P7_LINEAR_FAST.info.id: _P7_LINEAR_FAST,
}

DEFAULT_BY_CAPABILITY: dict[Capability, str] = {
    Capability.DECODE: _TENSOR_ADAPTIVE.info.id,
    Capability.DETECT: _TENSOR_P7.info.id,
}

__all__ = [
    "Capability",
    "Engine",
    "EngineInfo",
    "PageOutcome",
    "Region",
    "Roi",
    "DEFAULT_BY_CAPABILITY",
    "get_engine",
    "default_engine",
    "list_engines",
    "warm_all",
]


def get_engine(engine_id: str | None, capability: Capability) -> Engine:
    """Resolve an engine id, falling back to the default for the capability."""
    if not engine_id:
        return default_engine(capability)
    engine = _REGISTRY.get(engine_id)
    if engine is None:
        raise KeyError(f"unknown engine {engine_id!r}")
    if engine.info.capability is not capability:
        raise ValueError(
            f"engine {engine_id!r} provides {engine.info.capability.value}, "
            f"but {capability.value} was requested"
        )
    if not engine.info.available:
        raise RuntimeError(engine.info.unavailable_reason or f"engine {engine_id!r} is unavailable")
    return engine


def default_engine(capability: Capability) -> Engine:
    return _REGISTRY[DEFAULT_BY_CAPABILITY[capability]]


def list_engines(capability: Capability | None = None) -> list[EngineInfo]:
    """Return the production engines the UI may offer."""
    infos = [engine.info for engine in _REGISTRY.values()]
    if capability is not None:
        infos = [info for info in infos if info.capability is capability]
    return infos


def warm_all() -> dict[str, str]:
    """Load models and prime caches before the first request.

    Several engines have a non-thread-safe first call, so this runs once at startup
    rather than racing inside a worker pool.
    """
    status: dict[str, str] = {}
    for engine_id, engine in _REGISTRY.items():
        if not engine.info.available:
            status[engine_id] = "unavailable"
            continue
        try:
            engine.warm()
            status[engine_id] = "ready"
        except Exception as exc:  # noqa: BLE001 - a cold engine must not stop startup
            status[engine_id] = f"warm failed: {exc}"
    return status
