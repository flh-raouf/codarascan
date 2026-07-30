"""Ultra-low-latency 1-D localization using the native structure-tensor engine.

This is the production adapter for the clean-room Pipeline 8 implementation
vendored in ``apps/backend/localization``. It is a separate, explicit product
tier: no ZXing, payload decoding, learned model, GPU, or Pipeline 7 fallback is
constructed behind the user's choice.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .base import (
    CONTEXT_MARGIN_PX,
    Capability,
    Engine,
    EngineInfo,
    PageOutcome,
    Region,
    RegionStatus,
    Roi,
    region_center,
)


_LOCALIZATION_DIR = Path(__file__).resolve().parent.parent / "localization"
if str(_LOCALIZATION_DIR) not in sys.path:
    sys.path.append(str(_LOCALIZATION_DIR))

import sttg_localizer  # noqa: E402


_NATIVE_AVAILABLE = getattr(sttg_localizer, "_sttg_native", None) is not None
_WARM_LOCK = threading.Lock()


@dataclass(frozen=True)
class PreparedTensorPage:
    """All data needed by timed tensor-localizer compute."""

    prepared: sttg_localizer.PreparedPage
    full_width: int
    full_height: int
    offset_x: int
    offset_y: int
    roi: Roi | None
    preparation_timings: dict[str, float]


def _quad_to_tuple(
    quad: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    points = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    points[:, 0] += offset_x
    points[:, 1] += offset_y
    return tuple(round(float(value), 2) for value in points.reshape(-1))  # type: ignore[return-value]


class StructureTensorDetector:
    """Prepared-input adapter for the refined document STTG profile."""

    info = EngineInfo(
        id="sttg-linear-fast",
        label="Tensor fast localizer",
        capability=Capability.DETECT,
        summary=(
            "Native structure-tensor localization for ordinary 1-D barcodes on "
            "scanned documents. Refines oversized components at source resolution."
        ),
        speed_ms_per_page="~8.7 ms median engine · ~2.7 ms/page batch",
        accuracy_note=(
            "41/41 dossier recall with zero extras · rejects curved periodic "
            "texture · 1-D only · looser boxes than Coarse-to-fine under "
            "strict IoU"
        ),
        badge="Latest fast · 1D",
        available=_NATIVE_AVAILABLE,
        unavailable_reason=(
            None
            if _NATIVE_AVAILABLE
            else (
                "Native tensor extension is not built. Run "
                "`python localization/setup.py build_ext --inplace`."
            )
        ),
        options={},
    )

    def __init__(self) -> None:
        self._warmed = False

    @staticmethod
    def _config(options: dict[str, Any] | None) -> sttg_localizer.TensorConfig:
        del options
        return sttg_localizer.profile_config("document")

    def warm(self) -> None:
        if not self.info.available or self._warmed:
            return
        with _WARM_LOCK:
            if self._warmed:
                return
            config = self._config(None)
            probe = np.full((320, 320), 255, np.uint8)
            prepared = sttg_localizer.prepare_page(probe, config)
            sttg_localizer.locate_prepared(prepared, config)
            self._warmed = True

    def prepare_page(
        self,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedTensorPage:
        """Load, crop, gray-normalize, and resize before engine timing begins."""
        preparation_started = perf_counter()
        load_started = perf_counter()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        load_seconds = perf_counter() - load_started
        if gray is None:
            raise RuntimeError(f"unable to read page image: {path}")

        full_height, full_width = gray.shape
        offset_x = offset_y = 0
        if roi is not None:
            gray, offset_x, offset_y = roi.crop(
                gray,
                margin=CONTEXT_MARGIN_PX,
            )

        config = self._config(options)
        tensor_prepared = sttg_localizer.prepare_page(gray, config)
        return PreparedTensorPage(
            prepared=tensor_prepared,
            full_width=full_width,
            full_height=full_height,
            offset_x=offset_x,
            offset_y=offset_y,
            roi=roi,
            preparation_timings={
                "image_load_seconds": load_seconds,
                "tensor_preparation_seconds": (
                    tensor_prepared.preparation_seconds
                ),
                "total_seconds": perf_counter() - preparation_started,
            },
        )

    def analyze_prepared(
        self,
        page: int,
        prepared: PreparedTensorPage,
    ) -> PageOutcome:
        if not isinstance(prepared, PreparedTensorPage):
            raise TypeError(
                "Tensor fast localizer requires a PreparedTensorPage input"
            )
        if not self.info.available:
            raise RuntimeError(
                self.info.unavailable_reason
                or "native tensor extension is unavailable"
            )

        started = perf_counter()
        config = self._config(None)
        result = sttg_localizer.locate_prepared(prepared.prepared, config)
        regions = [
            Region(
                quad=_quad_to_tuple(
                    item.quad,
                    prepared.offset_x,
                    prepared.offset_y,
                ),
                kind="linear",
                confidence=float(item.confidence),
                sources=(item.source,),
                status=RegionStatus.LOCALIZED,
                extras={
                    "evidence": {
                        key: round(float(value), 6)
                        for key, value in item.metrics.items()
                    }
                },
            )
            for item in result.detections
        ]

        diagnostics = dict(result.diagnostics)
        if prepared.roi is not None:
            before = len(regions)
            regions = [
                region
                for region in regions
                if prepared.roi.contains_point(
                    *region_center(region.quad),
                    prepared.full_width,
                    prepared.full_height,
                )
            ]
            diagnostics.update(
                {
                    "context_margin_px": CONTEXT_MARGIN_PX,
                    "dropped_outside_zone": before - len(regions),
                    "roi_pixels": prepared.roi.to_pixels(
                        prepared.full_width,
                        prepared.full_height,
                    ),
                    "roi_applied": True,
                }
            )

        diagnostics.update(
            {
                "measurement_scope": (
                    "engine_compute_after_preprocessing"
                ),
                "page_size": {
                    "width": prepared.full_width,
                    "height": prepared.full_height,
                },
                "timings": {
                    key: round(float(value), 6)
                    for key, value in result.timings.items()
                },
                "preparation_timings": {
                    key: round(float(value), 6)
                    for key, value in prepared.preparation_timings.items()
                },
            }
        )
        return PageOutcome(
            page=page,
            regions=regions,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            diagnostics=diagnostics,
        )

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        prepared = self.prepare_page(path, roi=roi, options=options)
        return self.analyze_prepared(page, prepared)


ENGINE: Engine = StructureTensorDetector()
