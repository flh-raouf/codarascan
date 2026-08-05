"""Low-latency, high-precision 1-D localization using pipeline 7.

This is an explicit product tier, not an automatic shortcut. It runs only the
validated linear proposal and native-pixel verification branch, so it never
pays for Data Matrix or QR localization and never claims to support them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .base import CONTEXT_MARGIN_PX, Capability, Engine, EngineInfo, PageOutcome, Roi
from .detection_p7 import Pipeline7Detector, localizer


@dataclass(frozen=True)
class PreparedLinearPage:
    """The complete input contract consumed by Fast linear compute."""

    native_gray: np.ndarray
    detector_gray: np.ndarray
    config: Any
    full_width: int
    full_height: int
    offset_x: int
    offset_y: int
    roi: Roi | None
    preparation_timings: dict[str, float]


class FastLinearDetector(Pipeline7Detector):
    info = EngineInfo(
        id="p7-linear-fast",
        label="Fast linear localizer",
        capability=Capability.DETECT,
        summary=(
            "Verified pipeline-7 localization for 1-D barcodes only. Designed "
            "for scanned forms and batches known not to contain QR/Data Matrix."
        ),
        speed_ms_per_page="~15 ms median engine compute after preprocessing",
        accuracy_note=(
            "High-precision 1-D tier · intentionally ignores every 2-D symbol"
        ),
        badge="Previous fast · 1D",
        options={},
    )

    @staticmethod
    def _config(options: dict[str, Any] | None) -> Any:
        del options
        return Pipeline7Detector._config({"kinds": "linear"})

    def prepare_page(
        self,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedLinearPage:
        """Load and transform a page before it enters the engine compute queue."""
        preparation_started = perf_counter()
        load_started = perf_counter()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        load_seconds = perf_counter() - load_started
        if gray is None:
            raise RuntimeError(f"unable to read page image: {path}")

        full_height, full_width = gray.shape[:2]
        offset_x = offset_y = 0
        if roi is not None:
            gray, offset_x, offset_y = roi.crop(
                gray,
                margin=CONTEXT_MARGIN_PX,
            )

        config = self._config(options)
        scale, _scales = localizer.linear_profile(config)
        resize_started = perf_counter()
        detector_gray = (
            cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else gray
        )
        resize_seconds = perf_counter() - resize_started
        return PreparedLinearPage(
            native_gray=gray,
            detector_gray=detector_gray,
            config=config,
            full_width=full_width,
            full_height=full_height,
            offset_x=offset_x,
            offset_y=offset_y,
            roi=roi,
            preparation_timings={
                "image_load_seconds": load_seconds,
                "detector_resize_seconds": resize_seconds,
                "total_seconds": perf_counter() - preparation_started,
            },
        )

    def analyze_prepared(
        self,
        page: int,
        prepared: PreparedLinearPage,
    ) -> PageOutcome:
        """Time only localization compute over already-prepared arrays."""
        if not isinstance(prepared, PreparedLinearPage):
            raise TypeError("Fast linear requires a PreparedLinearPage input")
        started = perf_counter()
        return self._analyze_gray(
            page,
            prepared.native_gray,
            config=prepared.config,
            started=started,
            full_width=prepared.full_width,
            full_height=prepared.full_height,
            offset_x=prepared.offset_x,
            offset_y=prepared.offset_y,
            roi=prepared.roi,
            linear_work=prepared.detector_gray,
            diagnostics_extra={
                "measurement_scope": "engine_compute_after_preprocessing",
                "preparation_timings": {
                    key: round(value, 6)
                    for key, value in prepared.preparation_timings.items()
                },
            },
        )

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        """Convenience path; preparation still remains outside elapsed_ms."""
        prepared = self.prepare_page(path, roi=roi, options=options)
        return self.analyze_prepared(page, prepared)


ENGINE: Engine = FastLinearDetector()
