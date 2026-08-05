"""Tensor 1-D localization combined with the latest classical 2-D path.

The replacement is structural, not a post-processing filter:

    1-D: native structure tensor
    2-D: validated classical QR / Data Matrix geometry

One grayscale load and ROI crop are shared by both prepared branches.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from barcode_detection.engines.common.native import sttg_localizer

from barcode_detection.core.contracts import (
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
from barcode_detection.engines.common.classical_2d_localizer import (
    ENGINE as CLASSICAL_2D_DETECTOR,
)
from barcode_detection.engines.common.tensor_localizer import (
    ENGINE as TENSOR_ENGINE,
)
from barcode_detection.engines.common.tensor_localizer import PreparedTensorPage, _quad_to_tuple
from barcode_detection.engines.common.classical_2d_extractor import (
    PreparedClassical2DPage,
    prepare_loaded_gray,
)


KINDS = ("all", "linear", "2d")


@dataclass(frozen=True)
class PreparedTesseraLocalizerPage:
    """One shared page with independently prepared 1-D and 2-D branches."""

    tensor: PreparedTensorPage | None
    matrix: PreparedClassical2DPage | None
    kinds: str
    stage_parallel: bool
    preparation_timings: dict[str, float]


class TesseraLocalizer:
    """Tessera: the recommended fast 1-D plus validated 2-D localizer."""

    info = EngineInfo(
        id="tessera-localizer",
        label="Tessera Localizer",
        capability=Capability.DETECT,
        summary=(
            "Tessera combines the latest native Tensor 1-D localizer with the latest "
            "validated classical QR/Data Matrix geometry route. It shares one "
            "prepared page and returns geometry without decoding payloads."
        ),
        speed_ms_per_page="~24 ms/page native; Python fallback available",
        accuracy_note=(
            "Fast mixed tier for clear, ordinary documents; use Mosaic for "
            "small, noisy, damaged, or highly transformed symbols."
        ),
        badge="Recommended",
        available=TENSOR_ENGINE.info.available,
        unavailable_reason=TENSOR_ENGINE.info.unavailable_reason,
        options={
            "kinds": {
                "type": "enum",
                "values": list(KINDS),
                "default": "all",
                "label": "Symbol types",
                "value_timings": {
                    "all": "~24 ms/page",
                    "linear": "~6–9 ms/page",
                    "2d": "~9–18 ms/page",
                },
            },
            # Both profiles remain available to benchmark/direct callers. The
            # product frontend intentionally pins this engine to robust mode.
            "mode": {
                "type": "enum",
                "values": ["robust", "fast"],
                "default": "robust",
                "label": "2D localization mode",
                "value_timings": {
                    "robust": (
                        "~9–18 ms/page typical; hard recovery pages are higher"
                    ),
                    "fast": "~8–12 ms/page typical engine compute",
                },
            },
        },
    )

    def warm(self) -> None:
        TENSOR_ENGINE.warm()
        CLASSICAL_2D_DETECTOR.warm()

    @staticmethod
    def _kinds(options: dict[str, Any] | None) -> str:
        kinds = str((options or {}).get("kinds", "all")).strip().lower()
        if kinds not in KINDS:
            raise ValueError(f"kinds must be one of {KINDS}, got {kinds!r}")
        return kinds

    def prepare_page(
        self,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedTesseraLocalizerPage:
        options = dict(options or {})
        kinds = self._kinds(options)
        preparation_started = perf_counter()
        load_started = perf_counter()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        load_seconds = perf_counter() - load_started
        if gray is None:
            raise RuntimeError(f"unable to read page image: {path}")
        full_height, full_width = gray.shape[:2]
        target = gray
        offset_x = offset_y = 0
        if roi is not None:
            target, offset_x, offset_y = roi.crop(
                gray,
                margin=CONTEXT_MARGIN_PX,
            )

        tensor_started = perf_counter()
        tensor_native = (
            sttg_localizer.prepare_page(
                target,
                sttg_localizer.profile_config("document"),
            )
            if kinds in {"all", "linear"}
            else None
        )
        tensor_seconds = perf_counter() - tensor_started
        tensor = (
            PreparedTensorPage(
                prepared=tensor_native,
                full_width=full_width,
                full_height=full_height,
                offset_x=offset_x,
                offset_y=offset_y,
                roi=roi,
                preparation_timings={
                    "image_load_seconds": load_seconds,
                    "tensor_preparation_seconds": tensor_seconds,
                    "total_seconds": load_seconds + tensor_seconds,
                },
            )
            if tensor_native is not None
            else None
        )

        stage_parallel = bool(options.get("stage_parallel"))
        matrix_options = dict(options)
        matrix_options["stage_parallel"] = bool(
            stage_parallel and kinds == "2d"
        )
        matrix = (
            prepare_loaded_gray(
                target,
                options=matrix_options,
                offset_x=offset_x,
                offset_y=offset_y,
                page_width=full_width,
                page_height=full_height,
                roi=roi,
                load_seconds=load_seconds,
            )
            if kinds in {"all", "2d"}
            else None
        )
        return PreparedTesseraLocalizerPage(
            tensor=tensor,
            matrix=matrix,
            kinds=kinds,
            stage_parallel=stage_parallel,
            preparation_timings={
                "image_load_seconds": load_seconds,
                "tensor_preparation_seconds": (
                    tensor_seconds if tensor is not None else 0.0
                ),
                "matrix_preparation_seconds": (
                    0.0 if matrix is None else matrix.preparation_seconds
                ),
                "total_seconds": perf_counter() - preparation_started,
            },
        )

    @staticmethod
    def _tensor_regions(
        prepared: PreparedTensorPage,
        result: sttg_localizer.LocalizationResult,
    ) -> list[Region]:
        regions = [
            Region(
                quad=_quad_to_tuple(
                    detection.quad,
                    prepared.offset_x,
                    prepared.offset_y,
                ),
                kind="linear",
                confidence=float(detection.confidence),
                sources=(detection.source,),
                status=RegionStatus.LOCALIZED,
                extras={
                    "evidence": {
                        key: round(float(value), 6)
                        for key, value in detection.metrics.items()
                    }
                },
            )
            for detection in result.detections
        ]
        if prepared.roi is None:
            return regions
        return [
            region
            for region in regions
            if prepared.roi.contains_point(
                *region_center(region.quad),
                prepared.full_width,
                prepared.full_height,
            )
        ]

    def analyze_prepared(
        self,
        page: int,
        prepared: PreparedTesseraLocalizerPage,
    ) -> PageOutcome:
        if not isinstance(prepared, PreparedTesseraLocalizerPage):
            raise TypeError(
                "Tessera requires "
                "PreparedTesseraLocalizerPage input"
            )
        if not self.info.available:
            raise RuntimeError(
                self.info.unavailable_reason
                or "native tensor extension is unavailable"
            )

        started = perf_counter()

        def tensor_branch() -> tuple[list[Region], Any | None]:
            if prepared.tensor is None:
                return [], None
            result = sttg_localizer.locate_prepared(
                prepared.tensor.prepared,
                sttg_localizer.profile_config("document"),
            )
            return self._tensor_regions(prepared.tensor, result), result

        def matrix_branch() -> PageOutcome | None:
            if prepared.matrix is None:
                return None
            return CLASSICAL_2D_DETECTOR.analyze_prepared(
                page,
                prepared.matrix,
            )

        if (
            prepared.stage_parallel
            and prepared.tensor is not None
            and prepared.matrix is not None
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                tensor_future = pool.submit(tensor_branch)
                matrix_future = pool.submit(matrix_branch)
                tensor_regions, tensor_result = tensor_future.result()
                matrix_outcome = matrix_future.result()
        else:
            tensor_regions, tensor_result = tensor_branch()
            matrix_outcome = matrix_branch()

        regions = [
            *tensor_regions,
            *(
                matrix_outcome.regions
                if matrix_outcome is not None
                else ()
            ),
        ]
        page_width = (
            prepared.tensor.full_width
            if prepared.tensor is not None
            else prepared.matrix.page_width  # type: ignore[union-attr]
        )
        page_height = (
            prepared.tensor.full_height
            if prepared.tensor is not None
            else prepared.matrix.page_height  # type: ignore[union-attr]
        )

        diagnostics = {
            "mode": "tensor-linear-plus-classical-2d",
            "kinds": prepared.kinds,
            "pipeline7_invoked": False,
            "tensor": (
                None
                if tensor_result is None
                else {
                    **tensor_result.diagnostics,
                    "timings": {
                        key: round(float(value), 6)
                        for key, value in tensor_result.timings.items()
                    },
                }
            ),
            "classical_2d": (
                None
                if matrix_outcome is None
                else matrix_outcome.diagnostics
            ),
            "matrix_tensor_parallel": bool(
                prepared.stage_parallel
                and prepared.tensor is not None
                and prepared.matrix is not None
            ),
            "measurement_scope": "engine_compute_after_preprocessing",
            "preparation_timings": {
                key: round(float(value), 6)
                for key, value in prepared.preparation_timings.items()
            },
            "page_size": {
                "width": page_width,
                "height": page_height,
            },
        }
        active_roi = (
            prepared.tensor.roi
            if prepared.tensor is not None
            else prepared.matrix.roi  # type: ignore[union-attr]
        )
        if active_roi is not None:
            diagnostics.update(
                {
                    "roi_applied": True,
                    "roi_pixels": active_roi.to_pixels(
                        page_width,
                        page_height,
                    ),
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


ENGINE: Engine = TesseraLocalizer()
