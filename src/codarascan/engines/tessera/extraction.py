# SPDX-License-Identifier: Apache-2.0
"""Latest classical 2-D extraction plus Tensor-first 1-D ZXing decoding.

This engine composes the two current fast specialists:

    1-D: Tensor proposals -> candidate ZXing -> unresolved-only ZXing rescue
    2-D: bounded classical QR/Data Matrix cascade

One grayscale load and ROI crop are shared by both prepared branches.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from codarascan.core.contracts import (
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
from codarascan.engines.common.classical_2d_extractor import (
    ENGINE as CLASSICAL_2D_ENGINE,
)
from codarascan.engines.common.classical_2d_extractor import (
    PreparedClassical2DPage,
    prepare_loaded_gray,
)
from codarascan.engines.common.native import sttg_localizer
from codarascan.engines.common.tensor_decoder import (
    decode_tensor_candidate,
    decode_tensor_candidate_compact,
)
from codarascan.engines.common.tensor_localizer import (
    ENGINE as TENSOR_ENGINE,
)
from codarascan.engines.common.tensor_localizer import PreparedTensorPage, _quad_to_tuple
from codarascan.engines.common.vendored_decoder import (
    DEFAULT_KINDS,
    KINDS,
    normalize_engine_options,
)


@dataclass(frozen=True)
class PreparedTesseraExtractorPage:
    roi: Roi | None
    options: dict[str, Any]
    tensor: PreparedTensorPage | None
    matrix: PreparedClassical2DPage | None
    preparation_timings: dict[str, float]


class TesseraExtractor:
    """Tessera: the recommended fast 1-D plus classical 2-D extraction composition."""

    info = EngineInfo(
        id="tessera-extractor",
        label="Tessera Extractor",
        capability=Capability.DECODE,
        summary=(
            "Tessera combines the latest Tensor 1-D localizer and candidate decoder "
            "with the latest classical QR/Data Matrix cascade. It shares one "
            "prepared page and never runs OpenCV's linear detector."
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
                "default": DEFAULT_KINDS,
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
                "label": "2D processing mode",
                "value_timings": {
                    "robust": (
                        "~9–18 ms/page typical; hard recovery pages are higher"
                    ),
                    "fast": "~8–12 ms/page typical engine compute",
                },
            },
            "formats": {
                "type": "format-list",
                "default": [],
                "label": "Barcode formats",
                "help": (
                    "Restrict formats when known. Matrix formats use the latest "
                    "classical cascade; linear formats use Tensor proposals."
                ),
            },
        },
    )

    def warm(self) -> None:
        CLASSICAL_2D_ENGINE.warm()
        TENSOR_ENGINE.warm()

    def prepare_page(
        self,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedTesseraExtractorPage:
        options = normalize_engine_options(options)
        preparation_started = perf_counter()
        load_started = perf_counter()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        load_seconds = perf_counter() - load_started
        if gray is None:
            raise RuntimeError(f"unable to read page image: {path}")
        page_height, page_width = gray.shape[:2]
        target = gray
        offset_x = offset_y = 0
        if roi is not None:
            target, offset_x, offset_y = roi.crop(
                gray,
                margin=CONTEXT_MARGIN_PX,
            )

        linear_formats = options.get("linear_formats")
        matrix_formats = options.get("matrix_formats")
        tensor_started = perf_counter()
        tensor_prepared = (
            sttg_localizer.prepare_page(
                target,
                sttg_localizer.profile_config("document"),
            )
            if linear_formats
            else None
        )
        tensor_seconds = perf_counter() - tensor_started
        tensor = (
            PreparedTensorPage(
                prepared=tensor_prepared,
                full_width=page_width,
                full_height=page_height,
                offset_x=offset_x,
                offset_y=offset_y,
                roi=roi,
                preparation_timings={
                    "image_load_seconds": load_seconds,
                    "tensor_preparation_seconds": tensor_seconds,
                    "total_seconds": load_seconds + tensor_seconds,
                },
            )
            if tensor_prepared is not None
            else None
        )

        matrix_options = dict(options)
        # When both branches run on one isolated page, the outer executor
        # overlaps them. Keep the matrix scans single-threaded there to avoid
        # nested pools. A 2-D-only isolated page may use its two scan workers.
        matrix_options["stage_parallel"] = bool(
            options.get("stage_parallel") and not linear_formats
        )
        matrix = (
            prepare_loaded_gray(
                target,
                options=matrix_options,
                offset_x=offset_x,
                offset_y=offset_y,
                page_width=page_width,
                page_height=page_height,
                roi=roi,
                load_seconds=load_seconds,
            )
            if matrix_formats
            else None
        )
        return PreparedTesseraExtractorPage(
            roi=roi,
            options=options,
            tensor=tensor,
            matrix=matrix,
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

    def _matrix_outcome(
        self,
        page: int,
        prepared: PreparedTesseraExtractorPage,
    ) -> PageOutcome | None:
        if prepared.matrix is None:
            return None
        return CLASSICAL_2D_ENGINE.analyze_prepared(
            page,
            prepared.matrix,
        )

    @staticmethod
    def _linear_outcome(
        page: int,
        prepared: PreparedTesseraExtractorPage,
    ) -> PageOutcome | None:
        tensor = prepared.tensor
        linear_formats = prepared.options.get("linear_formats")
        if tensor is None or not linear_formats:
            return None

        started = perf_counter()
        localized = sttg_localizer.locate_prepared(
            tensor.prepared,
            sttg_localizer.profile_config("document"),
        )
        regions: list[Region] = []
        decoded_count = 0
        unresolved_count = 0
        decode_started = perf_counter()
        decode_policy = str(
            prepared.options.get("tensor_decode_policy", "exhaustive")
        )
        if decode_policy not in {"exhaustive", "compact-8"}:
            raise ValueError(
                "tensor_decode_policy must be exhaustive or compact-8"
            )
        for detection in localized.detections:
            if decode_policy == "compact-8":
                decoded, attempts = decode_tensor_candidate_compact(
                    tensor.prepared.native_gray,
                    detection,
                    linear_formats,
                    max_attempts=8,
                )
            else:
                decoded, attempts = decode_tensor_candidate(
                    tensor.prepared.native_gray,
                    detection,
                    linear_formats,
                )
            quad = _quad_to_tuple(
                detection.quad,
                tensor.offset_x,
                tensor.offset_y,
            )
            evidence = {
                key: round(float(value), 6)
                for key, value in detection.metrics.items()
            }
            if decoded is None:
                unresolved_count += 1
                region = Region(
                    quad=quad,
                    kind="linear",
                    confidence=float(detection.confidence),
                    sources=(detection.source, "tensor-zxing:unresolved"),
                    status=RegionStatus.REVIEW_CANDIDATE,
                    extras={
                        "attempts": list(attempts),
                        "evidence": evidence,
                    },
                )
            else:
                decoded_count += 1
                region = Region(
                    quad=quad,
                    kind="linear",
                    confidence=1.0,
                    value=decoded.text,
                    raw_bytes=decoded.raw_bytes,
                    symbology=decoded.symbology,
                    sources=(detection.source, decoded.route),
                    status=RegionStatus.DECODED,
                    extras={
                        "attempts": list(decoded.attempts),
                        "evidence": evidence,
                        "tensor_decode_route": decoded.route,
                    },
                )
            if tensor.roi is None or tensor.roi.contains_point(
                *region_center(region.quad),
                tensor.full_width,
                tensor.full_height,
            ):
                regions.append(region)

        decode_seconds = perf_counter() - decode_started
        return PageOutcome(
            page=page,
            regions=regions,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            diagnostics={
                **localized.diagnostics,
                "mode": "tensor-proposal-zxing",
                "whole_page_linear_zxing": False,
                "opencv_linear_proposals": False,
                "tensor_candidates": len(localized.detections),
                "tensor_decoded": decoded_count,
                "tensor_unresolved": unresolved_count,
                "tensor_decode_seconds": round(decode_seconds, 6),
                "tensor_decode_policy": decode_policy,
            },
        )

    def analyze_prepared(
        self,
        page: int,
        prepared: PreparedTesseraExtractorPage,
    ) -> PageOutcome:
        if not isinstance(prepared, PreparedTesseraExtractorPage):
            raise TypeError(
                "Tessera requires "
                "PreparedTesseraExtractorPage input"
            )
        started = perf_counter()
        run_matrix = bool(prepared.options.get("matrix_formats"))
        run_linear = bool(prepared.options.get("linear_formats"))
        stage_parallel = bool(
            prepared.options.get("stage_parallel")
            and run_matrix
            and run_linear
        )
        if stage_parallel:
            # The route enables this only for an isolated page. In batch mode,
            # page-level workers already saturate the machine and these nested
            # workers would oversubscribe it.
            with ThreadPoolExecutor(max_workers=2) as pool:
                matrix_future = pool.submit(
                    self._matrix_outcome,
                    page,
                    prepared,
                )
                linear_future = pool.submit(
                    self._linear_outcome,
                    page,
                    prepared,
                )
                matrix = matrix_future.result()
                linear = linear_future.result()
        else:
            matrix = self._matrix_outcome(page, prepared)
            linear = self._linear_outcome(page, prepared)
        regions = [
            *(matrix.regions if matrix is not None else ()),
            *(linear.regions if linear is not None else ()),
        ]
        diagnostics = {
            "mode": "classical-matrix-plus-tensor-linear",
            "matrix": None if matrix is None else matrix.diagnostics,
            "linear": None if linear is None else linear.diagnostics,
            "measurement_scope": "engine_compute_after_tensor_preparation",
            "matrix_linear_parallel": stage_parallel,
            "preparation_timings": {
                key: round(float(value), 6)
                for key, value in prepared.preparation_timings.items()
            },
        }
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


ENGINE: Engine = TesseraExtractor()
