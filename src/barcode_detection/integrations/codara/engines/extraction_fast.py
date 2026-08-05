"""Low-latency extraction for clear, large, ordinary dark-on-light barcodes.

This engine is intentionally a different product tier from the adaptive 5.1
extractor.  It performs one checksum/error-correction-valid ZXing-C++ page
scan and stops.  It has no proposal detector, error-location promotion,
enhancement portfolio, candidate decoder, restoration model, or fallback to
the expensive engine.  A miss is returned as a miss rather than quietly
spending the work the user opted out of.
"""

from __future__ import annotations

import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import zxingcpp

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
from .extraction_vendored import DEFAULT_KINDS, KINDS


_WARM_LOCK = threading.Lock()
_MATRIX_NAMES = {
    "Aztec",
    "Data Matrix",
    "MaxiCode",
    "Micro QR Code",
    "PDF417",
    "QR Code",
    "rMQR Code",
}


def _formats(options: dict[str, Any]) -> Any:
    if "matrix_formats" not in options and "linear_formats" not in options:
        return zxingcpp.barcode_formats_from_str("AllReadable")
    selected = [
        *list(options.get("matrix_formats") or ()),
        *list(options.get("linear_formats") or ()),
    ]
    return zxingcpp.barcode_formats_from_str(
        ",".join(str(value) for value in selected)
    )


def _point(value: Any) -> np.ndarray:
    return np.asarray([float(value.x), float(value.y)], np.float32)


def _quad(barcode: Any) -> np.ndarray | None:
    position = barcode.position
    points = np.asarray(
        [
            _point(position.top_left),
            _point(position.top_right),
            _point(position.bottom_right),
            _point(position.bottom_left),
        ],
        np.float32,
    )
    if abs(float(cv2.contourArea(points))) > 1e-6:
        return points

    # A valid linear read may expose its scan segment rather than a complete
    # rectangle.  Preserve the measured geometry with a two-pixel support band
    # instead of inventing a barcode height.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    start, end = points[first], points[second]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return None
    normal = np.asarray([-direction[1], direction[0]], np.float32) / length
    return np.asarray(
        [
            start - normal,
            end - normal,
            end + normal,
            start + normal,
        ],
        np.float32,
    )


def _containment(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(np.asarray(first, np.float32))
    second_hull = cv2.convexHull(np.asarray(second, np.float32))
    areas = (
        abs(float(cv2.contourArea(first_hull))),
        abs(float(cv2.contourArea(second_hull))),
    )
    if min(areas) <= 1e-6:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    return float(intersection) / min(areas)


def _deduplicate(regions: list[Region]) -> list[Region]:
    output: list[Region] = []
    for candidate in regions:
        candidate_quad = np.asarray(candidate.quad, np.float32).reshape(4, 2)
        duplicate = next(
            (
                prior
                for prior in output
                if prior.value == candidate.value
                and _containment(
                    np.asarray(prior.quad, np.float32).reshape(4, 2),
                    candidate_quad,
                )
                >= 0.80
            ),
            None,
        )
        if duplicate is None:
            output.append(candidate)
    return output


class FastDirectExtractor:
    """One valid-only ZXing pass over the prepared page raster."""

    info = EngineInfo(
        id="fast-direct-extractor",
        label="Fast extractor",
        capability=Capability.DECODE,
        summary=(
            "One direct decoder pass for clear, large, dark-on-light barcodes. "
            "Skips localization proposals, restoration, and recovery."
        ),
        speed_ms_per_page="~29 ms/page on the clear-large benchmark",
        accuracy_note="Fast tier · use Adaptive extractor for difficult pages",
        badge="Previous fast",
        options={
            "kinds": {
                "type": "enum",
                "values": list(KINDS),
                "default": DEFAULT_KINDS,
                "label": "Symbol types",
                "value_timings": {
                    "all": "~29 ms/page on clear, large pages",
                },
            },
            "formats": {
                "type": "format-list",
                "default": [],
                "label": "Barcode formats",
                "help": (
                    "Declare known formats when possible. The engine searches "
                    "only those families, reducing work and wrong-family reads."
                ),
            },
        },
    )

    def __init__(self) -> None:
        self._warmed = False

    def warm(self) -> None:
        with _WARM_LOCK:
            if self._warmed:
                return
            zxingcpp.barcode_formats_from_str("All")
            self._warmed = True

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        options = options or {}
        started = perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"unable to read page image: {path}")
        page_height, page_width = image.shape[:2]
        target = image
        offset_x = offset_y = 0
        if roi is not None:
            target, offset_x, offset_y = roi.crop(
                image,
                margin=CONTEXT_MARGIN_PX,
            )

        decode_started = perf_counter()
        reads = zxingcpp.read_barcodes(
            target,
            formats=_formats(options),
            try_rotate=True,
            # The fast tier assumes large symbols. Disabling the image pyramid
            # preserved recall on the frozen support-envelope holdout while
            # materially reducing latency.
            try_downscale=False,
            # Ordinary print is dark-on-light. Inverted-symbol recovery belongs
            # to the adaptive tier.
            try_invert=False,
            binarizer=zxingcpp.Binarizer.LocalAverage,
            return_errors=False,
        )
        decode_ms = (perf_counter() - decode_started) * 1000.0

        regions: list[Region] = []
        for barcode in reads:
            if not barcode.valid or not barcode.text:
                continue
            text = str(barcode.text)
            barcode_format = str(barcode.format)
            # Very short ITF has no mandatory checksum and is a known text-stroke
            # false-read class. This is the same conservative rule used by 5.1.
            if barcode_format == "ITF" and len(text) < 6:
                continue
            points = _quad(barcode)
            if points is None:
                continue
            points[:, 0] += offset_x
            points[:, 1] += offset_y
            region = Region(
                quad=tuple(float(value) for value in points.reshape(-1)),  # type: ignore[arg-type]
                kind="2d" if barcode_format in _MATRIX_NAMES else "linear",
                confidence=1.0,
                value=text,
                symbology=barcode_format,
                sources=("zxing:fast-valid-only-page-pass",),
                status=RegionStatus.DECODED,
                extras={"fast_path": True},
            )
            if roi is None or roi.contains_point(
                *region_center(region.quad),
                page_width,
                page_height,
            ):
                regions.append(region)

        return PageOutcome(
            page=page,
            regions=_deduplicate(regions),
            elapsed_ms=(perf_counter() - started) * 1000.0,
            diagnostics={
                "mode": "valid-only-single-pass",
                "decoder_ms": round(decode_ms, 4),
                "try_downscale": False,
                "try_invert": False,
                "recovery_stages": 0,
                "page_size": {"width": page_width, "height": page_height},
                "roi_applied": roi is not None,
            },
        )


ENGINE: Engine = FastDirectExtractor()
