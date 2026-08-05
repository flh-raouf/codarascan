"""Bounded ZXing decoding for native structure-tensor 1-D proposals.

The contract is intentionally proposal-first:

    Tensor localization -> native crop ZXing -> unresolved-only ZXing rescue

There is no whole-page linear ZXing pass, OpenCV linear proposal detector, or
fallback to the Adaptive Extractor's linear branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np
import zxingcpp

from barcode_detection.engines.common.native import sttg_localizer
from barcode_detection.engines.common.recovery import runtime as adaptive
from barcode_detection.engines.common.recovery.hybrid import Result
from barcode_detection.engines.common.recovery.zxing import (
    prepare_variant,
    scanline_consensus_variants,
)


@dataclass(frozen=True)
class TensorDecode:
    text: str
    symbology: str
    route: str
    attempts: tuple[str, ...]


def _acceptable(barcode: Any) -> bool:
    if barcode is None or not barcode.valid or not barcode.text:
        return False
    # ITF has no mandatory checksum. Short text strokes can therefore become a
    # checksum-free valid read; use the same guard as the Adaptive/Fast tiers.
    return str(barcode.format) != "ITF" or len(str(barcode.text)) >= 6


def _read(
    image: np.ndarray,
    formats: Any,
    attempt: str,
    attempts: list[str],
    *,
    try_rotate: bool = True,
    try_invert: bool = True,
) -> TensorDecode | None:
    attempts.append(attempt)
    try:
        barcode = zxingcpp.read_barcode(
            image,
            formats=formats,
            try_rotate=try_rotate,
            try_downscale=False,
            try_invert=try_invert,
            binarizer=zxingcpp.Binarizer.LocalAverage,
            return_errors=False,
        )
    except Exception:  # noqa: BLE001 - one failed crop is an unresolved result
        return None
    if not _acceptable(barcode):
        return None
    return TensorDecode(
        text=str(barcode.text),
        symbology=adaptive.normalize_format(str(barcode.format))
        or str(barcode.format),
        route=attempt,
        attempts=tuple(attempts),
    )


def decode_tensor_candidate_bounded(
    gray: np.ndarray,
    detection: sttg_localizer.Detection,
    formats: Any,
    *,
    max_attempts: int,
    internal_rotate: bool = False,
    internal_invert: bool = False,
) -> tuple[TensorDecode | None, tuple[str, ...]]:
    """Benchmarkable latency-bounded candidate decode.

    Unlike :func:`decode_tensor_candidate`, this policy never enters angle,
    phase, multi-binarizer, or scanline-consensus recovery. Tensor has already
    rectified the candidate, so the lean policy can also disable ZXing's own
    rotation and inversion searches. The production decoder remains unchanged
    unless a caller selects this function explicitly.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    attempts: list[str] = []
    for index, (name, variant) in enumerate(
        _fast_variants(gray, detection),
        start=1,
    ):
        if index > max_attempts:
            break
        bounded_name = f"{name}:bounded-{max_attempts}"
        decoded = _read(
            variant,
            formats,
            bounded_name,
            attempts,
            try_rotate=internal_rotate,
            try_invert=internal_invert,
        )
        if decoded is not None:
            return decoded, decoded.attempts
    return None, tuple(attempts)


def _compact_variants(
    gray: np.ndarray,
    detection: sttg_localizer.Detection,
) -> Iterable[tuple[str, np.ndarray]]:
    """Yield only transforms with unique wins on the regression corpus."""
    useful_fast = {
        "tensor-zxing:native",
        "tensor-zxing:cubic-1.5",
        "tensor-zxing:cubic-2",
        "tensor-zxing:wide-cubic-3",
        "tensor-zxing:wide-clahe-3",
    }
    for name, variant in _fast_variants(gray, detection):
        if name in useful_fast:
            yield name, variant

    for name, variant in _extended_angle_variants(gray, detection):
        if name == "tensor-zxing:angle-6":
            yield name, variant

    # This exact thin-Code-39 correction avoids entering the original nested
    # 84-call recipe search.
    variant, _variant_to_page = prepare_variant(
        gray,
        detection,
        0.0,
        0.05,
        -2.0,
        1.5,
        1.5,
        cv2.INTER_NEAREST,
        "otsu",
        0.0,
    )
    yield "tensor-zxing:compact-angle--2-otsu", variant

    # private corpus A has one legitimate value which only the median scanline
    # reconstruction recovers. Select that one view directly instead of
    # entering all scanline and advanced geometry combinations.
    for name, variant in scanline_consensus_variants(gray, detection):
        if name == "consensus-median-cubic3":
            yield "tensor-zxing:compact-consensus-median-cubic3", variant
            break


def decode_tensor_candidate_compact(
    gray: np.ndarray,
    detection: sttg_localizer.Detection,
    formats: Any,
    *,
    max_attempts: int = 8,
) -> tuple[TensorDecode | None, tuple[str, ...]]:
    """Small evidence-selected portfolio with a strict ZXing-call ceiling."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    attempts: list[str] = []
    for index, (name, variant) in enumerate(
        _compact_variants(gray, detection),
        start=1,
    ):
        if index > max_attempts:
            break
        decoded = _read(
            variant,
            formats,
            f"{name}:compact-{max_attempts}",
            attempts,
            try_rotate=False,
            try_invert=False,
        )
        if decoded is not None:
            return decoded, decoded.attempts
    return None, tuple(attempts)


def _fast_variants(
    gray: np.ndarray,
    detection: sttg_localizer.Detection,
) -> Iterable[tuple[str, np.ndarray]]:
    """Yield the frozen cheap-to-expensive crop sequence lazily."""
    base = adaptive.rectify_quad(
        gray,
        detection.quad,
        pad_long=0.04,
        pad_short=0.12,
    )
    yield "tensor-zxing:native", base
    for scale in (1.5, 2.0, 3.0):
        yield (
            f"tensor-zxing:cubic-{scale:g}",
            cv2.resize(
                base,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            ),
        )

    wide = adaptive.rectify_quad(
        gray,
        detection.quad,
        pad_long=0.25,
        pad_short=0.40,
    )
    yield (
        "tensor-zxing:wide-cubic-3",
        cv2.resize(
            wide,
            None,
            fx=3.0,
            fy=3.0,
            interpolation=cv2.INTER_CUBIC,
        ),
    )
    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(wide)
    yield (
        "tensor-zxing:wide-clahe-3",
        cv2.resize(
            enhanced,
            None,
            fx=3.0,
            fy=3.0,
            interpolation=cv2.INTER_CUBIC,
        ),
    )

    alternate = adaptive.rectify_quad(
        gray,
        detection.quad,
        pad_long=0.08,
        pad_short=0.24,
    )
    yield (
        "tensor-zxing:alternate-cubic-1.5",
        cv2.resize(
            alternate,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC,
        ),
    )


def _extended_angle_variants(
    gray: np.ndarray,
    detection: sttg_localizer.Detection,
) -> Iterable[tuple[str, np.ndarray]]:
    """Recover thin bar fields whose common edge angle exceeds their loose box."""
    patch = adaptive.rectify_quad(
        gray,
        detection.quad,
        pad_long=0.0,
        pad_short=0.0,
    )
    patch = cv2.copyMakeBorder(
        patch,
        30,
        30,
        40,
        40,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    height, width = patch.shape[:2]
    for angle in (-8.0, -6.0, 6.0, 8.0):
        transform = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0),
            angle,
            1.0,
        )
        rotated = cv2.warpAffine(
            patch,
            transform,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        enlarged = cv2.resize(
            rotated,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_NEAREST,
        )
        binary = cv2.threshold(
            enlarged,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        yield f"tensor-zxing:angle-{angle:g}", binary


def decode_tensor_candidate(
    gray: np.ndarray,
    detection: sttg_localizer.Detection,
    formats: Any,
) -> tuple[TensorDecode | None, tuple[str, ...]]:
    """Decode one Tensor proposal without any full-page linear scan."""
    attempts: list[str] = []
    for name, variant in _fast_variants(gray, detection):
        decoded = _read(variant, formats, name, attempts)
        if decoded is not None:
            return decoded, decoded.attempts

    for name, variant in _extended_angle_variants(gray, detection):
        decoded = _read(variant, formats, name, attempts)
        if decoded is not None:
            return decoded, decoded.attempts

    previous = Result(
        quad=np.asarray(detection.quad, np.float32),
        confidence=float(detection.confidence),
        sources={detection.source},
        attempts=list(attempts),
        structural_metrics=dict(detection.metrics),
    )
    advanced = adaptive.decode_advanced_linear(
        gray,
        detection,
        previous,
        formats=formats,
    )
    if advanced.decoded and advanced.text:
        normalized = adaptive.normalize_format(advanced.format) or (
            advanced.format or "Unknown"
        )
        if normalized == "Itf" and len(str(advanced.text)) < 6:
            return None, tuple(advanced.attempts)
        route = next(
            (
                source
                for source in advanced.sources
                if source.startswith("zxing:advanced:")
            ),
            "tensor-zxing:advanced",
        )
        return (
            TensorDecode(
                text=str(advanced.text),
                symbology=normalized,
                route=route,
                attempts=tuple(advanced.attempts),
            ),
            tuple(advanced.attempts),
        )
    return None, tuple(advanced.attempts)
