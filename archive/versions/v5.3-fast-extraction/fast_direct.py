"""Selected fast engine: one valid-only ZXing-C++ pass over one page."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import zxingcpp


MATRIX_NAMES = {
    "Aztec",
    "Data Matrix",
    "MaxiCode",
    "Micro QR Code",
    "PDF417",
    "QR Code",
    "rMQR Code",
}


def _point(value: Any) -> np.ndarray:
    return np.asarray([float(value.x), float(value.y)], np.float32)


def barcode_quad(barcode: Any) -> np.ndarray | None:
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

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    start, end = points[first], points[second]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return None
    normal = np.asarray([-direction[1], direction[0]], np.float32) / length
    return np.asarray(
        [start - normal, end - normal, end + normal, start + normal],
        np.float32,
    )


def containment(first: np.ndarray, second: np.ndarray) -> float:
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


def deduplicate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in values:
        if any(
            prior["text"] == candidate["text"]
            and containment(prior["quad"], candidate["quad"]) >= 0.80
            for prior in output
        ):
            continue
        output.append(candidate)
    return output


def direct_decode(
    image: np.ndarray,
    *,
    formats: Any = zxingcpp.BarcodeFormat.AllReadable,
    max_dimension: int | None = None,
    try_downscale: bool = False,
    try_invert: bool = False,
    binarizer: Any = zxingcpp.Binarizer.LocalAverage,
) -> list[dict[str, Any]]:
    """Read clear symbols without proposals, enhancement, retries, or fallback."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 1.0
    if max_dimension and max(gray.shape) > max_dimension:
        scale = max_dimension / max(gray.shape)
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    output: list[dict[str, Any]] = []
    for barcode in zxingcpp.read_barcodes(
        gray,
        formats=formats,
        try_rotate=True,
        try_downscale=try_downscale,
        try_invert=try_invert,
        binarizer=binarizer,
        return_errors=False,
    ):
        if not barcode.valid or not barcode.text:
            continue
        text = str(barcode.text)
        barcode_format = str(barcode.format)
        if barcode_format == "ITF" and len(text) < 6:
            continue
        quad = barcode_quad(barcode)
        if quad is None:
            continue
        quad /= scale
        output.append(
            {
                "quad": quad,
                "text": text,
                "format": barcode_format,
                "kind": "2d" if barcode_format in MATRIX_NAMES else "linear",
                "confidence": 1.0,
                "sources": ["zxing:fast-valid-only-page-pass"],
            }
        )
    return deduplicate(output)
