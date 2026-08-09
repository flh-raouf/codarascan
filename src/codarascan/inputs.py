# SPDX-License-Identifier: Apache-2.0
"""Image adapters and EXIF-aware normalization."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import InvalidImageError

ImageInput: TypeAlias = (
    str | Path | bytes | bytearray | memoryview | Image.Image | NDArray[Any]
)
UInt8Image: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    pixels: UInt8Image
    width: int
    height: int
    source: str


def _pillow_pixels(image: Image.Image) -> UInt8Image:
    oriented = ImageOps.exif_transpose(image)
    if oriented.mode in {"1", "L", "I", "I;16", "F"}:
        array = np.asarray(oriented.convert("L"), dtype=np.uint8)
        return np.ascontiguousarray(array)
    rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _from_pillow(image: Image.Image, source: str) -> NormalizedImage:
    try:
        image.load()
        pixels = _pillow_pixels(image)
    except (OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            f"unable to decode {source}", context={"source": source}
        ) from exc
    height, width = pixels.shape[:2]
    if width <= 0 or height <= 0:
        raise InvalidImageError("decoded image is empty", context={"source": source})
    return NormalizedImage(pixels=pixels, width=width, height=height, source=source)


def _from_array(value: NDArray[Any]) -> NormalizedImage:
    if value.dtype != np.uint8:
        raise InvalidImageError(
            f"NumPy image dtype must be uint8, got {value.dtype}",
            context={"dtype": str(value.dtype)},
        )
    if value.ndim == 2:
        pass
    elif value.ndim == 3 and value.shape[2] in {3, 4}:
        pass
    else:
        raise InvalidImageError(
            "NumPy images must be 2-D grayscale, 3-channel BGR, or 4-channel BGRA",
            context={"shape": list(value.shape)},
        )
    if value.size == 0 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise InvalidImageError("NumPy image is empty", context={"shape": list(value.shape)})
    # Always own a contiguous snapshot so caller mutation cannot race a scan.
    pixels = np.array(value, dtype=np.uint8, order="C", copy=True)
    return NormalizedImage(
        pixels=pixels,
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        source="numpy",
    )


def normalize_image(value: ImageInput) -> NormalizedImage:
    """Normalize a supported input into an owned OpenCV-compatible array."""

    if isinstance(value, np.ndarray):
        return _from_array(value)
    if isinstance(value, Image.Image):
        return _from_pillow(value, "Pillow image")
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.exists():
            raise InvalidImageError(
                f"image path does not exist: {path}", context={"path": str(path)}
            )
        if not path.is_file():
            raise InvalidImageError(
                f"image path is not a file: {path}", context={"path": str(path)}
            )
        try:
            with Image.open(path) as image:
                return _from_pillow(image, str(path))
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(
                f"unsupported or corrupted image: {path}", context={"path": str(path)}
            ) from exc
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if not data:
            raise InvalidImageError("encoded image bytes are empty")
        try:
            with Image.open(BytesIO(data)) as image:
                return _from_pillow(image, "encoded image bytes")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("unsupported or corrupted encoded image bytes") from exc
    raise InvalidImageError(
        f"unsupported image input type: {type(value).__name__}",
        context={"type": type(value).__name__},
    )


__all__ = ["ImageInput", "NormalizedImage", "normalize_image"]
