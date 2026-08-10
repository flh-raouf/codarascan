# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from codarascan.errors import InvalidImageError
from codarascan.inputs import normalize_image


def test_numpy_bgr_bgra_grayscale_and_noncontiguous_are_owned() -> None:
    for value in (
        np.zeros((12, 13), dtype=np.uint8),
        np.zeros((12, 13, 3), dtype=np.uint8),
        np.zeros((12, 13, 4), dtype=np.uint8),
        np.zeros((24, 26, 3), dtype=np.uint8)[::2, ::2],
    ):
        normalized = normalize_image(value)
        assert normalized.pixels.flags.c_contiguous
        assert normalized.pixels.flags.owndata
        assert (normalized.width, normalized.height) == (13, 12)


def test_numpy_snapshot_is_isolated_from_caller_mutation() -> None:
    source = np.arange(6 * 7, dtype=np.uint8).reshape(6, 7)
    expected = source.copy()
    normalized = normalize_image(source)
    source[:] = 0
    np.testing.assert_array_equal(normalized.pixels, expected)


@pytest.mark.parametrize(
    "value",
    [
        np.empty((0, 2), dtype=np.uint8),
        np.zeros((2, 2, 2), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=object),
    ],
)
def test_numpy_rejects_unsupported_arrays(value: np.ndarray) -> None:
    with pytest.raises(InvalidImageError):
        normalize_image(value)


def test_encoded_png_and_pillow_have_equivalent_pixels() -> None:
    image = Image.new("RGB", (7, 5), (10, 20, 30))
    stream = BytesIO()
    image.save(stream, "PNG")
    from_pillow = normalize_image(image)
    from_bytes = normalize_image(stream.getvalue())
    np.testing.assert_array_equal(from_pillow.pixels, from_bytes.pixels)


@pytest.mark.parametrize("wrapper", [bytes, bytearray, memoryview])
def test_every_documented_encoded_byte_container(wrapper: type) -> None:
    stream = BytesIO()
    Image.new("RGBA", (5, 4), (10, 20, 30, 128)).save(stream, "PNG")
    normalized = normalize_image(wrapper(stream.getvalue()))
    assert normalized.pixels.shape == (4, 5, 3)


@pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA", "CMYK", "I", "F"])
def test_documented_pillow_modes_normalize_deterministically(mode: str) -> None:
    image = Image.new(mode, (6, 5))
    normalized = normalize_image(image)
    assert normalized.pixels.dtype == np.uint8
    assert normalized.pixels.flags.c_contiguous
    assert normalized.pixels.shape[:2] == (5, 6)


def test_exif_orientation_is_applied_to_encoded_input() -> None:
    image = Image.new("RGB", (8, 5), "white")
    exif = Image.Exif()
    exif[274] = 6
    stream = BytesIO()
    image.save(stream, "JPEG", exif=exif)
    normalized = normalize_image(stream.getvalue())
    assert (normalized.width, normalized.height) == (5, 8)


@pytest.mark.parametrize(
    ("orientation", "operation"),
    [
        (1, None),
        (2, Image.Transpose.FLIP_LEFT_RIGHT),
        (3, Image.Transpose.ROTATE_180),
        (4, Image.Transpose.FLIP_TOP_BOTTOM),
        (5, Image.Transpose.TRANSPOSE),
        (6, Image.Transpose.ROTATE_270),
        (7, Image.Transpose.TRANSVERSE),
        (8, Image.Transpose.ROTATE_90),
    ],
)
def test_all_exif_orientation_values(orientation: int, operation: Image.Transpose | None) -> None:
    pixels = np.asarray(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels, "RGB")
    image.getexif()[274] = orientation
    expected_image = image.copy() if operation is None else image.transpose(operation)
    expected = cv2.cvtColor(np.asarray(expected_image), cv2.COLOR_RGB2BGR)
    np.testing.assert_array_equal(normalize_image(image).pixels, expected)


def test_corrupt_and_empty_bytes_are_rejected() -> None:
    for value in (b"", b"not an image"):
        with pytest.raises(InvalidImageError):
            normalize_image(value)


def test_invalid_paths_directories_and_types_are_typed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(InvalidImageError, match="does not exist"):
        normalize_image(missing)
    with pytest.raises(InvalidImageError, match="not a file"):
        normalize_image(tmp_path)
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError, match="corrupted"):
        normalize_image(invalid)
    with pytest.raises(InvalidImageError, match="unsupported image input type"):
        normalize_image(object())  # type: ignore[arg-type]
