# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest
import zxingcpp

from codarascan import Scanner
from codarascan.engines.common.native import loader


def _linear_fixture() -> np.ndarray:
    barcode = zxingcpp.create_barcode("CODE128-PARITY", zxingcpp.BarcodeFormat.Code128)
    symbol = np.asarray(
        zxingcpp.write_barcode_to_image(barcode, scale=2, add_quiet_zones=True)
    ).copy()
    height, width = symbol.shape
    canvas = np.full((height + 100, width + 100), 255, np.uint8)
    canvas[50 : 50 + height, 50 : 50 + width] = symbol
    return canvas


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    first_area = abs(float(cv2.contourArea(first)))
    second_area = abs(float(cv2.contourArea(second)))
    intersection, _ = cv2.intersectConvexConvex(first, second)
    return float(intersection) / (first_area + second_area - float(intersection))


@pytest.mark.native
@pytest.mark.integration
def test_native_and_forced_reference_match_documented_parity_tolerances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODARASCAN_FORCE_PYTHON", raising=False)
    loader._reset_for_tests()
    if loader.load_native() is None:
        pytest.skip("compiled Tessera extension is unavailable")
    image = _linear_fixture()
    native = Scanner(mode="fast", symbols="linear", formats=["code-128"]).scan_image(image)

    monkeypatch.setenv("CODARASCAN_FORCE_PYTHON", "1")
    loader._reset_for_tests()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reference = Scanner(mode="fast", symbols="linear", formats=["code-128"]).scan_image(image)

    assert native.metadata.backend == "native"
    assert reference.metadata.backend == "python-reference"
    assert [str(item.message) for item in caught] == [loader.FALLBACK_WARNING]
    assert len(native.symbols) == len(reference.symbols) == 1
    native_symbol, reference_symbol = native.symbols[0], reference.symbols[0]
    assert native_symbol.status == reference_symbol.status
    assert native_symbol.text == reference_symbol.text == "CODE128-PARITY"  # type: ignore[attr-defined]
    assert native_symbol.raw_bytes == reference_symbol.raw_bytes  # type: ignore[attr-defined]
    assert native_symbol.format == reference_symbol.format == "code-128"  # type: ignore[attr-defined]
    native_quad = np.asarray([(point.x, point.y) for point in native_symbol.quad])
    reference_quad = np.asarray([(point.x, point.y) for point in reference_symbol.quad])
    # Tessera's native connected-component boundary and Python reference boundary
    # need not be pixel-identical. They must describe the same physical symbol.
    assert _iou(native_quad, reference_quad) >= 0.75
    np.testing.assert_allclose(
        native_quad.mean(axis=0) / np.asarray([native.width, native.height]),
        reference_quad.mean(axis=0) / np.asarray([reference.width, reference.height]),
        atol=0.05,
    )
