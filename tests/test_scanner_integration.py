# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import zxingcpp
from PIL import Image

from codarascan import DecodedSymbolResult, Scanner, SymbolResult, to_json

PAYLOAD = "CODARASCAN-CONTRACT"


def _barcode_array(payload: str | bytes = PAYLOAD) -> np.ndarray:
    barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
    return np.asarray(
        zxingcpp.write_barcode_to_image(barcode, scale=12, add_quiet_zones=True)
    ).copy()


@pytest.mark.parametrize("mode", ["fast", "robust"])
def test_both_public_engines_satisfy_decoding_and_localization_contract(mode: str) -> None:
    image = _barcode_array()
    decoded = Scanner(mode=mode, symbols="2d", formats=["qr-code"]).scan_image(image)
    located = Scanner(
        mode=mode, symbols="2d", formats=["qr-code"], decode=False
    ).scan_image(image)

    assert len(decoded.symbols) == len(located.symbols) == 1
    assert isinstance(decoded.symbols[0], DecodedSymbolResult)
    assert decoded.symbols[0].text == PAYLOAD
    assert decoded.symbols[0].format == "qr-code"
    assert type(located.symbols[0]) is SymbolResult
    assert located.symbols[0].status.value == "localized"
    assert not hasattr(located.symbols[0], "text")


def test_all_documented_image_adapters_produce_equivalent_decode(tmp_path: Path) -> None:
    gray = _barcode_array()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgra = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    pillow = Image.fromarray(gray)
    stream = BytesIO()
    pillow.save(stream, "PNG")
    path = tmp_path / "barcode.png"
    pillow.save(path)
    read_only = gray.copy()
    read_only.flags.writeable = False
    non_contiguous = np.repeat(gray, 2, axis=0)[::2]

    scanner = Scanner(symbols="2d", formats=["qr-code"])
    results = [
        scanner.scan_image(value)
        for value in (
            path,
            str(path),
            stream.getvalue(),
            pillow,
            gray,
            bgr,
            bgra,
            read_only,
            non_contiguous,
        )
    ]
    assert all(result.symbols[0].text == PAYLOAD for result in results)  # type: ignore[attr-defined]
    assert {(result.width, result.height) for result in results} == {
        (gray.shape[1], gray.shape[0])
    }
    reference = np.asarray(
        [[point.x, point.y] for point in results[0].symbols[0].quad]
    )
    for result in results[1:]:
        actual = np.asarray([[point.x, point.y] for point in result.symbols[0].quad])
        np.testing.assert_allclose(actual, reference, atol=1.0)


def test_roi_geometry_is_returned_in_full_image_coordinates() -> None:
    barcode = _barcode_array()
    canvas = np.full((barcode.shape[0] + 80, barcode.shape[1] * 2 + 160), 255, np.uint8)
    top = 40
    left = canvas.shape[1] - barcode.shape[1] - 40
    canvas[top : top + barcode.shape[0], left : left + barcode.shape[1]] = barcode

    scanner = Scanner(symbols="2d", formats=["qr-code"])
    result = scanner.scan_image(canvas, roi=(0.5, 0.0, 0.5, 1.0))
    assert len(result.symbols) == 1
    symbol = result.symbols[0]
    assert min(point.x for point in symbol.quad) >= left
    for point, normalized in zip(symbol.quad, symbol.normalized_quad, strict=True):
        assert normalized.x == pytest.approx(point.x / result.width)
        assert normalized.y == pytest.approx(point.y / result.height)
        assert 0 <= normalized.x <= 1
        assert 0 <= normalized.y <= 1


def test_exact_format_filter_excludes_other_decoded_formats() -> None:
    result = Scanner(symbols="2d", formats=["data-matrix"]).scan_image(_barcode_array())
    assert result.symbols == ()


def test_binary_payload_round_trips_and_strict_json_has_no_nonfinite_values() -> None:
    raw = b"\x00\xffCODARA\nSCAN"
    result = Scanner(symbols="2d", formats=["qr-code"]).scan_image(
        _barcode_array(raw)
    )
    symbol = result.symbols[0]
    assert isinstance(symbol, DecodedSymbolResult)
    assert symbol.raw_bytes == raw
    serialized = to_json(result)
    assert "NaN" not in serialized and "Infinity" not in serialized


def test_diagnostics_are_opt_in_and_compact_by_default() -> None:
    scanner = Scanner(symbols="2d", formats=["qr-code"])
    compact = scanner.scan_image(_barcode_array())
    detailed = scanner.scan_image(_barcode_array(), diagnostics=True)
    assert "diagnostics" not in compact.to_dict()
    assert "diagnostics" in detailed.to_dict(include_diagnostics=True)
    assert detailed.diagnostics


def test_reusable_scanner_is_thread_safe_and_results_are_deterministic() -> None:
    scanner = Scanner(symbols="2d", formats=["qr-code"])
    image = _barcode_array()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: scanner.scan_image(image), range(8)))
    contracts = [
        [
            (
                symbol.status.value,
                getattr(symbol, "text", None),
                tuple((point.x, point.y) for point in symbol.quad),
            )
            for symbol in result.symbols
        ]
        for result in results
    ]
    assert all(contract == contracts[0] for contract in contracts)


def test_warm_is_idempotent_and_first_use_remains_lazy() -> None:
    scanner = Scanner(symbols="2d", formats=["qr-code"])
    assert scanner._engine is None
    scanner.warm()
    initialized = scanner._engine
    scanner.warm()
    assert scanner._engine is initialized
