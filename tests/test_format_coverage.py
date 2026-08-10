# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import zxingcpp

from codarascan import DecodedSymbolResult, FormatInfo, Scanner

from .format_fixtures import clean_fixture

FORMAT_INFOS = tuple(
    item
    for group in Scanner.supported_formats().values()
    for item in group
)
MATRIX_INFOS = tuple(item for item in FORMAT_INFOS if item.kind.value == "2d")


def _id(info: FormatInfo) -> str:
    return info.name


def test_catalog_is_the_installed_readable_catalog_without_non_formats() -> None:
    readable = {
        str(item)
        for item in zxingcpp.barcode_formats_list(zxingcpp.BarcodeFormat.AllReadable)
    }
    public = {item.zxing_name for item in FORMAT_INFOS}
    assert public == readable - {"Other barcode"}
    assert "EAN-2" not in public and "EAN-5" not in public
    assert len(public) == len(FORMAT_INFOS)


@pytest.mark.parametrize("info", FORMAT_INFOS, ids=_id)
def test_mosaic_decodes_every_public_format_clean_fixture(info: FormatInfo) -> None:
    image, expected_raw = clean_fixture(info)
    result = Scanner(
        mode="robust",
        symbols="2d" if info.kind.value == "2d" else "linear",
        formats=[info.name],
    ).scan_image(image)
    decoded = [item for item in result.symbols if isinstance(item, DecodedSymbolResult)]
    assert any(item.raw_bytes == expected_raw for item in decoded)


@pytest.mark.parametrize("info", MATRIX_INFOS, ids=_id)
def test_tessera_routes_every_public_matrix_format_clean_fixture(info: FormatInfo) -> None:
    image, expected_raw = clean_fixture(info)
    result = Scanner(mode="fast", symbols="2d", formats=[info.name]).scan_image(image)
    decoded = [item for item in result.symbols if isinstance(item, DecodedSymbolResult)]
    assert any(item.raw_bytes == expected_raw for item in decoded)
