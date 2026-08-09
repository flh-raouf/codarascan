# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from codarascan import (
    ROI,
    ConfigurationError,
    DecodedSymbolResult,
    DocumentResult,
    ErrorDetail,
    ImageResult,
    InternalProcessingError,
    Point,
    ScanMetadata,
    Scanner,
    SymbolKind,
    SymbolResult,
    SymbolStatus,
)
from codarascan import scanner as scanner_module
from codarascan.core.contracts import Region, RegionStatus


def _quad() -> tuple[Point, Point, Point, Point]:
    return (Point(1, 2), Point(3, 2), Point(3, 4), Point(1, 4))


def _normalized_quad() -> tuple[Point, Point, Point, Point]:
    return (Point(0.1, 0.2), Point(0.3, 0.2), Point(0.3, 0.4), Point(0.1, 0.4))


def test_scanner_configuration_is_normalized_hashable_and_immutable() -> None:
    first = Scanner(mode="FAST", symbols="all", formats=["QRCode", "qr-code"])
    second = Scanner(mode="fast", symbols="all", formats=("qr-code",))
    assert first == second
    assert hash(first) == hash(second)
    assert first.formats == ("qr-code",)
    with pytest.raises(FrozenInstanceError):
        first.mode = "robust"  # type: ignore[misc]


def test_every_public_scanner_combination_and_configuration_snapshot() -> None:
    for mode, symbols, decode in product(
        ("fast", "robust"), ("linear", "2d", "all"), (False, True)
    ):
        scanner = Scanner(mode=mode, symbols=symbols, decode=decode)
        assert scanner.mode.value == mode
        assert scanner.symbols.value == symbols
        assert scanner.decode is decode

    mutable = ["QRCode"]
    scanner = Scanner(symbols="2d", formats=mutable)
    mutable[0] = "data-matrix"
    assert scanner.formats == ("qr-code",)


def test_convenience_cache_reuses_only_equal_immutable_configuration() -> None:
    scanner_module._cached_scanner.cache_clear()
    first = scanner_module._cached_scanner(
        scanner_module.ScanMode.FAST,
        scanner_module.SymbolGroup.TWO_D,
        ("qr-code",),
        True,
    )
    second = scanner_module._cached_scanner(
        scanner_module.ScanMode.FAST,
        scanner_module.SymbolGroup.TWO_D,
        ("qr-code",),
        True,
    )
    different = scanner_module._cached_scanner(
        scanner_module.ScanMode.ROBUST,
        scanner_module.SymbolGroup.TWO_D,
        ("qr-code",),
        True,
    )
    assert first is second
    assert different is not first
    assert scanner_module._cached_scanner.cache_info().hits == 1


def test_tessera_public_route_always_uses_stronger_2d_profile() -> None:
    assert Scanner(mode="fast", symbols="2d")._engine_options()["mode"] == "robust"


@pytest.mark.parametrize("mode", ["auto", "", 1, True])
def test_scanner_rejects_invalid_mode(mode: object) -> None:
    with pytest.raises(ConfigurationError):
        Scanner(mode=mode)  # type: ignore[arg-type]


def test_scanner_rejects_contradictory_format_category() -> None:
    with pytest.raises(ConfigurationError, match="contradict"):
        Scanner(symbols="linear", formats=["qr-code"])


def test_supported_formats_are_grouped_without_stability_fields() -> None:
    groups = Scanner.supported_formats()
    assert set(groups) == {"1d", "2d"}
    assert {"qr-code", "data-matrix", "aztec", "pdf417", "maxicode"} <= {
        item.name for item in groups["2d"]
    }
    assert all("stability" not in item.to_dict() for group in groups.values() for item in group)


@pytest.mark.parametrize(
    "values",
    [
        (-0.1, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.5, 0.5, 0.6, 0.5),
        (float("nan"), 0.0, 1.0, 1.0),
        (True, 0.0, 1.0, 1.0),
    ],
)
def test_roi_rejects_invalid_values(values: tuple[object, object, object, object]) -> None:
    with pytest.raises(ConfigurationError):
        ROI(*values)  # type: ignore[arg-type]


def test_localized_symbol_has_no_payload_attributes_or_serialized_fields() -> None:
    symbol = SymbolResult(
        status=SymbolStatus.LOCALIZED,
        kind=SymbolKind.TWO_D,
        confidence=0.8,
        quad=_quad(),
        normalized_quad=_normalized_quad(),
        image_width=10,
        image_height=10,
    )
    assert not hasattr(symbol, "text")
    assert not hasattr(symbol, "value")
    assert not hasattr(symbol, "raw_bytes")
    assert not hasattr(symbol, "format")
    assert not {"text", "value", "raw_bytes", "format", "bounding_box"} & symbol.to_dict().keys()


def test_decoded_symbol_preserves_bytes_and_value_alias() -> None:
    raw = b"\x00\xffpayload"
    symbol = DecodedSymbolResult(
        status=SymbolStatus.DECODED,
        kind=SymbolKind.LINEAR,
        confidence=1.0,
        quad=_quad(),
        normalized_quad=_normalized_quad(),
        image_width=10,
        image_height=10,
        text="payload",
        raw_bytes=raw,
        format="code-128",
    )
    encoded = symbol.to_dict()
    assert symbol.value == symbol.text
    assert base64.b64decode(encoded["raw_bytes"]["data"]) == raw
    assert "bounding_box" not in encoded


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (SymbolStatus.DECODED, SymbolKind.LINEAR),
        (SymbolStatus.UNRESOLVED_LINEAR, SymbolKind.TWO_D),
        (SymbolStatus.UNRESOLVED_MATRIX, SymbolKind.LINEAR),
    ],
)
def test_symbol_models_reject_status_kind_contract_violations(
    status: SymbolStatus, kind: SymbolKind
) -> None:
    with pytest.raises(ValueError):
        SymbolResult(
            status=status,
            kind=kind,
            confidence=0.5,
            quad=_quad(),
            normalized_quad=_normalized_quad(),
            image_width=10,
            image_height=10,
        )


@pytest.mark.parametrize(
    "normalized_quad",
    [
        (Point(0.1, 0.2), Point(1.1, 0.2), Point(0.3, 0.4), Point(0.1, 0.4)),
        (Point(0.1, 0.2), Point(0.3, 0.4), Point(0.3, 0.2), Point(0.1, 0.4)),
        (Point(0.2, 0.2),) * 4,
    ],
)
def test_symbol_models_reject_out_of_bounds_or_nonconvex_geometry(
    normalized_quad: tuple[Point, Point, Point, Point],
) -> None:
    with pytest.raises(ValueError):
        SymbolResult(
            status=SymbolStatus.LOCALIZED,
            kind=SymbolKind.TWO_D,
            confidence=0.5,
            quad=_quad(),
            normalized_quad=normalized_quad,
            image_width=10,
            image_height=10,
        )


def test_result_models_enforce_container_and_completion_invariants() -> None:
    metadata = ScanMetadata("0.1.0", "tessera", "0.1.0", "3.1.1", "native", "fast")
    symbol = SymbolResult(
        status=SymbolStatus.LOCALIZED,
        kind=SymbolKind.TWO_D,
        confidence=0.5,
        quad=_quad(),
        normalized_quad=_normalized_quad(),
        image_width=10,
        image_height=10,
    )
    with pytest.raises(ValueError, match="dimensions"):
        ImageResult((symbol,), 20, 10, 1.0, metadata)
    with pytest.raises(ValueError, match="complete"):
        DocumentResult(
            (),
            (ErrorDetail("page_processing_error", "failed", page=1),),
            True,
            1.0,
            metadata,
        )


@pytest.mark.parametrize(
    "region",
    [
        Region(
            quad=(1, 2, 3, 2, 3, 4, 1, 4),
            kind="unknown",
            confidence=0.5,
        ),
        Region(
            quad=(1, 2, 3, 2, 3, 4, 1, 4),
            kind="matrix",
            confidence=0.5,
            status=RegionStatus.DECODED,
        ),
        Region(
            quad=(1, 2, 3, 2, 3, 4, 1, 4),
            kind="matrix",
            confidence=0.5,
            status=RegionStatus.DECODED,
            value="payload",
            symbology="not-a-format",
        ),
    ],
)
def test_engine_contract_violations_become_typed_internal_errors(region: Region) -> None:
    with pytest.raises(InternalProcessingError):
        Scanner(symbols="2d")._symbol(region, 10, 10, False)


def test_metadata_has_reproducibility_fields() -> None:
    assert set(ScanMetadata("0.1.0", "tessera", "0.1.0", "3.1.1", "native", "fast").to_dict()) == {
        "package_version",
        "engine",
        "engine_version",
        "decoder_version",
        "backend",
        "mode",
    }
