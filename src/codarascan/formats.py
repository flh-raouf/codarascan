# SPDX-License-Identifier: Apache-2.0
"""Installed ZXing-C++ format catalog shared by every public surface."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import zxingcpp

from .errors import UnsupportedFormatError


class FormatKind(str, Enum):
    LINEAR = "1d"
    TWO_D = "2d"


@dataclass(frozen=True, slots=True)
class FormatInfo:
    """One selectable format in the installed decoder.

    Deliberately contains no experimental, validated, tier, or stability field.
    """

    name: str
    label: str
    kind: FormatKind
    aliases: tuple[str, ...]
    zxing_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind.value,
            "aliases": list(self.aliases),
        }


_META_NAMES = {
    "NONE",
    "All",
    "AllReadable",
    "AllCreatable",
    "AllLinear",
    "AllMatrix",
    "AllGS1",
    "AllRetail",
    "AllIndustrial",
    "LinearCodes",
    "MatrixCodes",
    "DataBarExpanded",  # enum alias
    "DataBarLimited",  # enum alias
    # This is an ISO symbology-identifier catch-all, not a concrete reader with
    # a standards-compliant fixture or a useful exact-format selection.
    "OtherBarcode",
}

_MATRIX_ENUM_NAMES = {
    "PDF417",
    "CompactPDF417",
    "MicroPDF417",
    "Aztec",
    "AztecCode",
    "AztecRune",
    "QRCode",
    "QRCodeModel1",
    "QRCodeModel2",
    "MicroQRCode",
    "RMQRCode",
    "DataMatrix",
    "MaxiCode",
}

_CANONICAL_OVERRIDES = {
    "PZN": "pzn",
    "RMQRCode": "rmqr-code",
    "QRCode": "qr-code",
    "QRCodeModel1": "qr-code-model-1",
    "QRCodeModel2": "qr-code-model-2",
    "MicroQRCode": "micro-qr-code",
    "DataMatrix": "data-matrix",
    "MaxiCode": "maxicode",
    "CompactPDF417": "compact-pdf417",
    "MicroPDF417": "micro-pdf417",
    "PDF417": "pdf417",
    "ITF14": "itf-14",
    "UPCA": "upc-a",
    "UPCE": "upc-e",
}


def _slug(value: str) -> str:
    value = value.strip().lower().replace("/", "-")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _alias_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _build_catalog() -> tuple[FormatInfo, ...]:
    records: list[FormatInfo] = []
    seen_values: set[int] = set()
    readable_values = {
        int(value)
        for value in zxingcpp.barcode_formats_list(
            zxingcpp.barcode_formats_from_str("AllReadable")
        )
    }
    for enum_name, value in vars(zxingcpp.BarcodeFormat).items():
        if enum_name.startswith("_") or enum_name in _META_NAMES:
            continue
        if not isinstance(value, zxingcpp.BarcodeFormat):
            continue
        numeric = int(value)
        if numeric not in readable_values or numeric in seen_values:
            continue
        seen_values.add(numeric)
        label = str(value)
        canonical = _CANONICAL_OVERRIDES.get(enum_name, _slug(label))
        aliases = tuple(
            dict.fromkeys(
                alias
                for alias in (
                    enum_name,
                    label,
                    _slug(enum_name),
                    _slug(label),
                )
                if alias != canonical
            )
        )
        records.append(
            FormatInfo(
                name=canonical,
                label=label,
                kind=(
                    FormatKind.TWO_D
                    if enum_name in _MATRIX_ENUM_NAMES
                    else FormatKind.LINEAR
                ),
                aliases=aliases,
                zxing_name=label,
            )
        )
    return tuple(sorted(records, key=lambda item: (item.kind.value, item.name)))


_CATALOG = _build_catalog()
_BY_NAME = {item.name: item for item in _CATALOG}
_BY_ALIAS: dict[str, FormatInfo] = {}
for _item in _CATALOG:
    for _alias in (_item.name, _item.label, *_item.aliases):
        _BY_ALIAS[_alias_key(_alias)] = _item


def supported_formats() -> Mapping[str, tuple[FormatInfo, ...]]:
    """Return selectable formats grouped under the stable ``1d`` and ``2d`` keys."""

    return MappingProxyType(
        {
            "1d": tuple(item for item in _CATALOG if item.kind is FormatKind.LINEAR),
            "2d": tuple(item for item in _CATALOG if item.kind is FormatKind.TWO_D),
        }
    )


def resolve_format(value: str) -> FormatInfo:
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedFormatError(
            "format names must be non-empty strings",
            context={"parameter": "formats"},
        )
    resolved = _BY_ALIAS.get(_alias_key(value))
    if resolved is None:
        choices = sorted(_BY_NAME)
        raise UnsupportedFormatError(
            f"unknown barcode format {value!r}; supported choices: {', '.join(choices)}",
            context={"format": value, "supported": choices},
        )
    return resolved


def resolve_formats(values: Iterable[str]) -> tuple[FormatInfo, ...]:
    resolved: dict[str, FormatInfo] = {}
    for value in values:
        item = resolve_format(value)
        resolved[item.name] = item
    return tuple(resolved[name] for name in sorted(resolved))


def zxing_formats(values: Iterable[FormatInfo]) -> Any:
    names = ",".join(item.zxing_name for item in values)
    return zxingcpp.barcode_formats_from_str(names)


__all__ = ["FormatInfo", "FormatKind", "resolve_format", "resolve_formats", "supported_formats"]
