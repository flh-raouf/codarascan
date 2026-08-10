# SPDX-License-Identifier: Apache-2.0
"""Stable result, geometry, metadata, and error models."""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _validate_quad(
    quad: Quadrilateral,
    *,
    width: int | None,
    height: int | None,
    normalized: bool,
) -> None:
    if len(quad) != 4:
        raise ValueError("quadrilaterals must contain exactly four points")
    maximum_x = 1.0 if normalized else math.inf if width is None else float(width)
    maximum_y = 1.0 if normalized else math.inf if height is None else float(height)
    tolerance = 1e-9
    for point in quad:
        if not -tolerance <= point.x <= maximum_x + tolerance:
            raise ValueError("quadrilateral x coordinates must remain within the image")
        if not -tolerance <= point.y <= maximum_y + tolerance:
            raise ValueError("quadrilateral y coordinates must remain within the image")
    crosses = []
    for index in range(4):
        first = quad[index]
        second = quad[(index + 1) % 4]
        third = quad[(index + 2) % 4]
        crosses.append(
            (second.x - first.x) * (third.y - second.y)
            - (second.y - first.y) * (third.x - second.x)
        )
    if any(value <= tolerance for value in crosses):
        raise ValueError("quadrilaterals must be non-degenerate, convex, and ordered clockwise")


class SymbolKind(str, Enum):
    LINEAR = "linear"
    TWO_D = "2d"


class SymbolStatus(str, Enum):
    DECODED = "decoded"
    UNRESOLVED_LINEAR = "localized_unresolved_linear"
    UNRESOLVED_MATRIX = "localized_unresolved_matrix"
    LOCALIZED = "localized"
    REVIEW_CANDIDATE = "review_candidate"


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("point coordinates must be finite")

    def to_list(self) -> list[float]:
        return [self.x, self.y]


Quadrilateral = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class ScanMetadata:
    package_version: str
    engine: str
    engine_version: str
    decoder_version: str
    backend: str
    mode: str

    def __post_init__(self) -> None:
        for name in (
            "package_version",
            "engine",
            "engine_version",
            "decoder_version",
            "backend",
            "mode",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"metadata {name} must be a non-empty string")
        if self.mode not in {"fast", "robust"}:
            raise ValueError("metadata mode must be fast or robust")

    def to_dict(self) -> dict[str, str]:
        return {
            "package_version": self.package_version,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "decoder_version": self.decoder_version,
            "backend": self.backend,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class SymbolResult:
    status: SymbolStatus
    kind: SymbolKind
    confidence: float
    quad: Quadrilateral
    normalized_quad: Quadrilateral
    image_width: InitVar[int]
    image_height: InitVar[int]
    sources: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self, image_width: int, image_height: int) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        _validate_quad(
            self.quad,
            width=image_width,
            height=image_height,
            normalized=False,
        )
        _validate_quad(
            self.normalized_quad,
            width=None,
            height=None,
            normalized=True,
        )
        for point, normalized_point in zip(self.quad, self.normalized_quad, strict=True):
            if not math.isclose(
                normalized_point.x,
                point.x / image_width,
                rel_tol=0.0,
                abs_tol=1e-7,
            ) or not math.isclose(
                normalized_point.y,
                point.y / image_height,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError("normalized quadrilateral must match pixel geometry")
        if self.status is SymbolStatus.DECODED and type(self) is SymbolResult:
            raise ValueError("decoded status requires DecodedSymbolResult")
        if self.status is SymbolStatus.UNRESOLVED_LINEAR and self.kind is not SymbolKind.LINEAR:
            raise ValueError("unresolved linear status requires linear kind")
        if self.status is SymbolStatus.UNRESOLVED_MATRIX and self.kind is not SymbolKind.TWO_D:
            raise ValueError("unresolved matrix status requires 2d kind")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    def to_dict(self, *, include_diagnostics: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "status": self.status.value,
            "kind": self.kind.value,
            "confidence": self.confidence,
            "quad": [point.to_list() for point in self.quad],
            "normalized_quad": [point.to_list() for point in self.normalized_quad],
            "sources": list(self.sources),
        }
        if include_diagnostics and self.diagnostics:
            output["diagnostics"] = dict(self.diagnostics)
        return output


@dataclass(frozen=True, slots=True)
class DecodedSymbolResult(SymbolResult):
    text: str = ""
    raw_bytes: bytes = b""
    format: str = ""

    def __post_init__(self, image_width: int, image_height: int) -> None:
        SymbolResult.__post_init__(self, image_width, image_height)
        if self.status is not SymbolStatus.DECODED:
            raise ValueError("DecodedSymbolResult requires decoded status")
        if not isinstance(self.text, str):
            raise ValueError("decoded text must be a string")
        if not isinstance(self.raw_bytes, bytes):
            raise ValueError("decoded raw_bytes must be bytes")
        if not isinstance(self.format, str) or not self.format:
            raise ValueError("decoded format must be a non-empty canonical name")

    @property
    def value(self) -> str:
        return self.text

    def to_dict(self, *, include_diagnostics: bool = False) -> dict[str, Any]:
        output = SymbolResult.to_dict(self, include_diagnostics=include_diagnostics)
        output.update(
            {
                "text": self.text,
                "value": self.text,
                "raw_bytes": {
                    "encoding": "base64",
                    "data": base64.b64encode(self.raw_bytes).decode("ascii"),
                },
                "format": self.format,
            }
        )
        return output


@dataclass(frozen=True, slots=True)
class ImageResult:
    symbols: tuple[SymbolResult, ...]
    width: int
    height: int
    elapsed_ms: float
    metadata: ScanMetadata
    diagnostics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        for symbol in self.symbols:
            _validate_quad(symbol.quad, width=self.width, height=self.height, normalized=False)
            for point, normalized_point in zip(
                symbol.quad, symbol.normalized_quad, strict=True
            ):
                if not math.isclose(
                    normalized_point.x,
                    point.x / self.width,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ) or not math.isclose(
                    normalized_point.y,
                    point.y / self.height,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        "normalized quadrilateral must match containing image dimensions"
                    )

    def to_dict(self, *, include_diagnostics: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "type": "image",
            "width": self.width,
            "height": self.height,
            "elapsed_ms": self.elapsed_ms,
            "metadata": self.metadata.to_dict(),
            "symbols": [
                symbol.to_dict(include_diagnostics=include_diagnostics) for symbol in self.symbols
            ],
        }
        if include_diagnostics and self.diagnostics:
            output["diagnostics"] = dict(self.diagnostics)
        return output


@dataclass(frozen=True, slots=True)
class PageResult:
    page: int
    image: ImageResult

    def __post_init__(self) -> None:
        if self.page <= 0:
            raise ValueError("page numbers are one-based positive integers")

    def to_dict(self, *, include_diagnostics: bool = False) -> dict[str, Any]:
        output = self.image.to_dict(include_diagnostics=include_diagnostics)
        output["type"] = "page"
        output["page"] = self.page
        return output


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: str
    message: str
    page: int | None = None
    context: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _mapping(self.context))
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("error code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("error message must be a non-empty string")
        if self.page is not None and (
            isinstance(self.page, bool) or not isinstance(self.page, int) or self.page <= 0
        ):
            raise ValueError("error page must be a one-based positive integer")

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.page is not None:
            output["page"] = self.page
        if self.context:
            output["context"] = dict(self.context)
        return output


@dataclass(frozen=True, slots=True)
class DocumentResult:
    pages: tuple[PageResult, ...]
    errors: tuple[ErrorDetail, ...]
    complete: bool
    elapsed_ms: float
    metadata: ScanMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "errors", tuple(self.errors))
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        if self.complete and self.errors:
            raise ValueError("a complete document result cannot contain errors")

    def to_dict(self, *, include_diagnostics: bool = False) -> dict[str, Any]:
        return {
            "type": "document",
            "complete": self.complete,
            "elapsed_ms": self.elapsed_ms,
            "metadata": self.metadata.to_dict(),
            "pages": [page.to_dict(include_diagnostics=include_diagnostics) for page in self.pages],
            "errors": [error.to_dict() for error in self.errors],
        }


__all__ = [
    "DecodedSymbolResult",
    "DocumentResult",
    "ErrorDetail",
    "ImageResult",
    "PageResult",
    "Point",
    "Quadrilateral",
    "ScanMetadata",
    "SymbolKind",
    "SymbolResult",
    "SymbolStatus",
]
