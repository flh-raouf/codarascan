# SPDX-License-Identifier: Apache-2.0
"""Immutable, reusable, thread-safe CodaraScan Scanner façade."""

from __future__ import annotations

import math
import tempfile
import threading
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeVar, cast

import cv2

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .documents import DocumentInput, DocumentStream, ErrorPolicy, WorkerCount
    from .models import DocumentResult

from . import __version__
from .core.contracts import RegionStatus as EngineRegionStatus
from .core.contracts import Roi
from .engines import get_engine
from .errors import ConfigurationError, InternalProcessingError
from .formats import FormatInfo, FormatKind, resolve_format, resolve_formats, supported_formats
from .inputs import ImageInput, normalize_image
from .models import (
    DecodedSymbolResult,
    ImageResult,
    Point,
    ScanMetadata,
    SymbolKind,
    SymbolResult,
    SymbolStatus,
)


class ScanMode(str, Enum):
    FAST = "fast"
    ROBUST = "robust"
    PANORAMA = "panorama"


class SymbolGroup(str, Enum):
    LINEAR = "linear"
    TWO_D = "2d"
    ALL = "all"


EnumT = TypeVar("EnumT", bound=Enum)


def _enum_value(enum_type: type[EnumT], value: Any, parameter: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ConfigurationError(
            f"{parameter} must be a string or {enum_type.__name__}",
            context={"parameter": parameter, "value": repr(value)},
        )
    try:
        return enum_type(value.strip().lower())
    except ValueError as exc:
        choices = [member.value for member in enum_type]
        raise ConfigurationError(
            f"{parameter} must be one of {', '.join(choices)}, got {value!r}",
            context={"parameter": parameter, "value": value, "supported": choices},
        ) from exc


def _normalize_roi(value: Roi | tuple[float, float, float, float] | None) -> Roi | None:
    if value is None or isinstance(value, Roi):
        return value
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ConfigurationError("roi must contain normalized x, y, width, and height")
    values = tuple(value)
    if len(values) != 4:
        raise ConfigurationError(
            "roi must contain exactly four values: x, y, width, height",
            context={"length": len(values)},
        )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
        raise ConfigurationError("roi values must be finite real numbers")
    try:
        return Roi(*(float(item) for item in values))
    except ValueError as exc:
        raise ConfigurationError(str(exc), context={"parameter": "roi"}) from exc


def _decoder_version() -> str:
    try:
        return importlib_metadata.version("zxing-cpp")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


_STATUS_MAP = {
    EngineRegionStatus.DECODED: SymbolStatus.DECODED,
    EngineRegionStatus.UNRESOLVED_LINEAR: SymbolStatus.UNRESOLVED_LINEAR,
    EngineRegionStatus.UNRESOLVED_MATRIX: SymbolStatus.UNRESOLVED_MATRIX,
    EngineRegionStatus.LOCALIZED: SymbolStatus.LOCALIZED,
    EngineRegionStatus.REVIEW_CANDIDATE: SymbolStatus.REVIEW_CANDIDATE,
}


@dataclass(frozen=True, slots=True)
class Scanner:
    """A fixed scanning configuration safe to share across application threads."""

    mode: ScanMode | str = ScanMode.FAST
    symbols: SymbolGroup | str = SymbolGroup.ALL
    formats: tuple[str, ...] | list[str] | set[str] | str | None = None
    decode: bool = True
    _format_infos: tuple[FormatInfo, ...] = field(init=False, repr=False, compare=False)
    _engine: Any = field(init=False, default=None, repr=False, compare=False)
    _initialize_lock: threading.RLock = field(
        init=False, default_factory=threading.RLock, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        mode = _enum_value(ScanMode, self.mode, "mode")
        symbols = _enum_value(SymbolGroup, self.symbols, "symbols")
        if not isinstance(self.decode, bool):
            raise ConfigurationError(
                "decode must be a boolean", context={"parameter": "decode"}
            )
        if mode is ScanMode.PANORAMA and not self.decode:
            raise ConfigurationError("panorama mode requires decode=True")
        raw_formats: tuple[str, ...]
        if self.formats is None:
            raw_formats = ()
        elif isinstance(self.formats, str):
            raw_formats = (self.formats,)
        else:
            try:
                raw_formats = tuple(self.formats)
            except TypeError as exc:
                raise ConfigurationError("formats must be an iterable of names") from exc
        if any(not isinstance(value, str) for value in raw_formats):
            raise ConfigurationError("every format name must be a string")
        infos = resolve_formats(raw_formats)
        expected_kind = {
            SymbolGroup.LINEAR: FormatKind.LINEAR,
            SymbolGroup.TWO_D: FormatKind.TWO_D,
        }.get(symbols)
        contradictory = [
            item.name for item in infos if expected_kind is not None and item.kind is not expected_kind
        ]
        if contradictory:
            raise ConfigurationError(
                f"formats contradict symbols={symbols.value}: {', '.join(contradictory)}",
                context={
                    "parameter": "formats",
                    "symbols": symbols.value,
                    "formats": contradictory,
                },
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "formats", tuple(item.name for item in infos))
        object.__setattr__(self, "_format_infos", infos)

    @staticmethod
    def supported_formats() -> Any:
        return supported_formats()

    def _get_engine(self) -> Any:
        engine = self._engine
        if engine is not None:
            return engine
        with self._initialize_lock:
            engine = self._engine
            if engine is None:
                suffix = "extractor" if self.decode else "localizer"
                family = {
                    ScanMode.FAST: "tessera",
                    ScanMode.ROBUST: "mosaic",
                    ScanMode.PANORAMA: "panorama",
                }[cast(ScanMode, self.mode)]
                engine_id = f"{family}-{suffix}"
                from .core.contracts import Capability

                capability = Capability.DECODE if self.decode else Capability.DETECT
                engine = get_engine(engine_id, capability)
                object.__setattr__(self, "_engine", engine)
        return engine

    def warm(self) -> None:
        """Eagerly perform the same idempotent initialization used by the first scan."""

        with self._initialize_lock:
            self._get_engine().warm()

    def _engine_options(self) -> dict[str, Any]:
        # ZXing exposes ISBN as a readable semantic subtype, but its reader is
        # selected through EAN-13 and reports EAN-13. The public adapter applies
        # the ISBN prefix postcondition below so the exact filter remains exact.
        zxing_names = [
            "EAN-13" if item.name == "isbn" else item.zxing_name
            for item in self._format_infos
        ]
        return {
            "kinds": cast(SymbolGroup, self.symbols).value,
            "formats": zxing_names,
            # Fast is the public engine choice. Its 2-D branch is always the stronger
            # internal profile; the weaker profile is not selectable here.
            "mode": "robust",
            "stage_parallel": True,
            "save_crops": False,
            "save_overlays": False,
        }

    def _metadata(self, engine: Any) -> ScanMetadata:
        mode = cast(ScanMode, self.mode)
        backend = "python"
        if mode is ScanMode.FAST:
            try:
                from .engines.common.native import loader

                backend = loader.backend_name()
            except Exception:
                backend = "python-reference"
        elif mode is ScanMode.PANORAMA:
            backend = "hybrid"
        return ScanMetadata(
            package_version=__version__,
            engine=engine.info.id,
            engine_version=__version__,
            decoder_version=_decoder_version(),
            backend=backend,
            mode=mode.value,
        )

    @staticmethod
    def _quad(
        flat: tuple[float, ...], width: int, height: int
    ) -> tuple[tuple[Point, Point, Point, Point], tuple[Point, Point, Point, Point]]:
        if len(flat) != 8 or any(not math.isfinite(float(value)) for value in flat):
            raise InternalProcessingError("engine returned an invalid quadrilateral")
        points: list[Point] = []
        normalized: list[Point] = []
        # ZXing's 1-D endpoint convention can extend one module beyond the
        # decoded raster (commonly 7 px on generated DataBar samples). Treat a
        # small relative overrun as harmless coordinate drift, while retaining
        # a hard failure for materially invalid engine geometry.
        tolerance_x = max(1.0, width * 0.02)
        tolerance_y = max(1.0, height * 0.02)
        for index in range(0, 8, 2):
            x = float(flat[index])
            y = float(flat[index + 1])
            if (
                x < -tolerance_x
                or y < -tolerance_y
                or x > width + tolerance_x
                or y > height + tolerance_y
            ):
                raise InternalProcessingError(
                    "engine returned geometry outside the analyzed image",
                    context={"x": x, "y": y, "width": width, "height": height},
                )
            x = min(float(width), max(0.0, x))
            y = min(float(height), max(0.0, y))
            points.append(Point(x, y))
            normalized.append(Point(x / width, y / height))
        return tuple(points), tuple(normalized)  # type: ignore[return-value]

    def _symbol(self, region: Any, width: int, height: int, diagnostics: bool) -> SymbolResult:
        quad, normalized_quad = self._quad(tuple(region.quad), width, height)
        confidence = float(region.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise InternalProcessingError(
                "engine returned confidence outside [0, 1]",
                context={"confidence": confidence},
            )
        raw_kind = str(region.kind).lower()
        if raw_kind in {"2d", "matrix"}:
            kind = SymbolKind.TWO_D
        elif raw_kind in {"1d", "linear"}:
            kind = SymbolKind.LINEAR
        else:
            raise InternalProcessingError(
                "engine returned an unknown symbol kind",
                context={"kind": str(region.kind)},
            )
        status = _STATUS_MAP.get(region.status)
        if status is None:
            raise InternalProcessingError(
                "engine returned an unknown status", context={"status": str(region.status)}
            )
        if not self.decode:
            status = SymbolStatus.LOCALIZED
        common: dict[str, Any] = {
            "status": status,
            "kind": kind,
            "confidence": confidence,
            "quad": quad,
            "normalized_quad": normalized_quad,
            "image_width": width,
            "image_height": height,
            "sources": tuple(region.sources),
            "diagnostics": dict(region.extras) if diagnostics else {},
        }
        if self.decode and status is SymbolStatus.DECODED:
            if region.value is None:
                raise InternalProcessingError("decoded symbol has no payload")
            if not region.symbology:
                raise InternalProcessingError("decoded symbol has no format")
            try:
                format_name = resolve_format(str(region.symbology)).name
            except ConfigurationError as exc:
                raise InternalProcessingError(
                    "engine returned an unknown decoded format",
                    context={"format": str(region.symbology)},
                ) from exc
            raw_bytes = getattr(region, "raw_bytes", None)
            if raw_bytes is None:
                raw_bytes = str(region.value).encode("utf-8")
            return DecodedSymbolResult(
                **common,
                text=str(region.value),
                raw_bytes=bytes(raw_bytes),
                format=format_name,
            )
        return SymbolResult(**common)

    def _filtered_symbol(
        self, region: Any, width: int, height: int, diagnostics: bool
    ) -> SymbolResult | None:
        symbol = self._symbol(region, width, height, diagnostics)
        if not isinstance(symbol, DecodedSymbolResult):
            return symbol
        selected_names = {item.name for item in self._format_infos}
        isbn_only_for_ean13 = (
            "isbn" in selected_names
            and "ean-13" not in selected_names
            and "ean-upc" not in selected_names
        )
        if symbol.format == "ean-13" and isbn_only_for_ean13:
            if not symbol.text.startswith(("978", "979")):
                return None
            return DecodedSymbolResult(
                status=symbol.status,
                kind=symbol.kind,
                confidence=symbol.confidence,
                quad=symbol.quad,
                normalized_quad=symbol.normalized_quad,
                image_width=width,
                image_height=height,
                sources=symbol.sources,
                diagnostics=symbol.diagnostics,
                text=symbol.text,
                raw_bytes=symbol.raw_bytes,
                format="isbn",
            )
        return symbol

    def scan_image(
        self,
        image: ImageInput,
        *,
        roi: Roi | tuple[float, float, float, float] | None = None,
        diagnostics: bool = False,
    ) -> ImageResult:
        """Scan one path, encoded image, Pillow image, or OpenCV-style NumPy array."""

        if not isinstance(diagnostics, bool):
            raise ConfigurationError("diagnostics must be a boolean")
        normalized_roi = _normalize_roi(roi)
        started = perf_counter()
        normalized = normalize_image(image)
        engine = self._get_engine()
        try:
            with tempfile.TemporaryDirectory(prefix="codarascan-image-") as directory:
                path = Path(directory) / "input.png"
                if not cv2.imwrite(str(path), normalized.pixels):
                    raise InternalProcessingError("unable to create private scan input")
                outcome = engine.analyze_page(
                    1,
                    path,
                    roi=normalized_roi,
                    options=self._engine_options(),
                )
        except (ConfigurationError, InternalProcessingError):
            raise
        except Exception as exc:
            mode = cast(ScanMode, self.mode)
            raise InternalProcessingError(
                f"{mode.value} scan failed: {exc}",
                context={"mode": mode.value},
            ) from exc
        converted = (
            self._filtered_symbol(region, normalized.width, normalized.height, diagnostics)
            for region in outcome.regions
        )
        symbols = tuple(
            sorted(
                (symbol for symbol in converted if symbol is not None),
                key=lambda item: (
                    item.quad[0].y,
                    item.quad[0].x,
                    item.status.value,
                    getattr(item, "format", ""),
                    getattr(item, "text", ""),
                ),
            )
        )
        return ImageResult(
            symbols=symbols,
            width=normalized.width,
            height=normalized.height,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            metadata=self._metadata(engine),
            diagnostics=(dict(outcome.diagnostics) if diagnostics else {}),
        )

    def iter_document(
        self,
        document: DocumentInput,
        *,
        pages: Iterable[int] | None = None,
        workers: WorkerCount = 1,
        on_error: ErrorPolicy = "raise",
        roi: Roi | tuple[float, float, float, float] | None = None,
        diagnostics: bool = False,
    ) -> DocumentStream:
        """Stream selected one-based PDF pages in requested order."""

        from .documents import DocumentStream

        return DocumentStream(
            self,
            document,
            pages=pages,
            workers=workers,
            on_error=on_error,
            roi=roi,
            diagnostics=diagnostics,
        )

    def scan_document(
        self,
        document: DocumentInput,
        *,
        pages: Iterable[int] | None = None,
        workers: WorkerCount = 1,
        on_error: ErrorPolicy = "raise",
        roi: Roi | tuple[float, float, float, float] | None = None,
        diagnostics: bool = False,
    ) -> DocumentResult:
        """Scan selected one-based PDF pages into an ordered document result."""

        from .documents import scan_document

        return scan_document(
            self,
            document,
            pages=pages,
            workers=workers,
            on_error=on_error,
            roi=roi,
            diagnostics=diagnostics,
        )


@lru_cache(maxsize=64)
def _cached_scanner(
    mode: ScanMode, symbols: SymbolGroup, formats: tuple[str, ...], decode: bool
) -> Scanner:
    return Scanner(mode=mode, symbols=symbols, formats=formats, decode=decode)


def scan_image(
    image: ImageInput,
    *,
    mode: ScanMode | str = ScanMode.FAST,
    symbols: SymbolGroup | str = SymbolGroup.ALL,
    formats: tuple[str, ...] | list[str] | set[str] | str | None = None,
    decode: bool = True,
    roi: Roi | tuple[float, float, float, float] | None = None,
    diagnostics: bool = False,
) -> ImageResult:
    """Scan one image with a cached immutable Scanner."""

    configured = Scanner(mode=mode, symbols=symbols, formats=formats, decode=decode)
    scanner = _cached_scanner(
        cast(ScanMode, configured.mode),
        cast(SymbolGroup, configured.symbols),
        cast(tuple[str, ...], configured.formats),
        configured.decode,
    )
    return scanner.scan_image(image, roi=roi, diagnostics=diagnostics)


def iter_document(
    document: DocumentInput,
    *,
    mode: ScanMode | str = ScanMode.FAST,
    symbols: SymbolGroup | str = SymbolGroup.ALL,
    formats: tuple[str, ...] | list[str] | set[str] | str | None = None,
    decode: bool = True,
    pages: Iterable[int] | None = None,
    workers: WorkerCount = 1,
    on_error: ErrorPolicy = "raise",
    roi: Roi | tuple[float, float, float, float] | None = None,
    diagnostics: bool = False,
) -> DocumentStream:
    """Stream a PDF with a cached immutable scanner configuration."""

    configured = Scanner(mode=mode, symbols=symbols, formats=formats, decode=decode)
    scanner = _cached_scanner(
        cast(ScanMode, configured.mode),
        cast(SymbolGroup, configured.symbols),
        cast(tuple[str, ...], configured.formats),
        configured.decode,
    )
    return scanner.iter_document(
        document,
        pages=pages,
        workers=workers,
        on_error=on_error,
        roi=roi,
        diagnostics=diagnostics,
    )


def scan_document(
    document: DocumentInput,
    *,
    mode: ScanMode | str = ScanMode.FAST,
    symbols: SymbolGroup | str = SymbolGroup.ALL,
    formats: tuple[str, ...] | list[str] | set[str] | str | None = None,
    decode: bool = True,
    pages: Iterable[int] | None = None,
    workers: WorkerCount = 1,
    on_error: ErrorPolicy = "raise",
    roi: Roi | tuple[float, float, float, float] | None = None,
    diagnostics: bool = False,
) -> DocumentResult:
    """Scan a PDF with a cached immutable scanner configuration."""

    configured = Scanner(mode=mode, symbols=symbols, formats=formats, decode=decode)
    scanner = _cached_scanner(
        cast(ScanMode, configured.mode),
        cast(SymbolGroup, configured.symbols),
        cast(tuple[str, ...], configured.formats),
        configured.decode,
    )
    return scanner.scan_document(
        document,
        pages=pages,
        workers=workers,
        on_error=on_error,
        roi=roi,
        diagnostics=diagnostics,
    )


__all__ = [
    "ScanMode",
    "Scanner",
    "SymbolGroup",
    "iter_document",
    "scan_document",
    "scan_image",
]
