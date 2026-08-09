# SPDX-License-Identifier: Apache-2.0
"""Ordered, bounded PDF streaming built on pypdfium2 and image scanning."""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray

from .core.contracts import Roi
from .errors import (
    CodaraScanError,
    ConfigurationError,
    DocumentError,
    DocumentRenderError,
    PageProcessingError,
)
from .models import DocumentResult, ErrorDetail, ImageResult, PageResult, ScanMetadata

if TYPE_CHECKING:
    from .scanner import Scanner

DocumentInput: TypeAlias = str | Path | bytes | bytearray | memoryview
WorkerCount: TypeAlias = int | Literal["auto"]
ErrorPolicy: TypeAlias = Literal["raise", "collect"]
UInt8Image: TypeAlias = NDArray[np.uint8]

RENDER_DPI = 300
_PDFIUM_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class RenderedPage:
    pixels: UInt8Image
    strategy: str


def _document_source(value: DocumentInput) -> str | bytes:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.exists():
            raise DocumentError(
                f"document path does not exist: {path}",
                context={"path": str(path)},
            )
        if not path.is_file():
            raise DocumentError(
                f"document path is not a file: {path}",
                context={"path": str(path)},
            )
        if path.suffix.lower() != ".pdf":
            raise DocumentError(
                "the 0.1.0 document API supports PDF files only",
                context={"path": str(path), "type": path.suffix.lower()},
            )
        return str(path)
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if not data:
            raise DocumentError("PDF bytes are empty")
        if not data.lstrip().startswith(b"%PDF-"):
            raise DocumentError("document bytes are not a PDF")
        return data
    raise DocumentError(
        f"unsupported document input type: {type(value).__name__}",
        context={"type": type(value).__name__},
    )


def _open_pdf(source: str | bytes) -> pdfium.PdfDocument:
    try:
        with _PDFIUM_LOCK:
            document = pdfium.PdfDocument(source)
            page_count = len(document)
    except Exception as exc:
        raise DocumentRenderError("unable to open PDF document") from exc
    if page_count <= 0:
        document.close()
        raise DocumentRenderError("PDF document contains no pages")
    return document


def _selected_pages(pages: Iterable[int] | None, page_count: int) -> tuple[int, ...]:
    if pages is None:
        return tuple(range(1, page_count + 1))
    if isinstance(pages, (str, bytes)):
        raise ConfigurationError("pages must be an iterable of one-based integers")
    try:
        selected = tuple(pages)
    except TypeError as exc:
        raise ConfigurationError("pages must be an iterable of one-based integers") from exc
    if not selected:
        raise ConfigurationError("pages must not be empty")
    if any(isinstance(page, bool) or not isinstance(page, int) for page in selected):
        raise ConfigurationError("every selected page must be a one-based integer")
    seen: set[int] = set()
    duplicates: set[int] = set()
    for page in selected:
        if page in seen:
            duplicates.add(page)
        seen.add(page)
    if duplicates:
        raise ConfigurationError(
            f"duplicate page selections are not allowed: {sorted(duplicates)}",
            context={"pages": sorted(duplicates)},
        )
    invalid = [page for page in selected if page < 1 or page > page_count]
    if invalid:
        raise ConfigurationError(
            f"selected pages are outside the one-based range 1..{page_count}: {invalid}",
            context={"pages": invalid, "page_count": page_count},
        )
    return selected


def _worker_count(value: WorkerCount, page_count: int) -> int:
    if value == "auto":
        return max(1, min(os.cpu_count() or 1, page_count))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError("workers must be a positive integer or 'auto'")
    return min(value, page_count)


def _error_policy(value: str) -> ErrorPolicy:
    if value not in {"raise", "collect"}:
        raise ConfigurationError("on_error must be 'raise' or 'collect'")
    return cast(ErrorPolicy, value)


def _copy_bitmap(bitmap: pdfium.PdfBitmap) -> UInt8Image:
    try:
        pixels = np.asarray(bitmap.to_numpy(), dtype=np.uint8).copy(order="C")
    finally:
        bitmap.close()
    if pixels.ndim not in {2, 3} or pixels.size == 0:
        raise DocumentRenderError("PDFium returned an empty page bitmap")
    if pixels.ndim == 3 and pixels.shape[2] not in {3, 4}:
        raise DocumentRenderError(
            "PDFium returned an unsupported page bitmap",
            context={"shape": list(pixels.shape)},
        )
    return pixels


def _native_raster(page: pdfium.PdfPage) -> UInt8Image | None:
    """Reuse a sole full-page embedded raster only when page meaning is preserved."""

    if page.get_rotation() != 0:
        return None
    objects = list(page.get_objects())
    try:
        if len(objects) != 1 or not isinstance(objects[0], pdfium.PdfImage):
            return None
        image = objects[0]
        page_width, page_height = page.get_size()
        left, bottom, right, top = image.get_bounds()
        coverage = max(0.0, right - left) * max(0.0, top - bottom)
        page_area = page_width * page_height
        if page_area <= 0 or coverage / page_area < 0.98:
            return None
        quad = image.get_quad_points()
        xs = {round(point[0], 4) for point in quad}
        ys = {round(point[1], 4) for point in quad}
        if len(xs) != 2 or len(ys) != 2:
            return None
        return _copy_bitmap(image.get_bitmap(render=True, scale_to_original=True))
    finally:
        for page_object in objects:
            page_object.close()


def _render_page(document: pdfium.PdfDocument, page_number: int) -> RenderedPage:
    try:
        with _PDFIUM_LOCK:
            page = document[page_number - 1]
            try:
                native = _native_raster(page)
                if native is not None:
                    return RenderedPage(native, "native-embedded-raster")
                bitmap = page.render(
                    scale=RENDER_DPI / 72.0,
                    optimize_mode="print",
                    maybe_alpha=True,
                )
                return RenderedPage(
                    _copy_bitmap(bitmap),
                    f"pdfium-render-{RENDER_DPI}-dpi",
                )
            finally:
                page.close()
    except DocumentRenderError:
        raise
    except Exception as exc:
        raise DocumentRenderError(
            f"unable to render PDF page {page_number}",
            context={"page": page_number},
        ) from exc


def _error_detail(error: BaseException, page: int) -> ErrorDetail:
    if isinstance(error, CodaraScanError):
        context = dict(error.context)
        code = error.code
        message = error.message
    else:
        context = {}
        code = PageProcessingError.code
        message = "unexpected page processing failure"
    context.setdefault("page", page)
    return ErrorDetail(code=code, message=message, page=page, context=context)


class DocumentStream(Iterator[PageResult]):
    """Ordered PDF page iterator with bounded rendering and analysis work.

    Under ``on_error='collect'``, failed pages are omitted from iteration and
    accumulated in :attr:`errors`. Inspect :attr:`complete` after exhaustion.
    Closing the iterator stops scheduling and cancels work that has not begun.
    """

    def __init__(
        self,
        scanner: Scanner,
        document: DocumentInput,
        *,
        pages: Iterable[int] | None = None,
        workers: WorkerCount = 1,
        on_error: ErrorPolicy = "raise",
        roi: Roi | tuple[float, float, float, float] | None = None,
        diagnostics: bool = False,
    ) -> None:
        if not isinstance(diagnostics, bool):
            raise ConfigurationError("diagnostics must be a boolean")
        self._scanner = scanner
        # Validate input-specific configuration before PDF rendering or task
        # submission, matching scan_image's fail-before-engine contract.
        from .scanner import _normalize_roi

        normalized_roi = _normalize_roi(roi)
        self._source = _document_source(document)
        self._document = _open_pdf(self._source)
        try:
            self.selected_pages = _selected_pages(pages, len(self._document))
            self.workers = _worker_count(workers, len(self.selected_pages))
            self.on_error = _error_policy(on_error)
        except Exception:
            self._document.close()
            raise
        self._roi = normalized_roi
        self._diagnostics = diagnostics
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="codarascan-page",
        )
        self._pending: deque[tuple[int, Future[ImageResult]]] = deque()
        self._next_index = 0
        self._errors: list[ErrorDetail] = []
        self._yielded = 0
        self._started = perf_counter()
        self._closed = False

    @property
    def errors(self) -> tuple[ErrorDetail, ...]:
        return tuple(self._errors)

    @property
    def complete(self) -> bool:
        return (
            self._closed
            and not self._errors
            and self._yielded == len(self.selected_pages)
        )

    @property
    def elapsed_ms(self) -> float:
        return (perf_counter() - self._started) * 1000.0

    @property
    def metadata(self) -> ScanMetadata:
        engine = self._scanner._get_engine()
        return self._scanner._metadata(engine)

    def __iter__(self) -> DocumentStream:
        return self

    def __enter__(self) -> DocumentStream:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _record_or_raise(self, error: BaseException, page: int) -> None:
        if self.on_error == "collect":
            self._errors.append(_error_detail(error, page))
            return
        self.close()
        if isinstance(error, DocumentError):
            raise error
        if isinstance(error, CodaraScanError):
            raise PageProcessingError(
                f"failed to process PDF page {page}: {error.message}",
                context={"page": page, "cause": error.code},
            ) from error
        raise PageProcessingError(
            f"failed to process PDF page {page}", context={"page": page}
        ) from error

    def _fill(self) -> None:
        while (
            not self._closed
            and len(self._pending) < self.workers
            and self._next_index < len(self.selected_pages)
        ):
            page_number = self.selected_pages[self._next_index]
            self._next_index += 1
            try:
                rendered = _render_page(self._document, page_number)
            except BaseException as exc:
                self._record_or_raise(exc, page_number)
                continue
            future = self._executor.submit(
                self._scanner.scan_image,
                rendered.pixels,
                roi=self._roi,
                diagnostics=self._diagnostics,
            )
            self._pending.append((page_number, future))

    def __next__(self) -> PageResult:
        while True:
            if self._closed:
                raise StopIteration
            self._fill()
            if not self._pending:
                self._finish()
                raise StopIteration
            page_number, future = self._pending.popleft()
            try:
                image = future.result()
            except BaseException as exc:
                self._record_or_raise(exc, page_number)
                continue
            self._yielded += 1
            if self._next_index >= len(self.selected_pages) and not self._pending:
                self._finish()
            else:
                self._fill()
            return PageResult(page=page_number, image=image)

    def _finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        with _PDFIUM_LOCK:
            self._document.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _page, future in self._pending:
            future.cancel()
        self._pending.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        with _PDFIUM_LOCK:
            self._document.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def scan_document(
    scanner: Scanner,
    document: DocumentInput,
    *,
    pages: Iterable[int] | None = None,
    workers: WorkerCount = 1,
    on_error: ErrorPolicy = "raise",
    roi: Roi | tuple[float, float, float, float] | None = None,
    diagnostics: bool = False,
) -> DocumentResult:
    stream = DocumentStream(
        scanner,
        document,
        pages=pages,
        workers=workers,
        on_error=on_error,
        roi=roi,
        diagnostics=diagnostics,
    )
    try:
        page_results = tuple(stream)
        elapsed_ms = stream.elapsed_ms
        errors = stream.errors
        complete = not errors and len(page_results) == len(stream.selected_pages)
        return DocumentResult(
            pages=page_results,
            errors=errors,
            complete=complete,
            elapsed_ms=elapsed_ms,
            metadata=stream.metadata,
        )
    finally:
        stream.close()


__all__ = [
    "DocumentInput",
    "DocumentStream",
    "ErrorPolicy",
    "RENDER_DPI",
    "WorkerCount",
    "scan_document",
]
