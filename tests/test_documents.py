# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
import zxingcpp
from PIL import Image

from codarascan import (
    ConfigurationError,
    DocumentError,
    DocumentRenderError,
    ImageResult,
    PageProcessingError,
    ScanMetadata,
    Scanner,
    documents,
)


def _qr(payload: str, *, scale: int = 8) -> Image.Image:
    barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
    pixels = np.asarray(
        zxingcpp.write_barcode_to_image(barcode, scale=scale, add_quiet_zones=True)
    ).copy()
    return Image.fromarray(pixels).convert("RGB")


def _pdf(path: Path, payloads: list[str]) -> Path:
    images = [_qr(payload) for payload in payloads]
    images[0].save(
        path,
        "PDF",
        save_all=True,
        append_images=images[1:],
        resolution=150,
    )
    return path


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["fast", "robust"])
def test_pdf_smoke_uses_pdfium_and_preserves_requested_order(tmp_path: Path, mode: str) -> None:
    path = _pdf(tmp_path / "ordered.pdf", ["PAGE-1", "PAGE-2", "PAGE-3"])
    scanner = Scanner(mode=mode, symbols="2d", formats=["qr-code"])
    result = scanner.scan_document(path, pages=[3, 1], workers=2)

    assert result.complete
    assert result.errors == ()
    assert [page.page for page in result.pages] == [3, 1]
    assert [page.image.symbols[0].text for page in result.pages] == [  # type: ignore[attr-defined]
        "PAGE-3",
        "PAGE-1",
    ]


def test_pdf_bytes_and_streaming_page_results(tmp_path: Path) -> None:
    path = _pdf(tmp_path / "bytes.pdf", ["ONE", "TWO"])
    scanner = Scanner(symbols="2d", formats=["qr-code"])
    with scanner.iter_document(path.read_bytes(), workers="auto") as stream:
        pages = list(stream)
        assert stream.complete
        assert stream.errors == ()
    assert [page.page for page in pages] == [1, 2]


def test_same_normalized_roi_is_page_relative_across_different_page_sizes(
    tmp_path: Path,
) -> None:
    qr = _qr("PAGE-RELATIVE-ROI", scale=5)
    page_sizes = [(900, 600), (600, 900)]
    pages = []
    expected_offsets = []
    for width, height in page_sizes:
        page = Image.new("RGB", (width, height), "white")
        left, top = int(width * 0.62), int(height * 0.62)
        page.paste(qr, (left, top))
        pages.append(page)
        expected_offsets.append((left, top))
    path = tmp_path / "page-relative-roi.pdf"
    pages[0].save(
        path,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150,
    )
    for page in pages:
        page.close()

    result = Scanner(mode="robust", symbols="2d", formats=["qr-code"]).scan_document(
        path,
        roi=(0.5, 0.5, 0.5, 0.5),
        workers=2,
    )
    assert result.complete
    assert [(page.image.width, page.image.height) for page in result.pages] == page_sizes
    for page, (left, top) in zip(result.pages, expected_offsets, strict=True):
        symbol = page.image.symbols[0]
        assert min(point.x for point in symbol.quad) >= left
        assert min(point.y for point in symbol.quad) >= top


def test_native_raster_strategy_and_rotated_page_render_fallback(tmp_path: Path) -> None:
    original = _pdf(tmp_path / "native.pdf", ["NATIVE"])
    with documents._open_pdf(str(original)) as document:
        rendered = documents._render_page(document, 1)
    assert rendered.strategy == "native-embedded-raster"

    rotated = tmp_path / "rotated.pdf"
    source = pdfium.PdfDocument(original)
    page = source[0]
    page.set_rotation(90)
    page.close()
    source.save(rotated)
    source.close()
    with documents._open_pdf(str(rotated)) as document:
        rendered = documents._render_page(document, 1)
    assert rendered.strategy == "pdfium-render-300-dpi"


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "many"])
def test_invalid_worker_values_fail_before_iteration(tmp_path: Path, workers: object) -> None:
    path = _pdf(tmp_path / "workers.pdf", ["WORKERS"])
    with pytest.raises(ConfigurationError):
        Scanner().iter_document(path, workers=workers)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "pages",
    [[], [0], [-1], [2], [1, 1], [True], [1.0], "1-2"],
)
def test_invalid_page_selections_fail_predictably(tmp_path: Path, pages: object) -> None:
    path = _pdf(tmp_path / "pages.pdf", ["PAGES"])
    with pytest.raises(ConfigurationError):
        Scanner().iter_document(path, pages=pages)  # type: ignore[arg-type]


def test_out_of_order_completion_is_yielded_in_requested_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pdf(tmp_path / "concurrency.pdf", ["1", "2", "3"])

    def render(_document: object, page: int) -> documents.RenderedPage:
        return documents.RenderedPage(np.full((8, 8), page, np.uint8), "test")

    metadata = ScanMetadata("0.1.0", "test", "0.1.0", "test", "python", "robust")

    def analyze(
        _self: Scanner,
        image: np.ndarray,
        **_options: object,
    ) -> ImageResult:
        marker = int(image[0, 0])
        time.sleep({1: 0.06, 2: 0.03, 3: 0.0}[marker])
        return ImageResult((), 8, 8, 1.0, metadata)

    monkeypatch.setattr(documents, "_render_page", render)
    monkeypatch.setattr(Scanner, "scan_image", analyze)
    pages = list(Scanner(mode="robust").iter_document(path, pages=[3, 1, 2], workers=3))
    assert [page.page for page in pages] == [3, 1, 2]


def test_collect_policy_keeps_successes_and_typed_failed_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pdf(tmp_path / "partial.pdf", ["ONE", "TWO", "THREE"])
    original = documents._render_page

    def fail_second(document: pdfium.PdfDocument, page: int) -> documents.RenderedPage:
        if page == 2:
            raise DocumentRenderError("injected render failure", context={"page": 2})
        return original(document, page)

    monkeypatch.setattr(documents, "_render_page", fail_second)
    scanner = Scanner(mode="robust", symbols="2d", formats=["qr-code"])
    result = scanner.scan_document(path, workers=2, on_error="collect")
    assert not result.complete
    assert [page.page for page in result.pages] == [1, 3]
    assert len(result.errors) == 1
    assert result.errors[0].page == 2
    assert result.errors[0].code == DocumentRenderError.code


def test_raise_policy_is_fail_fast_for_page_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pdf(tmp_path / "raise.pdf", ["ONE", "TWO"])

    def fail(_document: object, page: int) -> documents.RenderedPage:
        raise DocumentRenderError("injected", context={"page": page})

    monkeypatch.setattr(documents, "_render_page", fail)
    with pytest.raises(DocumentRenderError):
        Scanner().scan_document(path)


def test_analysis_failures_become_page_processing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pdf(tmp_path / "analysis.pdf", ["ONE"])

    def fail(_self: Scanner, _image: np.ndarray, **_options: object) -> ImageResult:
        raise RuntimeError("private detail")

    monkeypatch.setattr(Scanner, "scan_image", fail)
    with pytest.raises(PageProcessingError) as caught:
        Scanner().scan_document(path)
    assert caught.value.context["page"] == 1


def test_close_stops_scheduling_new_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _pdf(tmp_path / "cancel.pdf", [str(index) for index in range(1, 7)])
    submitted: list[int] = []
    lock = threading.Lock()

    def render(_document: object, page: int) -> documents.RenderedPage:
        return documents.RenderedPage(np.full((8, 8), page, np.uint8), "test")

    metadata = ScanMetadata("0.1.0", "test", "0.1.0", "test", "python", "fast")

    def analyze(_self: Scanner, image: np.ndarray, **_options: object) -> ImageResult:
        with lock:
            submitted.append(int(image[0, 0]))
        time.sleep(0.02)
        return ImageResult((), 8, 8, 1.0, metadata)

    monkeypatch.setattr(documents, "_render_page", render)
    monkeypatch.setattr(Scanner, "scan_image", analyze)
    stream = Scanner().iter_document(path, workers=2)
    assert next(stream).page == 1
    stream.close()
    assert len(submitted) <= 3
    with pytest.raises(StopIteration):
        next(stream)


@pytest.mark.parametrize(
    "value",
    [b"", b"not a pdf", object()],
)
def test_invalid_document_inputs_are_typed(value: object) -> None:
    with pytest.raises(DocumentError):
        Scanner().scan_document(value)  # type: ignore[arg-type]


def test_invalid_roi_fails_before_page_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pdf(tmp_path / "roi.pdf", ["ROI"])
    called = False

    def render(_document: object, _page: int) -> documents.RenderedPage:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(documents, "_render_page", render)
    with pytest.raises(ConfigurationError):
        Scanner().scan_document(path, roi=(0.0, 0.0, 0.0, 1.0))
    assert not called
