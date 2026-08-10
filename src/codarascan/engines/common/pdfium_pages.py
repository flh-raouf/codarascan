# SPDX-License-Identifier: Apache-2.0
"""PDFium-backed compatibility helpers for legacy engine batch entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2

from codarascan import documents


def pdf_page_count(path: Path) -> int:
    document = documents._open_pdf(str(path))
    try:
        return len(document)
    finally:
        with documents._PDFIUM_LOCK:
            document.close()


def write_pdf_page(
    path: Path,
    page_number: int,
    destination: Path,
    dpi: int,
    *,
    native_only: bool = False,
) -> str | None:
    document = documents._open_pdf(str(path))
    try:
        if not 1 <= page_number <= len(document):
            raise ValueError(f"page {page_number} exceeds PDF page count {len(document)}")
        with documents._PDFIUM_LOCK:
            page = document[page_number - 1]
            try:
                pixels = documents._native_raster(page)
                strategy = "native-embedded-raster"
                if pixels is None:
                    if native_only:
                        return None
                    bitmap = page.render(
                        scale=dpi / 72.0,
                        optimize_mode="print",
                        maybe_alpha=True,
                    )
                    pixels = documents._copy_bitmap(bitmap)
                    strategy = f"pdfium-render-{dpi}-dpi"
            finally:
                page.close()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), pixels):
            raise RuntimeError(f"unable to write rendered PDF page: {destination}")
        return strategy
    finally:
        with documents._PDFIUM_LOCK:
            document.close()


def materialize_pdf_pages(
    path: Path,
    selected: Sequence[int],
    pages_directory: Path,
    dpi: int,
) -> list[tuple[int, Path, str]]:
    count = pdf_page_count(path)
    if any(page < 1 or page > count for page in selected):
        raise ValueError(f"selected page exceeds PDF page count {count}")
    output: list[tuple[int, Path, str]] = []
    for page_number in selected:
        destination = pages_directory / f"page-{page_number:04d}.png"
        strategy = write_pdf_page(path, page_number, destination, dpi)
        assert strategy is not None
        output.append((page_number, destination, strategy))
    return output


__all__ = ["materialize_pdf_pages", "pdf_page_count", "write_pdf_page"]
