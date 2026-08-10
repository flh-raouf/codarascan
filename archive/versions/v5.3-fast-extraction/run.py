#!/usr/bin/env python3
"""Run the selected bounded fast engine on images, directories, or PDFs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import zxingcpp

from fast_direct import direct_decode


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_pages(value: str | None) -> list[int] | None:
    if not value:
        return None
    pages = sorted({int(item) for item in value.split(",") if item.strip()})
    if not pages or min(pages) < 1:
        raise ValueError("pages must be positive comma-separated numbers")
    return pages


def selected_formats(value: str, kinds: str) -> Any:
    if value.strip():
        requested = zxingcpp.barcode_formats_from_str(value.replace(",", "|"))
    elif kinds == "linear":
        requested = zxingcpp.barcode_formats_from_str("AllLinear")
    elif kinds == "2d":
        requested = zxingcpp.barcode_formats_from_str("AllMatrix")
    else:
        requested = zxingcpp.barcode_formats_from_str("AllReadable")
    return requested


def collect_images(
    source: Path,
    work: Path,
    *,
    pages: list[int] | None,
    dpi: int,
) -> tuple[list[tuple[int, Path, str]], float]:
    started = perf_counter()
    if source.is_dir():
        images = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        selected = [
            (index, path, "input-image")
            for index, path in enumerate(images, start=1)
            if pages is None or index in pages
        ]
        return selected, perf_counter() - started
    if source.suffix.lower() != ".pdf":
        return [(1, source, "input-image")], perf_counter() - started

    raster_dir = work / "pages"
    raster_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "pdftoppm",
        "-gray",
        "-jpeg",
        "-jpegopt",
        "quality=95,optimize=y",
        "-r",
        str(dpi),
    ]
    if pages:
        # Render selected pages individually so sparse selections do not raster
        # every intervening page.
        output: list[tuple[int, Path, str]] = []
        for page in pages:
            prefix = raster_dir / f"page-{page:04d}"
            subprocess.run(
                [
                    *command,
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-singlefile",
                    str(source),
                    str(prefix),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            output.append((page, prefix.with_suffix(".jpg"), "rendered-pdf"))
        return output, perf_counter() - started

    prefix = raster_dir / "page"
    subprocess.run(
        [*command, str(source), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    images = sorted(raster_dir.glob("page-*.jpg"))
    return [
        (index, path, "rendered-pdf")
        for index, path in enumerate(images, start=1)
    ], perf_counter() - started


def prediction_json(value: dict[str, Any]) -> dict[str, Any]:
    quad = np.asarray(value["quad"], np.float32)
    return {
        "status": "decoded",
        "kind": value["kind"],
        "text": value["text"],
        "format": value["format"],
        "confidence": value["confidence"],
        "sources": value["sources"],
        "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in quad],
    }


def annotate(image: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for prediction in predictions:
        quad = np.rint(np.asarray(prediction["quad"], np.float32)).astype(np.int32)
        cv2.polylines(color, [quad], True, (24, 176, 70), 3, cv2.LINE_AA)
        anchor = tuple(int(value) for value in quad[0])
        cv2.putText(
            color,
            str(prediction["format"]),
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (24, 176, 70),
            2,
            cv2.LINE_AA,
        )
    return color


def process_page(
    item: tuple[int, Path, str],
    *,
    formats: Any,
    overlay_dir: Path | None,
) -> dict[str, Any]:
    page, path, mode = item
    started = perf_counter()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"unable to read page image: {path}")
    decode_started = perf_counter()
    predictions = direct_decode(image, formats=formats)
    decode_seconds = perf_counter() - decode_started
    artifact_seconds = 0.0
    overlay: str | None = None
    if overlay_dir is not None:
        artifact_started = perf_counter()
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"page-{page:04d}.png"
        if not cv2.imwrite(str(overlay_path), annotate(image, predictions)):
            raise RuntimeError(f"unable to write overlay: {overlay_path}")
        artifact_seconds = perf_counter() - artifact_started
        overlay = overlay_path.name
    return {
        "page": page,
        "source_image": str(path),
        "source_mode": mode,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "decoded_count": len(predictions),
        "results": [prediction_json(value) for value in predictions],
        "timings": {
            "decode_seconds": round(decode_seconds, 6),
            "processing_seconds": round(perf_counter() - started - artifact_seconds, 6),
            "artifact_seconds": round(artifact_seconds, 6),
        },
        **({"overlay": overlay} if overlay else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast valid-only barcode extraction for clear, large symbols"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kinds", choices=("all", "linear", "2d"), default="all")
    parser.add_argument("--formats", default="", help="Optional comma-separated ZXing formats")
    parser.add_argument("--pages")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overlays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    source = arguments.input.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    run_started = perf_counter()
    try:
        pages, extraction_seconds = collect_images(
            source,
            temporary,
            pages=parse_pages(arguments.pages),
            dpi=arguments.dpi,
        )
        workers = (
            max(1, min(arguments.workers, len(pages)))
            if arguments.workers > 0
            else max(1, min(8, os.cpu_count() or 1, len(pages)))
        )
        formats = selected_formats(arguments.formats, arguments.kinds)
        processing_started = perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(
                executor.map(
                    lambda item: process_page(
                        item,
                        formats=formats,
                        overlay_dir=temporary / "overlays" if arguments.overlays else None,
                    ),
                    pages,
                )
            )
        processing_seconds = perf_counter() - processing_started
        records.sort(key=lambda value: value["page"])
        latencies = [1000.0 * value["timings"]["processing_seconds"] for value in records]
        report = {
            "method": "5.3-fast-valid-only-page-pass",
            "configuration": {
                "kinds": arguments.kinds,
                "formats": arguments.formats or "all-readable",
                "try_rotate": True,
                "try_downscale": False,
                "try_invert": False,
                "recovery_stages": 0,
                "workers": workers,
            },
            "started_at": datetime.now(UTC).isoformat(),
            "pages": records,
            "summary": {
                "pages": len(records),
                "decoded": sum(value["decoded_count"] for value in records),
                "extraction_seconds": round(extraction_seconds, 6),
                "processing_seconds": round(processing_seconds, 6),
                "throughput_ms_per_page": round(
                    1000.0 * processing_seconds / max(1, len(records)),
                    3,
                ),
                "median_page_task_ms": round(median(latencies), 3) if latencies else None,
                "total_seconds": round(perf_counter() - run_started, 6),
            },
        }
        (temporary / "detections.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    summary = report["summary"]
    print(
        f"{summary['decoded']} decoded across {summary['pages']} pages; "
        f"processing={summary['processing_seconds']:.3f}s "
        f"({summary['throughput_ms_per_page']:.1f} ms/page throughput), "
        f"extraction={summary['extraction_seconds']:.3f}s"
    )
    print(f"wrote {output / 'detections.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
