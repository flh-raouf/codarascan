#!/usr/bin/env python3
"""Run the native structure-tensor 1D localizer on images or PDFs.

The measured engine begins only after image loading, grayscale conversion, and
detector resizing have completed. Those preparation costs remain visible in a
separate section of the JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np

from pipeline import (
    Detection,
    PreparedPage,
    TensorConfig,
    locate_prepared,
    prepare_page,
    profile_config,
    rectify,
)


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_OPENCV_THREADS = max(1, cv2.getNumThreads())


@dataclass(frozen=True)
class PageInput:
    page: int
    path: Path
    source_mode: str


@dataclass(frozen=True)
class PreparedInput:
    page: int
    path: Path
    source_mode: str
    prepared: PreparedPage
    load_seconds: float


def parse_pages(value: str) -> list[int]:
    pages: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise argparse.ArgumentTypeError(f"invalid page range: {item}")
            pages.update(range(start, end + 1))
        else:
            page = int(item)
            if page < 1:
                raise argparse.ArgumentTypeError("page numbers begin at 1")
            pages.add(page)
    if not pages:
        raise argparse.ArgumentTypeError("empty page selection")
    return sorted(pages)


def pdf_page_count(path: Path) -> int:
    completed = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)", completed.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("unable to determine PDF page count")
    return int(match.group(1))


def pdf_image_rows(path: Path) -> list[dict[str, int]]:
    completed = subprocess.run(
        ["pdfimages", "-list", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        rows.append(
            {
                "page": int(parts[0]),
                "number": int(parts[1]),
                "width": int(parts[3]),
                "height": int(parts[4]),
            }
        )
    return rows


def image_number(path: Path) -> int:
    match = re.search(r"-(\d+)(?:\.[^.]+)?$", path.name)
    return int(match.group(1)) if match else 10**9


def extract_pdf(
    path: Path,
    selected: Sequence[int],
    work: Path,
    dpi: int,
) -> list[PageInput]:
    extraction = work / ".native"
    pages_directory = work / ".pages"
    extraction.mkdir(parents=True, exist_ok=True)
    pages_directory.mkdir(parents=True, exist_ok=True)
    first, last = min(selected), max(selected)
    rows = [
        row
        for row in pdf_image_rows(path)
        if first <= row["page"] <= last
    ]
    prefix = extraction / "image"
    try:
        subprocess.run(
            [
                "pdfimages",
                "-f",
                str(first),
                "-l",
                str(last),
                "-j",
                str(path),
                str(prefix),
            ],
            check=True,
            capture_output=True,
        )
        files = sorted(
            (item for item in extraction.glob("image-*") if item.is_file()),
            key=image_number,
        )
        if len(files) != len(rows):
            raise RuntimeError("native PDF image manifest mismatch")
        best: dict[int, tuple[int, Path]] = {}
        for row, image_path in zip(rows, files):
            area = row["width"] * row["height"]
            incumbent = best.get(row["page"])
            if (
                row["page"] in selected
                and area >= 1_000_000
                and (incumbent is None or area > incumbent[0])
            ):
                best[row["page"]] = (area, image_path)
        output: list[PageInput] = []
        for page in selected:
            if page in best:
                source = best[page][1]
                destination = (
                    pages_directory
                    / f"page-{page:04d}{source.suffix.lower()}"
                )
                shutil.copy2(source, destination)
                output.append(
                    PageInput(page, destination, "native-embedded-raster")
                )
                continue
            prefix_path = pages_directory / f"render-{page:04d}"
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-r",
                    str(dpi),
                    "-gray",
                    "-jpeg",
                    "-singlefile",
                    str(path),
                    str(prefix_path),
                ],
                check=True,
                capture_output=True,
            )
            output.append(
                PageInput(
                    page,
                    prefix_path.with_suffix(".jpg"),
                    f"rendered-{dpi}-dpi",
                )
            )
        return output
    finally:
        shutil.rmtree(extraction, ignore_errors=True)


def collect_pages(
    input_path: Path,
    selected: list[int] | None,
    work: Path,
    dpi: int,
) -> list[PageInput]:
    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        pages = selected or list(range(1, len(images) + 1))
        if any(page > len(images) for page in pages):
            raise ValueError("selected page exceeds image count")
        return [
            PageInput(page, images[page - 1], "image-directory")
            for page in pages
        ]
    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        if selected and selected != [1]:
            raise ValueError("a single image only has page 1")
        return [PageInput(1, input_path, "image-file")]
    if input_path.suffix.lower() != ".pdf":
        raise ValueError("input must be a PDF, an image, or an image directory")
    count = pdf_page_count(input_path)
    pages = selected or list(range(1, count + 1))
    if any(page > count for page in pages):
        raise ValueError(f"selected page exceeds PDF page count {count}")
    return extract_pdf(input_path, pages, work, dpi)


def effective_workers(requested: int, count: int) -> int:
    if requested > 0:
        return max(1, min(requested, count))
    return max(1, min(8, count, os.cpu_count() or 1))


def prepare_input(page: PageInput, config: TensorConfig) -> PreparedInput:
    started = perf_counter()
    image = cv2.imread(str(page.path), cv2.IMREAD_GRAYSCALE)
    load_seconds = perf_counter() - started
    if image is None:
        raise RuntimeError(f"unable to read image: {page.path}")
    prepared = prepare_page(image, config)
    return PreparedInput(
        page.page,
        page.path,
        page.source_mode,
        prepared,
        load_seconds,
    )


def run_engine(
    page: PreparedInput,
    config: TensorConfig,
) -> tuple[dict[str, Any], list[Detection], Path]:
    result = locate_prepared(page.prepared, config)
    detections = result.detections
    payload = {
        "page": page.page,
        "source_page": page.page,
        "source_image": str(page.path),
        "source_mode": page.source_mode,
        "image_size": {
            "width": int(page.prepared.native_gray.shape[1]),
            "height": int(page.prepared.native_gray.shape[0]),
        },
        "count": len(detections),
        "detections": [
            {
                "quad": np.asarray(item.quad, np.float32).tolist(),
                "kind": "linear",
                "status": "localized",
                "confidence": float(item.confidence),
                "source": item.source,
                "evidence": {
                    key: float(value)
                    for key, value in item.metrics.items()
                },
            }
            for item in detections
        ],
        "diagnostics": result.diagnostics,
        "timings": {
            "load_seconds": page.load_seconds,
            "preparation_seconds": page.prepared.preparation_seconds,
            **result.timings,
            "artifact_seconds": 0.0,
        },
        "overlay": None,
        "crops": [],
    }
    return payload, detections, page.path


def annotate(gray: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index, detection in enumerate(detections, start=1):
        quad = np.rint(detection.quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [quad], True, (0, 190, 0), 4, cv2.LINE_AA)
        anchor = tuple(int(value) for value in quad.reshape(-1, 2)[0])
        cv2.putText(
            image,
            f"{index}: linear",
            (anchor[0], max(24, anchor[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 120, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def generate_artifacts(
    payload: dict[str, Any],
    detections: Sequence[Detection],
    source: Path,
    work: Path,
    overlays: bool,
    crops: bool,
) -> None:
    started = perf_counter()
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"unable to reload image for artifacts: {source}")
    page = int(payload["page"])
    if overlays:
        overlay_path = work / "overlays" / f"page-{page:04d}.jpg"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(overlay_path),
            annotate(gray, detections),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        payload["overlay"] = str(overlay_path.relative_to(work))
    if crops:
        crop_directory = work / "crops"
        crop_directory.mkdir(parents=True, exist_ok=True)
        for index, detection in enumerate(detections, start=1):
            crop_path = crop_directory / f"page-{page:04d}-{index:03d}.png"
            cv2.imwrite(str(crop_path), rectify(gray, detection.quad))
            payload["crops"].append(str(crop_path.relative_to(work)))
    payload["timings"]["artifact_seconds"] = perf_counter() - started


def summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(median(values)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def run(
    input_path: Path,
    output: Path,
    *,
    profile: str,
    pages: list[int] | None,
    workers: int,
    render_dpi: int,
    work_size: int | None,
    tile: int | None,
    overlays: bool,
    crops: bool,
    periodic_rescue: bool,
    overwrite: bool,
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output}")
    resolved_profile = (
        "document"
        if profile == "auto" and input_path.suffix.lower() == ".pdf"
        else "general"
        if profile == "auto"
        else profile
    )
    config = profile_config(
        resolved_profile,
        work_size=work_size,
        tile=tile,
    )
    if not periodic_rescue:
        config = replace(
            config,
            retain_low_transition_rescue=False,
            split_periodic_bands=False,
        )
    resolved_tile = config.aggregation_windows[0]
    work = output.parent / f".{output.name}.work-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True)
    overall_started = perf_counter()
    try:
        extraction_started = perf_counter()
        inputs = collect_pages(input_path, pages, work, render_dpi)
        extraction_seconds = perf_counter() - extraction_started
        worker_count = effective_workers(workers, len(inputs))
        cv2.setNumThreads(
            1 if worker_count > 1 else DEFAULT_OPENCV_THREADS
        )

        # Preparation is intentionally complete before the engine stopwatch.
        preparation_wall_started = perf_counter()
        if worker_count == 1:
            prepared = [prepare_input(page, config) for page in inputs]
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="sttg-prepare",
            ) as executor:
                prepared = list(
                    executor.map(
                        lambda page: prepare_input(page, config),
                        inputs,
                    )
                )
        preparation_wall_seconds = perf_counter() - preparation_wall_started

        engine_wall_started = perf_counter()
        if worker_count == 1:
            processed = [run_engine(page, config) for page in prepared]
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="sttg-engine",
            ) as executor:
                processed = list(
                    executor.map(
                        lambda page: run_engine(page, config),
                        prepared,
                    )
                )
        engine_wall_seconds = perf_counter() - engine_wall_started

        artifact_wall_seconds = 0.0
        if overlays or crops:
            artifact_started = perf_counter()
            if worker_count == 1:
                for payload, detections, source in processed:
                    generate_artifacts(
                        payload,
                        detections,
                        source,
                        work,
                        overlays,
                        crops,
                    )
            else:
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="sttg-artifact",
                ) as executor:
                    futures = [
                        executor.submit(
                            generate_artifacts,
                            payload,
                            detections,
                            source,
                            work,
                            overlays,
                            crops,
                        )
                        for payload, detections, source in processed
                    ]
                    for future in futures:
                        future.result()
            artifact_wall_seconds = perf_counter() - artifact_started

        payloads = sorted(
            (item[0] for item in processed),
            key=lambda item: item["page"],
        )
        if input_path.suffix.lower() == ".pdf":
            for page in payloads:
                page["source_image"] = str(input_path)
        engine_times_ms = [
            1000.0 * float(page["timings"]["engine_seconds"])
            for page in payloads
        ]
        preparation_times_ms = [
            1000.0
            * (
                float(page["timings"]["load_seconds"])
                + float(page["timings"]["preparation_seconds"])
            )
            for page in payloads
        ]
        result = {
            "input": str(input_path),
            "method": "native context-aware sparse structure tensor plus native-pixel physical verification",
            "decoding_performed": False,
            "zxing_used": False,
            "gpu_used": False,
            "configuration": {
                "profile": resolved_profile,
                "work_size": config.work_size,
                "tile": resolved_tile,
                "workers": worker_count,
                "render_dpi": render_dpi,
                "save_overlays": overlays,
                "save_crops": crops,
            },
            "summary": {
                "pages": len(payloads),
                "localized": sum(int(page["count"]) for page in payloads),
                "engine_wall_seconds": engine_wall_seconds,
                "engine_throughput_ms_per_page": (
                    1000.0 * engine_wall_seconds / max(1, len(payloads))
                ),
                "engine_latency_ms": summary(engine_times_ms),
                "preparation_wall_seconds": preparation_wall_seconds,
                "preparation_ms": summary(preparation_times_ms),
                "extraction_seconds": extraction_seconds,
                "artifact_wall_seconds": artifact_wall_seconds,
                "overall_wall_seconds": perf_counter() - overall_started,
            },
            "pages": payloads,
        }
        shutil.rmtree(work / ".pages", ignore_errors=True)
        (work / "detections.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        work.rename(output)
        for page in payloads:
            print(
                f"page {int(page['page']):04d}: {int(page['count'])} localized, "
                f"engine={1000.0 * float(page['timings']['engine_seconds']):.3f}ms",
                flush=True,
            )
        print(
            f"summary: {result['summary']['localized']} localized across "
            f"{len(payloads)} pages; engine wall={engine_wall_seconds:.4f}s "
            f"({result['summary']['engine_throughput_ms_per_page']:.3f}ms/page), "
            f"preparation={preparation_wall_seconds:.4f}s, "
            f"artifacts={artifact_wall_seconds:.4f}s",
            flush=True,
        )
        print(f"wrote {output / 'detections.json'}", flush=True)
        return result
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("8-structure-tensor-linear-localization/output/run"),
    )
    parser.add_argument(
        "--profile",
        choices=("auto", "general", "document"),
        default="auto",
        help="auto uses document for PDFs and general for images",
    )
    parser.add_argument("--pages", type=parse_pages)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--work-size", type=int)
    parser.add_argument("--tile", type=int)
    parser.add_argument("--overlays", action="store_true")
    parser.add_argument("--crops", action="store_true")
    parser.add_argument(
        "--no-periodic-rescue",
        action="store_true",
        help="disable document weak-parent retention and periodic-band splitting",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        run(
            arguments.input,
            arguments.output,
            profile=arguments.profile,
            pages=arguments.pages,
            workers=arguments.workers,
            render_dpi=arguments.render_dpi,
            work_size=arguments.work_size,
            tile=arguments.tile,
            overlays=arguments.overlays,
            crops=arguments.crops,
            periodic_rescue=not arguments.no_periodic_rescue,
            overwrite=arguments.overwrite,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
