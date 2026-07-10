#!/usr/bin/env python3
"""Hybrid, candidate-driven barcode pipeline.

The pipeline keeps localization and decoding separate:

* extract a native PDF raster when possible;
* gather whole-page semantic anchors from decoders;
* locate 1-D symbols using the deterministic OpenCV implementation;
* rectify each oriented proposal and try a bounded decoder portfolio;
* optionally run a regional matrix-only recovery sweep;
* validate, merge, annotate, and report timings and unresolved proposals.

It deliberately contains no expected barcode counts and never fabricates a
decoded value. An unresolved but structurally credible proposal is retained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import zxingcpp

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deterministic_barcode_locator.barcode_locator import (  # noqa: E402
    Detection as LocatorDetection,
    locate_page,
    order_quad,
    quad_overlap,
    rectify_quad,
)

try:
    from dynamsoft_barcode_reader_bundle import CaptureVisionRouter, EnumPresetTemplate, LicenseManager
except ImportError:
    CaptureVisionRouter = None
    EnumPresetTemplate = None
    LicenseManager = None


LINEAR_FORMATS = zxingcpp.barcode_formats_from_str("Code39,Code128")
MATRIX_FORMATS = zxingcpp.barcode_formats_from_str("DataMatrix,QRCode")
ALL_FORMATS = zxingcpp.barcode_formats_from_str("Code39,Code128,DataMatrix,QRCode")


def is_matrix_format(name: str | None) -> bool:
    normalized = (name or "").replace("_", " ").lower()
    return any(value in normalized for value in ("data matrix", "datamatrix", "qr code", "qrcode", "aztec"))


def normalize_format(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.replace("_", " ").replace("EXTENDED", "").strip().lower()
    if "data" in normalized and "matrix" in normalized:
        return "Data Matrix"
    if "code" in normalized and "128" in normalized:
        return "Code 128"
    if "code" in normalized and "39" in normalized:
        return "Code 39"
    if "qr" in normalized:
        return "QR Code"
    return " ".join(word.capitalize() for word in normalized.split())


@dataclass
class PipelineConfig:
    output: Path
    render_dpi: int = 300
    pages: list[int] | None = None
    angle_step: int = 0
    minimum_score: float = 0.52
    minimum_linear_length: float = 100.0
    tensor_fallback: bool = False
    engine: str = "auto"
    matrix_sweep: bool = True
    matrix_tile_size: int = 700
    matrix_tile_stride: int = 400
    save_crops: bool = True
    include_unresolved: bool = True


@dataclass
class Result:
    quad: np.ndarray
    text: str | None = None
    format: str | None = None
    confidence: float = 0.0
    status: str = "unresolved"
    sources: set[str] = field(default_factory=set)
    attempts: list[str] = field(default_factory=list)
    structural_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def decoded(self) -> bool:
        return bool(self.text is not None and self.format)

    @property
    def kind(self) -> str:
        return "matrix" if is_matrix_format(self.format) else "linear"

    def center(self) -> np.ndarray:
        return np.asarray(self.quad, dtype=np.float32).mean(axis=0)

    def to_json(self) -> dict[str, Any]:
        quad = np.asarray(self.quad, dtype=np.float32)
        minimum = quad.min(axis=0)
        maximum = quad.max(axis=0)
        return {
            "status": self.status,
            "kind": self.kind if self.decoded else "candidate",
            "decoded": self.decoded,
            "text": self.text,
            "format": self.format,
            "confidence": round(float(self.confidence), 4),
            "sources": sorted(self.sources),
            "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in quad],
            "aabb": {
                "x": round(float(minimum[0]), 2),
                "y": round(float(minimum[1]), 2),
                "width": round(float(maximum[0] - minimum[0]), 2),
                "height": round(float(maximum[1] - minimum[1]), 2),
            },
            "attempts": self.attempts,
            "structural_metrics": {key: round(float(value), 4) for key, value in self.structural_metrics.items()},
        }


class DynamsoftDecoder:
    """Small adapter around the optional commercial decoder."""

    def __init__(self, license_key: str | None):
        self.available = False
        self.router = None
        self.error: str | None = None
        if not license_key:
            self.error = "DYNAMSOFT_LICENSE_KEY is not set"
            return
        if CaptureVisionRouter is None or LicenseManager is None:
            self.error = "dynamsoft-barcode-reader-bundle is not installed"
            return
        code, message = LicenseManager.init_license(license_key)
        if code not in (0, 1):
            self.error = f"license initialization failed ({code}): {message}"
            return
        self.router = CaptureVisionRouter()
        self.available = True

    def read(self, image: np.ndarray) -> list[dict[str, Any]]:
        if not self.available or self.router is None:
            return []
        descriptor, name = tempfile.mkstemp(suffix=".png")
        os.close(descriptor)
        try:
            cv2.imwrite(name, np.ascontiguousarray(image))
            captured = self.router.capture(name, EnumPresetTemplate.PT_READ_BARCODES.value)
            if captured.get_error_code() != 0:
                return []
            decoded = captured.get_decoded_barcodes_result()
            items = [] if decoded is None else decoded.get_items() or []
            output: list[dict[str, Any]] = []
            for item in items:
                location = item.get_location()
                output.append(
                    {
                        "text": item.get_text(),
                        "format": normalize_format(item.get_format_string()),
                        "confidence": float(item.get_confidence() or 0) / 100.0,
                        "quad": np.asarray([[point.x, point.y] for point in location.points], dtype=np.float32),
                    }
                )
            return output
        finally:
            Path(name).unlink(missing_ok=True)


def zxing_quad(barcode: Any, scale: float = 1.0, offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    position = barcode.position
    ox, oy = offset
    return np.asarray(
        [
            [position.top_left.x / scale + ox, position.top_left.y / scale + oy],
            [position.top_right.x / scale + ox, position.top_right.y / scale + oy],
            [position.bottom_right.x / scale + ox, position.bottom_right.y / scale + oy],
            [position.bottom_left.x / scale + ox, position.bottom_left.y / scale + oy],
        ],
        dtype=np.float32,
    )


def zxing_read(
    image: np.ndarray,
    formats: Any = ALL_FORMATS,
    binarizer: Any = zxingcpp.Binarizer.LocalAverage,
    try_downscale: bool = False,
    return_errors: bool = False,
) -> list[Any]:
    try:
        return list(
            zxingcpp.read_barcodes(
                image,
                formats=formats,
                try_rotate=True,
                try_downscale=try_downscale,
                try_invert=True,
                binarizer=binarizer,
                return_errors=return_errors,
            )
        )
    except Exception:
        return []


def parse_page_selection(value: str) -> list[int]:
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    if not pages or min(pages) < 1:
        raise argparse.ArgumentTypeError("pages must contain positive page numbers")
    return sorted(pages)


def pdf_page_count(pdf: Path) -> int:
    if shutil.which("pdfinfo") is None:
        raise RuntimeError("pdfinfo is required for PDF input")
    completed = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True)
    for line in completed.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("pdfinfo did not report a page count")


def extract_native_page(pdf: Path, page_number: int, output_dir: Path) -> Path | None:
    """Extract the largest embedded raster on one PDF page.

    Single-scan pages retain their original JPEG samples. Pages without a
    dominant embedded image fall back to ordinary rendering.
    """
    if shutil.which("pdfimages") is None:
        return None
    temporary = output_dir / "native-work" / f"page-{page_number:04d}"
    temporary.mkdir(parents=True, exist_ok=True)
    prefix = temporary / "image"
    subprocess.run(
        ["pdfimages", "-f", str(page_number), "-l", str(page_number), "-j", str(pdf), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    candidates: list[tuple[int, Path]] = []
    for path in temporary.glob("image-*"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            candidates.append((int(image.shape[0] * image.shape[1]), path))
    if not candidates:
        return None
    area, source = max(candidates, key=lambda item: item[0])
    # Reject tiny logos/stamps on an otherwise vector page.
    if area < 1_000_000:
        return None
    suffix = source.suffix.lower() if source.suffix else ".png"
    destination = output_dir / "pages" / f"page-{page_number:04d}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def render_pdf_page(pdf: Path, page_number: int, output_dir: Path, dpi: int) -> Path:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm is required for PDF input")
    destination = output_dir / "pages" / f"page-{page_number:04d}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = destination.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-r",
            str(dpi),
            "-png",
            str(pdf),
            str(prefix),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return destination


def collect_pages(input_path: Path, config: PipelineConfig) -> list[tuple[int, Path, str]]:
    if input_path.suffix.lower() == ".pdf":
        count = pdf_page_count(input_path)
        selected = config.pages or list(range(1, count + 1))
        pages: list[tuple[int, Path, str]] = []
        for page_number in selected:
            if page_number > count:
                raise ValueError(f"page {page_number} exceeds PDF page count {count}")
            extracted = extract_native_page(input_path, page_number, config.output)
            if extracted is not None:
                pages.append((page_number, extracted, "native-embedded-raster"))
            else:
                pages.append((page_number, render_pdf_page(input_path, page_number, config.output, config.render_dpi), "rendered"))
        shutil.rmtree(config.output / "native-work", ignore_errors=True)
        return pages
    if input_path.is_dir():
        supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        paths = sorted(path for path in input_path.iterdir() if path.suffix.lower() in supported)
        return [(index, path, "input-image") for index, path in enumerate(paths, start=1)]
    return [(1, input_path, "input-image")]


def whole_page_results(image: np.ndarray, dynamsoft: DynamsoftDecoder | None) -> list[Result]:
    output: list[Result] = []
    for barcode in zxing_read(image, try_downscale=False):
        if not barcode.valid:
            continue
        output.append(
            Result(
                quad=zxing_quad(barcode),
                text=str(barcode.text),
                format=normalize_format(str(barcode.format)),
                confidence=1.0,
                status="decoded",
                sources={"zxing:whole-page"},
            )
        )
    if dynamsoft and dynamsoft.available:
        for item in dynamsoft.read(image):
            output.append(
                Result(
                    quad=item["quad"],
                    text=item["text"],
                    format=item["format"],
                    confidence=item["confidence"],
                    status="decoded",
                    sources={"dynamsoft:whole-page"},
                )
            )
    return output


def candidate_variants(image: np.ndarray, detection: LocatorDetection) -> Iterable[tuple[str, np.ndarray, Any]]:
    """A deliberately small, ordered set of high-value crop variants."""
    recipes = [
        ("native-p08-s10", 0.08, 0.10, 1.0, cv2.INTER_CUBIC, None, zxingcpp.Binarizer.LocalAverage),
        ("cubic15-p08-s10", 0.08, 0.10, 1.5, cv2.INTER_CUBIC, None, zxingcpp.Binarizer.LocalAverage),
        ("cubic20-p18-s18", 0.18, 0.18, 2.0, cv2.INTER_CUBIC, None, zxingcpp.Binarizer.LocalAverage),
        ("nearest15-fixed", 0.08, 0.05, 1.5, cv2.INTER_NEAREST, None, zxingcpp.Binarizer.FixedThreshold),
        ("lanczos30-wide", 0.40, 0.05, 3.0, cv2.INTER_LANCZOS4, None, zxingcpp.Binarizer.LocalAverage),
        ("clahe15", 0.03, 0.05, 1.5, cv2.INTER_CUBIC, "clahe", zxingcpp.Binarizer.LocalAverage),
        ("otsu15", 0.15, 0.05, 1.5, cv2.INTER_NEAREST, "otsu", zxingcpp.Binarizer.LocalAverage),
    ]
    for name, pad_long, pad_short, scale, interpolation, preprocessing, binarizer in recipes:
        crop = rectify_quad(image, detection.quad, pad_long=pad_long, pad_short=pad_short)
        if scale != 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interpolation)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if preprocessing == "clahe":
            crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        elif preprocessing == "otsu":
            crop = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        yield name, crop, binarizer


def decode_candidate(
    image: np.ndarray,
    detection: LocatorDetection,
    dynamsoft: DynamsoftDecoder | None,
) -> Result:
    result = Result(
        quad=np.asarray(detection.quad, dtype=np.float32),
        confidence=max(float(detection.proposal_score), float(detection.structural_score)),
        status="unresolved",
        sources=set(detection.sources) | {"deterministic:linear-proposal"},
        structural_metrics=dict(detection.metrics),
    )

    # The commercial decoder gets the clean native crop first. This recovers
    # undersampled symbols without spending time on a large variant sweep.
    if dynamsoft and dynamsoft.available:
        crop = rectify_quad(image, detection.quad, pad_long=0.12, pad_short=0.18)
        result.attempts.append("dynamsoft:native-p12-s18")
        reads = [item for item in dynamsoft.read(crop) if not is_matrix_format(item["format"])]
        if reads:
            best = max(reads, key=lambda item: item["confidence"])
            result.text = best["text"]
            result.format = best["format"]
            result.confidence = best["confidence"]
            result.status = "decoded"
            result.sources.add("dynamsoft:candidate")
            return result

    for name, crop, binarizer in candidate_variants(image, detection):
        result.attempts.append(f"zxing:{name}")
        reads = zxing_read(crop, formats=LINEAR_FORMATS, binarizer=binarizer, try_downscale=False)
        valid = [barcode for barcode in reads if barcode.valid and not is_matrix_format(str(barcode.format))]
        if valid:
            barcode = valid[0]
            result.text = str(barcode.text)
            result.format = normalize_format(str(barcode.format))
            result.confidence = 1.0
            result.status = "decoded"
            result.sources.add(f"zxing:candidate:{name}")
            return result
    return result


def tile_positions(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    positions = list(range(0, length - size + 1, stride))
    final = length - size
    if positions[-1] != final:
        positions.append(final)
    return positions


def safely_inside_tile(quad: np.ndarray, width: int, height: int, margin: int = 0) -> bool:
    minimum = quad.min(axis=0)
    maximum = quad.max(axis=0)
    return bool(minimum[0] > margin and minimum[1] > margin and maximum[0] < width - margin and maximum[1] < height - margin)


def matrix_sweep_results(
    image: np.ndarray,
    dynamsoft: DynamsoftDecoder | None,
    tile_size: int,
    stride: int,
) -> list[Result]:
    """Regional 2-D recovery without rotating or upscaling whole pages."""
    if not dynamsoft or not dynamsoft.available:
        return []
    height, width = image.shape[:2]
    output: list[Result] = []
    for y in tile_positions(height, tile_size, stride):
        for x in tile_positions(width, tile_size, stride):
            tile = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            for item in dynamsoft.read(tile):
                if not is_matrix_format(item["format"]):
                    continue
                local_quad = np.asarray(item["quad"], dtype=np.float32)
                # Matrix codes carry strong error correction. Require the
                # reported quadrilateral to remain inside the tile, but do not
                # impose an extra margin: the folded page-21 symbol ends only
                # two pixels before the successful recovery tile boundary.
                if not safely_inside_tile(local_quad, tile.shape[1], tile.shape[0]):
                    continue
                global_quad = local_quad + np.asarray([x, y], dtype=np.float32)
                output.append(
                    Result(
                        quad=global_quad,
                        text=item["text"],
                        format=item["format"],
                        confidence=item["confidence"],
                        status="decoded",
                        sources={"dynamsoft:matrix-sweep"},
                        attempts=[f"tile:{x},{y},{tile.shape[1]},{tile.shape[0]}"],
                    )
                )
    return output


def same_result(a: Result, b: Result) -> bool:
    overlap = quad_overlap(a.quad, b.quad)
    if a.decoded and b.decoded and a.text == b.text and a.format == b.format and overlap >= 0.10:
        return True
    return overlap >= 0.62 and (not a.decoded or not b.decoded or a.text == b.text)


def merge_results(results: Iterable[Result]) -> list[Result]:
    merged: list[Result] = []
    ranked = sorted(results, key=lambda item: (item.decoded, item.confidence), reverse=True)
    for candidate in ranked:
        existing = next((item for item in merged if same_result(item, candidate)), None)
        if existing is None:
            merged.append(candidate)
            continue
        existing.sources.update(candidate.sources)
        existing.attempts.extend(attempt for attempt in candidate.attempts if attempt not in existing.attempts)
        if candidate.decoded and (not existing.decoded or candidate.confidence > existing.confidence):
            existing.text = candidate.text
            existing.format = candidate.format
            existing.confidence = candidate.confidence
            existing.status = "decoded"
            existing.quad = candidate.quad
    # Do not retain an unresolved proposal fully explained by a decoded box.
    cleaned: list[Result] = []
    for candidate in merged:
        if not candidate.decoded and any(item.decoded and quad_overlap(item.quad, candidate.quad) >= 0.72 for item in merged):
            continue
        cleaned.append(candidate)
    return cleaned


def annotate(image: np.ndarray, results: Sequence[Result]) -> np.ndarray:
    output = image.copy()
    for index, result in enumerate(results, start=1):
        if result.decoded and result.kind == "matrix":
            color = (255, 80, 255)
        elif result.decoded:
            color = (40, 190, 40)
        else:
            color = (0, 165, 255)
        quad = np.round(result.quad).astype(np.int32)
        cv2.polylines(output, [quad], True, color, 3, cv2.LINE_AA)
        anchor = np.min(quad, axis=0)
        label = f"{index} {result.text or 'unresolved'}"
        cv2.putText(
            output,
            label[:38],
            (int(anchor[0]), max(24, int(anchor[1]) - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def write_result_crops(image: np.ndarray, results: Sequence[Result], output: Path, page_number: int) -> None:
    directory = output / "crops"
    directory.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results, start=1):
        if result.kind == "matrix":
            minimum = np.floor(result.quad.min(axis=0) - 40).astype(int)
            maximum = np.ceil(result.quad.max(axis=0) + 40).astype(int)
            x1, y1 = np.maximum(minimum, 0)
            x2 = min(image.shape[1], maximum[0])
            y2 = min(image.shape[0], maximum[1])
            crop = image[y1:y2, x1:x2]
        else:
            locator = LocatorDetection(quad=np.asarray(result.quad, dtype=np.float32))
            crop = rectify_quad(image, locator.quad, pad_long=0.12, pad_short=0.18)
        if crop.size:
            cv2.imwrite(str(directory / f"page-{page_number:04d}-barcode-{index:03d}.png"), crop)


def process_page(
    page_number: int,
    page_path: Path,
    source_mode: str,
    config: PipelineConfig,
    dynamsoft: DynamsoftDecoder | None,
) -> dict[str, Any]:
    page_started = perf_counter()
    image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read {page_path}")
    timings: dict[str, float] = {}

    started = perf_counter()
    anchors = whole_page_results(image, dynamsoft)
    timings["whole_page_decode_seconds"] = perf_counter() - started

    started = perf_counter()
    located, rejected = locate_page(
        image,
        angle_step=max(0, config.angle_step),
        minimum_score=config.minimum_score,
        use_tensor_fallback=config.tensor_fallback,
        allow_tensor_only=False,
        minimum_linear_length=config.minimum_linear_length,
    )
    timings["linear_localization_seconds"] = perf_counter() - started

    started = perf_counter()
    candidate_results = [decode_candidate(image, detection, dynamsoft) for detection in located if detection.kind == "linear"]
    timings["candidate_decode_seconds"] = perf_counter() - started

    started = perf_counter()
    matrices = (
        matrix_sweep_results(image, dynamsoft, config.matrix_tile_size, config.matrix_tile_stride)
        if config.matrix_sweep
        else []
    )
    timings["matrix_sweep_seconds"] = perf_counter() - started

    results = merge_results([*anchors, *candidate_results, *matrices])
    if not config.include_unresolved:
        results = [result for result in results if result.decoded]
    results.sort(key=lambda item: (round(float(item.center()[1]) / 20), float(item.center()[0])))

    overlay = annotate(image, results)
    overlay_path = config.output / "overlays" / f"page-{page_number:04d}.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)
    if config.save_crops:
        write_result_crops(image, results, config.output, page_number)

    timings["total_seconds"] = perf_counter() - page_started
    decoded_count = sum(result.decoded for result in results)
    return {
        "page": page_number,
        "source_image": str(page_path),
        "source_mode": source_mode,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "decoded_count": decoded_count,
        "unresolved_count": len(results) - decoded_count,
        "results": [result.to_json() for result in results],
        "rejected_linear_proposals": len(rejected),
        "overlay": str(overlay_path),
        "timings": {key: round(value, 4) for key, value in timings.items()},
    }


def run_pipeline(input_path: Path, config: PipelineConfig) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    config.output = config.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    config.output.mkdir(parents=True, exist_ok=True)

    license_key = os.getenv("DYNAMSOFT_LICENSE_KEY")
    dynamsoft = DynamsoftDecoder(license_key) if config.engine in {"auto", "dynamsoft"} else None
    if config.engine == "dynamsoft" and (not dynamsoft or not dynamsoft.available):
        raise RuntimeError(dynamsoft.error if dynamsoft else "Dynamsoft is unavailable")

    started = perf_counter()
    pages = collect_pages(input_path, config)
    extraction_seconds = perf_counter() - started
    page_payloads: list[dict[str, Any]] = []
    for page_number, page_path, source_mode in pages:
        payload = process_page(page_number, page_path, source_mode, config, dynamsoft)
        page_payloads.append(payload)
        print(
            f"page {page_number:04d}: {payload['decoded_count']} decoded, "
            f"{payload['unresolved_count']} unresolved, {payload['timings']['total_seconds']:.2f}s",
            flush=True,
        )

    decoded = sum(page["decoded_count"] for page in page_payloads)
    unresolved = sum(page["unresolved_count"] for page in page_payloads)
    payload = {
        "input": str(input_path),
        "method": "native raster + deterministic 1-D proposals + candidate decoder portfolio + regional 2-D recovery",
        "engines": {
            "zxing_cpp": True,
            "dynamsoft": bool(dynamsoft and dynamsoft.available),
            "dynamsoft_error": dynamsoft.error if dynamsoft and not dynamsoft.available else None,
        },
        "configuration": {
            "render_dpi": config.render_dpi,
            "angle_step": config.angle_step,
            "minimum_score": config.minimum_score,
            "minimum_linear_length": config.minimum_linear_length,
            "tensor_fallback": config.tensor_fallback,
            "matrix_sweep": config.matrix_sweep,
            "matrix_tile_size": config.matrix_tile_size,
            "matrix_tile_stride": config.matrix_tile_stride,
        },
        "summary": {
            "pages": len(page_payloads),
            "decoded": decoded,
            "unresolved": unresolved,
            "extraction_seconds": round(extraction_seconds, 4),
            "processing_seconds": round(sum(page["timings"]["total_seconds"] for page in page_payloads), 4),
        },
        "pages": page_payloads,
    }
    destination = config.output / "detections.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {destination}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PDF, page image, or directory of page images")
    parser.add_argument("--output", type=Path, default=Path("hybrid_barcode_pipeline/output"))
    parser.add_argument("--pages", type=parse_page_selection, help="PDF pages, for example 8,16,19,21")
    parser.add_argument("--render-dpi", type=int, default=300, help="fallback DPI for pages without a dominant raster")
    parser.add_argument("--angle-step", type=int, default=0, help="deterministic locator deskew spacing; 0 disables page rotations")
    parser.add_argument("--minimum-score", type=float, default=0.52)
    parser.add_argument("--minimum-linear-length", type=float, default=100.0)
    parser.add_argument("--tensor-fallback", action="store_true")
    parser.add_argument("--engine", choices=("auto", "zxing", "dynamsoft"), default="auto")
    parser.add_argument("--no-matrix-sweep", dest="matrix_sweep", action="store_false")
    parser.add_argument("--matrix-tile-size", type=int, default=700)
    parser.add_argument("--matrix-tile-stride", type=int, default=400)
    parser.add_argument("--no-crops", dest="save_crops", action="store_false")
    parser.add_argument("--decoded-only", dest="include_unresolved", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = PipelineConfig(
        output=arguments.output,
        render_dpi=arguments.render_dpi,
        pages=arguments.pages,
        angle_step=arguments.angle_step,
        minimum_score=arguments.minimum_score,
        minimum_linear_length=arguments.minimum_linear_length,
        tensor_fallback=arguments.tensor_fallback,
        engine=arguments.engine,
        matrix_sweep=arguments.matrix_sweep,
        matrix_tile_size=arguments.matrix_tile_size,
        matrix_tile_stride=arguments.matrix_tile_stride,
        save_crops=arguments.save_crops,
        include_unresolved=arguments.include_unresolved,
    )
    try:
        run_pipeline(arguments.input, config)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
