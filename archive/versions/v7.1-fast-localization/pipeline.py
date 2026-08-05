"""Fast CPU-only barcode localization for clear, adequately sized symbols."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_FAST_WEIGHTS = (
    Path(__file__).resolve().parent
    / "models"
    / "fast-locator-v3.torchscript"
)


@dataclass(frozen=True)
class Profile:
    name: str
    work_size: int
    confidence: float
    minimum_area: float
    weights: Path
    verify_structure: bool


PROFILES = {
    "fast": Profile(
        "fast",
        256,
        0.65,
        60.0,
        DEFAULT_FAST_WEIGHTS,
        False,
    ),
}


@dataclass
class Config:
    output: Path
    profile: str = "fast"
    weights: Path | None = None
    workers: int = 1
    render_dpi: int = 300
    pages: list[int] | None = None
    save_overlays: bool = False
    overwrite: bool = False


def parse_page_selection(value: str) -> list[int]:
    pages: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise ValueError(f"invalid page range: {token}")
            pages.update(range(start, end + 1))
        else:
            page = int(token)
            if page < 1:
                raise ValueError(f"invalid page: {token}")
            pages.add(page)
    if not pages:
        raise ValueError("page selection is empty")
    return sorted(pages)


def _similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_hull = cv2.convexHull(np.asarray(first, np.float32))
    second_hull = cv2.convexHull(np.asarray(second, np.float32))
    first_area = abs(float(cv2.contourArea(first_hull)))
    second_area = abs(float(cv2.contourArea(second_hull)))
    if min(first_area, second_area) <= 1e-6:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(
        first_hull,
        second_hull,
    )
    intersection = max(0.0, float(intersection))
    return intersection / min(first_area, second_area)


def deduplicate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in sorted(
        values,
        key=lambda item: float(item["confidence"]),
        reverse=True,
    ):
        if any(
            _similarity(candidate["quad"], prior["quad"]) >= 0.75
            for prior in output
        ):
            continue
        output.append(candidate)
    return output


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32).reshape(-1, 2)
    ordered = np.empty((4, 2), np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    if len({tuple(point) for point in ordered}) != 4:
        center = points.mean(axis=0)
        cycle = points[
            np.argsort(
                np.arctan2(
                    points[:, 1] - center[1],
                    points[:, 0] - center[0],
                )
            )
        ]
        ordered = np.roll(
            cycle,
            -int(np.argmin(cycle.sum(axis=1))),
            axis=0,
        )
    return ordered


def rectify(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    source = order_quad(quad)
    width = max(
        2,
        int(
            round(
                max(
                    np.linalg.norm(source[1] - source[0]),
                    np.linalg.norm(source[2] - source[3]),
                )
            )
        ),
    )
    height = max(
        2,
        int(
            round(
                max(
                    np.linalg.norm(source[3] - source[0]),
                    np.linalg.norm(source[2] - source[1]),
                )
            )
        ),
    )
    destination = np.asarray(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        np.float32,
    )
    patch = cv2.warpPerspective(
        gray,
        cv2.getPerspectiveTransform(source, destination),
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    if patch.shape[0] > patch.shape[1]:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    return patch


def binary_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.uint8).reshape(1, -1)
    return (
        cv2.threshold(
            values,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1].ravel()
        > 0
    )


def count_transition_runs(binary: np.ndarray) -> tuple[int, float]:
    if len(binary) < 2:
        return 0, 0.0
    runs: list[tuple[bool, int]] = []
    state, length = bool(binary[0]), 1
    for raw_value in binary[1:]:
        value = bool(raw_value)
        if value == state:
            length += 1
        else:
            runs.append((state, length))
            state, length = value, 1
    runs.append((state, length))
    compact: list[tuple[bool, int]] = []
    index = 0
    while index < len(runs):
        state, length = runs[index]
        if (
            0 < index < len(runs) - 1
            and length == 1
            and runs[index - 1][0] == runs[index + 1][0]
        ):
            if compact:
                prior_state, prior_length = compact[-1]
                compact[-1] = (
                    prior_state,
                    prior_length + 1 + runs[index + 1][1],
                )
            index += 2
            continue
        compact.append((state, length))
        index += 1
    return max(0, len(compact) - 1), float(np.mean(binary))


def linear_score(gray: np.ndarray, quad: np.ndarray) -> float:
    patch = rectify(gray, quad)
    height, width = patch.shape
    if width < 18 or height < 5:
        return 0.0
    strip = patch[
        max(0, int(round(height * 0.10))):
        min(height, max(3, int(round(height * 0.82))))
    ]
    gradient_x = np.abs(cv2.Scharr(strip, cv2.CV_32F, 1, 0))
    gradient_y = np.abs(cv2.Scharr(strip, cv2.CV_32F, 0, 1))
    x_energy = float(np.mean(gradient_x))
    y_energy = float(np.mean(gradient_y))
    orientation = x_energy / (x_energy + y_energy + 1e-6)
    threshold = float(np.percentile(gradient_x, 72.0))
    if threshold <= 1e-6:
        threshold = float(
            np.mean(gradient_x) + np.std(gradient_x)
        )
    support = np.mean(
        gradient_x >= max(threshold, 1e-6),
        axis=0,
    )
    persistent = support >= 0.43
    persistence = float(np.mean(persistent))
    support_strength = (
        float(np.mean(support[persistent]))
        if np.any(persistent)
        else 0.0
    )
    consensus = np.median(strip, axis=0).astype(np.uint8)
    binary = binary_signal(consensus)
    transitions, dark_fraction = count_transition_runs(binary)
    transition_rate = transitions / max(1, width)
    sites = np.flatnonzero(
        np.diff(binary.astype(np.int8)) != 0
    )
    agreements = []
    if len(sites):
        expanded = np.zeros(width, dtype=bool)
        for offset in range(-2, 3):
            expanded[
                np.clip(sites + offset, 0, width - 1)
            ] = True
        for row in np.linspace(
            0,
            strip.shape[0] - 1,
            min(9, strip.shape[0]),
            dtype=int,
        ):
            row_binary = binary_signal(strip[row])
            agreements.append(
                float(
                    np.mean(
                        row_binary[expanded] == binary[expanded]
                    )
                )
            )
    agreement = float(np.median(agreements)) if agreements else 0.0
    aspect = width / max(1.0, height)
    aspect_score = min(1.0, max(0.0, (aspect - 1.15) / 2.2))
    alternation = min(1.0, transitions / 18.0) * min(
        1.0,
        transition_rate / 0.055,
    )
    persistence_score = min(1.0, persistence / 0.16) * min(
        1.0,
        support_strength / 0.62,
    )
    dark_score = max(
        0.0,
        1.0 - abs(dark_fraction - 0.43) / 0.43,
    )
    return float(
        0.22 * orientation
        + 0.30 * persistence_score
        + 0.25 * alternation
        + 0.16 * agreement
        + 0.04 * aspect_score
        + 0.03 * dark_score
    )


def compact_matrix_evidence(
    gray: np.ndarray,
    quad: np.ndarray,
) -> tuple[float, float]:
    patch = rectify(gray, quad)
    binary = cv2.threshold(
        patch,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    height, width = patch.shape
    gradient_x = float(
        np.mean(np.abs(cv2.Scharr(patch, cv2.CV_32F, 1, 0)))
    )
    gradient_y = float(
        np.mean(np.abs(cv2.Scharr(patch, cv2.CV_32F, 0, 1)))
    )
    edge_balance = min(gradient_x, gradient_y) / max(
        1e-6,
        max(gradient_x, gradient_y),
    )
    aspect = width / max(1.0, float(height))
    square_score = max(
        0.0,
        1.0
        - abs(np.log(max(1e-6, aspect))) / np.log(2.2),
    )
    source_binary = binary > 0
    row_transitions = []
    column_transitions = []
    for position in np.linspace(0.15, 0.85, 7):
        row = source_binary[
            min(
                height - 1,
                int(round(position * (height - 1))),
            )
        ]
        column = source_binary[
            :,
            min(
                width - 1,
                int(round(position * (width - 1))),
            ),
        ]
        row_transitions.append(
            int(np.count_nonzero(np.diff(row.astype(np.int8))))
        )
        column_transitions.append(
            int(
                np.count_nonzero(
                    np.diff(column.astype(np.int8))
                )
            )
        )
    transition_score = min(
        1.0,
        min(
            float(np.median(row_transitions)),
            float(np.median(column_transitions)),
        )
        / 10.0,
    )
    dark_fraction = float(np.mean(binary > 0))
    occupancy_score = max(
        0.0,
        1.0 - abs(dark_fraction - 0.45) / 0.45,
    )
    base_score = (
        0.30 * edge_balance
        + 0.25 * square_score
        + 0.30 * transition_score
        + 0.15 * occupancy_score
    )
    return float(base_score), float(transition_score)


def structurally_verified(
    gray: np.ndarray,
    candidate: dict[str, Any],
) -> bool:
    quad = np.asarray(candidate["quad"], np.float32)
    rectangle = cv2.minAreaRect(quad)
    long_side = float(max(rectangle[1]))
    short_side = float(min(rectangle[1]))
    aspect = long_side / max(short_side, 1e-6)
    if aspect >= 1.60:
        candidate["kind"] = "linear"
        return linear_score(gray, quad) >= 0.55
    candidate["kind"] = "matrix_2d"
    base_score, transition_score = compact_matrix_evidence(
        gray,
        quad,
    )
    return base_score >= 0.45 and transition_score >= 0.25


class FastLocalizer:
    """31k-parameter grayscale segmentation locator."""

    def __init__(self, weights: Path, profile: Profile) -> None:
        import torch

        weights = weights.expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        torch.set_num_threads(1)
        self.torch = torch
        self.model = torch.jit.load(
            str(weights),
            map_location="cpu",
        ).eval()
        self.profile = profile
        self.lock = Lock()
        with torch.inference_mode():
            self.model(
                torch.ones(
                    (1, 1, profile.work_size, profile.work_size),
                    dtype=torch.float32,
                )
            )

    def __call__(self, gray: np.ndarray) -> list[dict[str, Any]]:
        source_height, source_width = gray.shape[:2]
        scale = self.profile.work_size / max(
            source_height,
            source_width,
        )
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        work = cv2.resize(
            gray,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        padded_width = ((width + 31) // 32) * 32
        padded_height = ((height + 31) // 32) * 32
        padded = np.full(
            (padded_height, padded_width),
            255,
            np.uint8,
        )
        padded[:height, :width] = work
        tensor = (
            self.torch.from_numpy(padded[None, None])
            .float()
            .div(255.0)
        )
        with self.lock, self.torch.inference_mode():
            heatmaps = self.torch.sigmoid(
                self.model(tensor)
            )[0].numpy()

        output: list[dict[str, Any]] = []
        kernel = np.ones((3, 3), np.uint8)
        for class_index, kind in enumerate(("linear", "matrix_2d")):
            heatmap = heatmaps[class_index, :height, :width]
            binary = (
                heatmap >= self.profile.confidence
            ).astype(np.uint8)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel,
            )
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.profile.minimum_area:
                    continue
                rectangle = cv2.minAreaRect(contour)
                if min(rectangle[1]) < 2:
                    continue
                aspect = max(rectangle[1]) / max(
                    1e-6,
                    min(rectangle[1]),
                )
                component_mask = np.zeros_like(binary)
                cv2.drawContours(
                    component_mask,
                    [contour],
                    -1,
                    1,
                    -1,
                )
                output.append(
                    {
                        "quad": cv2.boxPoints(rectangle) / scale,
                        # The two heatmaps jointly provide localization
                        # evidence. Geometry is more reliable than their
                        # class ordering after cross-domain fine-tuning.
                        "kind": (
                            "linear"
                            if aspect >= 1.60
                            else "matrix_2d"
                        ),
                        "confidence": float(
                            cv2.mean(
                                heatmap,
                                mask=component_mask,
                            )[0]
                        ),
                        "source": "fast-locator-v3",
                    }
                )
        output = deduplicate(output)
        if self.profile.verify_structure:
            output = [
                item
                for item in output
                if structurally_verified(gray, item)
            ]
        return output


def process_page(
    page: int,
    path: Path,
    source_mode: str,
    localizer: FastLocalizer,
) -> tuple[dict[str, Any], Path]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"unable to read image: {path}")
    started = perf_counter()
    detections = localizer(gray)
    localization_seconds = perf_counter() - started
    detections.sort(
        key=lambda item: (
            float(np.mean(item["quad"][:, 1])),
            float(np.mean(item["quad"][:, 0])),
        )
    )
    serialized = [
        {
            **item,
            "quad": [
                [round(float(x), 3), round(float(y), 3)]
                for x, y in item["quad"]
            ],
            "status": "localized",
            "payload": None,
        }
        for item in detections
    ]
    return (
        {
            "page": page,
            "source_mode": source_mode,
            "image_size": {
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
            },
            "count": len(serialized),
            "counts": {
                "linear": sum(
                    item["kind"] == "linear"
                    for item in serialized
                ),
                "matrix_2d": sum(
                    item["kind"] == "matrix_2d"
                    for item in serialized
                ),
            },
            "detections": serialized,
            "timings": {
                "localization_seconds": round(
                    localization_seconds,
                    6,
                )
            },
            "overlay": None,
        },
        path,
    )


def annotate(gray: np.ndarray, detections: Sequence[dict[str, Any]]) -> np.ndarray:
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index, detection in enumerate(detections, start=1):
        quad = np.rint(
            np.asarray(detection["quad"], np.float32)
        ).astype(np.int32)
        color = (
            (45, 185, 65)
            if detection["kind"] == "linear"
            else (220, 120, 20)
        )
        cv2.polylines(
            output,
            [quad],
            True,
            color,
            3,
            cv2.LINE_AA,
        )
        x, y = np.min(quad, axis=0)
        cv2.putText(
            output,
            f"{index} {detection['kind']}",
            (int(x), max(18, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def generate_overlay(
    payload: dict[str, Any],
    source_path: Path,
    work: Path,
) -> None:
    started = perf_counter()
    gray = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(source_path)
    destination = (
        work
        / "overlays"
        / f"page-{int(payload['page']):04d}.jpg"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(destination),
        annotate(gray, payload["detections"]),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    payload["overlay"] = str(destination.relative_to(work))
    payload["timings"]["artifact_seconds"] = round(
        perf_counter() - started,
        6,
    )


def pdf_page_count(path: Path) -> int:
    completed = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(
        r"^Pages:\s+(\d+)",
        completed.stdout,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError("unable to determine PDF page count")
    return int(match.group(1))


def image_number(path: Path) -> int:
    match = re.search(r"-(\d+)(?:\.[^.]+)?$", path.name)
    return int(match.group(1)) if match else 10**9


def pdf_image_rows(path: Path) -> list[dict[str, int]]:
    completed = subprocess.run(
        ["pdfimages", "-list", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if (
            len(parts) >= 5
            and parts[0].isdigit()
            and parts[1].isdigit()
        ):
            rows.append(
                {
                    "page": int(parts[0]),
                    "width": int(parts[3]),
                    "height": int(parts[4]),
                }
            )
    return rows


def extract_pdf(
    path: Path,
    selected: Sequence[int],
    work: Path,
    dpi: int,
) -> list[tuple[int, Path, str]]:
    extraction = work / ".native"
    page_directory = work / ".pages"
    extraction.mkdir(parents=True, exist_ok=True)
    page_directory.mkdir(parents=True, exist_ok=True)
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
            (
                item
                for item in extraction.glob("image-*")
                if item.is_file()
            ),
            key=image_number,
        )
        if len(files) != len(rows):
            raise RuntimeError("native image manifest mismatch")
        best: dict[int, tuple[int, Path]] = {}
        for row, image_path in zip(rows, files):
            area = row["width"] * row["height"]
            if (
                row["page"] in selected
                and area >= 1_000_000
                and area > best.get(
                    row["page"],
                    (0, image_path),
                )[0]
            ):
                best[row["page"]] = (area, image_path)
        output = []
        for page in selected:
            if page in best:
                source = best[page][1]
                destination = (
                    page_directory
                    / f"page-{page:04d}{source.suffix.lower()}"
                )
                shutil.copy2(source, destination)
                output.append(
                    (
                        page,
                        destination,
                        "native-embedded-raster",
                    )
                )
                continue
            destination = page_directory / f"render-{page:04d}"
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
                    str(destination),
                ],
                check=True,
                capture_output=True,
            )
            output.append(
                (
                    page,
                    destination.with_suffix(".jpg"),
                    f"rendered-{dpi}-dpi",
                )
            )
        return output
    finally:
        shutil.rmtree(extraction, ignore_errors=True)


def collect_pages(
    path: Path,
    config: Config,
    work: Path,
) -> list[tuple[int, Path, str]]:
    if path.is_dir():
        images = sorted(
            item
            for item in path.iterdir()
            if item.suffix.lower() in IMAGE_EXTENSIONS
        )
        selected = config.pages or list(range(1, len(images) + 1))
        if any(page > len(images) for page in selected):
            raise ValueError("selected page exceeds image count")
        return [
            (page, images[page - 1], "image-directory")
            for page in selected
        ]
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        if config.pages and config.pages != [1]:
            raise ValueError("a single image only has page 1")
        return [(1, path, "image-file")]
    if path.suffix.lower() != ".pdf":
        raise ValueError("input must be a PDF, image, or image directory")
    page_count = pdf_page_count(path)
    selected = config.pages or list(range(1, page_count + 1))
    if any(page > page_count for page in selected):
        raise ValueError(
            f"selected page exceeds PDF page count {page_count}"
        )
    return extract_pdf(
        path,
        selected,
        work,
        config.render_dpi,
    )


def effective_workers(requested: int, page_count: int) -> int:
    if requested > 0:
        return max(1, min(requested, page_count))
    return max(1, min(4, page_count, os.cpu_count() or 1))


def run_pipeline(input_path: Path, config: Config) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output = config.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output.exists() and not config.overwrite:
        raise FileExistsError(
            f"output exists; use --overwrite: {output}"
        )
    profile = PROFILES[config.profile]
    weights = config.weights or profile.weights
    localizer = FastLocalizer(weights, profile)
    work = output.parent / f".{output.name}.work-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True)
    wall_started = perf_counter()
    try:
        extraction_started = perf_counter()
        pages = collect_pages(input_path, config, work)
        extraction_seconds = perf_counter() - extraction_started
        workers = effective_workers(config.workers, len(pages))
        processing_started = perf_counter()
        if workers == 1:
            processed = [
                process_page(page, path, mode, localizer)
                for page, path, mode in pages
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="fast-localize",
            ) as executor:
                futures = [
                    executor.submit(
                        process_page,
                        page,
                        path,
                        mode,
                        localizer,
                    )
                    for page, path, mode in pages
                ]
                processed = [future.result() for future in futures]
        processing_seconds = perf_counter() - processing_started

        artifact_seconds = 0.0
        if config.save_overlays:
            started = perf_counter()
            for payload, source_path in processed:
                generate_overlay(payload, source_path, work)
            artifact_seconds = perf_counter() - started
        payloads = sorted(
            (item[0] for item in processed),
            key=lambda item: item["page"],
        )
        shutil.rmtree(work / ".pages", ignore_errors=True)
        result = {
            "input": str(input_path),
            "method": (
                "31k-parameter CPU segmentation model; localization only"
            ),
            "decoding_performed": False,
            "zxing_used": False,
            "gpu_used": False,
            "configuration": {
                "profile": profile.name,
                "work_size": profile.work_size,
                "confidence": profile.confidence,
                "minimum_area": profile.minimum_area,
                "weights": str(weights.resolve()),
                "structural_verification": profile.verify_structure,
                "workers": workers,
                "render_dpi": config.render_dpi,
                "save_overlays": config.save_overlays,
            },
            "summary": {
                "pages": len(payloads),
                "localized": sum(
                    page["count"] for page in payloads
                ),
                "extraction_seconds": round(
                    extraction_seconds,
                    6,
                ),
                "processing_wall_seconds": round(
                    processing_seconds,
                    6,
                ),
                "average_processing_wall_seconds_per_page": round(
                    processing_seconds / max(1, len(payloads)),
                    6,
                ),
                "page_localization_seconds_sum": round(
                    sum(
                        page["timings"]["localization_seconds"]
                        for page in payloads
                    ),
                    6,
                ),
                "artifact_wall_seconds": round(
                    artifact_seconds,
                    6,
                ),
                "wall_seconds": round(
                    perf_counter() - wall_started,
                    6,
                ),
            },
            "pages": payloads,
        }
        (work / "detections.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        work.rename(output)
        for page in payloads:
            print(
                f"page {page['page']:04d}: "
                f"{page['count']} localized, "
                f"localization="
                f"{page['timings']['localization_seconds']:.4f}s",
                flush=True,
            )
        print(
            f"summary: {result['summary']['localized']} localized "
            f"across {len(payloads)} pages; processing wall="
            f"{processing_seconds:.4f}s "
            f"({processing_seconds / max(1, len(payloads)):.4f}s/page), "
            f"artifacts={artifact_seconds:.4f}s, workers={workers}",
            flush=True,
        )
        return result
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
