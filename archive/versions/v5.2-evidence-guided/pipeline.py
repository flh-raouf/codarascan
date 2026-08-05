#!/usr/bin/env python3
"""Evidence-selected coarse-to-fine barcode localization and decoding."""
from __future__ import annotations

import json
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import zxingcpp

from models import TinyBarcodeLocator, TinyQRRestorer


ROOT = Path(__file__).resolve().parent
MATRIX_FORMATS = {"aztec", "datamatrix", "maxicode", "pdf417", "qrcode"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Settings:
    device: str = "cpu"
    work_size: int = 320
    confidence: float = 0.60
    mode: str = "accuracy"
    formats: frozenset[str] = frozenset()
    include_unresolved: bool = True


def _format_name(value: Any) -> str:
    return str(value).replace("BarcodeFormat.", "")


def _quad(value: Any) -> list[list[float]]:
    position = value.position
    points = [
        [float(position.top_left.x), float(position.top_left.y)],
        [float(position.top_right.x), float(position.top_right.y)],
        [float(position.bottom_right.x), float(position.bottom_right.y)],
        [float(position.bottom_left.x), float(position.bottom_left.y)],
    ]
    if abs(cv2.contourArea(np.asarray(points, np.float32))) > 1e-6:
        return points
    values = np.asarray(points, np.float32)
    distances = np.linalg.norm(values[:, None] - values[None, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    start, end = values[first], values[second]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return []
    normal = np.asarray([-direction[1], direction[0]], np.float32) / length
    return np.asarray(
        [start - normal, end - normal, end + normal, start + normal],
    ).tolist()


def _accepted_format(symbology: str, allowed: frozenset[str]) -> bool:
    if not allowed:
        return True
    normalized = "".join(character for character in symbology.lower() if character.isalnum())
    return normalized in allowed


def _normalized_format(symbology: str) -> str:
    return "".join(character for character in symbology.lower() if character.isalnum())


def _read(image: np.ndarray, allowed: frozenset[str]) -> list[dict[str, Any]]:
    output = []
    for barcode in zxingcpp.read_barcodes(
        image,
        try_rotate=True,
        try_downscale=True,
        try_invert=True,
        return_errors=False,
    ):
        symbology = _format_name(barcode.format)
        quad = _quad(barcode)
        if not barcode.valid or not barcode.text or not quad:
            continue
        if not _accepted_format(symbology, allowed):
            continue
        output.append(
            {
                "polygon": quad,
                "kind": "2d" if _normalized_format(symbology) in MATRIX_FORMATS else "1d",
                "symbology": symbology,
                "payload": str(barcode.text),
                "status": "decoded",
                "confidence": 1.0,
                "sources": ["zxing-full-image"],
                "evidence": {},
            }
        )
    return output


def _read_opencv(image: np.ndarray, allowed: frozenset[str]) -> list[dict[str, Any]]:
    """Collect complementary OpenCV 1D hypotheses for consensus voting."""
    detector = cv2.barcode.BarcodeDetector()
    try:
        detected, texts, formats, points = detector.detectAndDecodeWithType(image)
    except cv2.error:
        return []
    if not detected or points is None:
        return []
    output = []
    for index, polygon in enumerate(np.asarray(points, np.float32)):
        payload = str(texts[index]) if index < len(texts) and texts[index] else ""
        symbology = str(formats[index]) if index < len(formats) else "Unknown"
        if not payload or not _accepted_format(symbology, allowed):
            continue
        output.append(
            {
                "polygon": polygon.reshape(-1, 2).tolist(),
                "kind": "1d",
                "symbology": symbology,
                "payload": payload,
                "status": "decoded",
                "confidence": 1.0,
                "sources": ["opencv-crop"],
                "evidence": {},
            }
        )
    return output


def _similarity(first: Any, second: Any) -> tuple[float, float]:
    a = cv2.convexHull(np.asarray(first, np.float32))
    b = cv2.convexHull(np.asarray(second, np.float32))
    area_a, area_b = abs(cv2.contourArea(a)), abs(cv2.contourArea(b))
    if min(area_a, area_b) <= 1e-6:
        return 0.0, 0.0
    intersection = float(cv2.intersectConvexConvex(a, b)[0])
    union = area_a + area_b - intersection
    return intersection / max(union, 1e-6), intersection / min(area_a, area_b)


def _deduplicate(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    ranking = sorted(
        values,
        key=lambda item: (
            item.get("payload") is not None,
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    for candidate in ranking:
        duplicate = None
        for existing in accepted:
            iou, containment = _similarity(candidate["polygon"], existing["polygon"])
            same_payload_nearby = False
            if (
                candidate.get("payload")
                and candidate.get("payload") == existing.get("payload")
                and _normalized_format(str(candidate.get("symbology")))
                == _normalized_format(str(existing.get("symbology")))
            ):
                first = np.asarray(candidate["polygon"], np.float32)
                second = np.asarray(existing["polygon"], np.float32)
                distance = float(np.linalg.norm(first.mean(axis=0) - second.mean(axis=0)))
                span = max(
                    float(np.ptp(first[:, 0])),
                    float(np.ptp(first[:, 1])),
                    float(np.ptp(second[:, 0])),
                    float(np.ptp(second[:, 1])),
                )
                same_payload_nearby = distance <= 0.35 * max(span, 1.0)
            if iou >= 0.50 or containment >= 0.80 or same_payload_nearby:
                duplicate = existing
                break
        if duplicate is None:
            accepted.append(candidate)
        elif candidate.get("payload") and not duplicate.get("payload"):
            duplicate.update(candidate)
        elif candidate.get("payload") == duplicate.get("payload"):
            duplicate["sources"] = list(
                dict.fromkeys([*duplicate["sources"], *candidate["sources"]])
            )
    return accepted


def _crop(image: np.ndarray, polygon: Any, padding: float) -> np.ndarray:
    points = np.asarray(polygon, np.float32).reshape(-1, 2)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    pad_x = max(1.0, float(x2 - x1)) * padding
    pad_y = max(1.0, float(y2 - y1)) * padding
    height, width = image.shape[:2]
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x + 1)), min(height, int(y2 + pad_y + 1))
    return image[top:bottom, left:right]


def _payload_key(item: dict[str, Any]) -> tuple[str, str]:
    symbology, payload = str(item["symbology"]), str(item["payload"])
    digits = "".join(character for character in payload if character.isdigit())
    normalized = _normalized_format(symbology)
    if normalized == "upca" and len(digits) == 12:
        return "GTIN", digits
    if normalized == "ean13" and len(digits) == 13 and digits.startswith("0"):
        return "GTIN", digits[1:]
    return symbology, payload


class EvidenceGuidedPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device(settings.device)
        locator_checkpoint = torch.load(
            ROOT / "models" / "tiny-barcode-locator.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.locator = TinyBarcodeLocator(int(locator_checkpoint.get("width", 16)))
        self.locator.load_state_dict(locator_checkpoint["model"])
        self.locator.to(self.device).eval()
        restorer_checkpoint = torch.load(
            ROOT / "models" / "tiny-qr-restorer.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.restorer = TinyQRRestorer(int(restorer_checkpoint.get("width", 8)))
        self.restorer.load_state_dict(restorer_checkpoint["model"])
        self.restorer.to(self.device).eval()
        self._legacy_module: Any | None = None

    def locate(self, image: np.ndarray) -> list[dict[str, Any]]:
        source_height, source_width = image.shape[:2]
        size = self.settings.work_size
        scale = size / max(source_height, source_width)
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[:height, :width] = resized
        tensor = torch.from_numpy(
            canvas[:, :, ::-1].copy().transpose(2, 0, 1)[None],
        ).float().div(255.0).to(self.device)
        with torch.inference_mode():
            heatmaps = torch.sigmoid(self.locator(tensor))[0].cpu().numpy()
        minimum_area = max(10.0, 0.00012 * width * height)
        output = []
        for class_index, kind in enumerate(("1d", "2d")):
            heatmap = heatmaps[class_index, :height, :width]
            binary = (heatmap >= self.settings.confidence).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                if cv2.contourArea(contour) < minimum_area:
                    continue
                points = cv2.boxPoints(cv2.minAreaRect(contour)) / scale
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], -1, 1, -1)
                output.append(
                    {
                        "polygon": points.tolist(),
                        "kind": kind,
                        "symbology": "QRCode" if kind == "2d" else "Unknown",
                        "payload": None,
                        "status": "localized",
                        "confidence": float(cv2.mean(heatmap, mask=mask)[0]),
                        "sources": ["tiny-full-frame-locator"],
                        "evidence": {},
                    }
                )
        return _deduplicate(output)

    def _decode_consensus(
        self,
        image: np.ndarray,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        primary: list[tuple[str, dict[str, Any]]] = []
        for padding in (0.12, 0.24):
            for item in _read(
                _crop(image, detection["polygon"], padding),
                self.settings.formats,
            ):
                primary.append((f"pad-{padding:.2f}:original", item))
        votes: dict[tuple[str, str], set[str]] = defaultdict(set)
        examples: dict[tuple[str, str], dict[str, Any]] = {}
        for source, item in primary:
            key = _payload_key(item)
            votes[key].add(source)
            examples.setdefault(key, item)
        agreed = [key for key, sources in votes.items() if len(sources) >= 2]
        if len(agreed) == 1:
            selected = examples[agreed[0]]
            if self._kind_compatible(detection, selected):
                return self._accept(
                    detection,
                    selected,
                    len(votes[agreed[0]]),
                    False,
                )
            detection["evidence"]["decode_rejected"] = "detector_decoder_kind_mismatch"
            return detection

        hypotheses = list(primary)
        for padding in (0.12, 0.24):
            crop = _crop(image, detection["polygon"], padding)
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
            variants = [
                ("original", crop, True),
                ("clahe", clahe, True),
                ("otsu", cv2.threshold(
                    clahe,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )[1], False),
            ]
            if min(gray.shape) < 180:
                scale = min(4.0, 200.0 / max(min(gray.shape), 1))
                variants.extend(
                    [
                        (
                            f"{stage}-upscaled",
                            cv2.resize(
                                variant,
                                None,
                                fx=scale,
                                fy=scale,
                                interpolation=cv2.INTER_CUBIC,
                            ),
                            use_opencv,
                        )
                        for stage, variant, use_opencv in variants[:2]
                    ]
                )
            for stage, variant, use_opencv in variants:
                for item in _read(variant, self.settings.formats):
                    hypotheses.append((f"pad-{padding:.2f}:{stage}", item))
                if use_opencv:
                    for item in _read_opencv(variant, self.settings.formats):
                        hypotheses.append((f"pad-{padding:.2f}:opencv:{stage}", item))
        votes, examples = defaultdict(set), {}
        for source, item in hypotheses:
            key = _payload_key(item)
            votes[key].add(source)
            examples.setdefault(key, item)
        if not votes:
            detection["evidence"]["adaptive_escalation"] = True
            return detection
        ranked = sorted(votes, key=lambda key: len(votes[key]), reverse=True)
        winner = ranked[0]
        winner_votes = len(votes[winner])
        runner_up = len(votes[ranked[1]]) if len(ranked) > 1 else 0
        if winner_votes <= runner_up:
            detection["evidence"].update(
                {
                    "adaptive_escalation": True,
                    "decode_rejected": "ambiguous_payload_tie",
                }
            )
            return detection
        only_opencv_support = all(
            ":opencv:" in source for source in votes[winner]
        )
        if (
            self.settings.mode == "accuracy"
            and winner_votes < 2
            and only_opencv_support
        ):
            detection["evidence"].update(
                {
                    "adaptive_escalation": True,
                    "decode_rejected": "insufficient_independent_consensus",
                }
            )
            return detection
        if not self._kind_compatible(detection, examples[winner]):
            detection["evidence"]["decode_rejected"] = "detector_decoder_kind_mismatch"
            return detection
        return self._accept(
            detection,
            examples[winner],
            winner_votes - runner_up,
            True,
        )

    @staticmethod
    def _kind_compatible(
        detection: dict[str, Any],
        decoded: dict[str, Any],
    ) -> bool:
        decoded_kind = (
            "2d"
            if _normalized_format(str(decoded.get("symbology"))) in MATRIX_FORMATS
            else "1d"
        )
        return decoded_kind == detection.get("kind")

    @staticmethod
    def _accept(
        detection: dict[str, Any],
        decoded: dict[str, Any],
        margin: int,
        escalated: bool,
    ) -> dict[str, Any]:
        detection.update(
            {
                "payload": decoded["payload"],
                "symbology": decoded["symbology"],
                "status": "decoded",
                "sources": list(
                    dict.fromkeys(
                        [
                            *detection["sources"],
                            *decoded["sources"],
                            "adaptive-consensus",
                        ]
                    )
                ),
            }
        )
        detection["evidence"].update(
            {"decode_margin": margin, "adaptive_escalation": escalated}
        )
        return detection

    def _restore_qr(
        self,
        image: np.ndarray,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        crop = _crop(image, detection["polygon"], 0.18)
        if crop.size == 0:
            return detection
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        scale = min(1.0, 512.0 / max(height, width))
        work = gray if scale == 1.0 else cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        work_height, work_width = work.shape
        padded = cv2.copyMakeBorder(
            work,
            0,
            (-work_height) % 4,
            0,
            (-work_width) % 4,
            cv2.BORDER_REFLECT_101,
        )
        tensor = torch.from_numpy(
            padded.astype(np.float32)[None, None] / 255.0,
        ).to(self.device)
        with torch.inference_mode():
            restored = torch.sigmoid(self.restorer(tensor))[0, 0]
        restored = restored[:work_height, :work_width].mul(255).byte().cpu().numpy()
        variants = [
            ("continuous", restored),
            ("otsu", cv2.threshold(
                restored,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]),
        ]
        for stage, variant in variants:
            decoded = _read(variant, self.settings.formats)
            if decoded:
                detection = self._accept(detection, decoded[0], 1, True)
                detection["sources"].append(f"learned-qr-restoration:{stage}")
                return detection
        return detection

    def process(self, image: np.ndarray) -> dict[str, Any]:
        started = perf_counter()
        direct = _read(image, self.settings.formats)
        detections = self.locate(image)
        for detection in detections:
            self._decode_consensus(image, detection)
            if detection["payload"] is None and detection["kind"] == "2d":
                self._restore_qr(image, detection)
        legacy = self._legacy_rescue(image) if self.settings.mode == "accuracy" else []
        results = _deduplicate([*direct, *detections, *legacy])
        if not self.settings.include_unresolved:
            results = [item for item in results if item["payload"] is not None]
        return {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "decoded": sum(item["payload"] is not None for item in results),
            "localized": len(results),
            "results": results,
        }

    def _legacy_rescue(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run the proven classical 5.1 routes in accuracy mode.

        Those routes cover Code 39, Code 128, Data Matrix, and QR Code. They
        remain complementary to the learned general locator, especially on
        dense document pages.
        """
        supported = {"code39", "code128", "datamatrix", "qrcode"}
        if self.settings.formats and not (self.settings.formats & supported):
            return []
        if self._legacy_module is None:
            directory = ROOT.parent / "v5.1-adaptive-generalization"
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
            spec = importlib.util.spec_from_file_location(
                "barcode_pipeline_5_1_rescue",
                directory / "pipeline.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("unable to load the local 5.1 rescue pipeline")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self._legacy_module = module
        module = self._legacy_module
        with tempfile.TemporaryDirectory(prefix="barcode-5.2-rescue-") as temporary:
            work = Path(temporary)
            page_path = work / "page.png"
            cv2.imwrite(str(page_path), image)
            config = module.Config(
                output=work / "unused",
                formats="all",
                render_dpi=300,
                overwrite=True,
                save_crops=False,
                save_overlays=False,
                residual_proposals=True,
                include_review_candidates=False,
                workers=1,
            )
            page = module.process_page(
                1,
                page_path,
                "5.2-accuracy-rescue",
                config,
                work,
                True,
            )
        output = []
        for item in [*page["results"], *page["unresolved_matrix"]]:
            payload = item.get("text")
            symbology = str(item.get("format") or "Unknown")
            if not _accepted_format(symbology, self.settings.formats):
                continue
            output.append(
                {
                    "polygon": item["quad"],
                    "kind": (
                        "2d"
                        if _normalized_format(symbology) in MATRIX_FORMATS
                        else "1d"
                    ),
                    "symbology": symbology,
                    "payload": payload,
                    "status": "decoded" if payload else "localized",
                    "confidence": float(item.get("confidence", 0.5)),
                    "sources": [
                        "classical-5.1-accuracy-rescue",
                        *item.get("sources", []),
                    ],
                    "evidence": item.get("evidence", {}),
                }
            )
        return output


def collect_inputs(input_path: Path, dpi: int) -> tuple[list[tuple[str, Path]], tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_dir():
        return [
            (path.name, path)
            for path in sorted(input_path.iterdir())
            if path.suffix.lower() in IMAGE_SUFFIXES
        ], None
    if input_path.suffix.lower() != ".pdf":
        return [(input_path.name, input_path)], None
    temporary = tempfile.TemporaryDirectory(prefix="barcode-5.2-")
    work = Path(temporary.name)
    # Prefer the largest native raster on each page. Re-rendering a PDF can
    # erase one-pixel bars that are still present in the embedded source.
    try:
        listing = subprocess.run(
            ["pdfimages", "-list", str(input_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = []
        for line in listing.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5 or not parts[0].isdigit() or not parts[1].isdigit():
                continue
            rows.append(
                {
                    "page": int(parts[0]),
                    "width": int(parts[3]),
                    "height": int(parts[4]),
                }
            )
        native_prefix = work / "native"
        subprocess.run(
            ["pdfimages", "-j", str(input_path), str(native_prefix)],
            check=True,
            capture_output=True,
        )
        files = sorted(
            (path for path in work.glob("native-*") if path.is_file()),
            key=lambda path: int(
                re.search(r"-(\d+)(?:\.[^.]+)?$", path.name).group(1)
            ),
        )
        if rows and len(rows) == len(files):
            best: dict[int, tuple[int, Path]] = {}
            for row, path in zip(rows, files):
                area = row["width"] * row["height"]
                if area >= 1_000_000 and area > best.get(row["page"], (0, path))[0]:
                    best[row["page"]] = (area, path)
            page_count = max(row["page"] for row in rows)
            if len(best) == page_count:
                return [
                    (f"page-{page:04d}", best[page][1])
                    for page in range(1, page_count + 1)
                ], temporary
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        pass
    prefix = work / "page"
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", str(dpi), str(input_path), str(prefix)],
        check=True,
    )
    pages = sorted(work.glob("page-*.jpg"))
    return [
        (f"page-{index:04d}", path)
        for index, path in enumerate(pages, 1)
    ], temporary


def write_overlay(image: np.ndarray, results: list[dict[str, Any]], destination: Path) -> None:
    canvas = image.copy()
    for item in results:
        points = np.asarray(item["polygon"], np.int32).reshape(-1, 1, 2)
        color = (20, 190, 20) if item["payload"] else (0, 170, 255)
        cv2.polylines(canvas, [points], True, color, 3, cv2.LINE_AA)
        label = item["payload"] or f"unresolved {item['kind']}"
        anchor = tuple(points.reshape(-1, 2).min(axis=0))
        cv2.putText(canvas, label[:60], anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)


def run(
    input_path: Path,
    output: Path,
    settings: Settings,
    *,
    dpi: int = 300,
    overlays: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    pages, temporary = collect_inputs(input_path, dpi)
    pipeline = EvidenceGuidedPipeline(settings)
    page_results = []
    try:
        for index, (name, path) in enumerate(pages, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unable to read image: {path}")
            result = pipeline.process(image)
            result.update({"page": index, "source": name})
            page_results.append(result)
            if overlays:
                write_overlay(
                    image,
                    result["results"],
                    output / "overlays" / f"{index:04d}.jpg",
                )
    finally:
        if temporary is not None:
            temporary.cleanup()
    document = {
        "schema_version": "barcode-pipeline-5.2-v1",
        "input": str(input_path.resolve()),
        "settings": {
            "device": settings.device,
            "work_size": settings.work_size,
            "confidence": settings.confidence,
            "mode": settings.mode,
            "formats": sorted(settings.formats),
            "include_unresolved": settings.include_unresolved,
            "render_dpi": dpi,
        },
        "summary": {
            "pages": len(page_results),
            "decoded": sum(page["decoded"] for page in page_results),
            "localized": sum(page["localized"] for page in page_results),
            "latency_ms": round(sum(page["latency_ms"] for page in page_results), 3),
        },
        "pages": page_results,
    }
    (output / "detections.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return document
