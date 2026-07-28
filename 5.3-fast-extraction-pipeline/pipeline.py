"""Ultra-fast extraction candidates and the selected model-to-crop pipeline."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
import zxingcpp


MATRIX_NAMES = {
    "Aztec",
    "Data Matrix",
    "MaxiCode",
    "Micro QR Code",
    "PDF417",
    "QR Code",
    "rMQR Code",
}
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "models" / "fast-locator-v1.torchscript"


@dataclass(frozen=True)
class FastConfig:
    weights: Path = DEFAULT_WEIGHTS
    work_size: int = 256
    confidence_threshold: float = 0.38
    minimum_component_area: float = 14.0
    crop_padding_fraction: float = 0.18
    try_downscale: bool = True
    try_invert: bool = False


def _point(value: Any) -> tuple[float, float]:
    return float(value.x), float(value.y)


def _barcode_quad(barcode: Any, scale: float = 1.0) -> np.ndarray:
    position = barcode.position
    return np.asarray(
        [
            _point(position.top_left),
            _point(position.top_right),
            _point(position.bottom_right),
            _point(position.bottom_left),
        ],
        np.float32,
    ) / scale


def _kind(barcode: Any) -> str:
    return "2d" if str(barcode.format) in MATRIX_NAMES else "linear"


def _prediction(
    barcode: Any,
    *,
    quad: np.ndarray | None = None,
    source: str,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "quad": np.asarray(quad if quad is not None else _barcode_quad(barcode), np.float32),
        "text": str(barcode.text),
        "format": str(barcode.format),
        "kind": _kind(barcode),
        "confidence": float(confidence),
        "sources": [source],
    }


def _overlap(first: np.ndarray, second: np.ndarray) -> float:
    first = cv2.convexHull(np.asarray(first, np.float32))
    second = cv2.convexHull(np.asarray(second, np.float32))
    areas = abs(float(cv2.contourArea(first))), abs(float(cv2.contourArea(second)))
    if min(areas) <= 1e-6:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    return float(intersection) / min(areas)


def deduplicate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in values:
        duplicate = next(
            (
                prior
                for prior in output
                if prior["text"] == candidate["text"]
                and _overlap(prior["quad"], candidate["quad"]) >= 0.55
            ),
            None,
        )
        if duplicate is None:
            output.append(candidate)
        else:
            duplicate["sources"] = sorted(
                set(duplicate.get("sources", ())) | set(candidate.get("sources", ()))
            )
            duplicate["confidence"] = max(
                float(duplicate.get("confidence", 0.0)),
                float(candidate.get("confidence", 0.0)),
            )
    return output


def direct_decode(
    image: np.ndarray,
    *,
    formats: Any = zxingcpp.BarcodeFormat.AllReadable,
    max_dimension: int | None = None,
    try_downscale: bool = True,
    try_invert: bool = True,
    binarizer: Any = zxingcpp.Binarizer.LocalAverage,
) -> list[dict[str, Any]]:
    """One compiled full-image decoder pass; no proposal or recovery stages."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 1.0
    if max_dimension and max(gray.shape) > max_dimension:
        scale = max_dimension / max(gray.shape)
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    output: list[dict[str, Any]] = []
    for barcode in zxingcpp.read_barcodes(
        gray,
        formats=formats,
        try_rotate=True,
        try_downscale=try_downscale,
        try_invert=try_invert,
        binarizer=binarizer,
        return_errors=False,
    ):
        if barcode.valid and barcode.text:
            output.append(
                _prediction(
                    barcode,
                    quad=_barcode_quad(barcode, scale),
                    source="zxing:single-page-pass",
                )
            )
    return deduplicate(output)


class FastModelExtractor:
    """Tiny segmentation proposals followed by one native crop decode."""

    def __init__(self, config: FastConfig = FastConfig()) -> None:
        if not config.weights.is_file():
            raise FileNotFoundError(f"fast locator weights not found: {config.weights}")
        self.config = config
        self._load_lock = threading.Lock()
        self._model: Any | None = None

    def warm(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            # Small models are faster with one intra-op worker.  Codara already
            # parallelizes independent pages at the job level.
            torch.set_num_threads(1)
            self._model = torch.jit.load(
                str(self.config.weights),
                map_location="cpu",
            ).eval()
            probe = torch.ones((1, 1, 256, 256), dtype=torch.float32)
            with torch.inference_mode():
                self._model(probe)

    def localize(self, image: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        self.warm()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source_height, source_width = gray.shape[:2]
        scale = self.config.work_size / max(source_height, source_width)
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        work = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        padded_width = ((width + 31) // 32) * 32
        padded_height = ((height + 31) // 32) * 32
        padded = np.full((padded_height, padded_width), 255, np.uint8)
        padded[:height, :width] = work
        tensor = torch.from_numpy(padded[None, None]).float().div(255.0)
        started = perf_counter()
        with torch.inference_mode():
            heatmaps = torch.sigmoid(self._model(tensor))[0].numpy()
        inference_ms = (perf_counter() - started) * 1000.0

        output: list[dict[str, Any]] = []
        kernel = np.ones((3, 3), np.uint8)
        for class_index, kind in enumerate(("linear", "2d")):
            heatmap = heatmaps[class_index, :height, :width]
            binary = (heatmap >= self.config.confidence_threshold).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.config.minimum_component_area:
                    continue
                rectangle = cv2.minAreaRect(contour)
                if min(rectangle[1]) < 2:
                    continue
                component_mask = np.zeros_like(binary)
                cv2.drawContours(component_mask, [contour], -1, 1, -1)
                output.append(
                    {
                        "quad": cv2.boxPoints(rectangle) / scale,
                        "kind": kind,
                        "confidence": float(cv2.mean(heatmap, mask=component_mask)[0]),
                    }
                )
        return output, inference_ms

    def _rectified_crop(
        self,
        image: np.ndarray,
        quad: np.ndarray,
        kind: str,
    ) -> np.ndarray:
        points = np.asarray(quad, np.float32)
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        points = points[np.argsort(angles)]
        start = int(np.argmin(points.sum(axis=1)))
        points = np.roll(points, -start, axis=0)
        first_edge = points[1] - points[0]
        second_edge = points[2] - points[1]
        if first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0] < 0:
            points = points[[0, 3, 2, 1]]
        expansion = 1.0 + 2.0 * self.config.crop_padding_fraction
        expanded = center + (points - center) * expansion
        target_width = max(
            16,
            int(
                round(
                    max(
                        np.linalg.norm(expanded[1] - expanded[0]),
                        np.linalg.norm(expanded[2] - expanded[3]),
                    )
                )
            ),
        )
        target_height = max(
            16,
            int(
                round(
                    max(
                        np.linalg.norm(expanded[3] - expanded[0]),
                        np.linalg.norm(expanded[2] - expanded[1]),
                    )
                )
            ),
        )
        destination = np.asarray(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            np.float32,
        )
        crop = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(expanded, destination),
            (target_width, target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        if kind == "linear" and crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        return cv2.copyMakeBorder(
            crop,
            12,
            12,
            16,
            16,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    def __call__(
        self,
        image: np.ndarray,
        *,
        formats: Any = zxingcpp.BarcodeFormat.AllReadable,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        proposals, inference_ms = self.localize(gray)
        output: list[dict[str, Any]] = []
        decode_started = perf_counter()
        for proposal in proposals:
            crop = self._rectified_crop(
                gray,
                proposal["quad"],
                proposal["kind"],
            )
            reads = zxingcpp.read_barcodes(
                crop,
                formats=formats,
                try_rotate=True,
                try_downscale=self.config.try_downscale,
                try_invert=self.config.try_invert,
                binarizer=zxingcpp.Binarizer.LocalAverage,
                return_errors=False,
            )
            if not any(barcode.valid and barcode.text for barcode in reads):
                reads = zxingcpp.read_barcodes(
                    crop,
                    formats=formats,
                    try_rotate=True,
                    try_downscale=False,
                    try_invert=False,
                    binarizer=zxingcpp.Binarizer.FixedThreshold,
                    return_errors=False,
                )
            if (
                proposal["kind"] == "linear"
                and not any(barcode.valid and barcode.text for barcode in reads)
            ):
                clahe = cv2.createCLAHE(
                    clipLimit=2.0,
                    tileGridSize=(8, 8),
                ).apply(crop)
                reads = zxingcpp.read_barcodes(
                    clahe,
                    formats=formats,
                    try_rotate=True,
                    try_downscale=False,
                    try_invert=False,
                    binarizer=zxingcpp.Binarizer.LocalAverage,
                    return_errors=False,
                )
            if (
                proposal["kind"] == "linear"
                and not any(barcode.valid and barcode.text for barcode in reads)
            ):
                enlarged = cv2.resize(
                    crop,
                    None,
                    fx=2.0,
                    fy=2.0,
                    interpolation=cv2.INTER_CUBIC,
                )
                reads = zxingcpp.read_barcodes(
                    enlarged,
                    formats=formats,
                    try_rotate=True,
                    try_downscale=False,
                    try_invert=False,
                    binarizer=zxingcpp.Binarizer.LocalAverage,
                    return_errors=False,
                )
            if any(str(barcode.format) == "EAN-13" for barcode in reads):
                # A crop containing one valid EAN-13 can occasionally also
                # produce a conflicting UPC-E expansion.  The longer symbol
                # has more observed modules, so retain it and reject the
                # lower-evidence family interpretation.
                reads = [
                    barcode
                    for barcode in reads
                    if str(barcode.format) != "UPC-E"
                ]
            for barcode in reads:
                if not barcode.valid or not barcode.text:
                    continue
                if str(barcode.format) == "ITF" and len(str(barcode.text)) < 6:
                    continue
                output.append(
                    _prediction(
                        barcode,
                        quad=proposal["quad"],
                        source="fast-model:native-crop-zxing",
                        confidence=proposal["confidence"],
                    )
                )
        diagnostics = {
            "proposals": len(proposals),
            "model_inference_ms": inference_ms,
            "crop_decode_ms": (perf_counter() - decode_started) * 1000.0,
        }
        return deduplicate(output), diagnostics
