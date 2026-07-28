"""Independent CPU-fast localization candidates."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
import importlib.util
import sys

import cv2
import numpy as np
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
DEFAULT_MODEL = (
    Path(__file__).resolve().parent
    / "models"
    / "fast-locator-v3.torchscript"
)
PIPELINE7_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "7-cpu-coarse-to-fine-localization"
    / "pipeline.py"
)


def _point(value: Any) -> np.ndarray:
    return np.asarray([float(value.x), float(value.y)], np.float32)


def zxing_quad(value: Any) -> np.ndarray | None:
    position = value.position
    points = np.asarray(
        [
            _point(position.top_left),
            _point(position.top_right),
            _point(position.bottom_right),
            _point(position.bottom_left),
        ],
        np.float32,
    )
    if abs(float(cv2.contourArea(points))) > 1e-6:
        return points
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    start, end = points[first], points[second]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return None
    normal = np.asarray([-direction[1], direction[0]], np.float32) / length
    return np.asarray(
        [start - normal, end - normal, end + normal, start + normal],
        np.float32,
    )


def _similarity(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first_hull = cv2.convexHull(np.asarray(first, np.float32))
    second_hull = cv2.convexHull(np.asarray(second, np.float32))
    first_area = abs(float(cv2.contourArea(first_hull)))
    second_area = abs(float(cv2.contourArea(second_hull)))
    if min(first_area, second_area) <= 1e-6:
        return 0.0, 0.0
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    intersection = max(0.0, float(intersection))
    union = first_area + second_area - intersection
    return (
        intersection / max(1e-6, union),
        intersection / min(first_area, second_area),
    )


def deduplicate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in sorted(
        values,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    ):
        if any(
            max(*_similarity(candidate["polygon"], prior["polygon"])) >= 0.75
            for prior in output
        ):
            continue
        output.append(candidate)
    return output


def zxing_geometry(
    image: np.ndarray,
    *,
    include_errors: bool,
    lean: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for barcode in zxingcpp.read_barcodes(
        image,
        try_rotate=True,
        try_downscale=not lean,
        try_invert=not lean,
        return_errors=include_errors,
    ):
        quad = zxing_quad(barcode)
        if quad is None:
            continue
        barcode_format = str(barcode.format)
        output.append(
            {
                "polygon": quad,
                "kind": "2d" if barcode_format in MATRIX_NAMES else "1d",
                "symbology": barcode_format,
                "payload": None,
                "confidence": 1.0 if barcode.valid else 0.35,
                "status": "localized",
                "sources": [
                    "zxing:valid-or-error-geometry"
                    if include_errors
                    else "zxing:valid-geometry"
                ],
            }
        )
    return deduplicate(output)


class TinyModelLocalizer:
    """Synthetic-trained 31k-parameter segmentation model, localization only."""

    def __init__(
        self,
        weights: Path = DEFAULT_MODEL,
        *,
        work_size: int = 256,
        confidence: float = 0.25,
        minimum_area: float = 14.0,
        interpolation: int = cv2.INTER_AREA,
    ) -> None:
        import torch

        if not weights.is_file():
            raise FileNotFoundError(weights)
        torch.set_num_threads(1)
        self.torch = torch
        self.model = torch.jit.load(str(weights), map_location="cpu").eval()
        self.work_size = work_size
        self.confidence = confidence
        self.minimum_area = minimum_area
        self.interpolation = interpolation
        probe = torch.ones((1, 1, 256, 256), dtype=torch.float32)
        with torch.inference_mode():
            self.model(probe)

    def __call__(self, image: np.ndarray) -> list[dict[str, Any]]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source_height, source_width = gray.shape[:2]
        scale = self.work_size / max(source_height, source_width)
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        work = cv2.resize(
            gray,
            (width, height),
            interpolation=self.interpolation,
        )
        padded_width = ((width + 31) // 32) * 32
        padded_height = ((height + 31) // 32) * 32
        padded = np.full((padded_height, padded_width), 255, np.uint8)
        padded[:height, :width] = work
        tensor = self.torch.from_numpy(padded[None, None]).float().div(255.0)
        with self.torch.inference_mode():
            heatmaps = self.torch.sigmoid(self.model(tensor))[0].numpy()

        output: list[dict[str, Any]] = []
        kernel = np.ones((3, 3), np.uint8)
        for class_index, kind in enumerate(("1d", "2d")):
            heatmap = heatmaps[class_index, :height, :width]
            binary = (heatmap >= self.confidence).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.minimum_area:
                    continue
                rectangle = cv2.minAreaRect(contour)
                if min(rectangle[1]) < 2:
                    continue
                component_mask = np.zeros_like(binary)
                cv2.drawContours(component_mask, [contour], -1, 1, -1)
                output.append(
                    {
                        "polygon": cv2.boxPoints(rectangle) / scale,
                        "kind": kind,
                        "symbology": "unknown",
                        "payload": None,
                        "confidence": float(
                            cv2.mean(heatmap, mask=component_mask)[0]
                        ),
                        "status": "localized",
                        "sources": ["clean-tiny-segmentation-model"],
                    }
                )
        return deduplicate(output)


class ModelZXingUnion:
    def __init__(self, model: TinyModelLocalizer) -> None:
        self.model = model

    def __call__(self, image: np.ndarray) -> list[dict[str, Any]]:
        return deduplicate(
            [
                *self.model(image),
                *zxing_geometry(image, include_errors=True, lean=True),
            ]
        )


class TiledTinyModelLocalizer:
    """Run the tiny model on four overlapping page quadrants."""

    def __init__(
        self,
        model: TinyModelLocalizer,
        *,
        overlap: float = 0.12,
    ) -> None:
        self.model = model
        self.overlap = overlap

    def __call__(self, image: np.ndarray) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        overlap_x = int(round(width * self.overlap / 2.0))
        overlap_y = int(round(height * self.overlap / 2.0))
        middle_x = width // 2
        middle_y = height // 2
        x_ranges = (
            (0, min(width, middle_x + overlap_x)),
            (max(0, middle_x - overlap_x), width),
        )
        y_ranges = (
            (0, min(height, middle_y + overlap_y)),
            (max(0, middle_y - overlap_y), height),
        )
        output: list[dict[str, Any]] = []
        for y1, y2 in y_ranges:
            for x1, x2 in x_ranges:
                for candidate in self.model(image[y1:y2, x1:x2]):
                    shifted = dict(candidate)
                    polygon = np.asarray(
                        candidate["polygon"],
                        np.float32,
                    ).copy()
                    polygon[:, 0] += x1
                    polygon[:, 1] += y1
                    shifted["polygon"] = polygon
                    shifted["sources"] = [
                        *candidate.get("sources", []),
                        "overlapping-2x2-tile",
                    ]
                    output.append(shifted)
        return deduplicate(output)


def _load_pipeline7() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_fast_localization_pipeline7_verifier",
        PIPELINE7_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(PIPELINE7_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StructurallyVerifiedModel:
    """Tiny full-page model followed by bounded native-pixel verification."""

    def __init__(self, model: TinyModelLocalizer) -> None:
        self.model = model
        self.pipeline7 = _load_pipeline7()

    def __call__(self, image: np.ndarray) -> list[dict[str, Any]]:
        gray = (
            image
            if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )
        output: list[dict[str, Any]] = []
        for candidate in self.model(image):
            quad = np.asarray(candidate["polygon"], np.float32)
            if candidate["kind"] == "1d":
                metrics = self.pipeline7.linear_metrics(gray, quad)
                rectangle = cv2.minAreaRect(quad)
                long_side = float(max(rectangle[1]))
                short_side = float(min(rectangle[1]))
                if (
                    long_side < 18.0
                    or short_side < 5.0
                    or long_side / max(short_side, 1e-6) < 1.45
                    or metrics.get("score", 0.0) < 0.55
                    or metrics.get("orientation", 0.0) < 0.55
                    or metrics.get("persistence", 0.0) < 0.025
                    or metrics.get("transitions", 0.0) < 10.0
                    or metrics.get("scanline_agreement", 0.0) < 0.42
                    or not 0.012
                    <= metrics.get("transition_rate", 0.0)
                    <= 0.75
                ):
                    continue
                candidate["confidence"] = float(
                    0.45 * candidate["confidence"]
                    + 0.55 * metrics["score"]
                )
                candidate["sources"].append(
                    "pipeline7:native-linear-structure"
                )
                candidate["metrics"] = metrics
                output.append(candidate)
                continue

            qr = self.pipeline7.locate_qr_in_candidate(gray, quad)
            if qr is not None:
                candidate["polygon"] = np.asarray(qr.quad, np.float32)
                candidate["confidence"] = max(
                    float(candidate["confidence"]),
                    float(qr.confidence),
                )
                candidate["sources"].append("pipeline7:local-qr-structure")
                candidate["metrics"] = dict(qr.metrics)
                output.append(candidate)
                continue
            data_matrix_quad, metrics = (
                self.pipeline7.refine_data_matrix_candidate(gray, quad)
            )
            if data_matrix_quad is not None:
                candidate["polygon"] = np.asarray(
                    data_matrix_quad,
                    np.float32,
                )
                candidate["confidence"] = max(
                    float(candidate["confidence"]),
                    float(metrics.get("grid_score", 0.0)),
                )
                candidate["sources"].append(
                    "pipeline7:local-data-matrix-structure"
                )
                candidate["metrics"] = metrics
                output.append(candidate)
        return deduplicate(output)


class LightVerifiedModel:
    """Bounded native structure checks for elongated and compact proposals."""

    def __init__(self, model: TinyModelLocalizer) -> None:
        self.model = model
        self.pipeline7 = _load_pipeline7()

    def __call__(self, image: np.ndarray) -> list[dict[str, Any]]:
        gray = (
            image
            if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )
        output: list[dict[str, Any]] = []
        for candidate in self.model(image):
            quad = np.asarray(candidate["polygon"], np.float32)
            rectangle = cv2.minAreaRect(quad)
            long_side = float(max(rectangle[1]))
            short_side = float(min(rectangle[1]))
            aspect = long_side / max(short_side, 1e-6)
            if aspect < 1.60:
                metrics = self.pipeline7.matrix_proposal_metrics(
                    gray,
                    quad,
                )
                if (
                    metrics.get("base_score", 0.0) < 0.45
                    or metrics.get("transition_score", 0.0) < 0.25
                ):
                    continue
                candidate["kind"] = "2d"
                candidate["sources"].append(
                    "pipeline7:light-native-matrix-structure"
                )
                candidate["metrics"] = metrics
                output.append(candidate)
                continue
            metrics = self.pipeline7.linear_metrics(gray, quad)
            if metrics.get("score", 0.0) < 0.55:
                continue
            candidate["kind"] = "1d"
            candidate["confidence"] = float(
                0.45 * candidate["confidence"]
                + 0.55 * metrics["score"]
            )
            candidate["sources"].append(
                "pipeline7:light-native-linear-structure"
            )
            candidate["metrics"] = metrics
            output.append(candidate)
        return deduplicate(output)


def method(
    name: str,
    *,
    weights: Path = DEFAULT_MODEL,
    work_size: int = 256,
    confidence: float = 0.25,
    minimum_area: float = 14.0,
    interpolation: int = cv2.INTER_AREA,
) -> Any:
    if name == "zxing-valid-lean":
        return lambda image: zxing_geometry(
            image,
            include_errors=False,
            lean=True,
        )
    if name == "zxing-errors-lean":
        return lambda image: zxing_geometry(
            image,
            include_errors=True,
            lean=True,
        )
    if name == "tiny-model":
        return TinyModelLocalizer(
            weights,
            work_size=work_size,
            confidence=confidence,
            minimum_area=minimum_area,
            interpolation=interpolation,
        )
    if name == "tiny-model-zxing":
        return ModelZXingUnion(
            TinyModelLocalizer(
                weights,
                work_size=work_size,
                confidence=confidence,
                minimum_area=minimum_area,
                interpolation=interpolation,
            )
        )
    if name == "tiny-model-tiled":
        return TiledTinyModelLocalizer(
            TinyModelLocalizer(
                weights,
                work_size=work_size,
                confidence=confidence,
                minimum_area=minimum_area,
                interpolation=interpolation,
            )
        )
    if name == "tiny-model-verified":
        return StructurallyVerifiedModel(
            TinyModelLocalizer(
                weights,
                work_size=work_size,
                confidence=confidence,
                minimum_area=minimum_area,
                interpolation=interpolation,
            )
        )
    if name == "tiny-model-light-verified":
        return LightVerifiedModel(
            TinyModelLocalizer(
                weights,
                work_size=work_size,
                confidence=confidence,
                minimum_area=minimum_area,
                interpolation=interpolation,
            )
        )
    raise ValueError(name)


METHODS = (
    "zxing-valid-lean",
    "zxing-errors-lean",
    "tiny-model",
    "tiny-model-tiled",
    "tiny-model-light-verified",
    "tiny-model-verified",
    "tiny-model-zxing",
)
