from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zxingcpp

from .evaluate import canonical_gtin, canonical_symbology
from .geometry import polygon, similarity


Prediction = dict[str, Any]
Competitor = Callable[[np.ndarray], list[Prediction]]


MATRIX_SYMBOLOGIES = {"AZTEC", "DATA_MATRIX", "MAXICODE", "PDF417", "QR_CODE"}
DEFAULT_PIERO_WEIGHTS = (
    Path(__file__).resolve().parents[2]
    / "bin"
    / "notebooks"
    / "piero_yolov8s"
    / "models"
    / "YOLOV8s_Barcode_Detection.pt"
)
DEFAULT_BAFALO_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "bafalo-models"
    / "Segmentation Models"
    / "bafalo_scnn_192-448_0.pt"
)
DEFAULT_QR_RESTORER_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "generated"
    / "tiny-qrdn-restorer.pt"
)
DEFAULT_TINY_LOCATOR_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "generated"
    / "tiny-barcode-locator.pt"
)
DEFAULT_ADNET_LENET_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "ADNet"
    / "lenet-qrcode.pth"
)
BAFALO_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "external" / "BarBeR" / "BaFaLo"


def _point(value: Any) -> list[float]:
    return [float(value.x), float(value.y)]


def _zxing_quad(value: Any) -> list[list[float]]:
    position = value.position
    points = np.asarray([
        _point(position.top_left),
        _point(position.top_right),
        _point(position.bottom_right),
        _point(position.bottom_left),
    ], dtype=np.float32)
    if abs(float(cv2.contourArea(points))) <= 1e-6:
        # Linear readers sometimes report a scan segment rather than a box.
        # Give that measured segment a two-pixel support band so it remains a
        # valid localization polygon without pretending to know barcode height.
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
        start, end = points[first], points[second]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            return []
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float32) / length
        points = np.asarray([
            start - normal,
            end - normal,
            end + normal,
            start + normal,
        ], dtype=np.float32)
    return points.tolist()


def _deduplicate(values: list[Prediction], threshold: float = 0.55) -> list[Prediction]:
    accepted: list[Prediction] = []
    for candidate in sorted(values, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        try:
            candidate_polygon = polygon(candidate["polygon"], "candidate")
        except ValueError:
            continue
        duplicate = False
        for existing in accepted:
            existing_polygon = polygon(existing["polygon"], "existing")
            iou, intersection_over_smaller = similarity(candidate_polygon, existing_polygon)
            if iou >= threshold or intersection_over_smaller >= 0.80:
                duplicate = True
                # Preserve a valid payload even when an unresolved proposal had
                # slightly stronger geometric evidence.
                if candidate.get("payload") and not existing.get("payload"):
                    existing.update(candidate)
                break
        if not duplicate:
            accepted.append(candidate)
    return accepted


def zxing_direct(image: np.ndarray, *, include_errors: bool = False) -> list[Prediction]:
    output: list[Prediction] = []
    for barcode in zxingcpp.read_barcodes(
        image,
        try_rotate=True,
        try_downscale=True,
        try_invert=True,
        return_errors=include_errors,
    ):
        symbology = canonical_symbology(str(barcode.format))
        valid = bool(barcode.valid)
        quad = _zxing_quad(barcode)
        if not quad:
            continue
        output.append({
            "polygon": quad,
            "kind": "2d" if symbology in MATRIX_SYMBOLOGIES else "1d",
            "symbology": symbology,
            "payload": str(barcode.text) if valid else None,
            "confidence": 1.0 if valid else 0.35,
            "status": "decoded" if valid else "decoder_error",
            "sources": ["zxing-direct"],
            "evidence": {"error": str(barcode.error) if not valid else None},
        })
    return _deduplicate(output)


def opencv_direct(image: np.ndarray) -> list[Prediction]:
    output: list[Prediction] = []
    detector = cv2.barcode.BarcodeDetector()
    try:
        detected, decoded_info, decoded_types, points = detector.detectAndDecodeWithType(image)
    except cv2.error:
        detected, decoded_info, decoded_types, points = False, (), (), None
    if detected and points is not None:
        for index, quad in enumerate(np.asarray(points, dtype=np.float32)):
            text = str(decoded_info[index]) if index < len(decoded_info) and decoded_info[index] else None
            symbology = canonical_symbology(decoded_types[index] if index < len(decoded_types) else None)
            output.append({
                "polygon": quad.reshape(-1, 2).tolist(),
                "kind": "1d",
                "symbology": symbology,
                "payload": text,
                "confidence": 1.0 if text else 0.55,
                "status": "decoded" if text else "localized",
                "sources": ["opencv-barcode-direct"],
            })

    qr_detector = cv2.QRCodeDetector()
    try:
        detected, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(image)
    except cv2.error:
        detected, decoded_info, points = False, (), None
    if detected and points is not None:
        for index, quad in enumerate(np.asarray(points, dtype=np.float32)):
            text = str(decoded_info[index]) if index < len(decoded_info) and decoded_info[index] else None
            output.append({
                "polygon": quad.reshape(-1, 2).tolist(),
                "kind": "2d",
                "symbology": "QR_CODE",
                "payload": text,
                "confidence": 1.0 if text else 0.55,
                "status": "decoded" if text else "localized",
                "sources": ["opencv-qr-direct"],
            })
    return _deduplicate(output)


def zxing_opencv_direct(image: np.ndarray) -> list[Prediction]:
    """Fast, from-scratch direct-engine ensemble.

    ZXing error locations recover plausible symbols that fail checksum or
    format parsing; OpenCV contributes complementary 1D and QR geometry.
    """
    return _deduplicate(
        [
            # ZXing's return_errors mode is not a strict superset: on noisy
            # QR inputs it can omit valid results found by the normal pass.
            *zxing_direct(image, include_errors=False),
            *zxing_direct(image, include_errors=True),
            *opencv_direct(image),
        ],
        threshold=0.45,
    )


def restoration_decode(image: np.ndarray) -> list[Prediction]:
    """Staged classical restoration with an early valid-decode exit.

    This operationalizes the restoration-before-decoding family of papers
    without claiming that visual enhancement itself is success. Every stage
    must produce a checksum/error-correction-valid decoder result.
    """
    original = zxing_direct(image, include_errors=False)
    if original:
        return original
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    unsharp = cv2.addWeighted(
        gray,
        2.0,
        cv2.GaussianBlur(gray, (0, 0), 2.0),
        -1.0,
        0.0,
    )
    stages: list[tuple[str, np.ndarray, float]] = [
        ("clahe", clahe, 1.0),
        (
            "clahe-otsu",
            cv2.threshold(
                clahe,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1],
            1.0,
        ),
        ("unsharp", unsharp, 1.0),
        (
            "adaptive-threshold",
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5,
            ),
            1.0,
        ),
        (
            "bicubic-2x",
            cv2.resize(
                image,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            ),
            2.0,
        ),
    ]
    for stage_name, variant, scale in stages:
        decoded = zxing_direct(variant, include_errors=False)
        if not decoded:
            continue
        for prediction in decoded:
            if scale != 1.0:
                prediction["polygon"] = [
                    [float(point[0]) / scale, float(point[1]) / scale]
                    for point in prediction["polygon"]
                ]
            prediction["sources"] = [
                *prediction.get("sources", []),
                f"restoration:{stage_name}",
            ]
        return decoded
    return []


class LearnedQRRestorationDecode:
    """Paired QR restoration whose output is accepted only after valid decode."""

    def __init__(
        self,
        weights: Path = DEFAULT_QR_RESTORER_WEIGHTS,
        *,
        device: str = "cpu",
    ) -> None:
        import torch

        from .restoration import TinyQRRestorer

        if not weights.is_file():
            raise FileNotFoundError(f"QR restorer weights not found: {weights}")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.torch = torch
        self.device = torch.device(device)
        self.model = TinyQRRestorer(width=int(checkpoint.get("width", 8)))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        # Preserve the fastest and safest path for already-readable symbols.
        direct = zxing_direct(image, include_errors=False)
        if direct:
            return direct
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        height, width = gray.shape
        padded_height = ((height + 3) // 4) * 4
        padded_width = ((width + 3) // 4) * 4
        padded = cv2.copyMakeBorder(
            gray,
            0,
            padded_height - height,
            0,
            padded_width - width,
            cv2.BORDER_REFLECT_101,
        )
        tensor = self.torch.from_numpy(
            padded.astype(np.float32)[None, None] / 255.0,
        ).to(self.device)
        with self.torch.inference_mode():
            restored = self.torch.sigmoid(self.model(tensor))[0, 0]
        restored = (
            restored[:height, :width].clamp(0, 1).mul(255).byte().cpu().numpy()
        )
        # Continuous and binarized reconstructions fail differently. Both
        # remain hypotheses until the standard decoder validates a payload.
        variants = [
            ("learned-continuous", restored),
            ("learned-otsu", cv2.threshold(
                restored,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]),
        ]
        for stage_name, variant in variants:
            decoded = zxing_direct(variant, include_errors=False)
            if decoded:
                for prediction in decoded:
                    prediction["sources"] = [
                        *prediction.get("sources", []),
                        f"qr-restoration:{stage_name}",
                    ]
                return decoded
        return []


class ADNetLENetQRDecode:
    """Late-2025 external QR motion-deblurring checkpoint benchmark."""

    def __init__(
        self,
        weights: Path = DEFAULT_ADNET_LENET_WEIGHTS,
        *,
        device: str = "cpu",
    ) -> None:
        import torch

        if not weights.is_file():
            raise FileNotFoundError(f"ADNet LENet weights not found: {weights}")
        from .adnet_lenet import ExternalADNetLENet

        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.torch = torch
        self.functional = torch.nn.functional
        self.device = torch.device(device)
        self.model = ExternalADNetLENet(input_channels=3, base_channels=16)
        self.model.load_state_dict(checkpoint["params"])
        self.model.to(self.device).eval()

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        direct = zxing_direct(image, include_errors=False)
        if direct:
            return direct
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(
            rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0,
        ).to(self.device)
        pad_height = (8 - height % 8) % 8
        pad_width = (8 - width % 8) % 8
        tensor = self.functional.pad(
            tensor,
            (0, pad_width, 0, pad_height),
            mode="reflect",
        )
        with self.torch.inference_mode():
            restored = self.model(tensor)[:, :, :height, :width].clamp(0, 1)
        restored_rgb = (
            restored[0].mul(255).byte().permute(1, 2, 0).cpu().numpy()
        )
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
        variants = (
            ("continuous", restored_bgr),
            ("otsu", cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]),
        )
        for stage, variant in variants:
            decoded = zxing_direct(variant, include_errors=False)
            if decoded:
                for prediction in decoded:
                    prediction["sources"] = [
                        *prediction.get("sources", []),
                        f"external-adnet-lenet:{stage}",
                    ]
                return decoded
        return []


class PieroYoloV8s:
    """Adapter for the unmodified public Piero2411 YOLOv8s checkpoint."""

    def __init__(
        self,
        weights: Path = DEFAULT_PIERO_WEIGHTS,
        *,
        image_size: int = 1280,
        confidence: float = 0.10,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        if not weights.is_file():
            raise FileNotFoundError(f"Piero YOLO weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.image_size = image_size
        self.confidence = confidence
        self.device = device

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        result = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=0.45,
            device=self.device,
            save=False,
            verbose=False,
        )[0]
        output: list[Prediction] = []
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, class_id)).lower()
            kind = "2d" if "qr" in class_name else "1d"
            output.append({
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "kind": kind,
                "symbology": "QR_CODE" if kind == "2d" else "unknown",
                "payload": None,
                "confidence": float(box.conf[0]),
                "status": "localized",
                "sources": ["Piero2411/YOLOV8s-Barcode-Detection"],
            })
        return _deduplicate(output, threshold=0.55)


def _tile_windows(
    width: int,
    height: int,
    *,
    columns: int = 2,
    rows: int = 2,
    overlap_fraction: float = 0.15,
) -> list[tuple[int, int, int, int]]:
    """Return clamped, overlapping windows that cover the complete image."""
    if width <= 0 or height <= 0 or columns <= 0 or rows <= 0:
        return []
    tile_width = int(np.ceil(width / columns))
    tile_height = int(np.ceil(height / rows))
    overlap_x = int(round(tile_width * overlap_fraction))
    overlap_y = int(round(tile_height * overlap_fraction))
    windows = []
    for row in range(rows):
        for column in range(columns):
            nominal_x1 = column * tile_width
            nominal_y1 = row * tile_height
            nominal_x2 = min(width, (column + 1) * tile_width)
            nominal_y2 = min(height, (row + 1) * tile_height)
            windows.append((
                max(0, nominal_x1 - overlap_x),
                max(0, nominal_y1 - overlap_y),
                min(width, nominal_x2 + overlap_x),
                min(height, nominal_y2 + overlap_y),
            ))
    return windows


class PieroYoloV8sTiled(PieroYoloV8s):
    """Full-frame plus 2x2 overlap inference for small-code recovery."""

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        output = super().__call__(image)
        height, width = image.shape[:2]
        for tile_index, (x1, y1, x2, y2) in enumerate(
            _tile_windows(width, height),
        ):
            crop = image[y1:y2, x1:x2]
            for prediction in super().__call__(crop):
                mapped = dict(prediction)
                mapped["polygon"] = [
                    [float(point[0]) + x1, float(point[1]) + y1]
                    for point in prediction["polygon"]
                ]
                mapped["sources"] = [
                    *prediction.get("sources", []),
                    f"overlap-tile-{tile_index}",
                ]
                output.append(mapped)
        return _deduplicate(output, threshold=0.55)


def _axis_crop(image: np.ndarray, points: Any, padding_fraction: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    height, width = image.shape[:2]
    x1, y1 = values.min(axis=0)
    x2, y2 = values.max(axis=0)
    pad_x = padding_fraction * max(1.0, float(x2 - x1))
    pad_y = padding_fraction * max(1.0, float(y2 - y1))
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    return image[top:bottom, left:right]


def _decode_crop_variants(crop: np.ndarray) -> list[Prediction]:
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    variants: list[tuple[np.ndarray, bool]] = [
        (crop, True),
        (gray, False),
        (clahe, True),
        (binary, False),
    ]
    short_side = min(gray.shape[:2])
    if short_side < 160:
        scale = min(4.0, 180.0 / max(short_side, 1))
        variants.extend([
            (
                cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
                use_opencv,
            )
            for value, use_opencv in list(variants)
        ])
    for variant, use_opencv in variants:
        decoded = zxing_direct(variant, include_errors=False)
        if use_opencv:
            decoded.extend(
                item for item in opencv_direct(variant)
                if item.get("payload") is not None
            )
        valid = [item for item in decoded if item.get("payload") is not None]
        if valid:
            return valid
    return []


def _decode_crop_hypotheses(
    crop: np.ndarray,
    *,
    lite: bool = False,
) -> list[Prediction]:
    """Collect independent valid hypotheses instead of stopping at the first.

    A valid checksum is necessary but not sufficient: a damaged crop can still
    decode to another valid code.  Transform agreement supplies the ambiguity
    signal used by the consensus decoder below.
    """
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    unsharp = cv2.addWeighted(
        gray,
        1.8,
        cv2.GaussianBlur(gray, (0, 0), 1.5),
        -0.8,
        0.0,
    )
    variants: list[tuple[str, np.ndarray, bool]] = [
        ("original", crop, True),
        ("gray", gray, False),
        ("clahe", clahe, True),
        ("otsu", cv2.threshold(
            clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1], False),
    ]
    if lite:
        variants = [variants[0], variants[2], variants[3]]
    else:
        variants.extend([
            ("unsharp", unsharp, False),
            ("adaptive", cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5,
            ), False),
        ])
    short_side = min(gray.shape[:2])
    if short_side < 180:
        scale = min(4.0, 200.0 / max(short_side, 1))
        variants.extend([
            (
                f"{name}-upscaled",
                cv2.resize(value, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
                use_opencv,
            )
            for name, value, use_opencv in list(variants[:2] if lite else variants[:4])
        ])

    output: list[Prediction] = []
    for stage, variant, use_opencv in variants:
        for item in zxing_direct(variant, include_errors=False):
            if item.get("payload") is not None:
                candidate = dict(item)
                candidate["_vote_source"] = f"zxing:{stage}"
                output.append(candidate)
        if use_opencv:
            for item in opencv_direct(variant):
                if item.get("payload") is not None:
                    candidate = dict(item)
                    candidate["_vote_source"] = f"opencv:{stage}"
                    output.append(candidate)
    return output


def _payload_vote_key(item: Prediction) -> tuple[str, str]:
    raw = str(item.get("payload") or "")
    symbology = canonical_symbology(item.get("symbology"))
    gtin = canonical_gtin(raw, symbology)
    return ("GTIN", gtin) if gtin is not None else (symbology, raw)


def _decode_localized_candidates_consensus(
    image: np.ndarray,
    detections: list[Prediction],
    *,
    lite: bool = False,
) -> list[Prediction]:
    """Decode multiple paddings/transforms and reject tied payload evidence."""
    for detection in detections:
        hypotheses: list[Prediction] = []
        paddings = (0.12, 0.24) if lite else (0.10, 0.18, 0.30)
        for padding in paddings:
            crop = _axis_crop(image, detection["polygon"], padding)
            for item in _decode_crop_hypotheses(crop, lite=lite):
                candidate = dict(item)
                candidate["_vote_source"] = (
                    f"pad-{padding:.2f}:{item.get('_vote_source', 'decoder')}"
                )
                hypotheses.append(candidate)
        if not hypotheses:
            continue

        # One transform/engine contributes at most one vote to a payload.
        votes: dict[tuple[str, str], set[str]] = defaultdict(set)
        examples: dict[tuple[str, str], Prediction] = {}
        for item in hypotheses:
            key = _payload_vote_key(item)
            votes[key].add(str(item["_vote_source"]))
            examples.setdefault(key, item)
        ranked = sorted(
            votes,
            key=lambda key: (len(votes[key]), key),
            reverse=True,
        )
        winner = ranked[0]
        winner_votes = len(votes[winner])
        runner_up_votes = len(votes[ranked[1]]) if len(ranked) > 1 else 0
        if winner_votes <= runner_up_votes:
            detection["evidence"] = {
                **detection.get("evidence", {}),
                "decode_rejected": "ambiguous_payload_tie",
                "candidate_payloads": len(ranked),
            }
            continue
        selected = examples[winner]
        detection["payload"] = str(selected["payload"])
        detection["symbology"] = selected.get("symbology", detection.get("symbology"))
        detection["status"] = "decoded"
        detection["sources"] = list(dict.fromkeys([
            *detection.get("sources", []),
            *selected.get("sources", []),
            "detector-gated-consensus-decode",
        ]))
        detection["evidence"] = {
            **detection.get("evidence", {}),
            "decode_votes": winner_votes,
            "decode_margin": winner_votes - runner_up_votes,
            "candidate_payloads": len(ranked),
        }
    return detections


def _decode_localized_candidates_adaptive(
    image: np.ndarray,
    detections: list[Prediction],
) -> list[Prediction]:
    """Accept two-padding agreement; escalate only unresolved/discordant crops."""
    for detection in detections:
        primary: list[Prediction] = []
        for padding in (0.12, 0.24):
            crop = _axis_crop(image, detection["polygon"], padding)
            for item in zxing_direct(crop, include_errors=False):
                if item.get("payload") is not None:
                    candidate = dict(item)
                    candidate["_vote_source"] = f"pad-{padding:.2f}:zxing:original"
                    primary.append(candidate)
        votes: dict[tuple[str, str], set[str]] = defaultdict(set)
        examples: dict[tuple[str, str], Prediction] = {}
        for item in primary:
            key = _payload_vote_key(item)
            votes[key].add(str(item["_vote_source"]))
            examples.setdefault(key, item)
        agreed = [
            key for key, sources in votes.items()
            if len(sources) >= 2
        ]
        if len(agreed) == 1:
            selected = examples[agreed[0]]
            detection["payload"] = str(selected["payload"])
            detection["symbology"] = selected.get("symbology", detection.get("symbology"))
            detection["status"] = "decoded"
            detection["sources"] = list(dict.fromkeys([
                *detection.get("sources", []),
                *selected.get("sources", []),
                "detector-gated-adaptive-consensus",
            ]))
            detection["evidence"] = {
                **detection.get("evidence", {}),
                "decode_votes": len(votes[agreed[0]]),
                "decode_margin": len(votes[agreed[0]]),
                "adaptive_escalation": False,
            }
            continue
        _decode_localized_candidates_consensus(
            image,
            [detection],
            lite=True,
        )
        detection["evidence"] = {
            **detection.get("evidence", {}),
            "adaptive_escalation": True,
        }
    return detections


def _decode_localized_candidates(
    image: np.ndarray,
    detections: list[Prediction],
    whole_image: list[Prediction],
) -> list[Prediction]:
    """Transfer valid whole-image/crop payloads onto detector geometry."""
    from .fusion import fuse_predictions

    detections = fuse_predictions(detections, [whole_image], overlap_threshold=0.45)
    for detection in detections:
        if detection.get("payload") is not None:
            continue
        crop = _axis_crop(
            image,
            detection["polygon"],
            0.18 if detection.get("kind") == "1d" else 0.12,
        )
        payload_candidates = [
            item for item in _decode_crop_variants(crop)
            if item.get("payload") is not None
        ]
        if not payload_candidates:
            continue
        counts = Counter(str(item["payload"]) for item in payload_candidates)
        selected_payload, _ = counts.most_common(1)[0]
        selected = next(
            item for item in payload_candidates
            if str(item["payload"]) == selected_payload
        )
        detection["payload"] = selected_payload
        detection["symbology"] = selected.get("symbology", detection.get("symbology"))
        detection["status"] = "decoded"
        detection["sources"] = list(dict.fromkeys([
            *detection.get("sources", []),
            *selected.get("sources", []),
            "detector-gated-crop-decode",
        ]))
    return detections


class PieroYoloV8sDecode(PieroYoloV8s):
    """Learned geometry plus detector-gated, checksum-valid decoding."""

    def whole_image_candidates(self, image: np.ndarray) -> list[Prediction]:
        return [
            *zxing_direct(image, include_errors=False),
            *(item for item in opencv_direct(image) if item.get("payload") is not None),
        ]

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        detections = super().__call__(image)
        whole_image = self.whole_image_candidates(image)
        return _decode_localized_candidates(image, detections, whole_image)


class PieroYoloV8sDecodeFast(PieroYoloV8sDecode):
    """Latency-oriented cascade; ZXing runs only on unresolved detector crops."""

    def whole_image_candidates(self, image: np.ndarray) -> list[Prediction]:
        return [
            item for item in opencv_direct(image)
            if item.get("payload") is not None
        ]


class PublishedBaFaLo:
    """Exact published BaFaLo checkpoint adapter (external AGPL-3.0 code)."""

    def __init__(
        self,
        weights: Path = DEFAULT_BAFALO_WEIGHTS,
        *,
        work_size: int = 320,
        confidence_threshold: float = 0.4,
        minimum_component_area: float = 300.0,
    ) -> None:
        import importlib
        import sys
        import torch

        if not weights.is_file():
            raise FileNotFoundError(f"BaFaLo weights not found: {weights}")
        source_root = str(BAFALO_SOURCE_ROOT)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        # The checkpoint was serialized before fast_scnn_pico.py was renamed
        # to bafalo_scnn.py in the public repository.
        sys.modules.setdefault("fast_scnn_pico", importlib.import_module("bafalo_scnn"))
        torch.set_num_threads(1)
        self.torch = torch
        self.model = torch.load(weights, weights_only=False, map_location="cpu").eval()
        self.work_size = work_size
        self.confidence_threshold = confidence_threshold
        self.minimum_component_area = minimum_component_area

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        source_height, source_width = image.shape[:2]
        scale = float(self.work_size) / max(source_height, source_width)
        work_width = max(1, int(round(source_width * scale)))
        work_height = max(1, int(round(source_height * scale)))
        work = cv2.resize(
            image,
            (work_width, work_height),
            interpolation=cv2.INTER_CUBIC,
        )
        padded_width = ((work_width + 31) // 32) * 32
        padded_height = ((work_height + 31) // 32) * 32
        padded = np.pad(
            work,
            ((0, padded_height - work_height), (0, padded_width - work_width), (0, 0)),
        )
        tensor = self.torch.from_numpy(
            (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None],
        )
        with self.torch.inference_mode():
            heatmaps = self.torch.sigmoid(self.model(tensor))[0].cpu().numpy()

        output: list[Prediction] = []
        for class_index, kind in enumerate(("1d", "2d")):
            heatmap = heatmaps[class_index, :work_height, :work_width]
            binary = (heatmap > self.confidence_threshold).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                if cv2.contourArea(contour) <= self.minimum_component_area:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], -1, 1, -1)
                confidence = float(cv2.mean(heatmap, mask=mask)[0])
                x1, y1 = x / scale, y / scale
                x2, y2 = (x + width) / scale, (y + height) / scale
                output.append({
                    "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "kind": kind,
                    "symbology": "unknown",
                    "payload": None,
                    "confidence": confidence,
                    "status": "localized",
                    "sources": [
                        "BaFaLo published checkpoint fold 0",
                        "external AGPL-3.0 implementation",
                    ],
                })
        return _deduplicate(output, threshold=0.55)


class PublishedBaFaLoDecode(PublishedBaFaLo):
    """Published fast locator followed by checksum-valid staged crop decoding."""

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        detections = super().__call__(image)
        whole_image = [
            *zxing_direct(image, include_errors=False),
            *(item for item in opencv_direct(image) if item.get("payload") is not None),
        ]
        return _decode_localized_candidates(image, detections, whole_image)


class PublishedBaFaLoDecodeFast(PublishedBaFaLo):
    """Decode only small detector crops; avoids full-resolution decoder scans."""

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        return _decode_localized_candidates(image, super().__call__(image), [])


class PublishedBaFaLoDecodeConsensus(PublishedBaFaLo):
    """Published locator plus BLaDE-inspired multi-view payload consensus."""

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        return _decode_localized_candidates_consensus(image, super().__call__(image))


class PublishedBaFaLoDecodeConsensusLite(PublishedBaFaLo):
    """Pruned consensus intended to retain most safety at lower latency."""

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        return _decode_localized_candidates_consensus(
            image,
            super().__call__(image),
            lite=True,
        )


class PublishedBaFaLoDecodeAdaptiveConsensus(PublishedBaFaLo):
    """Two-view fast path with pruned consensus escalation."""

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        return _decode_localized_candidates_adaptive(image, super().__call__(image))


class IndependentTinyLocator:
    """Clean local compact segmentation model, trained by this benchmark lab."""

    def __init__(
        self,
        weights: Path = DEFAULT_TINY_LOCATOR_WEIGHTS,
        *,
        work_size: int = 320,
        confidence_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        import torch

        from .tiny_locator import TinyBarcodeLocator

        if not weights.is_file():
            raise FileNotFoundError(f"tiny locator weights not found: {weights}")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.torch = torch
        self.device = torch.device(device)
        self.model = TinyBarcodeLocator(width=int(checkpoint.get("width", 16)))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.work_size = work_size
        self.confidence_threshold = confidence_threshold

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        source_height, source_width = image.shape[:2]
        scale = self.work_size / max(source_height, source_width)
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.work_size, self.work_size, 3), 114, np.uint8)
        canvas[:height, :width] = resized
        tensor = self.torch.from_numpy(
            canvas[:, :, ::-1].copy().transpose(2, 0, 1)[None],
        ).float().div(255.0).to(self.device)
        with self.torch.inference_mode():
            heatmaps = self.torch.sigmoid(self.model(tensor))[0].cpu().numpy()
        output: list[Prediction] = []
        minimum_area = max(10.0, 0.00012 * width * height)
        for class_index, kind in enumerate(("1d", "2d")):
            heatmap = heatmaps[class_index, :height, :width]
            binary = (heatmap >= self.confidence_threshold).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < minimum_area:
                    continue
                rectangle = cv2.minAreaRect(contour)
                points = cv2.boxPoints(rectangle) / scale
                component_mask = np.zeros_like(binary)
                cv2.drawContours(component_mask, [contour], -1, 1, -1)
                output.append({
                    "polygon": points.tolist(),
                    "kind": kind,
                    "symbology": "QR_CODE" if kind == "2d" else "unknown",
                    "payload": None,
                    "confidence": float(cv2.mean(heatmap, mask=component_mask)[0]),
                    "status": "localized",
                    "sources": ["independent-tiny-barcode-locator"],
                })
        return _deduplicate(output, threshold=0.5)


class IndependentTinyDecodeAdaptiveConsensus(IndependentTinyLocator):
    """Clean locator plus the evidence-selected two-view decoder."""

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        return _decode_localized_candidates_adaptive(image, super().__call__(image))


def _ordered_quad(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    # The first point after angular ordering is usually top-left or top-right.
    # Rotate to the minimum x+y corner while preserving clockwise order.
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    first_edge = ordered[1] - ordered[0]
    second_edge = ordered[2] - ordered[1]
    cross_z = float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])
    if cross_z < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def _expanded_quad(value: Any, pad_x: float, pad_y: float) -> np.ndarray:
    points = _ordered_quad(value)
    top_left, top_right, bottom_right, bottom_left = points
    center = points.mean(axis=0)
    horizontal = (top_right - top_left) + (bottom_right - bottom_left)
    vertical = (bottom_left - top_left) + (bottom_right - top_right)
    horizontal /= max(float(np.linalg.norm(horizontal)), 1e-6)
    vertical /= max(float(np.linalg.norm(vertical)), 1e-6)
    half_width = 0.25 * (
        float(np.linalg.norm(top_right - top_left))
        + float(np.linalg.norm(bottom_right - bottom_left))
    ) * (1.0 + 2.0 * pad_x)
    half_height = 0.25 * (
        float(np.linalg.norm(bottom_left - top_left))
        + float(np.linalg.norm(bottom_right - top_right))
    ) * (1.0 + 2.0 * pad_y)
    return np.asarray([
        center - horizontal * half_width - vertical * half_height,
        center + horizontal * half_width - vertical * half_height,
        center + horizontal * half_width + vertical * half_height,
        center - horizontal * half_width + vertical * half_height,
    ], dtype=np.float32)


def _warp_candidate(
    image: np.ndarray,
    candidate: Prediction,
) -> tuple[np.ndarray, np.ndarray]:
    kind = candidate.get("kind")
    source = _expanded_quad(
        candidate["polygon"],
        pad_x=0.55 if kind == "1d" else 0.30,
        pad_y=0.30 if kind == "1d" else 0.30,
    )
    width = max(
        8,
        int(round(max(
            np.linalg.norm(source[1] - source[0]),
            np.linalg.norm(source[2] - source[3]),
        ))),
    )
    height = max(
        8,
        int(round(max(
            np.linalg.norm(source[3] - source[0]),
            np.linalg.norm(source[2] - source[1]),
        ))),
    )
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    page_to_crop = cv2.getPerspectiveTransform(source, destination)
    crop_to_page = np.linalg.inv(page_to_crop)
    crop = cv2.warpPerspective(
        image,
        page_to_crop,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, crop_to_page


def _map_prediction(item: Prediction, crop_to_page: np.ndarray) -> Prediction:
    points = np.asarray(item["polygon"], dtype=np.float32).reshape(1, -1, 2)
    mapped = cv2.perspectiveTransform(points, crop_to_page).reshape(-1, 2)
    result = dict(item)
    result["polygon"] = mapped.tolist()
    return result


def proposal_rescue(image: np.ndarray, *, retain_decoder_errors: bool = False) -> list[Prediction]:
    """Direct ensemble plus structure-tensor candidates confirmed on crops."""
    direct = zxing_opencv_direct(image)
    proposals = [
        item for item in structure_tensor(image)
        if float(item.get("confidence", 0.0)) >= 0.62
    ]
    proposals.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    rescued: list[Prediction] = []
    for proposal in proposals[:16]:
        proposal_polygon = polygon(proposal["polygon"], "proposal")
        if any(
            max(similarity(proposal_polygon, polygon(item["polygon"], "direct")))
            >= 0.45
            for item in [*direct, *rescued]
        ):
            continue
        crop, crop_to_page = _warp_candidate(image, proposal)
        variants: list[tuple[np.ndarray, np.ndarray]] = [(crop, crop_to_page)]
        if min(crop.shape[:2]) < 100:
            enlarged = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            scaled_to_crop = np.asarray(
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            variants.append((enlarged, crop_to_page @ scaled_to_crop))
        accepted_for_proposal: list[Prediction] = []
        for variant, variant_to_page in variants:
            local = [
                *zxing_direct(variant, include_errors=retain_decoder_errors),
                *opencv_direct(variant),
            ]
            for item in local:
                if item.get("payload") is None and not retain_decoder_errors:
                    continue
                mapped = _map_prediction(item, variant_to_page)
                mapped["sources"] = [
                    *proposal.get("sources", []),
                    *item.get("sources", []),
                    "proposal-rescue",
                ]
                if mapped.get("payload") is None:
                    mapped["confidence"] = max(
                        0.5,
                        min(0.75, 0.5 * float(proposal["confidence"]) + 0.25),
                    )
                accepted_for_proposal.append(mapped)
            if any(item.get("payload") is not None for item in accepted_for_proposal):
                break
        rescued.extend(_deduplicate(accepted_for_proposal, threshold=0.45))
    return _deduplicate([*direct, *rescued], threshold=0.45)


def _robust_threshold(response: np.ndarray, percentile: float) -> float:
    positive = response[response > 0]
    if positive.size == 0:
        return float("inf")
    high = float(np.percentile(positive, percentile))
    median = float(np.median(positive))
    mad = float(np.median(np.abs(positive - median)))
    # A MAD-derived threshold can exceed the observed maximum for sparse,
    # nearly bimodal edge maps. Cap that guard below the selected tail instead
    # of returning an impossible threshold.
    robust_guard = median + 3.5 * max(mad, 1e-6)
    return min(high, robust_guard)


def _component_predictions(
    response: np.ndarray,
    *,
    scale_to_source: float,
    kind: str,
    source: str,
    percentile: float,
    minimum_area: float,
) -> list[Prediction]:
    threshold = _robust_threshold(response, percentile)
    if not np.isfinite(threshold):
        return []
    mask = (response >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = response.shape
    image_area = float(height * width)
    output: list[Prediction] = []
    response_range = max(1e-6, float(response.max()) - threshold)
    for contour in contours:
        contour_area = abs(float(cv2.contourArea(contour)))
        if contour_area < minimum_area or contour_area > 0.75 * image_area:
            continue
        rectangle = cv2.minAreaRect(contour)
        side_a, side_b = map(float, rectangle[1])
        short_side, long_side = min(side_a, side_b), max(side_a, side_b)
        if short_side < 4 or long_side < 8:
            continue
        aspect = long_side / max(short_side, 1e-6)
        if kind == "1d" and not (1.25 <= aspect <= 25.0):
            continue
        if kind == "2d" and aspect > 4.0:
            continue
        local_mask = np.zeros_like(mask)
        cv2.drawContours(local_mask, [contour], -1, 255, -1)
        peak = float(cv2.mean(response, mask=local_mask)[0])
        confidence = 0.5 + 0.45 * min(1.0, max(0.0, (peak - threshold) / response_range))
        quad = cv2.boxPoints(rectangle).astype(np.float32) * scale_to_source
        output.append({
            "polygon": quad.tolist(),
            "kind": kind,
            "symbology": "unknown",
            "payload": None,
            "confidence": confidence,
            "status": "localized",
            "sources": [source],
            "evidence": {
                "component_area": contour_area,
                "aspect": aspect,
                "threshold": threshold,
            },
        })
    return output


def structure_tensor(
    image: np.ndarray,
    *,
    work_size: int = 960,
    include_linear: bool = True,
    include_matrix: bool = True,
) -> list[Prediction]:
    """From-scratch multi-scale structure-tensor competitor.

    This implements the reusable premise of Sörös rather than copying BarBeR's
    AGPL implementation: one dominant eigenvalue indicates parallel 1D texture;
    two strong eigenvalues indicate 2D grid/corner texture.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    resize_scale = min(1.0, float(work_size) / max(height, width))
    if resize_scale < 1.0:
        work = cv2.resize(gray, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
    else:
        work = gray
    normalized = work.astype(np.float32) / 255.0
    gx = cv2.Scharr(normalized, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(normalized, cv2.CV_32F, 0, 1)
    scale_to_source = 1.0 / resize_scale
    output: list[Prediction] = []
    minimum_area = max(18.0, 0.00003 * float(work.shape[0] * work.shape[1]))

    for window in (7, 15, 31):
        jxx = cv2.boxFilter(gx * gx, cv2.CV_32F, (window, window), normalize=True)
        jyy = cv2.boxFilter(gy * gy, cv2.CV_32F, (window, window), normalize=True)
        jxy = cv2.boxFilter(gx * gy, cv2.CV_32F, (window, window), normalize=True)
        trace = jxx + jyy
        discriminant = np.sqrt(np.maximum(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy))
        lambda_high = 0.5 * (trace + discriminant)
        lambda_low = 0.5 * (trace - discriminant)
        energy = np.sqrt(np.maximum(trace, 0.0))
        anisotropy = discriminant / np.maximum(trace, 1e-6)

        if include_linear:
            linear_response = energy * np.power(anisotropy, 1.5)
            output.extend(_component_predictions(
                linear_response,
                scale_to_source=scale_to_source,
                kind="1d",
                source=f"structure-tensor:linear:w{window}",
                percentile=98.7,
                minimum_area=minimum_area,
            ))
        if include_matrix:
            corner_balance = 2.0 * lambda_low / np.maximum(trace, 1e-6)
            matrix_response = energy * np.power(np.maximum(corner_balance, 0.0), 0.8)
            output.extend(_component_predictions(
                matrix_response,
                scale_to_source=scale_to_source,
                kind="2d",
                source=f"structure-tensor:matrix:w{window}",
                percentile=99.0,
                minimum_area=minimum_area,
            ))
    return _deduplicate(output, threshold=0.45)


def competitor(
    name: str,
    *,
    model_path: Path | None = None,
    image_size: int = 1280,
    confidence: float = 0.10,
    device: str = "cpu",
) -> Competitor:
    if name == "zxing-direct":
        return lambda image: zxing_direct(image, include_errors=False)
    if name == "zxing-direct-errors":
        return lambda image: zxing_direct(image, include_errors=True)
    if name == "opencv-direct":
        return opencv_direct
    if name == "zxing-opencv-direct":
        return zxing_opencv_direct
    if name == "restoration-decode":
        return restoration_decode
    if name == "learned-qr-restoration-decode":
        return LearnedQRRestorationDecode(
            model_path or DEFAULT_QR_RESTORER_WEIGHTS,
            device=device,
        )
    if name == "adnet-lenet-qr-decode":
        return ADNetLENetQRDecode(
            model_path or DEFAULT_ADNET_LENET_WEIGHTS,
            device=device,
        )
    if name == "proposal-rescue":
        return lambda image: proposal_rescue(image, retain_decoder_errors=False)
    if name == "proposal-rescue-errors":
        return lambda image: proposal_rescue(image, retain_decoder_errors=True)
    if name == "structure-tensor":
        return structure_tensor
    if name == "structure-tensor-linear":
        return lambda image: structure_tensor(image, include_linear=True, include_matrix=False)
    if name == "structure-tensor-matrix":
        return lambda image: structure_tensor(image, include_linear=False, include_matrix=True)
    if name == "piero-yolov8s":
        return PieroYoloV8s(
            model_path or DEFAULT_PIERO_WEIGHTS,
            image_size=image_size,
            confidence=confidence,
            device=device,
        )
    if name == "piero-yolov8s-tiled":
        return PieroYoloV8sTiled(
            model_path or DEFAULT_PIERO_WEIGHTS,
            image_size=image_size,
            confidence=confidence,
            device=device,
        )
    if name == "piero-yolov8s-decode":
        return PieroYoloV8sDecode(
            model_path or DEFAULT_PIERO_WEIGHTS,
            image_size=image_size,
            confidence=confidence,
            device=device,
        )
    if name == "piero-yolov8s-decode-fast":
        return PieroYoloV8sDecodeFast(
            model_path or DEFAULT_PIERO_WEIGHTS,
            image_size=image_size,
            confidence=confidence,
            device=device,
        )
    if name == "bafalo-published":
        return PublishedBaFaLo(model_path or DEFAULT_BAFALO_WEIGHTS)
    if name == "bafalo-scaled":
        return PublishedBaFaLo(
            model_path or DEFAULT_BAFALO_WEIGHTS,
            work_size=image_size,
            confidence_threshold=confidence,
        )
    if name == "bafalo-published-decode":
        return PublishedBaFaLoDecode(model_path or DEFAULT_BAFALO_WEIGHTS)
    if name == "bafalo-published-decode-fast":
        return PublishedBaFaLoDecodeFast(model_path or DEFAULT_BAFALO_WEIGHTS)
    if name == "bafalo-published-decode-consensus":
        return PublishedBaFaLoDecodeConsensus(model_path or DEFAULT_BAFALO_WEIGHTS)
    if name == "bafalo-published-decode-consensus-lite":
        return PublishedBaFaLoDecodeConsensusLite(model_path or DEFAULT_BAFALO_WEIGHTS)
    if name == "bafalo-published-decode-adaptive-consensus":
        return PublishedBaFaLoDecodeAdaptiveConsensus(model_path or DEFAULT_BAFALO_WEIGHTS)
    if name == "independent-tiny-locator":
        return IndependentTinyLocator(
            model_path or DEFAULT_TINY_LOCATOR_WEIGHTS,
            work_size=image_size,
            confidence_threshold=confidence,
            device=device,
        )
    if name == "independent-tiny-decode-adaptive-consensus":
        return IndependentTinyDecodeAdaptiveConsensus(
            model_path or DEFAULT_TINY_LOCATOR_WEIGHTS,
            work_size=image_size,
            confidence_threshold=confidence,
            device=device,
        )
    raise ValueError(f"unknown competitor: {name}")


COMPETITOR_NAMES = (
    "zxing-direct",
    "zxing-direct-errors",
    "opencv-direct",
    "zxing-opencv-direct",
    "restoration-decode",
    "learned-qr-restoration-decode",
    "adnet-lenet-qr-decode",
    "proposal-rescue",
    "proposal-rescue-errors",
    "structure-tensor",
    "structure-tensor-linear",
    "structure-tensor-matrix",
    "piero-yolov8s",
    "piero-yolov8s-tiled",
    "piero-yolov8s-decode",
    "piero-yolov8s-decode-fast",
    "bafalo-published",
    "bafalo-scaled",
    "bafalo-published-decode",
    "bafalo-published-decode-fast",
    "bafalo-published-decode-consensus",
    "bafalo-published-decode-consensus-lite",
    "bafalo-published-decode-adaptive-consensus",
    "independent-tiny-locator",
    "independent-tiny-decode-adaptive-consensus",
)
