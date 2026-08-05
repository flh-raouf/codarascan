"""Optional neural localizer. Disabled unless BARCODE_ENABLE_BAFALO=1.

Three independent reasons this is not, and should not become, the default:

1. **Licence.** BaFaLo is AGPL-3.0. Shipping it inside a distributed product places the
   whole product under AGPL. `benchmark-lab/RESEARCH_REPORT.md` reaches the same
   conclusion, which is why the selected research pipeline deliberately contains no
   BaFaLo code or weights.
2. **Accuracy on this workload.** On the project's own Quality Dossier it located 21 of 43
   symbols. For a separation feature whose entire job is "does this page carry a separator
   sheet", missing half the symbols is disqualifying regardless of speed. It also emits
   axis-aligned boxes, which are a poor crop for a rotated separator sheet.
3. **Load safety.** The published checkpoint is a pickled `nn.Module` that must be loaded
   with `weights_only=False` — arbitrary code execution from a weights file.

Its 3.4 ms figure is real, but it was measured on 320 px retail photographs holding one
large centred code, not 200-ppi A3 industrial scans. Nothing here reaches 1 ms anyway:
decoding an 8 MP A3 JPEG alone costs 4-7 ms before any detector runs.

Neither the AGPL source nor the weights are vendored into this application. Both are
loaded at runtime from operator-supplied paths, so enabling the flag is a deliberate act
by someone who has accepted the licence terms.

    BARCODE_ENABLE_BAFALO=1
    BARCODE_BAFALO_WEIGHTS=/path/to/bafalo_scnn_192-448_0.pt
    BARCODE_BAFALO_SOURCE=/path/to/BarBeR/BaFaLo
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .base import Capability, Engine, EngineInfo, PageOutcome, Region, RegionStatus, Roi

LICENSE_WARNING = (
    "BaFaLo is licensed AGPL-3.0. Enabling it subjects this application to AGPL "
    "obligations, including providing source to network users. It also located 21 of 43 "
    "symbols on the reference dossier, well below the default localizer."
)

_ENV_ENABLE = "BARCODE_ENABLE_BAFALO"
_ENV_WEIGHTS = "BARCODE_BAFALO_WEIGHTS"
_ENV_SOURCE = "BARCODE_BAFALO_SOURCE"

# Reference operating point from the benchmark adapter. Changing these invalidates every
# published BaFaLo measurement, so they are defaults rather than tuning knobs.
WORK_SIZE = 320
CONFIDENCE_THRESHOLD = 0.4
MINIMUM_COMPONENT_AREA = 300.0

_LOAD_LOCK = threading.Lock()


def is_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "0").strip().lower() in {"1", "true", "yes", "on"}


def _path_from_env(name: str, *, want_dir: bool = False) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    path = Path(raw)
    ok = path.is_dir() if want_dir else path.is_file()
    return path if ok else None


def _unavailable_reason() -> str | None:
    if not is_enabled():
        return f"Disabled. Set {_ENV_ENABLE}=1 to enable, having accepted the AGPL-3.0 terms."
    if _path_from_env(_ENV_WEIGHTS) is None:
        return f"Enabled but {_ENV_WEIGHTS} does not point at a readable checkpoint file."
    if _path_from_env(_ENV_SOURCE, want_dir=True) is None:
        return (
            f"Enabled but {_ENV_SOURCE} does not point at the BaFaLo source directory. "
            "The checkpoint is a pickled module and cannot be loaded without it."
        )
    try:
        import torch  # noqa: F401
    except ImportError:
        return "Enabled but PyTorch is not installed in this environment."
    return None


class BaFaLoDetector:
    """Neural localizer. Present so the localizer stays swappable; off by default."""

    def __init__(self) -> None:
        self._model = None
        self._torch = None
        self.info = self._build_info()

    def _build_info(self) -> EngineInfo:
        reason = _unavailable_reason()
        return EngineInfo(
            id="bafalo",
            label="BaFaLo neural localizer",
            capability=Capability.DETECT,
            summary=(
                "Ultra-light neural segmentation localizer. Very fast on small images, but "
                "measured well below the default localizer on scanned document pages."
            ),
            speed_ms_per_page="~3-5 ms inference, excluding page decoding",
            accuracy_note="Located 21 of 43 symbols on the reference dossier — not recommended",
            available=reason is None,
            unavailable_reason=reason,
            license_warning=LICENSE_WARNING,
        )

    def refresh(self) -> None:
        """Re-read the environment. Used by tests and by the registry at startup."""
        self.info = self._build_info()

    def warm(self) -> None:
        if self.info.available:
            self._load()

    def _load(self):
        with _LOAD_LOCK:
            if self._model is not None:
                return self._model
            import importlib
            import sys

            import torch

            source = _path_from_env(_ENV_SOURCE, want_dir=True)
            weights = _path_from_env(_ENV_WEIGHTS)
            if source is None or weights is None:
                raise RuntimeError(self.info.unavailable_reason or "BaFaLo is not configured")

            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            # The checkpoint was serialized before fast_scnn_pico.py was renamed to
            # bafalo_scnn.py upstream, so unpickling needs the old name to resolve.
            sys.modules.setdefault("fast_scnn_pico", importlib.import_module("bafalo_scnn"))

            # Page-level concurrency is managed by the caller; letting torch spawn its own
            # pool on top of that oversubscribes the container CPU quota.
            torch.set_num_threads(1)
            self._torch = torch
            self._model = torch.load(str(weights), weights_only=False, map_location="cpu").eval()
            return self._model

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        if not self.info.available:
            raise RuntimeError(self.info.unavailable_reason or "BaFaLo is not available")

        started = perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unable to read page image: {path}")

        offset_x = offset_y = 0
        full_height, full_width = image.shape[:2]
        if roi is not None:
            image, offset_x, offset_y = roi.crop(image)

        model = self._load()
        torch = self._torch

        source_height, source_width = image.shape[:2]
        scale = float(WORK_SIZE) / max(source_height, source_width)
        work_width = max(1, int(round(source_width * scale)))
        work_height = max(1, int(round(source_height * scale)))
        work = cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_CUBIC)

        # The network downsamples by 32, so both dimensions must be a multiple of it.
        padded_width = ((work_width + 31) // 32) * 32
        padded_height = ((work_height + 31) // 32) * 32
        padded = np.pad(
            work, ((0, padded_height - work_height), (0, padded_width - work_width), (0, 0))
        )
        tensor = torch.from_numpy((padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        with torch.inference_mode():
            heatmaps = torch.sigmoid(model(tensor))[0].cpu().numpy()

        regions: list[Region] = []
        for class_index, kind in enumerate(("linear", "matrix")):
            heatmap = heatmaps[class_index, :work_height, :work_width]
            binary = (heatmap > CONFIDENCE_THRESHOLD).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) <= MINIMUM_COMPONENT_AREA:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], -1, 1, -1)
                confidence = float(cv2.mean(heatmap, mask=mask)[0])
                # Axis-aligned only — the network emits a mask, not an oriented box.
                x1, y1 = x / scale + offset_x, y / scale + offset_y
                x2, y2 = (x + width) / scale + offset_x, (y + height) / scale + offset_y
                regions.append(
                    Region(
                        quad=(
                            round(x1, 2), round(y1, 2), round(x2, 2), round(y1, 2),
                            round(x2, 2), round(y2, 2), round(x1, 2), round(y2, 2),
                        ),
                        kind=kind,
                        confidence=confidence,
                        sources=("bafalo-published-checkpoint", "external AGPL-3.0 implementation"),
                        status=RegionStatus.LOCALIZED,
                    )
                )

        diagnostics: dict[str, Any] = {
            "work_size": WORK_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "page_size": {"width": full_width, "height": full_height},
            "license": "AGPL-3.0",
            "geometry": "axis-aligned",
        }
        if roi is not None:
            diagnostics["roi_pixels"] = roi.to_pixels(full_width, full_height)
            diagnostics["roi_applied"] = True

        return PageOutcome(
            page=page,
            regions=regions,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            diagnostics=diagnostics,
        )


ENGINE: Engine = BaFaLoDetector()
