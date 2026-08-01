"""Detection-only engine backed by the vendored CPU coarse-to-fine localizer.

Measured 2026-07-27 across 131 pages / 382 ground-truth symbols (see
`codara-integration-analysis/BENCHMARK.md`):

    page-level accuracy   100 % / 100 % / 98.33 %   (real dossier / balanced / stress)
    page-level precision  100 % on all three — it never flagged an empty page
    cost                  ~160 ms/page sequential with
                          `kinds=all`; ~30 ms latency / ~12 ms batch with `kinds=linear`;
                          ~162 ms latency / ~103 ms batch with `kinds=2d` on the same
                          dense corpus

`kinds=linear` is 2-3x cheaper but blind to Data Matrix and QR, which is why it collapses
to 82-84 % page accuracy on the 2-D-heavy sets. It is offered as an explicit choice for
operators who know their separator sheets are 1-D, never as a silent default.

The other three localizer knobs are FIXED, not options. Re-measured 2026-07-27 on the
same corpus (universal|legacy × gate on|off × qr 900|2000), they showed:

  * `legacy` buys nothing over `universal` — identical page accuracy everywhere, one
    symbol FEWER on stress, ~35 % slower. It is strictly dominated.
  * the empty-page gate removes expensive 2-D work from confidently negative pages. It is
    intentionally conservative: positive evidence or uncertainty always unlocks the full
    scan, while a small QR can still be invisible to the gate's thumbnail check. The gate
    is therefore fixed on as a measured speed optimization, with no independent threshold
    controls exposed to callers.
  * the one missed page (stress p2, an "extreme"-tier 31 px QR) is missed by every
    configuration, gate included — so no knob fixes it, and no knob should exist.

So the engine always runs `universal` with the early-skip gate enabled and the gate's QR
work size pinned at the validated 900-pixel screen. In `all` mode, pages with no linear evidence
are screened before the expensive 2-D stages; positive evidence or uncertainty always
falls through to the full scan. An operator cannot tune the gate thresholds from the UI.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .base import (
    CONTEXT_MARGIN_PX,
    Capability,
    Engine,
    EngineInfo,
    PageOutcome,
    Region,
    RegionStatus,
    Roi,
    region_center,
)

# The vendored localizer is a single self-contained module (cv2 + numpy only). It is
# imported by path rather than by package name because its research original is called
# `pipeline.py`, which would collide with this app's own `pipeline` package.
_LOCALIZATION_DIR = Path(__file__).resolve().parent.parent / "localization"
if str(_LOCALIZATION_DIR) not in sys.path:
    sys.path.append(str(_LOCALIZATION_DIR))

import localizer  # noqa: E402  (path must be set first)

KINDS = ("all", "linear", "2d")

# Fixed localizer configuration — see the module docstring for the measurements behind
# these three values. They are not exposed as engine options on purpose.
FIXED_LINEAR_PROFILE = "universal"
FIXED_EMPTY_PAGE_GATE = True
FIXED_EMPTY_GATE_QR_WORK_SIZE = 900

_WARM_LOCK = threading.Lock()


class Pipeline7Detector:
    """Locates barcodes. Never decodes — `Region.value` is always None by construction."""

    info = EngineInfo(
        id="p7-localizer",
        label="Coarse-to-fine localizer",
        capability=Capability.DETECT,
        summary=(
            "Classical CPU localizer. Proposes regions on a reduced image, then verifies "
            "against native pixels. Uses a conservative early-skip screen before expensive "
            "2-D work on pages with no linear evidence."
        ),
        speed_ms_per_page="~160 ms/page",
        accuracy_note="",
        badge="Base",
        options={
            "kinds": {
                "type": "enum",
                "values": list(KINDS),
                "default": "all",
                "label": "Symbol types",
                # Sequential per-page latency on the 131-page Quality/Balanced/Stress
                # corpus. `linear` is unchanged; the 2-D figures include physical QR and
                # ECC 200 verification and intentionally replace the old permissive
                # texture-only timing.
                "value_timings": {
                    "all": "~160 ms/page",
                    "linear": "~30 ms/page",
                    "2d": "~162 ms/page",
                },
            },
        },
    )

    def __init__(self) -> None:
        self._warmed = False

    def warm(self) -> None:
        """Prime the OpenCV detector caches once, off the request path."""
        with _WARM_LOCK:
            if self._warmed:
                return
            probe = np.full((320, 320), 255, dtype=np.uint8)
            config = self._config({})
            try:
                localizer.locate_regions(probe, config)
            except Exception:  # noqa: BLE001 - warming must never break startup
                pass
            self._warmed = True

    @staticmethod
    def _config(options: dict[str, Any] | None) -> Any:
        options = options or {}
        kinds = options.get("kinds", "all")
        if kinds not in KINDS:
            raise ValueError(f"kinds must be one of {KINDS}, got {kinds!r}")

        # Only `kinds` is honoured. The other localizer knobs are fixed (module
        # docstring); a stale client that still sends them gets the safe values, not an
        # error and not its own choice.
        return localizer.Config(
            # `output` is required by the dataclass but never touched on the in-memory
            # path; locate_regions writes nothing to disk.
            output=Path("."),
            kinds=kinds,
            linear_profile=FIXED_LINEAR_PROFILE,
            empty_page_gate=FIXED_EMPTY_PAGE_GATE,
            empty_gate_qr_work_size=FIXED_EMPTY_GATE_QR_WORK_SIZE,
            save_overlays=False,
            save_crops=False,
        )

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        started = perf_counter()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"unable to read page image: {path}")

        offset_x = offset_y = 0
        full_height, full_width = gray.shape[:2]
        if roi is not None:
            gray, offset_x, offset_y = roi.crop(gray, margin=CONTEXT_MARGIN_PX)

        config = self._config(options)
        return self._analyze_gray(
            page,
            gray,
            config=config,
            started=started,
            full_width=full_width,
            full_height=full_height,
            offset_x=offset_x,
            offset_y=offset_y,
            roi=roi,
        )

    def _analyze_gray(
        self,
        page: int,
        gray: np.ndarray,
        *,
        config: Any,
        started: float,
        full_width: int,
        full_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
        roi: Roi | None = None,
        linear_work: np.ndarray | None = None,
        diagnostics_extra: dict[str, Any] | None = None,
    ) -> PageOutcome:
        """Analyze arrays that are already prepared for this engine.

        The standard coarse-to-fine engine starts its timer before loading and calls
        this helper afterward. Prepared engines may start the timer immediately before
        this call and supply ``linear_work`` to bypass the resize stage.
        """
        detections, diagnostics, timings = localizer.locate_regions(
            gray,
            config,
            linear_work=linear_work,
        )

        regions = [
            Region(
                quad=_quad_to_tuple(item.quad, offset_x, offset_y),
                kind=item.kind,
                confidence=float(item.confidence),
                sources=(item.source,),
                status=RegionStatus.LOCALIZED,
            )
            for item in detections
        ]

        if roi is not None:
            # The crop was widened for decoding context only. A symbol whose centre lies
            # outside the drawn zone is not in the zone, so it is discarded here.
            before = len(regions)
            regions = [
                region
                for region in regions
                if roi.contains_point(*region_center(region.quad), full_width, full_height)
            ]
            diagnostics = dict(diagnostics)
            diagnostics["context_margin_px"] = CONTEXT_MARGIN_PX
            diagnostics["dropped_outside_zone"] = before - len(regions)

        diagnostics = dict(diagnostics)
        diagnostics["timings"] = {k: round(v, 6) for k, v in timings.items()}
        diagnostics["page_size"] = {"width": full_width, "height": full_height}
        if diagnostics_extra:
            diagnostics.update(diagnostics_extra)
        if roi is not None:
            diagnostics["roi_pixels"] = roi.to_pixels(full_width, full_height)
            diagnostics["roi_applied"] = True

        return PageOutcome(
            page=page,
            regions=regions,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            diagnostics=diagnostics,
        )


def _quad_to_tuple(
    quad: np.ndarray, offset_x: int, offset_y: int
) -> tuple[float, float, float, float, float, float, float, float]:
    """Flatten a 4x2 quad to 8 floats, translated back into full-page coordinates.

    When an ROI is in force the localizer sees only the crop, so every coordinate it
    returns is relative to the crop origin and must be shifted back before it reaches the
    client — otherwise overlays land in the wrong place.
    """
    points = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    points[:, 0] += offset_x
    points[:, 1] += offset_y
    return tuple(round(float(v), 2) for v in points.reshape(-1))  # type: ignore[return-value]


ENGINE: Engine = Pipeline7Detector()
