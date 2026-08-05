"""Shared decoder adapter backed by the local recovery runtime.

This fork is *ahead* of both research folders: it alone carries the field fixes for the
1.5x/2x/3x small-image upscale fallback, the ITF short-code false-positive filter, the
22-format ZXing map, `categorize_formats` and `collect_pages_extended`.

Measured 2026-07-26 (see the repository benchmark notes in
`benchmarks/results/TENSOR_HYBRID_BENCHMARK.md`), exact payload recall:

    real Quality Dossier   43/43   100 %
    balanced synthetic    121/122   99.18 %
    noisy stress          136/217   62.67 %

The evidence-guided alternative recovered exactly one more symbol out of 382 while costing
4-6x the time, and lost one on the real document. That is why this stays the default.

Unresolved 2-D geometry is now held behind the same format-specific physical
acceptance rule used by the robust classical 2-D extractor. Successful ZXing
decodes are unchanged; generic square/gradient texture is no longer surfaced
as a Data Matrix or QR location without an ECC-200 lattice or finder-pattern
proof.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from barcode_detection.engines.common.recovery import runtime as engine

from barcode_detection.core.contracts import (
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

_WARM_LOCK = threading.Lock()

_KIND_ALIASES = {"linear": "linear", "matrix": "matrix", "1d": "linear", "2d": "matrix"}

# Values for the `kinds` option, spelled the same as the localizer's so an operator meets one
# vocabulary across both engines.
KINDS = ("all", "linear", "2d")
DEFAULT_KINDS = "all"

#: An empty format collection. The vendored pipeline gates whole stages on the truthiness of
#: `matrix_formats` / `linear_formats` (pipeline.py:463 and :476, again at :1005 and :1098),
#: so handing it an empty one skips that stage rather than running it and discarding the
#: results. Passing None would not do: the pipeline re-derives both from the config when both
#: are None (pipeline.py:979).
def _no_formats() -> Any:
    return engine.zxingcpp.barcode_formats_from_str("")


def restrict_kinds(matrix_formats: Any, linear_formats: Any, kinds: str | None) -> tuple[Any, Any]:
    """Narrow the format pair to one symbol class.

    Composes with an explicit format list rather than overriding it: the list says *which*
    symbologies, this says which *class*, and asking for both gives the intersection. Asking
    for a class the format list has already excluded correctly yields nothing.
    """
    wanted = (kinds or DEFAULT_KINDS).strip().lower()
    if wanted in ("", DEFAULT_KINDS):
        return matrix_formats, linear_formats
    resolved = _KIND_ALIASES.get(wanted)
    if resolved == "linear":
        return _no_formats(), linear_formats
    if resolved == "matrix":
        return matrix_formats, _no_formats()
    raise ValueError(f"kinds must be one of {', '.join(KINDS)}, got {kinds!r}")


def normalize_engine_options(options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill the internal format handles expected by the local engine branches.

    Application services historically built these ZXing handles before calling
    an engine. The repository-owned public contract accepts ordinary options,
    so direct callers may provide ``formats`` as a list or comma-separated
    string—or omit it to scan all supported formats.
    """

    normalized = dict(options or {})
    if "formats" in normalized and isinstance(normalized["formats"], str):
        normalized["formats"] = [
            value.strip()
            for value in normalized["formats"].split(",")
            if value.strip()
        ]
    selected = normalized.get("formats")
    if isinstance(selected, (list, tuple, set)) and {
        str(value).strip().lower() for value in selected
    } <= {"all", "all readable", "all creatable"}:
        selected = None

    matrix_formats = normalized.get("matrix_formats")
    linear_formats = normalized.get("linear_formats")
    if matrix_formats is None and linear_formats is None:
        matrix_formats, linear_formats = engine.categorize_formats(selected)
    matrix_formats, linear_formats = restrict_kinds(
        matrix_formats,
        linear_formats,
        normalized.get("kinds"),
    )
    normalized["matrix_formats"] = matrix_formats
    normalized["linear_formats"] = linear_formats
    return normalized


class VendoredExtractor:
    """Locates *and* decodes. Populates `Region.value` and `Region.symbology`."""

    info = EngineInfo(
        id="adaptive-extractor",
        label="Adaptive extractor",
        capability=Capability.DECODE,
        summary=(
            "Classical detector and decoder with scale-aware recovery for small or "
            "awkward images. Reads the payload as well as locating the symbol."
        ),
        speed_ms_per_page="~203 ms/page sequential",
        accuracy_note="",
        badge="Base",
        options={
            "kinds": {
                "type": "enum",
                "values": list(KINDS),
                "default": DEFAULT_KINDS,
                "label": "Symbol types",
                # Sequential per-page latency on the 131-page Quality, balanced, and
                # stress corpus, measured from the current vendored engine on 2026-07-28.
                # The complete `all` run took 26.590 seconds, or 202.98 ms/page.
                "value_timings": {
                    "all": "~203 ms/page",
                    "linear": "~220 ms/page",
                    "2d": "~140 ms/page",
                },
            },
            "formats": {
                "type": "format-list",
                "default": [],
                "label": "Barcode formats",
                "help": "Restricting formats reduces both time and false reads. Empty means all.",
            },
        },
    )

    def __init__(self) -> None:
        self._warmed = False

    def warm(self) -> None:
        with _WARM_LOCK:
            if self._warmed:
                return
            # Touching categorize_formats forces the zxingcpp format tables to build, which
            # is the only meaningful one-time cost on this engine.
            try:
                engine.categorize_formats(None)
            except Exception:  # noqa: BLE001
                pass
            self._warmed = True

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        """Analyze a page with application options or a standalone workspace.

        A caller that supplies neither ``work_dir`` nor ``config`` gets a
        temporary workspace and no persisted crops/overlays by default. Pass a
        workspace explicitly when those artifacts need to survive the call.
        """

        options = normalize_engine_options(options)
        started = perf_counter()
        configured_work = options.get("work_dir")
        configured_config = options.get("config")
        if configured_work is None and configured_config is not None:
            configured_work = configured_config.output

        temporary = None
        if configured_work is None:
            temporary = tempfile.TemporaryDirectory(prefix="barcode-detection-")
            configured_work = temporary.name
        work = Path(configured_work).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)

        config = configured_config
        if config is None:
            config = engine.Config(
                output=work,
                barcode_formats=options.get("formats"),
                save_crops=bool(options.get("save_crops", False)),
                save_overlays=bool(options.get("save_overlays", False)),
                residual_proposals=bool(options.get("residual_proposals", True)),
                include_review_candidates=bool(
                    options.get("include_review_candidates", True)
                ),
                keep_unresolved_qr=bool(options.get("keep_unresolved_qr", False)),
                strict_matrix_acceptance=bool(
                    options.get("strict_matrix_acceptance", True)
                ),
                minimum_linear_score=float(
                    options.get("minimum_linear_score", 0.52)
                ),
                minimum_linear_length=float(
                    options.get("minimum_linear_length", 100.0)
                ),
            )

        try:
            target = path
            resolved_input = path.expanduser().resolve()
            staged_path = work / "pages" / f"page-{page:04d}{path.suffix.lower() or '.png'}"
            if not resolved_input.is_relative_to(work):
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved_input, staged_path)
                target = staged_path

            offset_x = offset_y = 0
            page_width = page_height = 0

            if roi is not None:
                # The vendored engine reads from disk and calls page_path.relative_to(work),
                # so the crop has to be a real file underneath the work directory.
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise RuntimeError(f"unable to read page image: {path}")
                page_height, page_width = image.shape[:2]
                cropped, offset_x, offset_y = roi.crop(image, margin=CONTEXT_MARGIN_PX)
                crop_dir = work / "roi-pages"
                crop_dir.mkdir(parents=True, exist_ok=True)
                target = crop_dir / f"page-{page:04d}.png"
                cv2.imwrite(str(target), cropped)

            payload = engine.process_page(
                page,
                target,
                "standalone-image",
                config,
                work,
                options["matrix_formats"],
                options["linear_formats"],
                bool(options.get("stage_parallel", False)),
            )
        finally:
            if temporary is not None:
                temporary.cleanup()

        # Each bucket is a distinct outcome and stays distinct all the way to the UI:
        # a Data Matrix that was found but would not decode is not the same as a weak
        # candidate worth a second look, and neither is the same as no detection.
        regions: list[Region] = []
        for bucket, status in (
            ("results", RegionStatus.DECODED),
            ("unresolved_matrix", RegionStatus.UNRESOLVED_MATRIX),
            ("review_candidates", RegionStatus.REVIEW_CANDIDATE),
        ):
            for raw in payload.get(bucket, []) or []:
                regions.append(_to_region(raw, offset_x, offset_y, status))

        if roi is not None:
            # Same contract as the localizer: the widened crop buys decoding context, not
            # a wider zone. Only symbols centred inside the drawn zone survive.
            before = len(regions)
            regions = [
                region
                for region in regions
                if roi.contains_point(*region_center(region.quad), page_width, page_height)
            ]
            dropped = before - len(regions)
        else:
            dropped = 0

        diagnostics = dict(payload.get("diagnostics") or {})
        if roi is not None:
            diagnostics["context_margin_px"] = CONTEXT_MARGIN_PX
            diagnostics["dropped_outside_zone"] = dropped
        diagnostics["timings"] = payload.get("timings", {})
        diagnostics["decoded_count"] = payload.get("decoded_count", 0)
        diagnostics["unresolved_count"] = payload.get("localized_unresolved_matrix_count", 0)
        diagnostics["review_count"] = payload.get("review_candidate_count", 0)
        if roi is not None:
            diagnostics["roi_pixels"] = roi.to_pixels(page_width, page_height)
            diagnostics["roi_applied"] = True
        size = payload.get("image_size") or {}
        diagnostics["page_size"] = (
            {"width": page_width, "height": page_height}
            if roi is not None
            else {"width": size.get("width", 0), "height": size.get("height", 0)}
        )

        return PageOutcome(
            page=page,
            regions=regions,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            diagnostics=diagnostics,
        )


def _to_region(
    raw: dict[str, Any], offset_x: int, offset_y: int, status: RegionStatus
) -> Region:
    quad = raw.get("quad") or []
    flat: list[float] = []
    if quad and isinstance(quad[0], (list, tuple)):
        for point in quad:
            flat.extend(float(v) for v in point)
    else:
        flat = [float(v) for v in quad]
    if len(flat) == 8 and (offset_x or offset_y):
        points = np.asarray(flat, dtype=np.float64).reshape(4, 2)
        points[:, 0] += offset_x
        points[:, 1] += offset_y
        flat = [float(v) for v in points.reshape(-1)]
    while len(flat) < 8:
        flat.append(0.0)

    text = raw.get("text") or raw.get("value")
    decoded = status is RegionStatus.DECODED
    # Forward per-region diagnostics that do not have a dedicated field on `Region`. The
    # operator opens them when an unresolved or review candidate is selected, and the
    # same fields are useful on a decoded one when something looks suspicious. Only
    # include keys the vendored pipeline actually populated, so the surfaced JSON does
    # not grow an empty object on every row.
    extras: dict[str, Any] = {}
    for key in ("attempts", "aabb", "evidence", "crop", "decoded"):
        if key in raw and raw[key] not in (None, "", [], {}):
            extras[key] = raw[key]
    return Region(
        quad=tuple(round(v, 2) for v in flat[:8]),  # type: ignore[arg-type]
        kind=_KIND_ALIASES.get(str(raw.get("kind") or ""), str(raw.get("kind") or "unknown")),
        confidence=float(raw.get("confidence", 1.0)),
        value=text if (decoded and text) else None,
        symbology=raw.get("format") if (decoded and text) else None,
        sources=tuple(raw.get("sources") or ()),
        status=status,
        extras=extras,
    )


ENGINE: Engine = VendoredExtractor()
