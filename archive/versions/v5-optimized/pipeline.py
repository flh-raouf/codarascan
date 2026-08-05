#!/usr/bin/env python3
"""Accuracy-gated, throughput-optimized barcode pipeline.

The implementation preserves the validated ZXing/OpenCV recovery cascade while
removing duplicated scans.  It uses one proposal-only OpenCV pass, grayscale
buffers, batch PDF image extraction, and page-level concurrency.  The default
``all`` mode remains restricted to the repository's supported Code 39,
Code 128, Data Matrix, and QR Code formats.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
import zxingcpp

from loader import install_aliases

install_aliases()

from deterministic_barcode_locator import barcode_locator as locator  # noqa: E402
from hybrid_barcode_pipeline import pipeline as hybrid  # noqa: E402
from zxing_only_barcode_pipeline import pipeline as advanced  # noqa: E402
from zxing_2d_barcode_pipeline import pipeline as previous  # noqa: E402


MATRIX_FORMATS = previous.MATRIX_FORMATS
LINEAR_FORMATS = previous.LINEAR_FORMATS
UNION_LINEAR_SCALES = tuple(
    sorted({value for _name, scales in locator.SCALE_PROFILES for value in scales})
)


@dataclass
class Config:
    output: Path
    formats: str = "all"
    render_dpi: int = 300
    pages: list[int] | None = None
    overwrite: bool = False
    save_crops: bool = True
    save_overlays: bool = True
    residual_proposals: bool = False
    include_review_candidates: bool = True
    keep_unresolved_qr: bool = False
    minimum_linear_score: float = 0.52
    minimum_linear_length: float = 100.0
    workers: int = 0
    linear_detector: str = "optimized"
    batch_pdf_extraction: bool = True


def _image_number(path: Path) -> int:
    match = re.search(r"-(\d+)(?:\.[^.]+)?$", path.name)
    return int(match.group(1)) if match else 10**9


def _pdf_image_rows(pdf: Path) -> list[dict[str, int]]:
    completed = subprocess.run(
        ["pdfimages", "-list", str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        try:
            rows.append(
                {
                    "page": int(parts[0]),
                    "number": int(parts[1]),
                    "width": int(parts[3]),
                    "height": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return rows


def collect_pages(input_path: Path, config: Config, work: Path) -> list[tuple[int, Path, str]]:
    """Extract a dense PDF page range with one Poppler process.

    Sparse selections and unusual PDFs fall back to the predecessor's proven
    per-page extraction/rendering path.
    """
    shared = hybrid.PipelineConfig(
        output=work,
        render_dpi=config.render_dpi,
        pages=config.pages,
        angle_step=0,
        minimum_score=config.minimum_linear_score,
        minimum_linear_length=config.minimum_linear_length,
        tensor_fallback=False,
        engine="zxing",
        matrix_sweep=False,
        save_crops=config.save_crops,
        include_unresolved=config.include_review_candidates,
    )
    if input_path.suffix.lower() != ".pdf" or not config.batch_pdf_extraction:
        return hybrid.collect_pages(input_path, shared)

    page_count = hybrid.pdf_page_count(input_path)
    selected = config.pages or list(range(1, page_count + 1))
    if any(page > page_count for page in selected):
        raise ValueError(f"requested page exceeds PDF page count {page_count}")
    span = max(selected) - min(selected) + 1
    if len(selected) < 4 or span > max(8, int(len(selected) * 1.6)):
        return hybrid.collect_pages(input_path, shared)

    extraction = work / "batch-native-work"
    pages_dir = work / "pages"
    extraction.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    prefix = extraction / "image"
    try:
        rows = [row for row in _pdf_image_rows(input_path) if min(selected) <= row["page"] <= max(selected)]
        subprocess.run(
            [
                "pdfimages",
                "-f",
                str(min(selected)),
                "-l",
                str(max(selected)),
                "-j",
                str(input_path),
                str(prefix),
            ],
            check=True,
            capture_output=True,
        )
        files = sorted((path for path in extraction.glob("image-*") if path.is_file()), key=_image_number)
        if len(files) != len(rows):
            raise RuntimeError(f"pdfimages manifest mismatch: {len(rows)} rows, {len(files)} files")
        best: dict[int, tuple[int, Path]] = {}
        for row, path in zip(rows, files):
            area = row["width"] * row["height"]
            if row["page"] not in selected or area < 1_000_000:
                continue
            if area > best.get(row["page"], (0, path))[0]:
                best[row["page"]] = (area, path)

        output: list[tuple[int, Path, str]] = []
        for page in selected:
            native = best.get(page)
            if native is not None:
                suffix = native[1].suffix.lower() or ".img"
                destination = pages_dir / f"page-{page:04d}{suffix}"
                shutil.copy2(native[1], destination)
                output.append((page, destination, "native-embedded-raster-batch"))
            else:
                destination = hybrid.render_pdf_page(input_path, page, pages_dir, config.render_dpi)
                output.append((page, destination, f"rendered-{config.render_dpi}-dpi"))
        return output
    except Exception:
        shutil.rmtree(extraction, ignore_errors=True)
        return hybrid.collect_pages(input_path, shared)
    finally:
        shutil.rmtree(extraction, ignore_errors=True)


def _linear_detector_configs(mode: str) -> list[tuple[str, Sequence[float], float]]:
    if mode == "exhaustive":
        return [
            (f"{name}:g{int(threshold)}", scales, threshold)
            for name, scales in locator.SCALE_PROFILES
            for threshold in (48.0, 64.0)
        ]
    return [("union:g48", UNION_LINEAR_SCALES, 48.0)]


def visual_linear_proposals(gray: np.ndarray, mode: str) -> list[locator.Detection]:
    proposals: list[locator.Detection] = []
    for name, scales, threshold in _linear_detector_configs(mode):
        detector = locator.make_detector(scales, gray.shape, threshold)
        try:
            found, points = detector.detectMulti(gray)
        except cv2.error:
            continue
        if not found or points is None:
            continue
        for quad in np.asarray(points, dtype=np.float32):
            detection = locator.Detection(
                quad=quad,
                sources={f"opencv:optimized:{name}"},
                proposal_score=0.48,
            )
            long_side, short_side = detection.long_short()
            if long_side < 18 or short_side < 5 or long_side / (short_side + locator.EPS) < 1.15:
                continue
            proposals.append(detection)
    return locator.deduplicate(proposals)


def verify_linear_proposals(
    gray: np.ndarray,
    proposals: Sequence[locator.Detection],
    config: Config,
) -> tuple[list[locator.Detection], list[locator.Detection]]:
    eligible: list[locator.Detection] = []
    rejected: list[locator.Detection] = []
    decode_floor = max(0.35, config.minimum_linear_score - 0.16)
    for detection in proposals:
        detection.metrics = locator.barcode_structure_metrics(gray, detection.quad)
        detection.structural_score = detection.metrics.get("score", 0.0)
        detection.accepted = locator.accept_linear(
            detection,
            config.minimum_linear_score,
            allow_tensor_only=False,
            minimum_linear_length=config.minimum_linear_length,
        )
        # The predecessor decoded at this lower score before making its final
        # accept/reject decision.  Retain that escape hatch without decoding the
        # candidate twice.
        if detection.accepted or detection.structural_score >= decode_floor:
            eligible.append(detection)
        else:
            rejected.append(detection)
    return eligible, rejected


def _scan_matrix(image: np.ndarray, binarizer: Any) -> list[Any]:
    return previous.safe_zxing_read(
        image,
        MATRIX_FORMATS,
        binarizer,
        try_downscale=True,
        return_errors=True,
    )


def _scan_linear(image: np.ndarray) -> list[Any]:
    return previous.safe_zxing_read(
        image,
        LINEAR_FORMATS,
        zxingcpp.Binarizer.LocalAverage,
        try_downscale=False,
        return_errors=False,
    )


def fast_stage(
    gray: np.ndarray,
    config: Config,
    parallel: bool,
) -> tuple[dict[str, list[Any]], list[locator.Detection], float]:
    jobs: dict[str, Callable[[], Any]] = {}
    if config.formats in {"2d", "all"}:
        jobs["matrix-local"] = lambda: _scan_matrix(gray, zxingcpp.Binarizer.LocalAverage)
        jobs["matrix-global"] = lambda: _scan_matrix(gray, zxingcpp.Binarizer.GlobalHistogram)
    if config.formats in {"1d", "all"}:
        jobs["linear-zxing"] = lambda: _scan_linear(gray)
        jobs["linear-opencv"] = lambda: visual_linear_proposals(gray, config.linear_detector)
    started = perf_counter()
    if parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {name: executor.submit(function) for name, function in jobs.items()}
            values = {name: future.result() for name, future in futures.items()}
    else:
        values = {name: function() for name, function in jobs.items()}
    proposals = values.pop("linear-opencv", [])
    return values, proposals, perf_counter() - started


def parse_matrix_reads(reads_by_name: dict[str, list[Any]]) -> tuple[list[previous.Result], list[previous.MatrixAnchor], dict[str, Any]]:
    decoded: list[previous.Result] = []
    invalid: list[previous.MatrixAnchor] = []
    diagnostics: dict[str, Any] = {"reads": {}, "invalid_errors": {}}
    for key, source_name in (("matrix-local", "local-average"), ("matrix-global", "global-histogram")):
        reads = reads_by_name.get(key, [])
        diagnostics["reads"][source_name] = len(reads)
        for barcode in reads:
            quad = previous.barcode_quad(barcode)
            if abs(float(cv2.contourArea(quad))) < 9.0:
                continue
            if barcode.valid and barcode.text:
                decoded.append(
                    previous.Result(
                        quad=quad,
                        text=str(barcode.text),
                        format=hybrid.normalize_format(str(barcode.format)),
                        status="decoded",
                        confidence=1.0,
                        sources={f"zxing:matrix-whole-page:{source_name}:downscale"},
                        attempts=[f"whole-page:{source_name}:downscale"],
                    )
                )
            else:
                error = str(getattr(barcode, "error", None) or "invalid")
                error_key = error.split(" @ ", 1)[0]
                diagnostics["invalid_errors"][error_key] = diagnostics["invalid_errors"].get(error_key, 0) + 1
                invalid.append(
                    previous.MatrixAnchor(
                        quad=quad,
                        format=hybrid.normalize_format(str(barcode.format)),
                        error=error,
                        sources={f"zxing:invalid-matrix:{source_name}:downscale"},
                        attempts=[f"whole-page:{source_name}:downscale:return-errors"],
                    )
                )
    decoded = previous.merge_results(decoded)
    invalid = [
        anchor
        for anchor in previous.merge_anchors(invalid)
        if not any(locator.quad_overlap(anchor.quad, result.quad) >= 0.28 for result in decoded)
    ]
    diagnostics["decoded_unique"] = len(decoded)
    diagnostics["invalid_unique_unexplained"] = len(invalid)
    return decoded, invalid, diagnostics


def parse_linear_reads(reads: Sequence[Any]) -> list[previous.Result]:
    output: list[previous.Result] = []
    for barcode in reads:
        if not barcode.valid or not barcode.text:
            continue
        output.append(
            previous.Result(
                quad=previous.barcode_quad(barcode),
                text=str(barcode.text),
                format=hybrid.normalize_format(str(barcode.format)),
                status="decoded",
                confidence=1.0,
                sources={"zxing:linear-whole-page"},
                attempts=["whole-page:linear:local-average"],
            )
        )
    return previous.merge_results(output)


def _context_crop(
    image: np.ndarray,
    center: np.ndarray,
    width: int,
    height: int,
    fraction_x: float,
    fraction_y: float,
) -> tuple[np.ndarray, int, int]:
    x1 = int(round(float(center[0]) - fraction_x * width))
    y1 = int(round(float(center[1]) - fraction_y * height))
    x1 = min(max(0, x1), max(0, image.shape[1] - width))
    y1 = min(max(0, y1), max(0, image.shape[0] - height))
    return image[y1 : min(image.shape[0], y1 + height), x1 : min(image.shape[1], x1 + width)], x1, y1


def recover_matrix_anchor(image: np.ndarray, anchor: previous.MatrixAnchor) -> previous.Result | None:
    result = previous.recover_invalid_anchor(image, anchor)
    if result is not None:
        return result
    # Preserve more page context for edge-adjacent and partially covered matrix
    # symbols. The first recipe is the general-purpose native-scale variant.
    recipes = (
        (500, 500, 0.80, 0.75, 1.0, "offset500-native"),
        (500, 500, 0.65, 0.60, 1.5, "offset500-cubic15"),
        (700, 700, 0.80, 0.60, 1.5, "offset700-cubic15"),
    )
    for width, height, fraction_x, fraction_y, scale, name in recipes:
        crop, offset_x, offset_y = _context_crop(image, anchor.center(), width, height, fraction_x, fraction_y)
        variant = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for binarizer_name, binarizer in (
            ("local-average", zxingcpp.Binarizer.LocalAverage),
            ("global-histogram", zxingcpp.Binarizer.GlobalHistogram),
        ):
            reads = previous.safe_zxing_read(
                variant,
                MATRIX_FORMATS,
                binarizer,
                try_downscale=True,
                return_errors=False,
            )
            mapped: list[tuple[float, Any, np.ndarray]] = []
            for barcode in reads:
                if not barcode.valid or not barcode.text:
                    continue
                fmt = hybrid.normalize_format(str(barcode.format))
                if anchor.format and fmt != anchor.format:
                    continue
                quad = previous.barcode_quad(barcode, scale=scale, offset=(offset_x, offset_y))
                distance = float(np.linalg.norm(quad.mean(axis=0) - anchor.center()))
                mapped.append((distance, barcode, quad))
            if not mapped:
                continue
            distance, barcode, quad = min(mapped, key=lambda item: item[0])
            ordered_anchor = locator.order_quad(np.asarray(anchor.quad, dtype=np.float32))
            anchor_width = max(
                float(np.linalg.norm(ordered_anchor[1] - ordered_anchor[0])),
                float(np.linalg.norm(ordered_anchor[2] - ordered_anchor[3])),
            )
            anchor_height = max(
                float(np.linalg.norm(ordered_anchor[3] - ordered_anchor[0])),
                float(np.linalg.norm(ordered_anchor[2] - ordered_anchor[1])),
            )
            # A context window can contain several nearby matrix symbols.  A
            # checksum-error anchor may only be claimed by a read that remains
            # close to that anchor; otherwise a valid neighbour would erase the
            # genuinely unresolved location on crowded pages.
            maximum_distance = max(45.0, 0.90 * max(anchor_width, anchor_height))
            if distance > maximum_distance:
                continue
            return previous.Result(
                quad=quad,
                text=str(barcode.text),
                format=hybrid.normalize_format(str(barcode.format)),
                status="decoded",
                confidence=1.0,
                sources=set(anchor.sources) | {"zxing:matrix-offset-context-recovery"},
                attempts=anchor.attempts + [f"invalid-anchor:{name}:{binarizer_name}:downscale"],
                evidence={"prior_error": anchor.error, **anchor.evidence},
            )
    return None


def unresolved_matrix_from_anchor(
    image: np.ndarray,
    anchor: previous.MatrixAnchor,
    keep_weak_qr: bool,
) -> previous.Result | None:
    """Promote decoder-error anchors only when their physical structure agrees.

    ZXing checksum errors over dense mechanical drawings can resemble enormous
    QR symbols.  Genuine unresolved QR samples in the regression suites retain
    a roughly square, 40--55% dark module field; the real-document false
    anchors contain only 16--20% dark ink.  The extra gate is deliberately
    applied only to unresolved QR results, never to checksum-valid decodes.
    """
    result = previous.unresolved_from_anchor(image, anchor)
    if result is None or anchor.format != "QR Code" or keep_weak_qr:
        return result
    structure = result.evidence.get("structure", {})
    if (
        float(structure.get("dark_fraction", 0.0)) < 0.30
        or float(structure.get("square_score", 0.0)) < 0.75
        or float(structure.get("transition_score", 0.0)) < 0.55
    ):
        return None
    result.sources.add("verification:qr-module-occupancy")
    return result


def legacy_local_linear_decode(image: np.ndarray, detection: locator.Detection) -> previous.Result | None:
    patch = locator.rectify_quad(image, detection.quad, pad_long=0.08, pad_short=0.24)
    patch = cv2.copyMakeBorder(patch, 16, 16, 20, 20, cv2.BORDER_CONSTANT, value=255)
    try:
        barcode = zxingcpp.read_barcode(
            patch,
            formats=LINEAR_FORMATS,
            try_rotate=True,
            try_downscale=False,
            try_invert=True,
            return_errors=False,
        )
    except Exception:
        barcode = None
    if barcode is None or not barcode.valid or not barcode.text:
        return None
    return previous.Result(
        quad=np.asarray(detection.quad, dtype=np.float32),
        text=str(barcode.text),
        format=hybrid.normalize_format(str(barcode.format)),
        status="decoded",
        confidence=1.0,
        sources=set(detection.sources) | {"deterministic:linear-proposal", "zxing:legacy-deskewed-candidate"},
        attempts=["candidate:legacy-deskewed-native"],
        evidence={"structural_metrics": dict(detection.metrics)},
    )


def _convert_hybrid_result(result: Any) -> previous.Result:
    return previous.Result(
        quad=np.asarray(result.quad, dtype=np.float32),
        text=result.text,
        format=result.format,
        status="decoded" if result.decoded else "review_candidate",
        confidence=float(result.confidence),
        sources=set(result.sources),
        attempts=list(result.attempts),
        evidence={"structural_metrics": dict(result.structural_metrics)},
    )


def linear_paths(
    image: np.ndarray,
    whole_page: Sequence[previous.Result],
    proposals: Sequence[locator.Detection],
    config: Config,
) -> tuple[list[previous.Result], list[previous.Result], dict[str, Any]]:
    eligible, rejected = verify_linear_proposals(image, proposals, config)
    decoded: list[previous.Result] = list(whole_page)
    review: list[previous.Result] = []
    explained = 0
    rejected_by_physics = 0
    rejected_after_decode = 0
    advanced_attempted = 0
    for detection in eligible:
        if any(locator.quad_overlap(detection.quad, item.quad) >= 0.28 for item in whole_page):
            explained += 1
            continue

        direct = legacy_local_linear_decode(image, detection)
        if direct is not None:
            decoded.append(direct)
            continue

        result = hybrid.decode_candidate(image, detection, None)
        if not result.decoded:
            advanced_attempted += 1
            result = advanced.decode_advanced_linear(image, detection, result)
        converted = _convert_hybrid_result(result)
        if converted.decoded:
            decoded.append(converted)
            continue
        if not detection.accepted:
            rejected_after_decode += 1
            continue
        metrics = dict(result.structural_metrics)
        if float(metrics.get("dark_fraction", 0.0)) < 0.08 or float(metrics.get("transition_rate", 0.0)) < 0.03:
            rejected_by_physics += 1
            continue
        if config.include_review_candidates:
            review.append(converted)

    decoded = previous.merge_results(decoded)
    review = [item for item in review if not any(locator.quad_overlap(item.quad, dec.quad) >= 0.70 for dec in decoded)]
    return decoded, review, {
        "raw_visual_proposals": len(proposals),
        "eligible_visual_proposals": len(eligible),
        "locator_rejected": len(rejected) + rejected_after_decode,
        "linear_whole_page_decoded": len(whole_page),
        "linear_decoded": len(decoded),
        "linear_review": len(review),
        "linear_rejected_by_physics": rejected_by_physics,
        "linear_proposals_explained_by_anchor": explained,
        "advanced_linear_candidates": advanced_attempted,
        "detector_mode": config.linear_detector,
        "detector_calls": len(_linear_detector_configs(config.linear_detector)),
    }


def process_page(
    page_number: int,
    page_path: Path,
    source_mode: str,
    config: Config,
    work: Path,
    stage_parallel: bool,
) -> dict[str, Any]:
    page_started = perf_counter()
    started = perf_counter()
    image = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"unable to read {page_path}")
    timings: dict[str, float] = {"load_seconds": perf_counter() - started}

    reads, proposals, fast_seconds = fast_stage(image, config, stage_parallel)
    timings["parallel_fast_stage_seconds"] = fast_seconds
    decoded: list[previous.Result] = []
    unresolved: list[previous.Result] = []
    review: list[previous.Result] = []
    diagnostics: dict[str, Any] = {}

    if config.formats in {"2d", "all"}:
        matrix_decoded, invalid_anchors, matrix_diagnostics = parse_matrix_reads(reads)
        decoded.extend(matrix_decoded)
        started = perf_counter()
        recovered = 0
        invalid_qr_discarded = 0
        for anchor in invalid_anchors:
            result = recover_matrix_anchor(image, anchor)
            if result is not None:
                decoded.append(result)
                recovered += 1
                continue
            unresolved_result = unresolved_matrix_from_anchor(
                image,
                anchor,
                keep_weak_qr=config.keep_unresolved_qr,
            )
            if unresolved_result is not None:
                unresolved.append(unresolved_result)
            elif anchor.format == "QR Code":
                invalid_qr_discarded += 1

        residual_count = residual_decoded = residual_unresolved = 0
        if config.residual_proposals:
            covered = [np.asarray(item.quad, dtype=np.float32) for item in [*decoded, *unresolved]]
            residual = previous.visual_matrix_proposals(image, covered)
            residual_count = len(residual)
            for anchor in residual:
                result = recover_matrix_anchor(image, anchor)
                if result is not None:
                    decoded.append(result)
                    residual_decoded += 1
                    continue
                unresolved_result = unresolved_matrix_from_anchor(
                    image,
                    anchor,
                    keep_weak_qr=config.keep_unresolved_qr,
                )
                if unresolved_result is not None:
                    unresolved.append(unresolved_result)
                    residual_unresolved += 1
        timings["matrix_recovery_seconds"] = perf_counter() - started
        diagnostics["matrix"] = {
            **matrix_diagnostics,
            "error_guided_recovered": recovered,
            "invalid_qr_not_promoted": invalid_qr_discarded,
            "localized_unresolved": len(unresolved),
            "residual_proposals_enabled": config.residual_proposals,
            "residual_proposals": residual_count,
            "residual_decoded": residual_decoded,
            "residual_unresolved": residual_unresolved,
        }

    if config.formats in {"1d", "all"}:
        started = perf_counter()
        linear_whole = parse_linear_reads(reads.get("linear-zxing", []))
        linear_decoded, linear_review, linear_diagnostics = linear_paths(
            image,
            linear_whole,
            proposals,
            config,
        )
        timings["linear_verify_decode_seconds"] = perf_counter() - started
        decoded.extend(linear_decoded)
        review.extend(linear_review)
        diagnostics["linear"] = linear_diagnostics

    decoded = previous.merge_results(decoded)
    unresolved = [item for item in previous.merge_results(unresolved) if not any(locator.quad_overlap(item.quad, dec.quad) >= 0.28 for dec in decoded)]
    review = [item for item in review if not any(locator.quad_overlap(item.quad, dec.quad) >= 0.70 for dec in decoded)]
    decoded.sort(key=lambda item: (round(float(item.center()[1]) / 20), float(item.center()[0])))
    unresolved.sort(key=lambda item: (float(item.center()[1]), float(item.center()[0])))
    review.sort(key=lambda item: (float(item.center()[1]), float(item.center()[0])))

    started = perf_counter()
    if config.save_crops:
        for category, values in (("decoded", decoded), ("unresolved", unresolved), ("review", review)):
            for index, result in enumerate(values, start=1):
                previous.write_crop(image, result, work, page_number, category, index)
    overlay_relative: str | None = None
    if config.save_overlays:
        overlay_path = Path("overlays") / f"page-{page_number:04d}.png"
        (work / overlay_path).parent.mkdir(parents=True, exist_ok=True)
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(work / overlay_path), previous.annotate(color, decoded, unresolved, review))
        overlay_relative = overlay_path.as_posix()
    timings["artifact_seconds"] = perf_counter() - started
    timings["total_seconds"] = perf_counter() - page_started

    payload: dict[str, Any] = {
        "page": page_number,
        "source_image": page_path.relative_to(work).as_posix(),
        "source_mode": source_mode,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "decoded_count": len(decoded),
        "localized_unresolved_matrix_count": len(unresolved),
        "review_candidate_count": len(review),
        "results": [item.to_json() for item in decoded],
        "unresolved_matrix": [item.to_json() for item in unresolved],
        "review_candidates": [item.to_json() for item in review],
        "diagnostics": diagnostics,
        "timings": {key: round(float(value), 4) for key, value in timings.items()},
    }
    if overlay_relative is not None:
        payload["overlay"] = overlay_relative
    return payload


def validate_run(work: Path, pages: Sequence[dict[str, Any]], config: Config) -> dict[str, int]:
    missing: list[str] = []
    expected_crops: set[str] = set()
    for page in pages:
        if not (work / page["source_image"]).is_file():
            missing.append(page["source_image"])
        if config.save_overlays and not (work / page["overlay"]).is_file():
            missing.append(page["overlay"])
        for key in ("results", "unresolved_matrix", "review_candidates"):
            for result in page[key]:
                crop = result.get("crop")
                if crop:
                    expected_crops.add(crop)
                    if not (work / crop).is_file():
                        missing.append(crop)
    actual_crops = {path.relative_to(work).as_posix() for path in (work / "crops").rglob("*.png")} if (work / "crops").exists() else set()
    orphan = actual_crops - expected_crops
    if missing or orphan or (config.save_crops and actual_crops != expected_crops):
        raise RuntimeError(f"run integrity failure: missing={missing[:8]} orphan={sorted(orphan)[:8]}")
    return {"expected_crops": len(expected_crops), "actual_crops": len(actual_crops), "orphan_crops": 0}


def _resolved_workers(requested: int, page_count: int) -> int:
    if requested > 0:
        return max(1, min(requested, page_count))
    return max(1, min(8, os.cpu_count() or 1, page_count))


def run_pipeline(input_path: Path, config: Config) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output = config.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    work, run_id = previous.prepare_work_directory(output)
    run_started = perf_counter()
    started_at = previous.utc_now()
    try:
        extraction_started = perf_counter()
        sources = collect_pages(input_path, config, work)
        extraction_seconds = perf_counter() - extraction_started
        workers = _resolved_workers(config.workers, len(sources))
        stage_parallel = workers == 1

        arguments = [(page, path, mode, config, work, stage_parallel) for page, path, mode in sources]
        if workers == 1:
            pages = [process_page(*values) for values in arguments]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="barcode-page") as executor:
                futures = [executor.submit(process_page, *values) for values in arguments]
                pages = [future.result() for future in futures]
        pages.sort(key=lambda item: item["page"])
        task_timing_note = (
            "overlaps other page tasks"
            if workers > 1
            else "page tasks are sequential; internal detector scans may overlap"
        )
        for page in pages:
            print(
                f"page {page['page']:04d}: {page['decoded_count']} decoded, "
                f"{page['localized_unresolved_matrix_count']} unresolved, "
                f"{page['review_candidate_count']} review, "
                f"task={page['timings']['total_seconds']:.3f}s ({task_timing_note})",
                flush=True,
            )

        integrity = validate_run(work, pages, config)
        processing_sum = sum(page["timings"]["total_seconds"] for page in pages)
        wall_seconds = perf_counter() - run_started
        payload = {
            "schema_version": 3,
            "run": {
                "id": run_id,
                "started_at": started_at,
                "completed_at": previous.utc_now(),
                "input": str(input_path),
                "input_sha256": previous.sha256_file(input_path) if input_path.is_file() else None,
                "pipeline_sha256": previous.sha256_file(Path(__file__)),
                "output_root": ".",
            },
            "method": "parallel native-raster ZXing + fused proposal-only OpenCV + targeted recovery",
            "configuration": {
                "formats": config.formats,
                "render_dpi": config.render_dpi,
                "pages": config.pages,
                "workers_requested": config.workers,
                "workers_used": workers,
                "linear_detector": config.linear_detector,
                "batch_pdf_extraction": config.batch_pdf_extraction,
                "save_crops": config.save_crops,
                "save_overlays": config.save_overlays,
                "residual_proposals": config.residual_proposals,
                "keep_unresolved_qr": config.keep_unresolved_qr,
                "include_review_candidates": config.include_review_candidates,
                "minimum_linear_score": config.minimum_linear_score,
                "minimum_linear_length": config.minimum_linear_length,
            },
            "summary": {
                "pages": len(pages),
                "decoded": sum(page["decoded_count"] for page in pages),
                "localized_unresolved_matrix": sum(page["localized_unresolved_matrix_count"] for page in pages),
                "review_candidates": sum(page["review_candidate_count"] for page in pages),
                "extraction_seconds": round(extraction_seconds, 4),
                "page_task_elapsed_sum_seconds": round(processing_sum, 4),
                "mean_page_task_elapsed_seconds": round(processing_sum / max(1, len(pages)), 4),
                # Retained for compatibility with manifests created before the
                # terminal reporting was clarified. This is a sum of overlapping
                # task elapsed times, not CPU time and not end-to-end wall time.
                "processing_cpu_wall_sum_seconds": round(processing_sum, 4),
                "wall_seconds": round(wall_seconds, 4),
                "effective_wall_seconds_per_page": round(wall_seconds / max(1, len(pages)), 4),
                "throughput_pages_per_second": round(len(pages) / max(wall_seconds, 1e-9), 3),
            },
            "integrity": integrity,
            "pages": pages,
        }
        (work / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        os.replace(work, output)
        print(
            f"summary: {len(pages)} pages completed in {wall_seconds:.3f}s wall time "
            f"with {workers} worker(s); throughput-equivalent "
            f"{1000.0 * wall_seconds / max(1, len(pages)):.1f} ms/page; "
            f"{'overlapping ' if workers > 1 else ''}page-task sum {processing_sum:.3f}s",
            flush=True,
        )
        print(f"wrote {output / 'detections.json'}")
        return payload
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("archive/versions/v5-optimized/output/run"))
    parser.add_argument("--formats", choices=("2d", "1d", "all"), default="all")
    parser.add_argument("--pages", type=hybrid.parse_page_selection)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto, 1 = single-page latency mode")
    parser.add_argument("--linear-detector", choices=("optimized", "exhaustive"), default="optimized")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-crops", dest="save_crops", action="store_false")
    parser.add_argument("--no-overlays", dest="save_overlays", action="store_false")
    parser.add_argument("--no-batch-pdf-extraction", dest="batch_pdf_extraction", action="store_false")
    parser.add_argument("--residual-proposals", action="store_true")
    parser.add_argument("--keep-unresolved-qr", action="store_true")
    parser.add_argument("--no-review-candidates", dest="include_review_candidates", action="store_false")
    parser.add_argument("--minimum-linear-score", type=float, default=0.52)
    parser.add_argument("--minimum-linear-length", type=float, default=100.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = Config(
        output=arguments.output,
        formats=arguments.formats,
        render_dpi=arguments.render_dpi,
        pages=arguments.pages,
        overwrite=arguments.overwrite,
        save_crops=arguments.save_crops,
        save_overlays=arguments.save_overlays,
        residual_proposals=arguments.residual_proposals,
        include_review_candidates=arguments.include_review_candidates,
        keep_unresolved_qr=arguments.keep_unresolved_qr,
        minimum_linear_score=arguments.minimum_linear_score,
        minimum_linear_length=arguments.minimum_linear_length,
        workers=arguments.workers,
        linear_detector=arguments.linear_detector,
        batch_pdf_extraction=arguments.batch_pdf_extraction,
    )
    try:
        run_pipeline(arguments.input, config)
    except Exception as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
