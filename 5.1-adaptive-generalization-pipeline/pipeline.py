#!/usr/bin/env python3
"""Adaptive extension of project 5 for small and variably scaled symbols."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import zxingcpp

from adaptive_generalization import (
    accept_short,
    dense_layout_candidate_allowed,
    dense_layout_enabled,
    input_diagnostics,
    module_metrics,
    relative_contexts,
    scale_plan,
    split_dense_linear_quad,
)
from ean13_recovery import checksum_valid as ean13_checksum_valid
from ean13_recovery import qr_geometry_proposals
from ean13_recovery import template_evidence as repeated_ean13_template_evidence


ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = ROOT / "5-optimized pipeline"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))
spec = importlib.util.spec_from_file_location("barcode_pipeline_v5", LEGACY_DIR / "pipeline.py")
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load project 5")
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)


@dataclass
class Config(legacy.Config):
    residual_proposals: bool = True
    adaptive_generalization: bool = True


_legacy_scan_matrix = legacy._scan_matrix
_legacy_scan_linear = legacy._scan_linear
_legacy_recover_matrix_anchor = legacy.recover_matrix_anchor
_legacy_process_page = legacy.process_page
_legacy_visual_matrix_proposals = legacy.previous.visual_matrix_proposals
_legacy_collect_pages = legacy.collect_pages
_legacy_unresolved_matrix_from_anchor = legacy.unresolved_matrix_from_anchor


def collect_pages(input_path: Path, config: Config, work: Path) -> list[tuple[int, Path, str]]:
    """Copy image-directory inputs into the atomic work tree.

    Project 5 assumed every source already lived below its temporary output and
    failed while serializing external image directories.  The adaptive project
    fixes that independently of barcode accuracy.
    """
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if input_path.is_file() and input_path.suffix.lower() in supported:
        pages = work / "pages"
        pages.mkdir(parents=True, exist_ok=True)
        destination = pages / f"page-0001{input_path.suffix.lower()}"
        shutil.copy2(input_path, destination)
        return [(1, destination, "input-image-file")]
    if not input_path.is_dir():
        return _legacy_collect_pages(input_path, config, work)
    sources = sorted(path for path in input_path.iterdir() if path.suffix.lower() in supported)
    pages = work / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    output: list[tuple[int, Path, str]] = []
    for index, source in enumerate(sources, start=1):
        destination = pages / f"page-{index:04d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        output.append((index, destination, "input-image-directory"))
    return output


def visual_linear_proposals(gray: np.ndarray, mode: str, adaptive: bool = True) -> list[Any]:
    output: list[Any] = []
    for plan in scale_plan(gray.shape, adaptive):
        work = gray if plan.scale == 1.0 else cv2.resize(gray, None, fx=plan.scale, fy=plan.scale, interpolation=plan.interpolation)
        for name, scales, threshold in legacy._linear_detector_configs(mode):
            detector = legacy.locator.make_detector(scales, work.shape, threshold)
            try:
                found, points = detector.detectMulti(work)
            except cv2.error:
                continue
            if not found or points is None:
                continue
            for quad in np.asarray(points, np.float32):
                detection = legacy.locator.Detection(
                    quad=quad / plan.scale,
                    sources={f"opencv:adaptive:{name}:scale-{plan.scale:g}"},
                    proposal_score=0.48 if plan.scale == 1.0 else 0.46,
                )
                long_side, short_side = detection.long_short()
                if long_side >= 14 and short_side >= 3 and long_side / (short_side + legacy.locator.EPS) >= 1.15:
                    output.append(detection)
    merged = legacy.locator.deduplicate(output)
    if len(merged) < 12:
        return merged
    refined: list[Any] = []
    for detection in merged:
        children = split_dense_linear_quad(gray, detection.quad)
        if not children:
            refined.append(detection)
            continue
        for index, child in enumerate(children, start=1):
            refined.append(
                legacy.locator.Detection(
                    quad=child,
                    sources=set(detection.sources) | {f"geometry:dense-linear-split:{index}/{len(children)}"},
                    proposal_score=max(0.0, detection.proposal_score - 0.01),
                )
            )
    return legacy.locator.deduplicate(refined)


def verify_linear_proposals(gray: np.ndarray, proposals: Sequence[Any], config: Config) -> tuple[list[Any], list[Any]]:
    eligible, rejected = [], []
    decode_floor = max(0.35, config.minimum_linear_score - 0.16)
    for detection in proposals:
        detection.metrics = legacy.locator.barcode_structure_metrics(gray, detection.quad)
        if config.adaptive_generalization:
            detection.metrics.update(module_metrics(gray, detection.quad, legacy.locator.rectify_quad))
        detection.structural_score = detection.metrics.get("score", 0.0)
        normal = legacy.locator.accept_linear(
            detection,
            config.minimum_linear_score,
            allow_tensor_only=False,
            minimum_linear_length=config.minimum_linear_length,
        )
        adaptive = config.adaptive_generalization and accept_short(
            detection, config.minimum_linear_score, config.minimum_linear_length
        )
        detection.accepted = bool(normal)
        detection.metrics["legacy_length_gate_passed"] = float(normal)
        detection.metrics["adaptive_short_recovery_candidate"] = float(adaptive)
        (eligible if detection.accepted or adaptive or detection.structural_score >= decode_floor else rejected).append(detection)
    return eligible, rejected


def linear_paths(
    image: np.ndarray,
    whole_page: Sequence[Any],
    proposals: Sequence[Any],
    config: Config,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    eligible, rejected = verify_linear_proposals(image, proposals, config)
    decoded, review = list(whole_page), []
    dense_enabled = bool(
        config.adaptive_generalization
        and dense_layout_enabled(proposals, sum(bool(detection.accepted) for detection in eligible))
    )
    explained = rejected_by_physics = rejected_after_decode = advanced_attempted = 0
    repeated_ean13_recovered = 0
    ean13_templates = [
        item
        for item in whole_page
        if "".join(character for character in str(item.format).lower() if character.isalnum()) == "ean13"
        and item.text is not None
        and ean13_checksum_valid(item.text)
    ]
    for detection in eligible:
        if any(legacy.locator.quad_overlap(detection.quad, item.quad) >= 0.28 for item in whole_page):
            explained += 1
            continue
        direct = legacy.legacy_local_linear_decode(image, detection)
        if direct is not None:
            decoded.append(direct)
            continue
        result = legacy.hybrid.decode_candidate(image, detection, None)
        if not result.decoded:
            advanced_attempted += 1
            result = legacy.advanced.decode_advanced_linear(image, detection, result)
        converted = legacy._convert_hybrid_result(result)
        if converted.decoded:
            decoded.append(converted)
            continue
        dense_candidate = bool(dense_enabled and dense_layout_candidate_allowed(detection))
        if dense_candidate and ean13_templates:
            matches: list[tuple[float, Any, dict[str, float]]] = []
            for template in ean13_templates:
                evidence = repeated_ean13_template_evidence(
                    image,
                    detection.quad,
                    template.text or "",
                    legacy.locator.rectify_quad,
                )
                if evidence is not None:
                    matches.append((float(evidence["ean13_template_correlation"]), template, evidence))
            matches.sort(key=lambda item: item[0], reverse=True)
            if matches and (len(matches) == 1 or matches[0][0] - matches[1][0] >= 0.02):
                correlation, template, evidence = matches[0]
                decoded.append(
                    legacy.previous.Result(
                        quad=np.asarray(detection.quad, np.float32),
                        text=template.text,
                        format=template.format,
                        status="decoded",
                        confidence=min(0.97, 0.70 + 0.25 * correlation + 0.50 * evidence["ean13_template_margin"]),
                        sources=set(detection.sources) | {"deterministic:ean13-repeated-template", "context:dense-barcode-layout"},
                        attempts=["deterministic:ean13-template-fit"],
                        evidence={"structural_metrics": dict(detection.metrics), "template_anchor_source": "independent-zxing-page-read", **evidence},
                    )
                )
                repeated_ean13_recovered += 1
                continue
        if not detection.accepted and not dense_candidate:
            rejected_after_decode += 1
            continue
        metrics = dict(result.structural_metrics)
        fails_physics = (
            float(metrics.get("dark_fraction", 0.0)) < 0.08
            or float(metrics.get("transition_rate", 0.0)) < 0.03
        )
        if fails_physics and not dense_candidate:
            rejected_by_physics += 1
            continue
        if config.include_review_candidates:
            if dense_candidate:
                converted.sources.add("context:dense-barcode-layout")
                converted.evidence["dense_layout_candidate_review"] = True
                if ean13_templates:
                    long_side, _short_side = detection.long_short()
                    upper_bound = float(long_side) / 95.0
                    converted.evidence["resolution_diagnostics"] = {
                        "candidate_pixels_per_ean13_module_upper_bound": round(upper_bound, 4),
                        "likely_undersampled": bool(upper_bound < 1.25),
                        "reason": "candidate width is too close to the 95-module EAN-13 minimum for reliable independent payload recovery",
                    }
            review.append(converted)
    decoded = legacy.previous.merge_results(decoded)
    review = [
        item
        for item in review
        if not any(legacy.locator.quad_overlap(item.quad, decoded_item.quad) >= 0.70 for decoded_item in decoded)
    ]
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
        "repeated_ean13_template_recovered": repeated_ean13_recovered,
        "dense_layout_short_review_enabled": dense_enabled,
        "dense_split_proposals": sum(
            any(source.startswith("geometry:dense-linear-split:") for source in detection.sources)
            for detection in proposals
        ),
        "detector_mode": config.linear_detector,
        "detector_calls": len(legacy._linear_detector_configs(config.linear_detector)),
    }


def scan_matrix(image: np.ndarray, binarizer: Any, adaptive: bool) -> list[tuple[Any, float]]:
    native = _legacy_scan_matrix(image, binarizer)
    output = [(barcode, 1.0) for barcode in native]
    if any(barcode.valid and barcode.text for barcode in native) or not adaptive:
        return output
    for plan in scale_plan(image.shape, True)[1:]:
        work = cv2.resize(image, None, fx=plan.scale, fy=plan.scale, interpolation=plan.interpolation)
        reads = legacy.previous.safe_zxing_read(
            work, legacy.MATRIX_FORMATS, binarizer, try_downscale=False, return_errors=True
        )
        output.extend((barcode, plan.scale) for barcode in reads)
        if any(barcode.valid and barcode.text for barcode in reads):
            break
    return output


def scan_linear(image: np.ndarray, adaptive: bool) -> list[tuple[Any, float]]:
    native = _legacy_scan_linear(image)
    if native or not adaptive:
        return [(barcode, 1.0) for barcode in native]
    for plan in scale_plan(image.shape, True)[1:]:
        work = cv2.resize(image, None, fx=plan.scale, fy=plan.scale, interpolation=plan.interpolation)
        reads = legacy.previous.safe_zxing_read(
            work,
            zxingcpp.barcode_formats_from_str("AllLinear"),
            zxingcpp.Binarizer.LocalAverage,
            try_downscale=False,
            return_errors=False,
        )
        if reads:
            return [(barcode, plan.scale) for barcode in reads]
    return []


def fast_stage(gray: np.ndarray, config: Config, parallel: bool) -> tuple[dict[str, list[Any]], list[Any], float]:
    jobs: dict[str, Callable[[], Any]] = {}
    if config.formats in {"2d", "all"}:
        jobs["matrix-local"] = lambda: scan_matrix(gray, zxingcpp.Binarizer.LocalAverage, config.adaptive_generalization)
        jobs["matrix-global"] = lambda: scan_matrix(gray, zxingcpp.Binarizer.GlobalHistogram, config.adaptive_generalization)
        if config.adaptive_generalization:
            jobs["matrix-qr-geometry"] = lambda: qr_geometry_proposals(gray)
    if config.formats in {"1d", "all"}:
        jobs["linear-zxing"] = lambda: scan_linear(gray, config.adaptive_generalization)
        jobs["linear-opencv"] = lambda: visual_linear_proposals(gray, config.linear_detector, config.adaptive_generalization)
    started = perf_counter()
    if parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {name: executor.submit(function) for name, function in jobs.items()}
            values = {name: future.result() for name, future in futures.items()}
    else:
        values = {name: function() for name, function in jobs.items()}
    return values, values.pop("linear-opencv", []), perf_counter() - started


def parse_matrix_reads(reads_by_name: dict[str, list[Any]]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    decoded, invalid = [], []
    diagnostics: dict[str, Any] = {"reads": {}, "invalid_errors": {}, "scales": []}
    used_scales: set[float] = set()
    for key, source in (("matrix-local", "local-average"), ("matrix-global", "global-histogram")):
        reads = reads_by_name.get(key, [])
        diagnostics["reads"][source] = len(reads)
        for item in reads:
            barcode, scale = item if isinstance(item, tuple) else (item, 1.0)
            used_scales.add(float(scale))
            quad = legacy.previous.barcode_quad(barcode, scale=scale)
            if abs(float(cv2.contourArea(quad))) < 9.0:
                continue
            if barcode.valid and barcode.text:
                decoded.append(legacy.previous.Result(
                    quad=quad, text=str(barcode.text), format=legacy.hybrid.normalize_format(str(barcode.format)),
                    status="decoded", confidence=1.0,
                    sources={f"zxing:matrix-whole-page:{source}:scale-{scale:g}"},
                    attempts=[f"whole-page:{source}:scale-{scale:g}"],
                ))
            else:
                error = str(getattr(barcode, "error", None) or "invalid")
                diagnostics["invalid_errors"][error.split(" @ ", 1)[0]] = diagnostics["invalid_errors"].get(error.split(" @ ", 1)[0], 0) + 1
                invalid.append(legacy.previous.MatrixAnchor(
                    quad=quad, format=legacy.hybrid.normalize_format(str(barcode.format)), error=error,
                    sources={f"zxing:invalid-matrix:{source}:scale-{scale:g}"},
                    attempts=[f"whole-page:{source}:scale-{scale:g}:return-errors"],
                ))
    qr_geometry = reads_by_name.get("matrix-qr-geometry", [])
    for quad, scale in qr_geometry:
        if any(legacy.locator.quad_overlap(quad, result.quad) >= 0.28 for result in decoded):
            continue
        invalid.append(
            legacy.previous.MatrixAnchor(
                quad=np.asarray(quad, np.float32),
                format="QR Code",
                error="opencv-qr-geometry",
                sources={f"opencv:qr-geometry:scale-{scale:g}"},
                attempts=[f"opencv:qr-detect-multi:scale-{scale:g}"],
                confidence=0.90,
                evidence={"geometry_only": True, "proposal_scale": scale},
            )
        )
    decoded = legacy.previous.merge_results(decoded)
    invalid = [anchor for anchor in legacy.previous.merge_anchors(invalid) if not any(legacy.locator.quad_overlap(anchor.quad, result.quad) >= 0.28 for result in decoded)]
    diagnostics.update(
        decoded_unique=len(decoded),
        invalid_unique_unexplained=len(invalid),
        scales=sorted(used_scales),
        opencv_qr_geometry_proposals=len(qr_geometry),
    )
    return decoded, invalid, diagnostics


def parse_linear_reads(reads: Sequence[Any]) -> list[Any]:
    output = []
    for item in reads:
        barcode, scale = item if isinstance(item, tuple) else (item, 1.0)
        if barcode.valid and barcode.text:
            output.append(legacy.previous.Result(
                quad=legacy.previous.barcode_quad(barcode, scale=scale), text=str(barcode.text),
                format=legacy.hybrid.normalize_format(str(barcode.format)), status="decoded", confidence=1.0,
                sources={f"zxing:linear-whole-page:scale-{scale:g}"}, attempts=[f"whole-page:linear:scale-{scale:g}"],
            ))
    return legacy.previous.merge_results(output)


def recover_matrix_anchor(image: np.ndarray, anchor: Any) -> Any | None:
    recovered = _legacy_recover_matrix_anchor(image, anchor)
    if recovered is not None:
        return recovered
    for width, height, fx, fy, scale, name in relative_contexts(image.shape, anchor.quad):
        crop, offset_x, offset_y = legacy._context_crop(image, anchor.center(), width, height, fx, fy)
        work = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for bin_name, binarizer in (("local", zxingcpp.Binarizer.LocalAverage), ("global", zxingcpp.Binarizer.GlobalHistogram)):
            reads = legacy.previous.safe_zxing_read(work, legacy.MATRIX_FORMATS, binarizer, try_downscale=True, return_errors=False)
            for barcode in reads:
                if not barcode.valid or not barcode.text:
                    continue
                quad = legacy.previous.barcode_quad(barcode, scale=scale, offset=(offset_x, offset_y))
                if float(np.linalg.norm(quad.mean(axis=0) - anchor.center())) > max(45.0, 1.1 * max(cv2.minAreaRect(np.asarray(anchor.quad, np.float32))[1])):
                    continue
                return legacy.previous.Result(
                    quad=quad, text=str(barcode.text), format=legacy.hybrid.normalize_format(str(barcode.format)),
                    status="decoded", confidence=1.0, sources=set(anchor.sources) | {"zxing:matrix-relative-context"},
                    attempts=list(anchor.attempts) + [f"{name}:{bin_name}"], evidence={"prior_error": anchor.error, **anchor.evidence},
                )
    return None


def visual_matrix_proposals(image: np.ndarray, covered: Sequence[np.ndarray], limit: int = 12) -> list[Any]:
    if image.size > 4_000_000 or max(image.shape[:2]) > 1800:
        return []
    plan = scale_plan(image.shape, True)[-1]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    work = gray if plan.scale == 1.0 else cv2.resize(gray, None, fx=plan.scale, fy=plan.scale, interpolation=plan.interpolation)
    bgr = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    proposals = _legacy_visual_matrix_proposals(bgr, [np.asarray(item) * plan.scale for item in covered], limit)
    for proposal in proposals:
        proposal.quad = np.asarray(proposal.quad, np.float32) / plan.scale
        proposal.sources.add(f"adaptive:matrix-scale-{plan.scale:g}")
    return proposals


def unresolved_matrix_from_anchor(image: np.ndarray, anchor: Any, keep_weak_qr: bool) -> Any | None:
    if str(getattr(anchor, "error", "")) == "visual-matrix-proposal":
        return None
    if str(getattr(anchor, "error", "")) == "opencv-qr-geometry":
        structural = _legacy_unresolved_matrix_from_anchor(image, anchor, keep_weak_qr)
        evidence = dict(getattr(anchor, "evidence", {}))
        if structural is not None:
            evidence.update(structural.evidence)
        return legacy.previous.Result(
            quad=np.asarray(anchor.quad, np.float32),
            text=None,
            format="QR Code",
            status="localized_unresolved_matrix",
            confidence=max(0.90, float(anchor.confidence)),
            sources=set(anchor.sources) | {"verification:opencv-qr-finder-geometry"},
            attempts=list(anchor.attempts),
            evidence={"decoder_error": anchor.error, **evidence},
        )
    return _legacy_unresolved_matrix_from_anchor(image, anchor, keep_weak_qr)


def process_page(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _legacy_process_page(*args, **kwargs)
    page_path = Path(args[1] if len(args) > 1 else kwargs["page_path"])
    image = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if image is not None:
        payload.setdefault("diagnostics", {})["adaptive_generalization"] = {
            "enabled": bool(getattr(args[3] if len(args) > 3 else kwargs["config"], "adaptive_generalization", True)),
            "proposal_scales": [plan.scale for plan in scale_plan(image.shape, True)],
            "input_quality": input_diagnostics(image),
        }
    return payload


legacy.visual_linear_proposals = visual_linear_proposals
legacy.verify_linear_proposals = verify_linear_proposals
legacy.linear_paths = linear_paths
legacy.fast_stage = fast_stage
legacy.parse_matrix_reads = parse_matrix_reads
legacy.parse_linear_reads = parse_linear_reads
legacy.recover_matrix_anchor = recover_matrix_anchor
legacy.unresolved_matrix_from_anchor = unresolved_matrix_from_anchor
legacy.previous.visual_matrix_proposals = visual_matrix_proposals
legacy.process_page = process_page
legacy.collect_pages = collect_pages


def run_pipeline(input_path: Path, config: Config) -> dict[str, Any]:
    payload = legacy.run_pipeline(input_path, config)
    payload["configuration"]["adaptive_generalization"] = config.adaptive_generalization
    payload["method"] = "project 5 + adaptive scale-space, module evidence, and relative matrix recovery"
    destination = config.output.expanduser().resolve() / "detections.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("5.1-adaptive-generalization-pipeline/output/run"))
    parser.add_argument("--formats", choices=("2d", "1d", "all"), default="all")
    parser.add_argument("--pages", type=legacy.hybrid.parse_page_selection)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--linear-detector", choices=("optimized", "exhaustive"), default="optimized")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-crops", dest="save_crops", action="store_false")
    parser.add_argument("--no-overlays", dest="save_overlays", action="store_false")
    parser.add_argument("--no-review-candidates", dest="include_review_candidates", action="store_false")
    parser.add_argument("--no-adaptive-generalization", dest="adaptive_generalization", action="store_false")
    parser.add_argument("--no-residual-proposals", dest="residual_proposals", action="store_false")
    parser.add_argument("--minimum-linear-score", type=float, default=0.52)
    parser.add_argument("--minimum-linear-length", type=float, default=100.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = Config(
        output=arguments.output, formats=arguments.formats, render_dpi=arguments.render_dpi,
        pages=arguments.pages, overwrite=arguments.overwrite, save_crops=arguments.save_crops,
        save_overlays=arguments.save_overlays, residual_proposals=arguments.residual_proposals,
        include_review_candidates=arguments.include_review_candidates,
        minimum_linear_score=arguments.minimum_linear_score,
        minimum_linear_length=arguments.minimum_linear_length, workers=arguments.workers,
        linear_detector=arguments.linear_detector, adaptive_generalization=arguments.adaptive_generalization,
    )
    try:
        run_pipeline(arguments.input, config)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
