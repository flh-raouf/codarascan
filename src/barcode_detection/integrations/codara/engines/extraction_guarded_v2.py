"""The stable guarded v2 copy of Base with the second-generation review gate.

The Base proposal, decoder, matrix path, and decoded-result contract are still
delegated to the same vendored extractor. Guarded v2 changes only which
undecoded linear candidates are surfaced for operator review.
"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from .base import Engine, EngineInfo, PageOutcome, RegionStatus
from .extraction_vendored import VendoredExtractor
from .linear_review_gate_v2 import (
    ReviewGateV2Config,
    assess_linear_review_v2,
)


class GuardedExtractorV2(VendoredExtractor):
    """Run Base exactly, then apply only the v2 review-presentation gate."""

    info = EngineInfo(
        id="guarded-adaptive-extractor-v2",
        label="Guarded adaptive extractor v2",
        capability=VendoredExtractor.info.capability,
        summary=(
            "Base proposal and decoder recall with the Guarded v2 physical-evidence "
            "gate, including a narrow rescue for compressed long document barcodes."
        ),
        speed_ms_per_page="~250 ms/page sequential",
        accuracy_note=(
            "Base decode path unchanged; v2 review gate preserves thin long "
            "administrative barcodes while rejecting sparse hatching"
        ),
        badge="Guarded v2",
        options={
            "kinds": VendoredExtractor.info.options["kinds"],
            "formats": VendoredExtractor.info.options["formats"],
        },
    )

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi=None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        started = perf_counter()
        options = options or {}
        outcome = super().analyze_page(page, path, roi=roi, options=options)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return outcome

        config = _review_gate_config(options)
        visible = []
        suppressed: list[dict[str, Any]] = []
        input_review_count = 0
        accepted_review_count = 0
        for region in outcome.regions:
            if (
                region.status is not RegionStatus.REVIEW_CANDIDATE
                or region.kind not in {"linear", "1d"}
            ):
                visible.append(region)
                continue
            input_review_count += 1
            structural: dict[str, Any] = {}
            evidence = region.extras.get("evidence") if isinstance(region.extras, dict) else None
            if isinstance(evidence, dict) and isinstance(evidence.get("structural_metrics"), dict):
                structural = evidence["structural_metrics"]
            decision = assess_linear_review_v2(
                gray,
                region.quad,
                base_metrics=structural,
                sources=region.sources,
                config=config,
            )
            extras = dict(region.extras)
            merged_evidence = dict(evidence) if isinstance(evidence, dict) else {}
            merged_evidence["review_gate"] = decision.evidence
            extras["evidence"] = merged_evidence
            annotated = replace(
                region,
                confidence=round(float(decision.score), 4),
                extras=extras,
            )
            if decision.accepted:
                visible.append(annotated)
                accepted_review_count += 1
            else:
                suppressed.append(
                    {
                        "quad": list(region.quad),
                        "kind": region.kind,
                        "base_confidence": round(float(region.confidence), 4),
                        "reason": decision.reason,
                        "evidence": decision.evidence,
                    }
                )
                if bool(options.get("expose_suppressed_candidates", False)):
                    visible.append(annotated)

        diagnostics = dict(outcome.diagnostics)
        diagnostics["guarded_review_v2"] = {
            "validation": "review-gate-v2",
            "input_review_candidates": input_review_count,
            "accepted_review_candidates": accepted_review_count,
            "suppressed_review_candidates": len(suppressed),
            "suppressed_candidates": suppressed,
            "decode_path_preserved": True,
            "base_engine": "adaptive-extractor",
        }
        diagnostics["guarded_review_v2_seconds"] = round(perf_counter() - started, 6)
        return replace(outcome, regions=visible, diagnostics=diagnostics)


def _review_gate_config(options: dict[str, Any]) -> ReviewGateV2Config:
    values = options.get("review_gate_v2_config")
    if not isinstance(values, dict):
        values = options.get("review_gate_config")
    if not isinstance(values, dict):
        return ReviewGateV2Config()
    allowed = {field.name for field in fields(ReviewGateV2Config)}
    return ReviewGateV2Config(**{key: value for key, value in values.items() if key in allowed})


ENGINE: Engine = GuardedExtractorV2()
