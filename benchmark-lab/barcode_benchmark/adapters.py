from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluate import canonical_kind, canonical_symbology
from .io import PREDICTION_SCHEMA, image_index


def _prediction(item: dict[str, Any]) -> dict[str, Any]:
    sources = item.get("sources")
    if not isinstance(sources, list):
        source = item.get("source")
        sources = [str(source)] if source else []
    return {
        "polygon": item.get("quad"),
        "kind": canonical_kind(item.get("kind")),
        "symbology": canonical_symbology(item.get("format")),
        "payload": item.get("text"),
        "confidence": item.get("confidence", 0.0),
        "status": item.get("status", "unknown"),
        "sources": sources,
    }


def adapt_current_detections(
    dataset: dict[str, Any],
    detections: dict[str, Any],
    *,
    pipeline: str,
    command: str | None = None,
) -> dict[str, Any]:
    truth_images = list(image_index(dataset, "dataset").values())
    pages = detections.get("pages")
    if not isinstance(pages, list):
        raise ValueError("detections: 'pages' must be a list")
    if len(truth_images) != len(pages):
        raise ValueError(
            f"dataset/detection page mismatch: {len(truth_images)} != {len(pages)}; "
            "positional adaptation is refused unless all images are present"
        )
    output_images: list[dict[str, Any]] = []
    for truth, page in zip(truth_images, pages):
        image_size = page.get("image_size", {})
        if (
            isinstance(image_size, dict)
            and image_size.get("width") is not None
            and image_size.get("height") is not None
            and (int(image_size["width"]), int(image_size["height"])) != (int(truth["width"]), int(truth["height"]))
        ):
            raise ValueError(
                f"{truth['id']}: prediction dimensions "
                f"{(image_size['width'], image_size['height'])} do not match "
                f"dataset dimensions {(truth['width'], truth['height'])}"
            )
        candidates: list[dict[str, Any]] = []
        for key in ("results", "unresolved_matrix", "review_candidates", "detections"):
            values = page.get(key, [])
            if isinstance(values, list):
                candidates.extend(_prediction(item) for item in values)
        timings = page.get("timings", {})
        latency_ms = None
        if isinstance(timings, dict):
            if isinstance(timings.get("total_seconds"), (int, float)):
                latency_ms = 1000.0 * float(timings["total_seconds"])
            else:
                stage_values = [
                    float(value)
                    for key, value in timings.items()
                    if key != "artifact_seconds" and isinstance(value, (int, float))
                ]
                if stage_values:
                    latency_ms = 1000.0 * sum(stage_values)
        output_images.append({
            "id": truth["id"],
            "predictions": candidates,
            "timing": {"latency_ms": latency_ms, "peak_rss_mb": None},
            "diagnostics": {
                "source_page": page.get("page"),
                "source_image": page.get("source_image"),
            },
        })
    return {
        "schema_version": PREDICTION_SCHEMA,
        "run": {
            "pipeline": pipeline,
            "command": command,
            "adapter": "existing-project-detections-v1",
            "source_method": detections.get("method"),
            "source_configuration": detections.get("configuration", {}),
        },
        "images": output_images,
    }
