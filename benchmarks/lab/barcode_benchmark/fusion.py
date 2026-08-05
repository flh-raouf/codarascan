from __future__ import annotations

from copy import deepcopy
from typing import Any

from .geometry import polygon, similarity


Prediction = dict[str, Any]


def fuse_predictions(
    primary: list[Prediction],
    supplements: list[list[Prediction]],
    *,
    overlap_threshold: float = 0.45,
) -> list[Prediction]:
    """Preserve primary geometry while attaching or adding complementary evidence."""
    output = deepcopy(primary)
    for group in supplements:
        for candidate in group:
            candidate_polygon = polygon(candidate.get("polygon"), "supplement")
            best_index: int | None = None
            best_score = 0.0
            for index, existing in enumerate(output):
                existing_polygon = polygon(existing.get("polygon"), "fused")
                iou, intersection_over_smaller = similarity(candidate_polygon, existing_polygon)
                score = max(iou, intersection_over_smaller)
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index is None or best_score < overlap_threshold:
                output.append(deepcopy(candidate))
                continue

            existing = output[best_index]
            existing["sources"] = list(dict.fromkeys([
                *existing.get("sources", []),
                *candidate.get("sources", []),
            ]))
            if candidate.get("payload") and not existing.get("payload"):
                # Payload evidence may improve the primary detection, but its
                # tight/scanline geometry must not replace the learned box.
                for key in ("payload", "symbology", "status"):
                    if candidate.get(key) is not None:
                        existing[key] = candidate[key]
            existing["confidence"] = max(
                float(existing.get("confidence", 0.0)),
                float(candidate.get("confidence", 0.0)),
            )
    return output
