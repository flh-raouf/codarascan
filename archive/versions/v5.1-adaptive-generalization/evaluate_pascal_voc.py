#!/usr/bin/env python3
"""Evaluate a detections manifest against co-located Pascal VOC XML files."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def gt_objects(image_path: Path) -> list[dict[str, Any]]:
    xml_path = image_path.with_suffix(".xml")
    if not xml_path.is_file():
        return []
    root = ET.parse(xml_path).getroot()
    output = []
    for item in root.findall("object"):
        box = item.find("bndbox")
        if box is None:
            continue
        x1 = float(box.findtext("xmin", "0"))
        y1 = float(box.findtext("ymin", "0"))
        x2 = float(box.findtext("xmax", "0"))
        y2 = float(box.findtext("ymax", "0"))
        output.append({
            "class": (item.findtext("name", "unknown") or "unknown").lower(),
            "quad": np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32),
            "long_side": max(x2 - x1, y2 - y1),
        })
    return output


def similarity(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    area_first = abs(float(cv2.contourArea(first)))
    area_second = abs(float(cv2.contourArea(second)))
    intersection, _ = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    union = area_first + area_second - float(intersection)
    return float(intersection) / max(1e-6, union), float(intersection) / max(1e-6, min(area_first, area_second))


def size_bin(value: float) -> str:
    if value < 60:
        return "<60"
    if value < 100:
        return "60-99"
    if value < 200:
        return "100-199"
    if value < 400:
        return "200-399"
    return ">=400"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("detections", type=Path)
    args = parser.parse_args()
    images = sorted(path for path in args.dataset.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    payload = json.loads(args.detections.read_text())
    if len(images) != len(payload["pages"]):
        raise SystemExit(f"image/page mismatch: {len(images)} != {len(payload['pages'])}")

    totals: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_size: dict[str, Counter[str]] = defaultdict(Counter)
    for image, page in zip(images, payload["pages"]):
        ground_truth = gt_objects(image)
        predictions = [*page.get("results", []), *page.get("unresolved_matrix", []), *page.get("review_candidates", [])]
        candidates = [np.asarray(item["quad"], np.float32) for item in predictions]
        # Permissive containment matching and strict IoU matching answer two
        # different questions and must not share the same greedy assignment.
        # A very tight barcode quad can have excellent containment but poor IoU
        # with a loose Pascal-VOC rectangle.  Reusing that permissive choice for
        # the strict metric incorrectly made improved tight boxes look like a
        # strict-IoU regression.
        used: set[int] = set()
        strict_used: set[int] = set()
        for truth in ground_truth:
            totals["ground_truth"] += 1
            by_class[truth["class"]]["ground_truth"] += 1
            by_size[size_bin(truth["long_side"])]["ground_truth"] += 1
            ranked = []
            for index, proposal in enumerate(candidates):
                if index in used:
                    continue
                iou, containment = similarity(truth["quad"], proposal)
                if iou >= 0.50 or containment >= 0.50:
                    ranked.append((max(iou, containment), index, iou))
            if ranked:
                _, index, _iou = max(ranked)
                used.add(index)
                totals["matched"] += 1
                by_class[truth["class"]]["matched"] += 1
                by_size[size_bin(truth["long_side"])]["matched"] += 1
            strict_ranked = []
            for index, proposal in enumerate(candidates):
                if index in strict_used:
                    continue
                iou, _containment = similarity(truth["quad"], proposal)
                if iou >= 0.50:
                    strict_ranked.append((iou, index))
            if strict_ranked:
                _iou, index = max(strict_ranked)
                strict_used.add(index)
                totals["strict_iou_matched"] += 1
        totals["predictions"] += len(predictions)
        totals["unassigned_predictions"] += len(predictions) - len(used)

    result = {
        "totals": dict(totals),
        "permissive_recall": round(totals["matched"] / max(1, totals["ground_truth"]), 4),
        "strict_iou_recall": round(totals["strict_iou_matched"] / max(1, totals["ground_truth"]), 4),
        "by_class": {key: dict(value) for key, value in sorted(by_class.items())},
        "by_long_side_px": {key: dict(by_size[key]) for key in ("<60", "60-99", "100-199", "200-399", ">=400")},
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
