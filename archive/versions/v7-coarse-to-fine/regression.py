#!/usr/bin/env python3
"""Geometrically compare two localization-only detections.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def family(kind: str) -> str:
    return "linear" if kind == "linear" else "2d"


def overlap_over_smaller(first: list[list[float]], second: list[list[float]]) -> float:
    first_hull = cv2.convexHull(np.asarray(first, dtype=np.float32)).reshape(-1, 2)
    second_hull = cv2.convexHull(np.asarray(second, dtype=np.float32)).reshape(-1, 2)
    first_area = abs(cv2.contourArea(first_hull))
    second_area = abs(cv2.contourArea(second_hull))
    if min(first_area, second_area) <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    return float(intersection) / min(float(first_area), float(second_area))


def pages_by_number(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    return {int(page["page"]): list(page["detections"]) for page in payload["pages"]}


def compare(actual: dict[str, Any], reference: dict[str, Any], threshold: float) -> tuple[int, int, int]:
    actual_pages = pages_by_number(actual)
    reference_pages = pages_by_number(reference)
    matched_total = missing_total = extra_total = 0
    for page_number in sorted(set(actual_pages) | set(reference_pages)):
        expected = reference_pages.get(page_number, [])
        observed = actual_pages.get(page_number, [])
        edges = sorted(
            (
                overlap_over_smaller(left["quad"], right["quad"]),
                left_index,
                right_index,
            )
            for left_index, left in enumerate(expected)
            for right_index, right in enumerate(observed)
            if family(left["kind"]) == family(right["kind"])
        )
        used_expected: set[int] = set()
        used_observed: set[int] = set()
        for score, left_index, right_index in reversed(edges):
            if score < threshold:
                break
            if left_index in used_expected or right_index in used_observed:
                continue
            used_expected.add(left_index)
            used_observed.add(right_index)
        missing = [expected[index] for index in range(len(expected)) if index not in used_expected]
        extra = [observed[index] for index in range(len(observed)) if index not in used_observed]
        matched_total += len(used_expected)
        missing_total += len(missing)
        extra_total += len(extra)
        if missing or extra:
            print(
                f"page {page_number:04d}: matched={len(used_expected)} "
                f"missing={len(missing)} extra={len(extra)}"
            )
            for detection in missing:
                print(f"  missing {detection['kind']} at {detection.get('aabb')}")
            for detection in extra:
                print(f"  extra   {detection['kind']} at {detection.get('aabb')}")
    return matched_total, missing_total, extra_total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--minimum-overlap", type=float, default=0.20)
    parser.add_argument(
        "--allow-extra",
        type=int,
        default=0,
        help="permit this many new candidates while still failing any regression",
    )
    arguments = parser.parse_args()
    actual = json.loads(arguments.actual.read_text(encoding="utf-8"))
    reference = json.loads(arguments.reference.read_text(encoding="utf-8"))
    matched, missing, extra = compare(actual, reference, arguments.minimum_overlap)
    print(f"matched={matched} missing={missing} extra={extra}")
    if actual.get("decoding_performed") is not False:
        print("FAIL: actual output does not explicitly guarantee decoding_performed=false")
        return 1
    if missing or extra > arguments.allow_extra:
        return 1
    print("PASS: no localization regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
