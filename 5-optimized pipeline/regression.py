#!/usr/bin/env python3
"""Accuracy and artifact regression checks for optimized-pipeline runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


FORMAT_NAMES = {
    "code39": "Code 39",
    "code128": "Code 128",
    "datamatrix": "Data Matrix",
    "qrcode": "QR Code",
}


def normalize_format(value: str | None) -> str:
    compact = "".join(character for character in str(value or "") if character.isalnum()).lower()
    return FORMAT_NAMES.get(compact, str(value or "Unknown"))


@dataclass(frozen=True)
class Expected:
    page: int
    format: str
    text: str
    index: int
    quad: tuple[tuple[float, float], ...] | None = None


def _quad_from_row(row: dict[str, str]) -> tuple[tuple[float, float], ...] | None:
    keys = [(f"q{index}_x", f"q{index}_y") for index in range(4)]
    if not all(x in row and y in row and row[x] and row[y] for x, y in keys):
        return None
    return tuple((float(row[x]), float(row[y])) for x, y in keys)


def load_expected(path: Path, default_format: str | None) -> list[Expected]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        output: list[Expected] = []
        if "symbols" in payload:
            rows = payload["symbols"]
            for ordinal, row in enumerate(rows, start=1):
                output.append(
                    Expected(
                        page=int(row["page"]),
                        format=normalize_format(row.get("format") or default_format),
                        text=str(row.get("value", row.get("text", ""))),
                        index=int(row.get("index", ordinal)),
                        quad=tuple(tuple(map(float, point)) for point in row["page_quad"])
                        if row.get("page_quad")
                        else None,
                    )
                )
            return output
        for page in payload.get("pages", []):
            for ordinal, symbol in enumerate(page.get("symbols", []), start=1):
                output.append(
                    Expected(
                        page=int(page["page"]),
                        format=normalize_format(symbol.get("format") or default_format),
                        text=str(symbol.get("text", symbol.get("value", ""))),
                        index=ordinal,
                    )
                )
        return output

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for ordinal, row in enumerate(rows, start=1):
        output.append(
            Expected(
                page=int(row["page"]),
                format=normalize_format(row.get("format") or default_format),
                text=str(row.get("value", row.get("text", ""))),
                index=int(row.get("index") or ordinal),
                quad=_quad_from_row(row),
            )
        )
    return output


def decoded_counter(payload: dict[str, Any]) -> Counter[tuple[int, str, str]]:
    return Counter(
        (int(page["page"]), normalize_format(result.get("format")), str(result.get("text")))
        for page in payload["pages"]
        for result in page.get("results", [])
        if result.get("decoded", True) and result.get("text") is not None
    )


def all_outputs(payload: dict[str, Any]) -> Iterable[tuple[int, dict[str, Any]]]:
    for page in payload["pages"]:
        for key in ("results", "unresolved_matrix", "review_candidates"):
            for result in page.get(key, []):
                yield int(page["page"]), result


def overlap_over_smaller(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    smaller = min(abs(float(cv2.contourArea(a))), abs(float(cv2.contourArea(b))))
    return float(intersection) / max(1.0, smaller)


def localization_metrics(
    payload: dict[str, Any],
    expected: list[Expected],
) -> tuple[int, int, list[tuple[int, str]]]:
    expected_by_page: dict[int, list[Expected]] = defaultdict(list)
    for item in expected:
        if item.quad is not None:
            expected_by_page[item.page].append(item)
    if not expected_by_page:
        return 0, 0, []

    matched: set[tuple[int, int]] = set()
    false_outputs: list[tuple[int, str]] = []
    for page, result in all_outputs(payload):
        scores = [
            (overlap_over_smaller(result["quad"], item.quad), item)
            for item in expected_by_page.get(page, [])
        ]
        if not scores or max(scores, key=lambda pair: pair[0])[0] < 0.30:
            false_outputs.append((page, str(result.get("status", "unknown"))))
            continue
        item = max(scores, key=lambda pair: pair[0])[1]
        matched.add((item.page, item.index))
    total = sum(len(values) for values in expected_by_page.values())
    return len(matched), total, false_outputs


def artifact_errors(payload: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    referenced_crops: set[str] = set()
    for page in payload["pages"]:
        for key in ("source_image", "overlay"):
            value = page.get(key)
            if value and not (root / value).is_file():
                errors.append(f"missing {value}")
        for _page, result in ((page["page"], item) for key in ("results", "unresolved_matrix", "review_candidates") for item in page.get(key, [])):
            crop = result.get("crop")
            if crop:
                referenced_crops.add(crop)
                if not (root / crop).is_file():
                    errors.append(f"missing {crop}")
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "crops").rglob("*.png")
    } if (root / "crops").exists() else set()
    for path in sorted(actual - referenced_crops):
        errors.append(f"orphan {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--default-format")
    parser.add_argument("--baseline", type=Path, help="require every unique baseline decode to remain present")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--require-no-unresolved", action="store_true")
    parser.add_argument("--require-no-review", action="store_true")
    parser.add_argument("--minimum-localization-recall", type=float, default=0.0)
    args = parser.parse_args()

    payload = json.loads(args.detections.read_text(encoding="utf-8"))
    expected = load_expected(args.expected, args.default_format)
    expected_counts = Counter((item.page, item.format, item.text) for item in expected)
    actual_counts = decoded_counter(payload)
    missing = expected_counts - actual_counts
    unexpected = actual_counts - expected_counts
    decoded_matches = sum((actual_counts & expected_counts).values())
    localization_matches, localization_total, false_localizations = localization_metrics(payload, expected)
    unresolved = sum(len(page.get("unresolved_matrix", [])) for page in payload["pages"])
    reviews = sum(len(page.get("review_candidates", [])) for page in payload["pages"])
    artifacts = artifact_errors(payload, args.detections.resolve().parent)

    baseline_missing: Counter[tuple[int, str, str]] = Counter()
    new_unexpected = Counter(unexpected)
    if args.baseline:
        baseline = decoded_counter(json.loads(args.baseline.read_text(encoding="utf-8")))
        # Duplicated boxes are deliberately ignored here: the accuracy contract
        # is one retained instance of every unique page/format/payload.
        baseline_unique = Counter({key: 1 for key in baseline})
        actual_unique = Counter({key: 1 for key in actual_counts})
        baseline_missing = baseline_unique - actual_unique
        inherited_unexpected = Counter({key: 1 for key in baseline if key not in expected_counts})
        new_unexpected -= inherited_unexpected

    failed = bool(new_unexpected or false_localizations or artifacts or baseline_missing)
    if args.require_exact and missing:
        failed = True
    if args.require_no_unresolved and unresolved:
        failed = True
    if args.require_no_review and reviews:
        failed = True
    localization_recall = localization_matches / localization_total if localization_total else 0.0
    if localization_total and localization_recall < args.minimum_localization_recall:
        failed = True

    print(f"decoded exact: {decoded_matches}/{sum(expected_counts.values())}")
    print(f"decoded unexpected: {sum(unexpected.values())}")
    if localization_total:
        print(f"localized: {localization_matches}/{localization_total} ({localization_recall:.2%})")
        print(f"unmatched output locations: {len(false_localizations)}")
    print(f"unresolved={unresolved} review={reviews} artifact_errors={len(artifacts)}")
    if missing:
        print("missing:")
        for key, count in list(missing.items())[:12]:
            print(f"  {count} x {key}")
        if len(missing) > 12:
            print(f"  ... {len(missing) - 12} additional unique missing values")
    if unexpected:
        label = "unexpected (inherited from baseline):" if not new_unexpected and args.baseline else "unexpected:"
        print(label)
        for key, count in unexpected.items():
            print(f"  {count} x {key}")
    if new_unexpected:
        print("new unexpected versus baseline:")
        for key, count in new_unexpected.items():
            print(f"  {count} x {key}")
    if baseline_missing:
        print("lost versus baseline:")
        for key in baseline_missing:
            print(f"  {key}")
    for error in artifacts[:12]:
        print(f"artifact: {error}")
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
