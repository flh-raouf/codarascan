#!/usr/bin/env python3
"""Validate exact payloads and artifact integrity for a pipeline run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def normalize_expected_format(value: str) -> str:
    normalized = value.replace("_", " ").strip().lower()
    if "data" in normalized and "matrix" in normalized:
        return "Data Matrix"
    if "qr" in normalized:
        return "QR Code"
    if "128" in normalized:
        return "Code 128"
    if "39" in normalized:
        return "Code 39"
    return " ".join(word.capitalize() for word in normalized.split())


def expected_from_csv(path: Path, default_format: str | None) -> dict[int, Counter]:
    output: dict[int, Counter] = defaultdict(Counter)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_format = row.get("format") or default_format
            if not source_format:
                raise ValueError("expected CSV has no format column; pass --default-format")
            output[int(row["page"])][(normalize_expected_format(source_format), row["value"])] += 1
    return dict(output)


def actual_from_json(payload: dict) -> dict[int, Counter]:
    output: dict[int, Counter] = {}
    for page in payload["pages"]:
        output[int(page["page"])] = Counter(
            (item["format"], item["text"])
            for item in page.get("results", [])
            if item.get("decoded") and item.get("text") is not None
        )
    return output


def check_artifacts(root: Path, payload: dict) -> list[str]:
    errors: list[str] = []
    referenced_crops: set[str] = set()
    for page in payload["pages"]:
        for key in ("source_image", "overlay"):
            relative = page[key]
            if not (root / relative).is_file():
                errors.append(f"missing artifact: {relative}")
        for result_key in ("results", "unresolved_matrix", "review_candidates"):
            for item in page.get(result_key, []):
                crop = item.get("crop")
                if crop:
                    referenced_crops.add(crop)
                    if not (root / crop).is_file():
                        errors.append(f"missing crop: {crop}")
    actual_crops = {path.relative_to(root).as_posix() for path in (root / "crops").rglob("*.png")} if (root / "crops").exists() else set()
    for crop in sorted(actual_crops - referenced_crops):
        errors.append(f"orphan crop: {crop}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--expected-csv", type=Path, required=True)
    parser.add_argument("--default-format", help="format used when the CSV has no format column, e.g. 'Code 128'")
    parser.add_argument("--require-no-unresolved", action="store_true")
    parser.add_argument("--require-no-review", action="store_true")
    arguments = parser.parse_args()

    payload = json.loads(arguments.detections.read_text(encoding="utf-8"))
    actual = actual_from_json(payload)
    expected = expected_from_csv(arguments.expected_csv, arguments.default_format)
    failed = False
    for page in sorted(set(actual) | set(expected)):
        missing = expected.get(page, Counter()) - actual.get(page, Counter())
        unexpected = actual.get(page, Counter()) - expected.get(page, Counter())
        if missing or unexpected:
            failed = True
            print(f"page {page:04d}: FAIL")
            for value, count in missing.items():
                print(f"  missing {count} x {value}")
            for value, count in unexpected.items():
                print(f"  unexpected {count} x {value}")

    artifact_errors = check_artifacts(arguments.detections.parent, payload)
    for error in artifact_errors:
        failed = True
        print(f"artifact FAIL: {error}")
    if arguments.require_no_unresolved and payload["summary"]["localized_unresolved_matrix"]:
        failed = True
        print(f"FAIL: {payload['summary']['localized_unresolved_matrix']} unresolved matrix results")
    if arguments.require_no_review and payload["summary"]["review_candidates"]:
        failed = True
        print(f"FAIL: {payload['summary']['review_candidates']} review candidates")
    if failed:
        return 1
    total = sum(sum(values.values()) for values in actual.values())
    print(
        f"PASS: {total} exact symbols across {len(expected)} pages; "
        f"unresolved={payload['summary']['localized_unresolved_matrix']} "
        f"review={payload['summary']['review_candidates']} artifacts=clean"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
