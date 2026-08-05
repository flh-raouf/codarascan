#!/usr/bin/env python3
"""Compare pipeline output with a payload-level regression manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def entries_by_page(payload: dict, results_key: str) -> dict[int, Counter]:
    pages: dict[int, Counter] = {}
    for page in payload["pages"]:
        values = page.get(results_key, [])
        pages[int(page["page"])] = Counter(
            (item["format"], item["text"])
            for item in values
            if item.get("decoded", True) and item.get("text") is not None
        )
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", type=Path)
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path(__file__).resolve().parent / "dossier_expected.json",
    )
    args = parser.parse_args()
    actual = entries_by_page(json.loads(args.detections.read_text()), "results")
    expected = entries_by_page(json.loads(args.expected.read_text()), "symbols")
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
    if failed:
        return 1
    total = sum(sum(values.values()) for values in actual.values())
    print(f"PASS: {total} exact symbols across {len(expected)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

