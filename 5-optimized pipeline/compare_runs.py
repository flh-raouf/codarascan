#!/usr/bin/env python3
"""Compare throughput and unique decoded payloads between two run manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from regression import decoded_counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    optimized = json.loads(args.optimized.read_text(encoding="utf-8"))
    before = baseline["summary"]
    after = optimized["summary"]
    before_wall = float(before["wall_seconds"])
    after_wall = float(after["wall_seconds"])
    before_unique = set(decoded_counter(baseline))
    after_unique = set(decoded_counter(optimized))
    lost = before_unique - after_unique
    gained = after_unique - before_unique
    pages = int(after["pages"])

    print(f"wall: {before_wall:.4f}s -> {after_wall:.4f}s ({before_wall / after_wall:.2f}x)")
    print(f"effective optimized time: {1000 * after_wall / max(1, pages):.1f} ms/page")
    print(f"unique decoded: {len(before_unique)} -> {len(after_unique)}")
    print(f"lost unique payloads: {len(lost)}")
    print(f"gained unique payloads: {len(gained)}")
    for item in sorted(lost)[:12]:
        print(f"  lost {item}")
    for item in sorted(gained)[:12]:
        print(f"  gained {item}")
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
