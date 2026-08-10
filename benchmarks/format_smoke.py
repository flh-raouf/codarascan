#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit per-format clean-fixture decode and latency evidence as strict JSON."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

from codarascan import DecodedSymbolResult, Scanner
from tests.format_fixtures import clean_fixture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fast", "robust"), default="robust")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        help="also persist the strict JSON record at this path",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")

    records = []
    for group, infos in Scanner.supported_formats().items():
        for info in infos:
            image, expected = clean_fixture(info)
            scanner = Scanner(
                mode=arguments.mode,
                symbols="2d" if group == "2d" else "linear",
                formats=[info.name],
            )
            scanner.warm()
            durations = []
            decoded = False
            wrong_payload = False
            for _ in range(arguments.repeats):
                started = perf_counter()
                result = scanner.scan_image(image)
                durations.append((perf_counter() - started) * 1000.0)
                payloads = [
                    symbol.raw_bytes
                    for symbol in result.symbols
                    if isinstance(symbol, DecodedSymbolResult)
                ]
                decoded = decoded or expected in payloads
                wrong_payload = wrong_payload or any(payload != expected for payload in payloads)
            ordered = sorted(durations)
            p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
            records.append(
                {
                    "format": info.name,
                    "kind": group,
                    "decoded": decoded,
                    "wrong_payload": wrong_payload,
                    "median_ms": statistics.median(durations),
                    "p95_ms": ordered[p95_index],
                }
            )
    serialized = json.dumps(
        {
            "schema_version": 1,
            "mode": arguments.mode,
            "repeats": arguments.repeats,
            "formats": records,
        },
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return (
        0
        if all(record["decoded"] and not record["wrong_payload"] for record in records)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
