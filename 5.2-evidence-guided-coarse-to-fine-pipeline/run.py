#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import Settings, run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evidence-guided coarse-to-fine barcode pipeline",
    )
    parser.add_argument("input", type=Path, help="image, image directory, or PDF")
    parser.add_argument("--output", type=Path, default=Path("5.2-output"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument(
        "--mode",
        default="accuracy",
        choices=("accuracy", "balanced"),
        help="accuracy adds the proven classical document rescue routes",
    )
    parser.add_argument(
        "--formats",
        default="all",
        help="comma-separated ZXing names, e.g. EAN13,UPCA; default all",
    )
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--decoded-only", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    formats = frozenset()
    if arguments.formats.lower() != "all":
        formats = frozenset(
            "".join(character for character in value.lower() if character.isalnum())
            for value in arguments.formats.split(",")
            if value.strip()
        )
    document = run(
        arguments.input.expanduser().resolve(),
        arguments.output.expanduser().resolve(),
        Settings(
            device=arguments.device,
            mode=arguments.mode,
            formats=formats,
            include_unresolved=not arguments.decoded_only,
        ),
        dpi=arguments.render_dpi,
        overlays=not arguments.no_overlays,
        overwrite=arguments.overwrite,
    )
    summary = document["summary"]
    print(
        f"{summary['pages']} page(s), {summary['decoded']} decoded, "
        f"{summary['localized']} localized, {summary['latency_ms']:.1f} ms",
    )
    print(arguments.output.expanduser().resolve() / "detections.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
