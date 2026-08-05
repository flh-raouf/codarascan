#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import (
    PROFILES,
    Config,
    parse_page_selection,
    run_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast CPU barcode localization without decoding",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="fast",
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--pages", type=parse_page_selection)
    parser.add_argument("--overlays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    run_pipeline(
        arguments.input,
        Config(
            output=arguments.output,
            profile=arguments.profile,
            weights=arguments.weights,
            workers=arguments.workers,
            render_dpi=arguments.render_dpi,
            pages=arguments.pages,
            save_overlays=arguments.overlays,
            overwrite=arguments.overwrite,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
