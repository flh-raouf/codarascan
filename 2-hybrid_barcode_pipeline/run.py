#!/usr/bin/env python3
"""Command-line entry point for the hybrid barcode pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from numbered_project_aliases import install

install(REPOSITORY_ROOT)

from hybrid_barcode_pipeline.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
