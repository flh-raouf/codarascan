#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numbered_project_aliases import install

install(ROOT)

from zxing_only_barcode_pipeline.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
