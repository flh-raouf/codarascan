"""Load the numbered predecessor folders under import-safe package aliases.

The repository folders deliberately carry numeric prefixes for presentation,
but Python package names cannot contain hyphens.  Registering namespace aliases
lets the optimized pipeline reuse the validated geometry and recovery bricks
without renaming or modifying the existing implementations.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _namespace(alias: str, directory: str) -> None:
    if alias in sys.modules:
        return
    path = ROOT / directory
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    module.__package__ = alias
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[alias] = module


def install_aliases() -> None:
    """Install aliases in dependency order."""
    _namespace("deterministic_barcode_locator", "1-deterministic_barcode_locator")
    _namespace("hybrid_barcode_pipeline", "2-hybrid_barcode_pipeline")
    _namespace("zxing_only_barcode_pipeline", "3-zxing_only_barcode_pipeline")
    _namespace("zxing_2d_barcode_pipeline", "4-zxing_2d_barcode_pipeline")

