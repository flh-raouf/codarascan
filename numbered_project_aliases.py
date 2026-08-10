"""Expose archived project folders under their original Python package names.

The historical pipeline modules still use import-safe package names even though
their repository folders carry release prefixes. The CLI launchers call
:func:`install` before importing a pipeline, so the archived layout remains
usable without duplicate directories or symlinks.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path


PACKAGE_FOLDERS = {
    "deterministic_barcode_locator": "archive/versions/v1-deterministic-locator",
    "hybrid_barcode_pipeline": "archive/versions/v2-hybrid-pipeline",
    "zxing_only_barcode_pipeline": "archive/versions/v3-zxing-only",
    "zxing_2d_barcode_pipeline": "archive/versions/v4-zxing-2d",
}


def install(repository_root: Path) -> None:
    """Register lightweight package aliases for the numbered directories."""
    root = repository_root.resolve()
    for package_name, folder_name in PACKAGE_FOLDERS.items():
        if package_name in sys.modules:
            continue
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"missing project folder: {folder}")
        package = types.ModuleType(package_name)
        package.__file__ = str(folder / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(folder)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[package_name] = package
