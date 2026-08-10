# SPDX-License-Identifier: Apache-2.0
"""Mosaic, the guarded robust engine line."""

from __future__ import annotations

from typing import Any


def load_detection() -> Any:
    """Load Mosaic's local detector."""

    from .detection import ENGINE

    return ENGINE


def load_extraction() -> Any:
    """Load Mosaic's local extractor."""

    from .extraction import ENGINE

    return ENGINE


def __getattr__(name: str) -> Any:
    if name == "DETECTION_ENGINE":
        return load_detection()
    if name == "EXTRACTION_ENGINE":
        return load_extraction()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DETECTION_ENGINE", "EXTRACTION_ENGINE", "load_detection", "load_extraction"]
