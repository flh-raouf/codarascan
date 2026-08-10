# SPDX-License-Identifier: Apache-2.0
"""Lazy, isolated loading for the packaged Tessera native extension."""

from __future__ import annotations

import importlib
import os
import threading
import warnings
from types import ModuleType

FALLBACK_WARNING = (
    "CodaraScan could not load the native Tessera extension. "
    "Falling back to the slower Python implementation."
)


class NativeFallbackWarning(RuntimeWarning):
    """The optional-at-runtime Tessera accelerator could not be loaded."""


_LOCK = threading.Lock()
_ATTEMPTED = False
_MODULE: ModuleType | None = None
_ERROR: BaseException | None = None
_WARNED = False


def load_native() -> ModuleType | None:
    """Return the native module or a warned Python-reference fallback."""

    global _ATTEMPTED, _MODULE, _ERROR, _WARNED
    if _ATTEMPTED:
        return _MODULE
    with _LOCK:
        if not _ATTEMPTED:
            try:
                if os.environ.get("CODARASCAN_FORCE_PYTHON", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    raise ImportError("forced Python-reference backend")
                _MODULE = importlib.import_module("codarascan._sttg_native")
            except Exception as exc:  # loader/ABI failures must not break scanning
                _ERROR = exc
                _MODULE = None
            _ATTEMPTED = True
            if _MODULE is None and not _WARNED:
                warnings.warn(FALLBACK_WARNING, NativeFallbackWarning, stacklevel=2)
                _WARNED = True
    return _MODULE


def backend_name() -> str:
    return "native" if load_native() is not None else "python-reference"


def load_error() -> BaseException | None:
    load_native()
    return _ERROR


def _reset_for_tests() -> None:
    global _ATTEMPTED, _MODULE, _ERROR, _WARNED
    with _LOCK:
        _ATTEMPTED = False
        _MODULE = None
        _ERROR = None
        _WARNED = False


__all__ = [
    "FALLBACK_WARNING",
    "NativeFallbackWarning",
    "backend_name",
    "load_error",
    "load_native",
]
