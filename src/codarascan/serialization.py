# SPDX-License-Identifier: Apache-2.0
"""Strict deterministic JSON serialization for public result models."""

from __future__ import annotations

import json
from typing import Any, Protocol

from .errors import SerializationError


class SerializableResult(Protocol):
    def to_dict(self, *, include_diagnostics: bool = False) -> dict[str, Any]: ...


def to_json(
    result: SerializableResult,
    *,
    include_diagnostics: bool = False,
    indent: int | None = None,
) -> str:
    try:
        return json.dumps(
            result.to_dict(include_diagnostics=include_diagnostics),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"result cannot be serialized safely: {exc}") from exc


__all__ = ["to_json"]
