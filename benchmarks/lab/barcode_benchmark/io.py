from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


DATASET_SCHEMA = "barcode-benchmark-dataset-v1"
PREDICTION_SCHEMA = "barcode-benchmark-predictions-v1"
REPORT_SCHEMA = "barcode-benchmark-report-v1"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write JSON so an interrupted benchmark cannot look complete."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def require_schema(value: dict[str, Any], expected: str, source: str) -> None:
    actual = value.get("schema_version")
    if actual != expected:
        raise ValueError(f"{source}: expected schema {expected!r}, got {actual!r}")


def image_index(value: dict[str, Any], source: str) -> dict[str, dict[str, Any]]:
    images = value.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{source}: 'images' must be a list")
    output: dict[str, dict[str, Any]] = {}
    for position, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"{source}: image {position} must be an object")
        identifier = image.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{source}: image {position} has no stable string id")
        if identifier in output:
            raise ValueError(f"{source}: duplicate image id {identifier!r}")
        output[identifier] = image
    return output
