from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("pipeline_5_2", ROOT / "pipeline.py")
assert spec is not None and spec.loader is not None
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)


def test_format_normalization_and_allowlist() -> None:
    allowed = frozenset({"ean13", "upca"})
    assert pipeline._accepted_format("EAN-13", allowed)
    assert pipeline._accepted_format("UPC-A", allowed)
    assert not pipeline._accepted_format("Code 128", allowed)
    assert pipeline._normalized_format("QR Code") == "qrcode"


def test_gtin_payload_key_equates_zero_prefixed_ean13_and_upca() -> None:
    ean = {"symbology": "EAN-13", "payload": "0012345678905"}
    upc = {"symbology": "UPC-A", "payload": "012345678905"}
    assert pipeline._payload_key(ean) == pipeline._payload_key(upc)


def test_crop_is_clamped_to_image() -> None:
    image = np.zeros((20, 30, 3), np.uint8)
    crop = pipeline._crop(
        image,
        [[-5, -2], [25, -2], [25, 15], [-5, 15]],
        0.25,
    )
    assert 0 < crop.shape[0] <= 20
    assert 0 < crop.shape[1] <= 30


def test_deduplicate_prefers_decoded_result() -> None:
    polygon = [[1, 1], [10, 1], [10, 10], [1, 10]]
    unresolved = {
        "polygon": polygon,
        "payload": None,
        "confidence": 0.99,
        "sources": ["locator"],
    }
    decoded = {
        "polygon": polygon,
        "payload": "123",
        "confidence": 1.0,
        "sources": ["decoder"],
    }
    result = pipeline._deduplicate([unresolved, decoded])
    assert len(result) == 1
    assert result[0]["payload"] == "123"


def test_deduplicate_merges_nearby_views_of_same_payload() -> None:
    first = {
        "polygon": [[0, 0], [100, 0], [100, 10], [0, 10]],
        "symbology": "Code 39",
        "payload": "ABC",
        "confidence": 1.0,
        "sources": ["full"],
    }
    second = {
        "polygon": [[0, 20], [100, 20], [100, 40], [0, 40]],
        "symbology": "Code 39",
        "payload": "ABC",
        "confidence": 0.8,
        "sources": ["crop"],
    }
    assert len(pipeline._deduplicate([first, second])) == 1
