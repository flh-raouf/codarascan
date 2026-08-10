# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest
from PIL import Image

from codarascan import Scanner, to_json

from .format_fixtures import clean_fixture


def _schema() -> dict[str, object]:
    resource = files("codarascan").joinpath("schemas/result.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def test_published_result_schema_is_valid_and_accepts_compact_result_shapes() -> None:
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    info = next(item for item in Scanner.supported_formats()["2d"] if item.name == "qr-code")
    image, _ = clean_fixture(info)
    for decode in (True, False):
        result = Scanner(symbols="2d", formats=["qr-code"], decode=decode).scan_image(image)
        jsonschema.validate(json.loads(to_json(result)), schema)


def test_published_schema_accepts_document_results(tmp_path: Path) -> None:
    schema = _schema()
    info = next(item for item in Scanner.supported_formats()["2d"] if item.name == "qr-code")
    image, _ = clean_fixture(info)
    path = tmp_path / "schema.pdf"
    Image.fromarray(image).convert("RGB").save(path, "PDF", resolution=150)
    result = Scanner(symbols="2d", formats=["qr-code"]).scan_document(path)
    jsonschema.validate(json.loads(to_json(result)), schema)


def test_published_golden_fixtures_validate_against_their_shared_schemas() -> None:
    package = files("codarascan").joinpath("schemas")
    result_schema = json.loads(package.joinpath("result.schema.json").read_text())
    stream_schema = json.loads(package.joinpath("stream.schema.json").read_text())
    for name in ("result.decoded.example.json", "result.localized.example.json"):
        jsonschema.validate(json.loads(package.joinpath(name).read_text()), result_schema)
    jsonschema.validate(
        json.loads(package.joinpath("stream.example.json").read_text()),
        stream_schema,
    )


def test_schema_rejects_out_of_range_geometry_and_status_kind_mismatch() -> None:
    package = files("codarascan").joinpath("schemas")
    schema = json.loads(package.joinpath("result.schema.json").read_text())
    localized = json.loads(package.joinpath("result.localized.example.json").read_text())
    out_of_range = deepcopy(localized)
    out_of_range["symbols"][0]["normalized_quad"][0][0] = 1.1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(out_of_range, schema)
    wrong_kind = deepcopy(localized)
    wrong_kind["symbols"][0]["kind"] = "linear"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(wrong_kind, schema)
