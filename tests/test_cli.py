# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import zxingcpp
from PIL import Image

from codarascan import documents
from codarascan.cli import (
    EXIT_INVALID,
    EXIT_NO_SYMBOLS,
    EXIT_PARTIAL,
    EXIT_PROCESSING,
    EXIT_SUCCESS,
    build_parser,
    main,
)


def _qr_path(path: Path, payload: str = "CLI-PAYLOAD") -> Path:
    barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
    image = np.asarray(
        zxingcpp.write_barcode_to_image(barcode, scale=8, add_quiet_zones=True)
    ).copy()
    Image.fromarray(image).save(path)
    return path


def _pdf_path(path: Path, payloads: tuple[str, ...] = ("PAGE-1", "PAGE-2")) -> Path:
    images = [Image.open(_qr_path(path.parent / f"{payload}.png", payload)).convert("RGB") for payload in payloads]
    images[0].save(path, "PDF", save_all=True, append_images=images[1:], resolution=150)
    for image in images:
        image.close()
    return path


def _schema(name: str) -> dict[str, object]:
    return json.loads(files("codarascan").joinpath(f"schemas/{name}").read_text())


def test_help_and_version_are_public_cli_surfaces(
    capsys: pytest.CaptureFixture[str],
) -> None:
    help_text = build_parser().format_help()
    assert "{image,document}" in help_text
    assert "--version" in help_text
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == "codarascan 0.1.0 (Apache-2.0)\n"


def test_image_json_uses_shared_schema_and_stdout_is_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _qr_path(tmp_path / "qr.png")
    code = main(["image", str(path), "--mode", "robust", "--symbols", "2d", "--json"])
    captured = capsys.readouterr()
    assert code == EXIT_SUCCESS
    payload = json.loads(captured.out)
    jsonschema.validate(payload, _schema("result.schema.json"))
    assert captured.err == ""


def test_image_ndjson_is_one_shared_result_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _qr_path(tmp_path / "qr.png")
    code = main(["image", str(path), "--mode", "robust", "--ndjson"])
    lines = capsys.readouterr().out.splitlines()
    assert code == EXIT_SUCCESS and len(lines) == 1
    jsonschema.validate(json.loads(lines[0]), _schema("result.schema.json"))


def test_no_symbol_and_invalid_input_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blank = tmp_path / "blank.png"
    Image.new("L", (120, 80), 255).save(blank)
    assert main(["image", str(blank), "--mode", "robust", "--json"]) == EXIT_NO_SYMBOLS
    capsys.readouterr()
    assert main(["image", str(tmp_path / "missing.png"), "--json"]) == EXIT_INVALID
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does not exist" in captured.err


def test_processing_failure_has_its_documented_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codarascan import InternalProcessingError, cli

    path = _qr_path(tmp_path / "failure.png")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise InternalProcessingError("injected processing failure")

    monkeypatch.setattr(cli.Scanner, "scan_image", fail)
    assert main(["image", str(path), "--json"]) == EXIT_PROCESSING
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "injected processing failure" in captured.err


def test_document_json_and_ndjson_validate_shared_schemas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _pdf_path(tmp_path / "pages.pdf")
    common = ["document", str(path), "--mode", "robust", "--symbols", "2d"]
    assert main([*common, "--pages", "2,1", "--workers", "2", "--json"]) == EXIT_SUCCESS
    complete = json.loads(capsys.readouterr().out)
    jsonschema.validate(complete, _schema("result.schema.json"))
    assert [page["page"] for page in complete["pages"]] == [2, 1]

    assert main([*common, "--ndjson"]) == EXIT_SUCCESS
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["type"] for line in lines] == ["page", "page", "document_summary"]
    for page in lines[:-1]:
        jsonschema.validate(page, _schema("result.schema.json"))
    jsonschema.validate(lines[-1], _schema("stream.schema.json"))


def test_partial_document_ndjson_returns_partial_exit_and_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _pdf_path(tmp_path / "partial.pdf")
    original = documents._render_page

    def fail_second(document: object, page: int) -> documents.RenderedPage:
        if page == 2:
            from codarascan import DocumentRenderError

            raise DocumentRenderError("injected", context={"page": 2})
        return original(document, page)  # type: ignore[arg-type]

    monkeypatch.setattr(documents, "_render_page", fail_second)
    code = main(
        [
            "document",
            str(path),
            "--mode",
            "robust",
            "--symbols",
            "2d",
            "--on-error",
            "collect",
            "--ndjson",
        ]
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == EXIT_PARTIAL
    assert lines[-1]["type"] == "document_summary"
    assert not lines[-1]["complete"] and lines[-1]["errors"][0]["page"] == 2


def test_decode_false_cli_never_exposes_payload_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _qr_path(tmp_path / "localized.png")
    assert (
        main(
            [
                "image",
                str(path),
                "--mode",
                "robust",
                "--symbols",
                "2d",
                "--no-decode",
                "--json",
            ]
        )
        == EXIT_SUCCESS
    )
    symbol = json.loads(capsys.readouterr().out)["symbols"][0]
    assert not {"text", "value", "raw_bytes", "format"} & symbol.keys()


def test_debug_output_is_unique_and_sensitivity_marked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _qr_path(tmp_path / "debug.png")
    root = tmp_path / "debug"
    arguments = [
        "image",
        str(path),
        "--mode",
        "robust",
        "--json",
        "--debug-output",
        str(root),
    ]
    assert main(arguments) == EXIT_SUCCESS
    capsys.readouterr()
    assert main(arguments) == EXIT_SUCCESS
    capsys.readouterr()
    runs = list(root.iterdir())
    assert len(runs) == 2 and runs[0] != runs[1]
    assert all((run / "SENSITIVE_DATA_WARNING.txt").is_file() for run in runs)
    assert all((run / "result.json").is_file() for run in runs)
