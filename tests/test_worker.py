# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any, BinaryIO

import jsonschema
import numpy as np
import pytest
import zxingcpp
from PIL import Image

from codarascan.worker import (
    PROTOCOL_VERSION,
    ProtocolError,
    WorkerServer,
    read_frame,
    write_frame,
)


def _schema() -> dict[str, object]:
    return json.loads(files("codarascan").joinpath("schemas/worker.schema.json").read_text())


def _request(
    request_id: Any, operation: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "id": request_id,
        "operation": operation,
        "params": params or {},
    }


def _qr_bytes(payload: bytes = b"WORKER\nPAYLOAD") -> bytes:
    barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
    image = np.asarray(
        zxingcpp.write_barcode_to_image(barcode, scale=8, add_quiet_zones=True)
    ).copy()
    output = io.BytesIO()
    Image.fromarray(image).save(output, "PNG")
    return output.getvalue()


def _pdf_bytes(page_count: int) -> bytes:
    with Image.open(io.BytesIO(_qr_bytes(b"WORKER-CANCEL"))) as source:
        pages = [source.convert("RGB") for _ in range(page_count)]
    output = io.BytesIO()
    pages[0].save(
        output,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150,
    )
    for page in pages:
        page.close()
    return output.getvalue()


def _write(stream: BinaryIO, request: dict[str, Any]) -> None:
    write_frame(stream, request)


def _read(stream: BinaryIO) -> dict[str, Any]:
    response = read_frame(stream)
    assert response is not None
    jsonschema.validate(response, _schema())
    return response


def _worker(*args: str) -> subprocess.Popen[bytes]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.Popen(
        [sys.executable, "-m", "codarascan.worker", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )


def test_frame_round_trip_supports_large_embedded_newlines() -> None:
    value = _request("large", "capabilities", {"note": ("line\n" * 50_000)})
    stream = io.BytesIO()
    write_frame(stream, value)
    stream.seek(0)
    assert read_frame(stream) == value


def test_frame_round_trip_tolerates_split_reads_and_writes() -> None:
    class SplitWriter(io.BytesIO):
        def write(self, value: bytes | bytearray | memoryview) -> int:
            return super().write(bytes(value[:3]))

    class SplitReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 2) if size >= 0 else 2)

    value = _request("split", "capabilities", {"payload": "line\n" * 100})
    writer = SplitWriter()
    write_frame(writer, value)
    assert read_frame(SplitReader(writer.getvalue())) == value


@pytest.mark.parametrize("payload", [b"\x00", b"\x00\x00\x00\x05{}"])
def test_truncated_frames_are_protocol_errors(payload: bytes) -> None:
    with pytest.raises(ProtocolError):
        read_frame(io.BytesIO(payload))


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_nonfinite_json_numbers_are_protocol_errors(constant: bytes) -> None:
    payload = b'{"value":' + constant + b"}"
    frame = len(payload).to_bytes(4, "big") + payload
    with pytest.raises(ProtocolError):
        read_frame(io.BytesIO(frame))


@pytest.mark.integration
@pytest.mark.parametrize("args", [(), ("--workers", "4")])
def test_persistent_worker_capabilities_scanner_scan_release_and_shutdown(
    args: tuple[str, ...],
) -> None:
    process = _worker(*args)
    assert process.stdin and process.stdout and process.stderr
    _write(process.stdin, _request(1, "capabilities"))
    capabilities = _read(process.stdout)
    assert capabilities["ok"] and capabilities["result"]["protocol_version"] == 1

    _write(
        process.stdin,
        _request(
            2,
            "create_scanner",
            {
                "scanner_id": "main",
                "scanner": {"mode": "robust", "symbols": "2d", "formats": ["qr-code"]},
            },
        ),
    )
    assert _read(process.stdout)["result"] == {"reused": False, "scanner_id": "main"}

    encoded = base64.b64encode(_qr_bytes()).decode("ascii")
    _write(
        process.stdin,
        _request(
            3,
            "scan_image",
            {"scanner_id": "main", "input": {"base64": encoded}},
        ),
    )
    scanned = _read(process.stdout)
    symbol = scanned["result"]["symbols"][0]
    assert scanned["id"] == 3
    assert base64.b64decode(symbol["raw_bytes"]["data"]) == b"WORKER\nPAYLOAD"

    _write(process.stdin, _request(4, "release_scanner", {"scanner_id": "main"}))
    assert _read(process.stdout)["result"]["released"]
    _write(process.stdin, _request(5, "shutdown"))
    assert _read(process.stdout)["result"]["shutdown"]
    assert process.wait(timeout=10) == 0
    assert process.stderr.read() == b""


@pytest.mark.integration
def test_worker_matches_concurrent_request_ids_and_returns_structured_errors() -> None:
    process = _worker()
    assert process.stdin and process.stdout
    _write(
        process.stdin,
        _request(
            "create",
            "create_scanner",
            {
                "scanner_id": "concurrent",
                "scanner": {"mode": "robust", "symbols": "2d", "formats": ["qr-code"]},
            },
        ),
    )
    _read(process.stdout)
    encoded = base64.b64encode(_qr_bytes(b"CONCURRENT")).decode("ascii")
    for request_id in ("first", "second"):
        _write(
            process.stdin,
            _request(
                request_id,
                "scan_image",
                {"scanner_id": "concurrent", "input": {"base64": encoded}},
            ),
        )
    responses = {_read(process.stdout)["id"], _read(process.stdout)["id"]}
    assert responses == {"first", "second"}

    _write(
        process.stdin,
        {"protocol": 999, "id": "bad-version", "operation": "capabilities"},
    )
    mismatch = _read(process.stdout)
    assert not mismatch["ok"] and mismatch["error"]["code"] == "protocol_error"
    _write(process.stdin, _request("shutdown", "shutdown"))
    _read(process.stdout)
    assert process.wait(timeout=10) == 0


@pytest.mark.integration
def test_unknown_operation_is_reported_without_requiring_a_scanner() -> None:
    process = _worker()
    assert process.stdin and process.stdout
    _write(process.stdin, _request("unknown", "does_not_exist"))
    response = _read(process.stdout)
    assert not response["ok"]
    assert response["error"] == {
        "code": "protocol_error",
        "context": {"operation": "does_not_exist"},
        "message": "unknown worker operation",
    }
    _write(process.stdin, _request("shutdown", "shutdown"))
    _read(process.stdout)
    assert process.wait(timeout=10) == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    "worker_request",
    [
        {
            "protocol": PROTOCOL_VERSION,
            "id": "extra",
            "operation": "capabilities",
            "unexpected": True,
        },
        {
            "protocol": PROTOCOL_VERSION,
            "id": True,
            "operation": "capabilities",
        },
    ],
)
def test_worker_rejects_requests_outside_the_published_envelope(
    worker_request: dict[str, Any],
) -> None:
    process = _worker()
    assert process.stdin and process.stdout
    _write(process.stdin, worker_request)
    response = _read(process.stdout)
    assert not response["ok"]
    assert response["error"]["code"] == "protocol_error"
    _write(process.stdin, _request("shutdown", "shutdown"))
    _read(process.stdout)
    assert process.wait(timeout=10) == 0


@pytest.mark.integration
def test_worker_cancels_an_active_document_stream_and_remains_reusable() -> None:
    process = _worker()
    assert process.stdin and process.stdout and process.stderr
    _write(
        process.stdin,
        _request(
            "create",
            "create_scanner",
            {
                "scanner_id": "cancel",
                "scanner": {"mode": "robust", "symbols": "2d", "formats": ["qr-code"]},
            },
        ),
    )
    _read(process.stdout)
    page_count = 30
    _write(
        process.stdin,
        _request(
            "scan",
            "iter_document",
            {
                "scanner_id": "cancel",
                "input": {"base64": base64.b64encode(_pdf_bytes(page_count)).decode("ascii")},
                "workers": 1,
            },
        ),
    )
    first = _read(process.stdout)
    assert first["id"] == "scan" and first["event"] == "page"
    _write(process.stdin, _request("cancel-request", "cancel", {"id": "scan"}))

    pages_seen = 1
    cancel_response: dict[str, Any] | None = None
    scan_response: dict[str, Any] | None = None
    while cancel_response is None or scan_response is None:
        response = _read(process.stdout)
        if response["id"] == "scan" and response["event"] == "page":
            pages_seen += 1
        elif response["id"] == "scan":
            scan_response = response
        elif response["id"] == "cancel-request":
            cancel_response = response

    assert cancel_response["ok"]
    assert cancel_response["result"]["cancel_requested"] is True
    assert not scan_response["ok"]
    assert scan_response["error"]["code"] == "protocol_error"
    assert "cancelled" in scan_response["error"]["message"]
    assert pages_seen < page_count

    _write(process.stdin, _request("capabilities-after-cancel", "capabilities"))
    assert _read(process.stdout)["ok"]
    _write(process.stdin, _request("shutdown", "shutdown"))
    _read(process.stdout)
    assert process.wait(timeout=10) == 0
    assert process.stderr.read() == b""


def test_worker_module_contains_no_network_transport() -> None:
    source = files("codarascan").joinpath("worker.py").read_text()
    assert "import socket" not in source
    assert "HTTPServer" not in source


@pytest.mark.parametrize("workers", [0, -1, True, False, 1.5, "4", "auto", None])
def test_server_rejects_invalid_worker_count(workers: object) -> None:
    from codarascan.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="workers must be a positive integer"):
        WorkerServer(io.BytesIO(), io.BytesIO(), workers=workers)  # type: ignore[arg-type]


@pytest.mark.parametrize("workers", [None, 4])
def test_server_limits_concurrent_requests(workers: int | None) -> None:
    import threading

    kwargs = {} if workers is None else {"workers": workers}
    server = WorkerServer(io.BytesIO(), io.BytesIO(), **kwargs)
    count = workers or 1
    started = threading.Barrier(count + 1)
    release = threading.Event()
    extra_started = threading.Event()

    def block() -> None:
        started.wait(timeout=5)
        assert release.wait(timeout=5)

    try:
        futures = [server._executor.submit(block) for _ in range(count)]
        started.wait(timeout=5)
        extra = server._executor.submit(extra_started.set)
        assert not extra_started.wait(timeout=0.1)
        release.set()
        for future in futures:
            future.result(timeout=5)
        extra.result(timeout=5)
        assert extra_started.is_set()
    finally:
        release.set()
        server._executor.shutdown(wait=True)


@pytest.mark.parametrize("command", ["codarascan.worker", "codarascan.cli"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "auto"])
def test_worker_cli_rejects_invalid_count(command: str, value: str) -> None:
    args = [sys.executable, "-m", command]
    if command == "codarascan.cli":
        args.append("_worker")
    result = subprocess.run(
        [*args, "--workers", value], input=b"", capture_output=True, timeout=10,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert b"workers must be a positive integer" in result.stderr
