# SPDX-License-Identifier: Apache-2.0
"""Private versioned framed-JSON worker used by backend language adapters."""

from __future__ import annotations

import base64
import binascii
import json
import os
import struct
import sys
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, BinaryIO, cast

from . import __version__
from .errors import CodaraScanError, ProtocolError
from .scanner import Scanner

PROTOCOL_VERSION = 1
_HEADER = struct.Struct(">I")


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _read_header(stream: BinaryIO) -> bytes | None:
    header = bytearray()
    while len(header) < _HEADER.size:
        chunk = stream.read(_HEADER.size - len(header))
        if not chunk:
            if not header:
                return None
            raise ProtocolError("truncated worker frame header")
        header.extend(chunk)
    return bytes(header)


def _write_all(stream: BinaryIO, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise ProtocolError("worker stream stopped during frame write")
        remaining = remaining[written:]


def read_frame(stream: BinaryIO) -> dict[str, Any] | None:
    header = _read_header(stream)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    payload = bytearray()
    while len(payload) < length:
        chunk = stream.read(length - len(payload))
        if not chunk:
            raise ProtocolError("truncated worker frame payload")
        payload.extend(chunk)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError("worker frame is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("worker frame root must be a JSON object")
    return value


def write_frame(stream: BinaryIO, value: dict[str, Any]) -> None:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("worker response is not safely serializable") from exc
    _write_all(stream, _HEADER.pack(len(payload)))
    _write_all(stream, payload)
    stream.flush()


def _request_input(value: Any) -> str | bytes:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ProtocolError("input must be a path string or path/base64 object")
    if set(value) == {"path"} and isinstance(value["path"], str):
        return value["path"]
    if set(value) == {"base64"} and isinstance(value["base64"], str):
        try:
            return base64.b64decode(value["base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProtocolError("input.base64 is not valid Base64") from exc
    raise ProtocolError("input object must contain exactly one string path or base64 field")


def _error_payload(error: BaseException) -> dict[str, Any]:
    if isinstance(error, CodaraScanError):
        return {
            "code": error.code,
            "message": error.message,
            "context": dict(error.context),
        }
    return {
        "code": "internal_processing_error",
        "message": "unexpected worker processing failure",
        "context": {},
    }


class WorkerServer:
    def __init__(self, reader: BinaryIO, writer: BinaryIO) -> None:
        self.reader = reader
        self.writer = writer
        self._write_lock = threading.Lock()
        self._scanner_lock = threading.RLock()
        self._scanners: dict[str, Scanner] = {}
        self._active_lock = threading.RLock()
        self._active: dict[Any, threading.Event] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, os.cpu_count() or 1),
            thread_name_prefix="codarascan-worker",
        )

    def _write(self, value: dict[str, Any]) -> None:
        with self._write_lock:
            write_frame(self.writer, value)

    def _envelope(
        self,
        request_id: Any,
        *,
        result: Any = None,
        error: BaseException | None = None,
        event: str | None = None,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {
            "protocol": PROTOCOL_VERSION,
            "id": request_id,
            "ok": error is None,
        }
        if event is not None:
            output["event"] = event
        if error is None:
            output["result"] = result
        else:
            output["error"] = _error_payload(error)
        return output

    def _scanner(self, scanner_id: Any) -> Scanner:
        if not isinstance(scanner_id, str) or not scanner_id:
            raise ProtocolError("scanner_id must be a non-empty string")
        with self._scanner_lock:
            scanner = self._scanners.get(scanner_id)
        if scanner is None:
            raise ProtocolError("unknown scanner_id", context={"scanner_id": scanner_id})
        return scanner

    def _handle(
        self,
        request_id: Any,
        operation: str,
        params: dict[str, Any],
        cancelled: threading.Event,
    ) -> Any:
        if operation == "capabilities":
            return {
                "package_version": __version__,
                "protocol_version": PROTOCOL_VERSION,
                "formats": {
                    key: [item.to_dict() for item in values]
                    for key, values in Scanner.supported_formats().items()
                },
                "operations": [
                    "capabilities",
                    "create_scanner",
                    "warm",
                    "scan_image",
                    "scan_document",
                    "iter_document",
                    "release_scanner",
                    "cancel",
                    "shutdown",
                ],
            }
        if operation == "create_scanner":
            scanner_config = params.get("scanner", {})
            if not isinstance(scanner_config, dict):
                raise ProtocolError("scanner configuration must be an object")
            scanner = Scanner(**scanner_config)
            scanner_id = params.get("scanner_id") or uuid.uuid4().hex
            if not isinstance(scanner_id, str) or not scanner_id:
                raise ProtocolError("scanner_id must be a non-empty string")
            with self._scanner_lock:
                existing = self._scanners.get(scanner_id)
                if existing is not None and existing != scanner:
                    raise ProtocolError(
                        "scanner_id already has a different configuration",
                        context={"scanner_id": scanner_id},
                    )
                self._scanners[scanner_id] = scanner
            return {"scanner_id": scanner_id, "reused": existing is not None}
        if operation == "release_scanner":
            scanner_id = params.get("scanner_id")
            if not isinstance(scanner_id, str) or not scanner_id:
                raise ProtocolError("scanner_id must be a non-empty string")
            with self._scanner_lock:
                released = self._scanners.pop(scanner_id, None) is not None
            return {"scanner_id": scanner_id, "released": released}

        if operation not in {"warm", "scan_image", "scan_document", "iter_document"}:
            raise ProtocolError("unknown worker operation", context={"operation": operation})

        scanner = self._scanner(params.get("scanner_id"))
        if operation == "warm":
            scanner.warm()
            return {"warmed": True}
        if operation == "scan_image":
            result = scanner.scan_image(
                _request_input(params.get("input")),
                roi=params.get("roi"),
                diagnostics=params.get("diagnostics", False),
            )
            if cancelled.is_set():
                raise ProtocolError("request cancelled")
            return result.to_dict(include_diagnostics=params.get("diagnostics", False))
        if operation == "scan_document":
            document_result = scanner.scan_document(
                _request_input(params.get("input")),
                pages=params.get("pages"),
                workers=params.get("workers", 1),
                on_error=params.get("on_error", "raise"),
                roi=params.get("roi"),
                diagnostics=params.get("diagnostics", False),
            )
            if cancelled.is_set():
                raise ProtocolError("request cancelled")
            return document_result.to_dict(include_diagnostics=params.get("diagnostics", False))
        if operation == "iter_document":
            diagnostics = params.get("diagnostics", False)
            stream = scanner.iter_document(
                _request_input(params.get("input")),
                pages=params.get("pages"),
                workers=params.get("workers", 1),
                on_error=params.get("on_error", "raise"),
                roi=params.get("roi"),
                diagnostics=diagnostics,
            )
            page_count = 0
            try:
                for page in stream:
                    if cancelled.is_set():
                        stream.close()
                        raise ProtocolError("request cancelled")
                    page_count += 1
                    self._write(
                        self._envelope(
                            request_id,
                            result=page.to_dict(include_diagnostics=diagnostics),
                            event="page",
                        )
                    )
                return {
                    "type": "document_summary",
                    "complete": stream.complete,
                    "page_count": page_count,
                    "errors": [error.to_dict() for error in stream.errors],
                    "elapsed_ms": stream.elapsed_ms,
                    "metadata": stream.metadata.to_dict(),
                }
            finally:
                stream.close()
        raise AssertionError(f"unhandled validated worker operation: {operation}")

    def _completed(self, request_id: Any, future: Future[Any]) -> None:
        try:
            result = future.result()
            response = self._envelope(request_id, result=result, event="complete")
        except BaseException as exc:
            response = self._envelope(request_id, error=exc, event="complete")
        with self._active_lock:
            self._active.pop(request_id, None)
        self._write(response)

    @staticmethod
    def _validate_request(request: dict[str, Any]) -> tuple[Any, str, dict[str, Any]]:
        unexpected = sorted(set(request) - {"protocol", "id", "operation", "params"})
        if unexpected:
            raise ProtocolError(
                "worker request contains unknown fields",
                context={"fields": unexpected},
            )
        if request.get("protocol") != PROTOCOL_VERSION:
            raise ProtocolError(
                "incompatible worker protocol version",
                context={
                    "expected": PROTOCOL_VERSION,
                    "received": request.get("protocol"),
                },
            )
        if (
            "id" not in request
            or isinstance(request["id"], (bool, dict, list))
            or not isinstance(request["id"], (str, int, float, type(None)))
        ):
            raise ProtocolError("worker request requires a scalar id")
        operation = request.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ProtocolError("worker request requires an operation")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise ProtocolError("worker request params must be an object")
        return request["id"], operation, params

    @staticmethod
    def _safe_response_id(value: Any) -> str | int | float | None:
        if isinstance(value, bool) or not isinstance(value, (str, int, float, type(None))):
            return None
        return cast(str | int | float | None, value)

    def run(self) -> int:
        while True:
            try:
                request = read_frame(self.reader)
            except ProtocolError as exc:
                self._write(self._envelope(None, error=exc, event="complete"))
                return 2
            if request is None:
                self._executor.shutdown(wait=True, cancel_futures=False)
                return 0
            try:
                request_id, operation, params = self._validate_request(request)
            except ProtocolError as exc:
                self._write(
                    self._envelope(
                        self._safe_response_id(request.get("id")),
                        error=exc,
                        event="complete",
                    )
                )
                continue
            if operation == "cancel":
                target = params.get("id")
                with self._active_lock:
                    event = self._active.get(target)
                if event is not None:
                    event.set()
                self._write(
                    self._envelope(
                        request_id,
                        result={"id": target, "cancel_requested": event is not None},
                        event="complete",
                    )
                )
                continue
            if operation == "shutdown":
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._write(
                    self._envelope(
                        request_id,
                        result={"shutdown": True},
                        event="complete",
                    )
                )
                return 0
            with self._active_lock:
                if request_id in self._active:
                    self._write(
                        self._envelope(
                            request_id,
                            error=ProtocolError("request id is already active"),
                            event="complete",
                        )
                    )
                    continue
                event = threading.Event()
                self._active[request_id] = event
            future = self._executor.submit(self._handle, request_id, operation, params, event)

            def completed_callback(completed: Future[Any], current_id: Any = request_id) -> None:
                self._completed(current_id, completed)

            future.add_done_callback(completed_callback)


def main() -> int:
    return WorkerServer(sys.stdin.buffer, sys.stdout.buffer).run()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL_VERSION",
    "WorkerServer",
    "main",
    "read_frame",
    "write_frame",
]
