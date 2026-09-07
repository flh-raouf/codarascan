#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke an installed CodaraScan artifact without importing the source tree."""

from __future__ import annotations

import importlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import zxingcpp
from PIL import Image

import codarascan
from codarascan import Scanner

_HEADER = struct.Struct(">I")


def worker_round_trip(process: subprocess.Popen[bytes], request: dict[str, object]) -> dict[str, object]:
    assert process.stdin and process.stdout
    payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
    process.stdin.write(_HEADER.pack(len(payload)) + payload)
    process.stdin.flush()
    header = process.stdout.read(_HEADER.size)
    assert len(header) == _HEADER.size
    length = _HEADER.unpack(header)[0]
    response = json.loads(process.stdout.read(length))
    assert isinstance(response, dict)
    return response


def barcode(payload: str, format_: zxingcpp.BarcodeFormat, scale: int) -> np.ndarray:
    value = zxingcpp.create_barcode(payload, format_)
    return np.asarray(
        zxingcpp.write_barcode_to_image(value, scale=scale, add_quiet_zones=True)
    ).copy()


def main() -> int:
    package_path = Path(codarascan.__file__).resolve()
    source_root = Path(__file__).resolve().parents[1] / "src"
    if source_root in package_path.parents:
        raise AssertionError(f"artifact smoke imported the source tree: {package_path}")

    importlib.import_module("codarascan._sttg_native")

    qr = barcode("ARTIFACT-QR", zxingcpp.BarcodeFormat.QRCode, 8)
    code128 = barcode("ARTIFACT-CODE128", zxingcpp.BarcodeFormat.Code128, 2)
    canvas = np.full((code128.shape[0] + 100, code128.shape[1] + 100), 255, np.uint8)
    canvas[50 : 50 + code128.shape[0], 50 : 50 + code128.shape[1]] = code128

    fast_qr = Scanner(mode="fast", symbols="2d", formats=["qr-code"]).scan_image(qr)
    robust_qr = Scanner(mode="robust", symbols="2d", formats=["qr-code"]).scan_image(qr)
    fast_linear = Scanner(
        mode="fast", symbols="linear", formats=["code-128"]
    ).scan_image(canvas)
    panorama_linear = Scanner(
        mode="panorama", symbols="linear", formats=["code-128"]
    ).scan_image(canvas)
    assert fast_qr.symbols[0].text == robust_qr.symbols[0].text == "ARTIFACT-QR"  # type: ignore[attr-defined]
    assert fast_linear.symbols[0].text == "ARTIFACT-CODE128"  # type: ignore[attr-defined]
    assert panorama_linear.symbols[0].text == "ARTIFACT-CODE128"  # type: ignore[attr-defined]
    assert panorama_linear.metadata.engine == "panorama-extractor"
    assert fast_qr.metadata.backend == "native"

    with tempfile.TemporaryDirectory(prefix="codarascan-artifact-") as directory:
        root = Path(directory)
        image_path = root / "qr.png"
        pdf_path = root / "qr.pdf"
        Image.fromarray(qr).save(image_path)
        Image.fromarray(qr).convert("RGB").save(pdf_path, "PDF", resolution=150)
        document = Scanner(
            mode="robust", symbols="2d", formats=["qr-code"]
        ).scan_document(pdf_path)
        assert document.complete and document.pages[0].image.symbols[0].text == "ARTIFACT-QR"  # type: ignore[attr-defined]

        command = [
            sys.executable,
            "-m",
            "codarascan.cli",
            "image",
            str(image_path),
            "--mode",
            "robust",
            "--symbols",
            "2d",
            "--json",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        assert json.loads(completed.stdout)["symbols"][0]["text"] == "ARTIFACT-QR"
        assert completed.stderr == ""
        version = subprocess.run(
            [sys.executable, "-m", "codarascan.cli", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert version.stdout == f"codarascan {codarascan.__version__} (Apache-2.0)\n"
        assert version.stderr == ""

        worker = subprocess.Popen(
            [sys.executable, "-m", "codarascan.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        capabilities = worker_round_trip(
            worker,
            {"protocol": 1, "id": "artifact", "operation": "capabilities"},
        )
        assert capabilities["ok"] is True
        shutdown = worker_round_trip(
            worker,
            {"protocol": 1, "id": "shutdown", "operation": "shutdown"},
        )
        assert shutdown["ok"] is True
        assert worker.wait(timeout=10) == 0
        assert worker.stderr and worker.stderr.read() == b""

        fallback_environment = dict(os.environ)
        fallback_environment["CODARASCAN_FORCE_PYTHON"] = "1"
        fallback = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import numpy as np; from codarascan import Scanner; "
                    "r=Scanner(mode='fast',symbols='linear',decode=False)"
                    ".scan_image(np.full((80,120),255,np.uint8)); "
                    "assert r.metadata.backend=='python-reference'"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=fallback_environment,
        )
        assert fallback.stdout == ""
        assert fallback.stderr.count(
            "CodaraScan could not load the native Tessera extension. "
            "Falling back to the slower Python implementation."
        ) == 1

    print(f"artifact smoke passed: {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
