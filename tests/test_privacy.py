# SPDX-License-Identifier: Apache-2.0
"""Offline and temporary-artifact behavior at the public package boundary."""

from __future__ import annotations

import socket
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import zxingcpp

from codarascan import InternalProcessingError, Scanner


def _qr() -> np.ndarray:
    return np.asarray(
        zxingcpp.write_barcode_to_image(
            zxingcpp.create_barcode("PRIVATE-OFFLINE", zxingcpp.BarcodeFormat.QRCode),
            scale=8,
            add_quiet_zones=True,
        )
    ).copy()


def test_warm_and_scan_are_offline_when_network_entry_points_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CodaraScan attempted network access")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    scanner = Scanner(mode="robust", symbols="2d", formats=["qr-code"])
    scanner.warm()
    assert scanner.scan_image(_qr()).symbols


@pytest.mark.parametrize("fail", [False, True])
def test_default_scan_leaves_no_persistent_artifacts_on_success_or_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    scratch = tmp_path / "scratch"
    working = tmp_path / "working"
    scratch.mkdir()
    working.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.chdir(working)
    scanner = Scanner(mode="robust", symbols="2d", formats=["qr-code"])
    if fail:
        engine = scanner._get_engine()

        def injected(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected scan failure")

        monkeypatch.setattr(engine, "analyze_page", injected)
        with pytest.raises(InternalProcessingError):
            scanner.scan_image(_qr(), diagnostics=True)
    else:
        scanner.scan_image(_qr(), diagnostics=True)

    assert list(working.iterdir()) == []
    assert list(scratch.iterdir()) == []


def test_installed_runtime_contains_no_network_client_imports() -> None:
    runtime = Path(__file__).resolve().parents[1] / "src" / "codarascan"
    source = "\n".join(path.read_text(errors="ignore") for path in runtime.rglob("*.py"))
    assert "import socket" not in source
    assert "from socket" not in source
    assert "import requests" not in source
    assert "import urllib" not in source
