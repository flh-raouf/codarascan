# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import numpy as np
import pytest

from codarascan import Scanner
from codarascan.engines.common.native import loader


@pytest.fixture(autouse=True)
def reset_loader(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CODARASCAN_FORCE_PYTHON", raising=False)
    loader._reset_for_tests()
    yield
    loader._reset_for_tests()


@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("missing module"),
        OSError("missing shared library"),
        RuntimeError("import-time exception"),
    ],
)
def test_loader_failure_warns_exactly_once_and_returns_reference_backend(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def fail_import(_name: str) -> ModuleType:
        raise failure

    monkeypatch.setattr(loader.importlib, "import_module", fail_import)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert loader.load_native() is None
        assert loader.load_native() is None
    assert [str(item.message) for item in caught] == [loader.FALLBACK_WARNING]
    assert caught[0].category is loader.NativeFallbackWarning
    assert loader.backend_name() == "python-reference"
    assert loader.load_error() is failure


def test_concurrent_first_load_emits_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(_name: str) -> ModuleType:
        raise ImportError("concurrent failure")

    monkeypatch.setattr(loader.importlib, "import_module", fail_import)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=8) as executor:
            assert list(executor.map(lambda _: loader.load_native(), range(32))) == [None] * 32
    assert [str(item.message) for item in caught] == [loader.FALLBACK_WARNING]


def test_forced_reference_scanning_warns_once_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODARASCAN_FORCE_PYTHON", "1")
    image = np.full((180, 320), 255, dtype=np.uint8)
    scanner = Scanner(mode="fast", symbols="linear", decode=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = scanner.scan_image(image)
        second = scanner.scan_image(image)
    assert first.metadata.backend == second.metadata.backend == "python-reference"
    assert [str(item.message) for item in caught] == [loader.FALLBACK_WARNING]


def test_successful_native_load_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("codarascan._sttg_native")
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: module)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert loader.load_native() is module
        assert loader.load_native() is module
    assert caught == []
    assert loader.backend_name() == "native"
