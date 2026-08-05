import pytest
import numpy as np


cv2 = pytest.importorskip("cv2")
zxingcpp = pytest.importorskip("zxingcpp")

from barcode_detection.core.contracts import Capability  # noqa: E402
from barcode_detection.engines import load_registry  # noqa: E402


def test_local_registry_loads_without_codara(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODARA_BACKEND_ROOT", raising=False)

    registry = load_registry()
    infos = registry.list_engines()

    assert {(info.label, info.capability.value) for info in infos} == {
        ("Tessera Localizer", "detect"),
        ("Tessera Extractor", "decode"),
        ("Mosaic Localizer", "detect"),
        ("Mosaic Extractor", "decode"),
    }


def test_extractors_accept_the_public_standalone_contract(tmp_path) -> None:
    payload = "barcode-detection-runtime-smoke"
    barcode = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
    image = np.asarray(
        zxingcpp.write_barcode_to_image(barcode, scale=8, add_quiet_zones=True)
    )
    page = tmp_path / "qr.png"
    assert cv2.imwrite(str(page), image)

    registry = load_registry()
    for engine_id in ("tessera-extractor", "mosaic-extractor"):
        outcome = registry.get_engine(engine_id, Capability.DECODE).analyze_page(1, page)
        assert any(region.value == payload for region in outcome.regions)
