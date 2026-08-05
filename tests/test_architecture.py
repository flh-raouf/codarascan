from pathlib import Path

from barcode_detection.engines import ENGINE_CATALOG


ROOT = Path(__file__).resolve().parents[1]


def test_current_engine_catalog_is_dependency_free() -> None:
    assert [(spec.name, spec.capability, spec.role) for spec in ENGINE_CATALOG] == [
        ("Tessera", "detect", "recommended"),
        ("Tessera", "decode", "recommended"),
        ("Mosaic", "detect", "base"),
        ("Mosaic", "decode", "base"),
    ]
    for spec in ENGINE_CATALOG:
        assert (ROOT / spec.source).is_file()


def test_historical_releases_are_archived() -> None:
    old_top_level_names = (
        "1-deterministic_barcode_locator",
        "2-hybrid_barcode_pipeline",
        "3-zxing_only_barcode_pipeline",
        "4-zxing_2d_barcode_pipeline",
        "5-optimized pipeline",
        "5.1-adaptive-generalization-pipeline",
        "5.2-evidence-guided-coarse-to-fine-pipeline",
        "5.3-fast-extraction-pipeline",
        "6-localization-only pipeline",
        "7-cpu-coarse-to-fine-localization",
        "7.1-fast-localization-pipeline",
        "8-structure-tensor-linear-localization",
    )
    assert all(not (ROOT / name).exists() for name in old_top_level_names)
    assert (ROOT / "archive/versions/v1-deterministic-locator").is_dir()
    assert (ROOT / "archive/versions/v8-structure-tensor").is_dir()


def test_shared_contract_has_one_stable_home() -> None:
    assert (ROOT / "src/barcode_detection/core/contracts.py").is_file()
    assert (ROOT / "src/barcode_detection/engines/common/base.py").is_file()
    assert (ROOT / "src/barcode_detection/integrations/codara/engines/base.py").is_file()
    assert (ROOT / "third_party/zxing-cpp").is_dir()


def test_active_engine_source_is_not_only_documentation() -> None:
    active_files = (
        "src/barcode_detection/engines/tessera/detection.py",
        "src/barcode_detection/engines/tessera/extraction.py",
        "src/barcode_detection/engines/mosaic/detection.py",
        "src/barcode_detection/engines/mosaic/extraction.py",
        "src/barcode_detection/engines/common/tensor_localizer.py",
        "src/barcode_detection/engines/common/vendored_decoder.py",
        "src/barcode_detection/engines/common/recovery/runtime.py",
        "src/barcode_detection/engines/common/native/sttg_localizer.py",
        "src/barcode_detection/engines/registry.py",
    )
    assert all((ROOT / path).is_file() for path in active_files)

    compatibility = ROOT / "src/barcode_detection/integrations/codara/engines"
    assert "barcode_detection.engines.tessera" in (
        compatibility / "detection_tensor_p7.py"
    ).read_text()
    assert "barcode_detection.engines.mosaic" in (
        compatibility / "extraction_guarded_v3.py"
    ).read_text()


def test_active_engine_sources_do_not_require_codara() -> None:
    active_root = ROOT / "src/barcode_detection/engines"
    active_sources = tuple(active_root.rglob("*.py"))
    assert active_sources
    assert all("CODARA_BACKEND_ROOT" not in path.read_text() for path in active_sources)
    assert all("from pipeline" not in path.read_text() for path in active_sources)
    assert all("from localization" not in path.read_text() for path in active_sources)


def test_codara_boundary_contains_only_compatibility_imports() -> None:
    compatibility_root = ROOT / "src/barcode_detection/integrations/codara"
    sources = tuple(compatibility_root.rglob("*.py"))
    assert sources
    assert all("CODARA_BACKEND_ROOT" not in path.read_text() for path in sources)
    assert all("from pipeline" not in path.read_text() for path in sources)
    assert all("from localization" not in path.read_text() for path in sources)
    assert not (compatibility_root / "runtime.py").exists()
