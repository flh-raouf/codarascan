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
    assert (ROOT / "src/barcode_detection/integrations/codara/engines/base.py").is_file()
    assert (ROOT / "third_party/zxing-cpp").is_dir()
