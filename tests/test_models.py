import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_catalog_is_complete_and_truthful() -> None:
    catalog = json.loads((ROOT / "models/catalog.json").read_text())
    entries = catalog["entries"]
    ids = {entry["id"] for entry in entries}

    assert catalog["schema_version"] == 1
    assert {"tessera-runtime", "mosaic-runtime"} <= ids
    assert all(entry["status"] in {"current", "historical"} for entry in entries)

    for entry in entries:
        if entry["status"] == "current":
            assert entry["artifacts"] == []
            assert entry["runtime"] == "barcode_detection"
        else:
            assert entry["artifacts"]
            assert all((ROOT / artifact).is_file() for artifact in entry["artifacts"])
            artifact_roots = {(ROOT / artifact).parent for artifact in entry["artifacts"]}
            assert all((artifact_root / "README.md").is_file() for artifact_root in artifact_roots)
