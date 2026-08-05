# Codara integration snapshot

These adapters are the newer engine work that was developed in the Codara application and copied into this repository for lineage and architecture work.

They are not currently a self-contained runtime. Several modules import Codara-owned packages such as `localization`, `pipeline`, and `classical_2d`. Set `CODARA_BACKEND_ROOT` to the Codara `apps/backend` directory before loading the registry, and keep the snapshot synchronized with the relevant Codara commit when changing it. Do not claim that `pip install -e .` makes the adapters runnable by itself.

The shared result vocabulary was promoted to [`barcode_detection.core`](../../core/__init__.py), while `engines/base.py` remains a compatibility bridge for the copied relative imports.

The selected registry entries are:

- Tessera detection: `detection_tensor_p7.py`
- Tessera extraction: `extraction_tensor_adaptive.py`
- Mosaic detection: `detection_guarded_v3.py`
- Mosaic extraction: `extraction_guarded_v3.py`

The remaining modules are retained comparison adapters and are intentionally not all exposed by the product registry.
