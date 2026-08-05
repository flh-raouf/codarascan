# Copied Codara engine adapters

The Python files in this directory are the reviewable adapter snapshot. The
registry is in `registry.py`; package imports are lazy so repository tooling can
inspect the shared contract without importing Codara's missing runtime modules.

The relative `.base` imports are kept through a compatibility bridge. The
canonical shared definitions are in `src/barcode_detection/core/contracts.py`.

To load the registry from a Codara checkout, set `CODARA_BACKEND_ROOT` to its
`apps/backend` directory before importing the registry.
