# Barcode Detection

This repository contains the independent Tessera and Mosaic barcode engines,
their benchmark history, and the experiments that led to them.

The repository is deliberately split into three layers:

1. `src/barcode_detection/` is the stable package boundary. It contains the shared contracts, local recovery runtime, native adapter, and active Tessera/Mosaic implementations.
2. `benchmarks/` contains reproducible evaluation code, curated synthetic fixtures, and committed results.
3. `archive/` preserves experiments and numbered historical releases without presenting them as current production code.

## Start here

- [Architecture](docs/architecture/README.md) — the boundaries and data flow.
- [Lineage](docs/architecture/lineage.md) — how the experiments became versions, then Codara engines, Tessera, and Mosaic.
- [Engine runtime](src/barcode_detection/engines/README.md) — the local implementation and product boundaries.
- [Version archive](archive/versions/README.md) — the numbered releases in order.
- [Codara compatibility](src/barcode_detection/integrations/codara/README.md) — legacy import paths retained for the application checkout.
- [Model catalog](models/README.md) — local historical artifacts versus externally-owned final runtime assets.
- [Contributing](CONTRIBUTING.md) — setup, boundaries, and hygiene rules for changes.
- [Benchmarks](benchmarks/README.md) — reproducible evaluation code and committed fixtures.
- [Reports](reports/README.md) — authored reports and their provenance.

## Repository map

```text
src/barcode_detection/
├── core/                         shared contracts and domain vocabulary
├── engines/                      active Tessera/Mosaic implementations and registry
└── integrations/codara/          compatibility imports and comparison adapters

archive/
├── experiments/                  isolated atoms and pipeline notebooks
├── versions/                     official releases 1 through 8
└── README.md                     archive policy and hygiene rules
benchmarks/                       benchmark lab, fixtures, and committed results
docs/                             architecture and lineage documentation
models/                           model catalog and promotion provenance
reports/                          authored technical reports
tests/                            repository-level architecture checks
third_party/                      vendored dependencies and Git submodules
```

## Local setup

The active engine runtime is self-contained in this checkout. The historical
pipelines and optional benchmark experiments have additional dependency sets.

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[test]"
```

Run the repository-level checks with:

```bash
.venv/bin/python -m pytest
```

For the active engines, install the project with its default dependencies and
load an engine directly:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from barcode_detection.core.contracts import Capability
from barcode_detection.engines import get_engine

engine = get_engine("tessera-extractor", Capability.DECODE)
outcome = engine.analyze_page(1, Path("page-0001.png"))
print([(region.value, region.symbology) for region in outcome.regions])
PY
```

The registry also exposes `list_engines()` and `warm_all()` for applications
that need discovery or startup priming.

For benchmark tests, install the optional benchmark group and run them
explicitly. Historical pipelines remain isolated under `archive/versions/` and
use their local README and requirements file.

## Data and confidentiality

Raw dossier inputs and generated run directories remain local and ignored. Only
explicitly curated synthetic fixtures, manifests, reports, and source files
belong in Git. See [.gitignore](.gitignore) before adding a new dataset or
output. Codara remains an optional application consumer, not a runtime
dependency of this package.
