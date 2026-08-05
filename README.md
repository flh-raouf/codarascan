# Barcode Detection

This repository records the evolution of the barcode-detection work from exploratory experiments to the final Tessera and Mosaic engine lineages.

The repository is deliberately split into three layers:

1. `src/barcode_detection/` is the stable package boundary. It contains shared contracts, engine metadata, and the Codara integration snapshot used to document the current production lineages.
2. `apps/` is reserved for user-facing applications. The API, CLI, and TUI boundaries are documented there; their runtime implementations remain in the Codara application until they are migrated here.
3. `archive/` preserves experiments and numbered historical releases without presenting them as current production code.

## Start here

- [Architecture](docs/architecture/README.md) — the boundaries and data flow.
- [Lineage](docs/architecture/lineage.md) — how the experiments became versions, then Codara engines, Tessera, and Mosaic.
- [Version archive](archive/versions/README.md) — the numbered releases in order.
- [Codara integration](src/barcode_detection/integrations/codara/README.md) — what is copied here and what still lives in Codara.
- [Benchmarks](benchmarks/README.md) — reproducible evaluation code and committed fixtures.
- [Reports](reports/README.md) — authored reports and their provenance.

## Repository map

```text
src/barcode_detection/
├── core/                         shared contracts and domain vocabulary
├── engines/                      stable engine catalog and Tessera/Mosaic boundaries
└── integrations/codara/          copied adapters whose runtime dependencies are in Codara

apps/                             API, CLI, and TUI application boundaries
archive/
├── experiments/                  bin/ and notebook experiments
├── versions/                     official releases 1 through 8
└── codara/                       lineage notes for the external application
benchmarks/                       benchmark lab, fixtures, and committed results
docs/                             architecture and product documentation
models/                           model provenance and promotion notes
reports/                          authored technical reports
tests/                            repository-level architecture checks
third_party/                      vendored dependencies and Git submodules
```

## Local setup

The historical pipelines and the Codara adapters have their own dependency sets. The root package intentionally has no heavyweight dependency list because the current runtime is not self-contained in this checkout.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

Run the repository-level checks with:

```bash
.venv/bin/python -m pytest
```

For a historical pipeline, use its local README and requirements file under `archive/versions/`. For the current Tessera/Mosaic runtime, use the Codara checkout and follow the integration notes; this repository does not contain the missing Codara-side `pipeline` and `localization` packages.

## Data and confidentiality

Raw dossier inputs and generated run directories remain local and ignored. Only explicitly curated synthetic fixtures, manifests, reports, and source files belong in Git. See [.gitignore](.gitignore) before adding a new dataset or output.
