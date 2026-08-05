# Contributing

Thanks for helping improve the barcode engines or their research record.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest
```

Initialize the ZXing-C++ reference submodule when working on native or
low-level recovery code:

```bash
git submodule update --init --recursive
```

The optional native structure-tensor extension can be built from the repository
root with:

```bash
.venv/bin/python src/barcode_detection/engines/common/native/setup.py build_ext --inplace
```

The Python fallback remains supported, so the extension is not required for
the test suite or for ordinary development.

## Repository boundaries

- Put current engine behavior under `src/barcode_detection/engines/`.
- Put shared result types under `src/barcode_detection/core/`.
- Put new evaluation code and curated fixtures under `benchmarks/`.
- Keep numbered releases and exploratory notebooks frozen under `archive/`.
- Keep Codara-specific compatibility code under `src/barcode_detection/integrations/codara/`.

Before committing, run the default tests and `git diff --check`. Do not commit
raw dossier inputs, generated outputs, caches, downloaded weights, or machine-
specific virtual environments. New model artifacts need provenance, checksum,
and license information before they can be promoted.
