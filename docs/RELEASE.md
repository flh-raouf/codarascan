# Release policy and verification

## 0.x publication floor

Every 0.x release must:

- pass Ruff, strict public-boundary mypy, unit/integration tests, and native
  fallback/parity checks;
- build wheel and sdist artifacts and install them outside the source tree;
- smoke fast/robust linear and matrix scans, PDFium PDF scanning, CLI JSON, and
  framed worker output from installed artifacts;
- include the native extension in official wheels and verify forced fallback;
- include Apache-2.0 license, notice, attribution, schemas, and native source;
- exclude corpora, confidential data, caches, local outputs, credentials, and
  machine paths;
- publish the compatibility matrix, benchmark status, known limitations,
  changelog, and checksums/provenance.

Recommended local evidence commands:

```bash
python -m ruff check src/codarascan tests tools benchmarks/format_smoke.py
python -m mypy
python -m pytest --cov=codarascan --cov-report=term-missing
python -m build
python tools/inspect_artifacts.py dist/*.whl dist/*.tar.gz
python tools/dependency_manifest.py dist/*.whl --output dist/dependency-manifest.json
python benchmarks/format_smoke.py --mode robust --repeats 3
git diff --check
```

The exact local pre-publication results for 0.1.0 are recorded in
[RELEASE_EVIDENCE_0.1.0.md](RELEASE_EVIDENCE_0.1.0.md). Platform-matrix and
publication results are appended only after the protected release workflow is
explicitly approved and run.

The release workflow builds CPython 3.11-3.14 wheels on Linux x86_64/ARM64,
macOS Intel/Apple Silicon, and Windows x86_64. Artifact upload and PyPI trusted
publishing are separate jobs. Publishing requires an explicit workflow input
and protected `pypi` environment approval; building a tag alone is not
publication authority.

## 1.0 blocking gates

Stable 1.0 additionally requires frozen per-format Tessera/Mosaic accuracy,
false-positive, zero-wrong-decode, duplicate, geometry, degraded-case, latency,
throughput, and peak-memory thresholds; every input/status/error contract;
native/reference and prior-Poppler/PDFium parity; all official wheels; schema
parity across Python/CLI/worker/Codara and any included Node wrapper; complete
Codara package consumption without copied engines; and licensing, privacy,
offline, upgrade, and artifact reviews.

0.1.0 does not claim these production gates.
