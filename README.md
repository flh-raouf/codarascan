# CodaraScan

CodaraScan is an offline Python package for locating and decoding barcodes in
images and PDF documents. It combines two explicit engines behind one stable
API: Tessera for low latency (`mode="fast"`) and Mosaic for stronger recovery
(`mode="robust"`). Results preserve quadrilateral geometry, localization-only
and unresolved states, Unicode text, and original payload bytes.

> **0.1.0 is alpha software.** APIs and schemas may change during 0.x. Pin an
> exact version and evaluate your own corpus before production use. All
> concrete formats readable by the installed ZXing-C++ release are selectable,
> but degraded-image validation depth varies by format. Stable 1.0 remains
> blocked on the production gates in [release policy](docs/RELEASE.md).

## Install

CodaraScan supports CPython 3.11 through 3.14.

```bash
python -m pip install codarascan==0.1.0
```

PDF support is included in the normal installation through pypdfium2. No
Poppler executable, server, network connection, runtime model download, or
Codara application is required.

## Python API

```python
from codarascan import Scanner

scanner = Scanner(
    mode="fast",                 # Tessera; use "robust" for Mosaic
    symbols="all",               # "linear", "2d", or "all"
    formats=["qr-code", "code-128"],
    decode=True,
)
scanner.warm()                   # optional eager initialization

result = scanner.scan_image("shipping-label.png")
for symbol in result.symbols:
    print(symbol.status, symbol.quad)
    if hasattr(symbol, "text"):
        print(symbol.format, symbol.text, symbol.raw_bytes)
```

`scan_image` accepts image paths, encoded image bytes, Pillow images, 2-D
`uint8` grayscale arrays, and 3/4-channel `uint8` BGR/BGRA arrays. EXIF
orientation is applied when metadata exists. NumPy arrays are analyzed exactly
as supplied and copied to isolate scans from caller mutation.

PDF scanning uses one-based page numbers and keeps requested order even with
parallel analysis:

```python
document = scanner.scan_document(
    "batch.pdf",
    pages=[3, 1, 2],
    workers=4,                    # default 1; positive integer or "auto"
    on_error="collect",           # default "raise"
)

with scanner.iter_document("large.pdf", workers=2) as pages:
    for page in pages:
        persist(page)
    print(pages.complete, pages.errors)
```

`iter_document` keeps at most the selected worker count in flight and does not
accumulate returned pages. Both image and document calls accept an optional
normalized `(x, y, width, height)` ROI and opt-in diagnostics.

One-off `scan_image`, `scan_document`, and `iter_document` functions reuse
cached immutable scanner configurations. See the complete [API contract](docs/API.md).

## Results

Stable symbol statuses are:

- `decoded`
- `localized_unresolved_linear`
- `localized_unresolved_matrix`
- `localized` when decoding is disabled
- `review_candidate`

Decoded results expose `text`, its `value` alias, exact `raw_bytes`, canonical
`format`, kind, confidence, sources, pixel quad, normalized quad, and analyzed
dimensions. Localization-only results do not have payload or format
attributes. Confidence is an engine-specific evidence score, not a probability
and not directly comparable across engines; status is the semantic signal.

`to_json(result)` is deterministic, rejects non-finite numbers, and serializes
raw bytes as Base64. Schemas and shared golden fixtures ship in
`codarascan/schemas` for consumer contract tests.

## Formats

```python
groups = Scanner.supported_formats()
print([item.name for item in groups["1d"]])
print([item.name for item in groups["2d"]])
```

The capability is generated from the installed ZXing-C++ readable catalog and
contains 27 linear and 13 matrix selections in the pinned 0.1.0 dependency
range. It deliberately has no experimental/stability field. See
[compatibility and format status](docs/COMPATIBILITY.md) and the
[0.1.0 format evidence](benchmarks/results/FORMAT_STATUS_0.1.0.md).

## CLI

```bash
codarascan image label.png --mode fast --symbols all --json
codarascan document dossier.pdf --pages 1-4,7 --workers auto --ndjson
```

Human output is the default. `--json` emits one shared-schema result;
`--ndjson` streams page objects and a final document summary. Standard output
is reserved for results. Warnings and debug-output locations use standard
error. Exit codes are 0 (symbols), 1 (successful no-symbol scan), 2 (invalid
usage/input), 3 (processing failure), and 4 (partial document result).

The private `codarascan _worker` command provides a persistent, versioned,
length-prefixed JSON protocol for backend adapters without opening a network
port. See [worker protocol](docs/WORKER_PROTOCOL.md).

## Native Tessera acceleration

Official wheels are expected to contain `codarascan._sttg_native`. Source
installs build it when a supported compiler is available. If it cannot load,
scanning continues through the Python reference backend and emits exactly one
process-level warning:

> CodaraScan could not load the native Tessera extension. Falling back to the slower Python implementation.

Set `CODARASCAN_FORCE_PYTHON=1` to exercise the reference path. Result metadata
records `native` or `python-reference`.

## Offline, privacy, and resources

Import, warm, image scanning, PDF rendering, CLI, and worker operations make no
network requests and emit no telemetry. Inputs, decoded payloads, rendered
pages, diagnostics, and explicit debug output may be sensitive. Default scans
leave no persistent crops, overlays, rendered pages, or payload files.

CodaraScan intentionally imposes no hidden limits on input bytes, dimensions,
pages, workers, CPU, memory, or results. Large images, 300-DPI PDFs, broad
format searches, diagnostics, and high worker counts can exhaust resources.
Callers own quotas, isolation, timeouts, admission control, and cancellation.

## Development

```bash
git submodule update --init --recursive
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/ruff check src/codarascan tests
.venv/bin/mypy
.venv/bin/python -m pytest
.venv/bin/python -m build
```

Research history remains under `archive/`, `benchmarks/`, and the authored
barcode report. Those assets are not installed as runtime package data. See
[contributing](CONTRIBUTING.md), [migration notes](MIGRATION.md),
[known limitations](docs/KNOWN_LIMITATIONS.md), and [changelog](CHANGELOG.md).

CodaraScan is licensed under Apache-2.0. Third-party provenance is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
