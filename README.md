# CodaraScan — Offline Barcode Detection for Images and PDFs

[![PyPI](https://img.shields.io/pypi/v/codarascan.svg?logo=pypi&logoColor=white)](https://pypi.org/project/codarascan/)
[![Python](https://img.shields.io/pypi/pyversions/codarascan.svg?logo=python&logoColor=white)](https://pypi.org/project/codarascan/)
[![CI](https://github.com/flh-raouf/codarascan/actions/workflows/ci.yml/badge.svg)](https://github.com/flh-raouf/codarascan/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/codarascan.svg)](https://github.com/flh-raouf/codarascan/blob/main/LICENSE)

**One local Python API for detecting, localizing, and decoding 40 selectable 1D and 2D barcode formats in images and PDF documents.**

CodaraScan is an open-source, offline barcode detector, barcode localizer, decoder, and document scanner for Python. It finds multiple linear and matrix barcodes, returns their quadrilateral coordinates, and preserves decoded text and original payload bytes. Multi-page PDF processing, ordered concurrency, streaming results, and a command-line interface are included—without a cloud API, telemetry, or runtime model download.

For teams looking for a universal barcode detection layer, CodaraScan provides one stable interface across QR Code, Data Matrix, PDF417, Aztec, MaxiCode, Code 128, Code 39, EAN, UPC, ITF, DataBar, and other formats. The exact [40-format catalog](#supported-barcode-formats) and its validation boundaries are documented below; “multi-format” does not mean that every degraded barcode is guaranteed to decode.

[Install](#installation) · [Quick start](#quick-start) ·
[PDF batch scanning](#scan-a-multi-page-pdf) · [CLI](#command-line-interface) ·
[Formats](#supported-barcode-formats) · [API reference](https://github.com/flh-raouf/codarascan/blob/main/docs/API.md) ·
[PyPI](https://pypi.org/project/codarascan/)

> **Alpha notice — CodaraScan 0.1.0 is alpha software.** Its APIs and schemas
> may change during 0.x. Pin an exact version and evaluate it against your own documents before
> production use. See the [known limitations](https://github.com/flh-raouf/codarascan/blob/main/docs/KNOWN_LIMITATIONS.md)
> and [release policy](https://github.com/flh-raouf/codarascan/blob/main/docs/RELEASE.md).

## Why CodaraScan?

Most barcode libraries focus on decoding a clean, tightly cropped symbol. Document workflows need more: finding an unknown number of barcodes on a page, retaining their geometry, processing long PDFs in a controlled order, and handling symbols that can be located but not decoded. CodaraScan exposes those states explicitly.

| Capability | What CodaraScan provides |
|---|---|
| Barcode detection | Finds multiple 1D/linear and 2D/matrix symbols in an image or PDF page |
| Barcode localization | Returns clockwise pixel and normalized quadrilaterals, even when decoding is disabled |
| Barcode decoding | Returns Unicode text, exact raw payload bytes, and a canonical format name |
| Document scanning | Reads selected PDF pages, preserves requested order, and supports bounded parallel work |
| Batch processing | Scans complete or selected multi-page PDFs and can stream pages without accumulating them |
| Format coverage | 40 selectable formats: 27 linear and 13 matrix selections in version 0.1.0 |
| Recovery profiles | `fast` Tessera mode for lower latency; `robust` Mosaic mode for stronger recovery |
| Local processing | No server, network request, telemetry, external executable, or runtime model download |
| Integration | Typed Python API, deterministic JSON, CLI, schemas, and a private persistent worker protocol |
| Deployment | Native wheels for supported Linux, macOS, and Windows targets; Python fallback for Tessera |

### Common use cases

- Extract QR Codes and barcodes from scanned PDF documents.
- Localize multiple barcodes on shipping labels, forms, invoices, and archive pages.
- Process document batches while keeping page results in deterministic order.
- Read retail, logistics, inventory, manufacturing, and library barcode formats.
- Return barcode coordinates for redaction, cropping, indexing, annotation, or review.
- Run barcode recognition inside private, air-gapped, or offline environments.
- Add a local barcode reader to Python, OpenCV, Pillow, desktop, or backend workflows.

## Installation

CodaraScan supports CPython 3.11, 3.12, 3.13, and 3.14.

```bash
python -m pip install codarascan
```

For reproducible alpha deployments, pin the current release:

```bash
python -m pip install codarascan==0.1.0
```


## Quick start

### Detect and decode barcodes in an image

```python
from codarascan import DecodedSymbolResult, Scanner

scanner = Scanner(
    mode="fast",       # "fast" (Tessera) or "robust" (Mosaic)
    symbols="all",     # "linear", "2d", or "all"
    decode=True,
)

result = scanner.scan_image("shipping-label.png")

for symbol in result.symbols:
    print("status:", symbol.status.value)
    print("corners:", [(point.x, point.y) for point in symbol.quad])

    if isinstance(symbol, DecodedSymbolResult):
        print("format:", symbol.format)
        print("text:", symbol.text)
        print("raw bytes:", symbol.raw_bytes)
```

`result.symbols` may contain zero, one, or multiple detections. Every symbol has full-image pixel geometry in `quad` and resolution-independent coordinates in `normalized_quad`.

### Scan only selected barcode formats

Narrowing the format set can reduce unnecessary work and makes the expected contract explicit:

```python
from codarascan import Scanner

scanner = Scanner(
    mode="robust",
    symbols="all",
    formats=["qr-code", "data-matrix", "code-128", "ean-13"],
)

result = scanner.scan_image("warehouse-label.jpg")
```

Canonical names and common aliases are accepted. Unknown formats and contradictions such as `symbols="linear"` with `formats=["qr-code"]` fail at configuration time.

### Localize barcodes without decoding

Use detection-only mode when you need coordinates for crops, overlays, redaction, indexing, or a separate decoding pipeline:

```python
from codarascan import Scanner

localizer = Scanner(symbols="all", decode=False)
result = localizer.scan_image("document.png")

for symbol in result.symbols:
    assert symbol.status.value == "localized"
    print(symbol.kind.value, symbol.quad, symbol.normalized_quad)
```

Localization-only results deliberately have no `text`, `raw_bytes`, or `format` attributes.

### Scan a region of interest

The ROI is `(x, y, width, height)` normalized to the oriented full image. The returned barcode coordinates remain relative to the full image.

```python
result = scanner.scan_image(
    "form.png",
    roi=(0.50, 0.00, 0.50, 0.40),  # top-right portion of the page
)
```

### Scan a multi-page PDF

Page numbers are one-based. Requested order is preserved even when pages are processed concurrently.

```python
from codarascan import Scanner

scanner = Scanner(mode="robust", symbols="all")

document = scanner.scan_document(
    "archive-batch.pdf",
    pages=[3, 1, 2],
    workers=4,          # positive integer or "auto"
    on_error="collect", # keep successful pages and report page errors
)

for page in document.pages:
    print(f"page {page.page}: {len(page.image.symbols)} symbol(s)")

print("complete:", document.complete)
print("errors:", document.errors)
```

Use `on_error="raise"` for fail-fast behavior. With `"collect"`, failed pages are recorded as typed errors while successful pages remain available.

### Stream a large PDF with bounded memory

`iter_document` yields ordered page results without storing all returned pages:

```python
from codarascan import Scanner

scanner = Scanner(mode="fast")

with scanner.iter_document("large-document.pdf", workers="auto") as pages:
    for page in pages:
        save_result(page)

    print("complete:", pages.complete)
    print("errors:", pages.errors)
```

At most the selected worker count is in flight. Access and rendering are serialized; owned pixel snapshots are then analyzed concurrently.

### Use the one-off functions

For short scripts, the module-level functions reuse cached immutable scanner configurations:

```python
from codarascan import scan_document, scan_image

image_result = scan_image("label.png", mode="fast", formats=["code-128"])
pdf_result = scan_document("forms.pdf", mode="robust", workers=2)
```

Long-running applications can create a `Scanner` once, optionally call `scanner.warm()`, and safely share the immutable configuration across threads.

## Barcode detection, localization, and decoding

These terms describe different parts of barcode recognition:

- **Localization** (also spelled *localisation*) describes where a detected barcode is. CodaraScan returns four clockwise corners rather than only an axis-aligned bounding box.
- **Decoding** interprets the bars or modules and returns the payload and barcode symbology.

A difficult symbol may be localized without being decoded. CodaraScan keeps that information instead of silently discarding the region.

| Status | Meaning |
|---|---|
| `decoded` | Geometry, format, Unicode text, and raw payload bytes are available |
| `localized_unresolved_linear` | A linear/1D barcode was confidently localized, but its payload could not be decoded |
| `localized_unresolved_matrix` | A matrix/2D barcode was confidently localized, but its payload could not be decoded |
| `localized` | Geometry-only result produced with `decode=False` |
| `review_candidate` | A barcode-like region was found with weaker evidence and may require human review |

Confidence is an engine-specific evidence score in `[0, 1]`, not a calibrated probability. It is not directly comparable between Tessera and Mosaic. Treat `status` as the primary semantic signal.

## Fast vs. robust barcode scanning

CodaraScan makes the performance/recovery choice explicit. It does not switch engines automatically.

| Mode | Engine | Best fit | Trade-off |
|---|---|---|---|
| `mode="fast"` | Tessera | Low-latency document scanning and common workflows | Less exhaustive recovery |
| `mode="robust"` | Mosaic | Difficult inputs where stronger recovery is worth more work | Higher CPU time and latency |

Start with `fast`, measure on representative inputs, and choose `robust` where it materially improves your corpus. Avoid choosing a mode from synthetic benchmarks alone.

## Supported barcode formats

CodaraScan 0.1.0 exposes **40 selectable barcode formats** generated from the installed ZXing-C++ readable catalog: **27 linear/1D selections** and **13 matrix/2D selections**.

### 1D and linear barcodes

`codabar`, `code-128`, `code-32`, `code-39`, `code-39-extended`, `code-39-standard`, `code-93`, `databar`, `databar-expanded`, `databar-expanded-stacked`, `databar-limited`, `databar-omni`, `databar-stacked`, `databar-stacked-omni`, `dx-film-edge`, `ean-13`, `ean-8`, `ean-upc`, `isbn`, `itf`, `itf-14`, `pzn`, `telepen`, `telepen-alpha`, `telepen-numeric`, `upc-a`, `upc-e`.

### 2D and matrix barcodes

`aztec`, `aztec-code`, `aztec-rune`, `compact-pdf417`, `data-matrix`, `maxicode`, `micro-pdf417`, `micro-qr-code`, `pdf417`, `qr-code`, `qr-code-model-1`, `qr-code-model-2`, `rmqr-code`.

Inspect the catalog programmatically instead of hard-coding it:

```python
from codarascan import supported_formats

groups = supported_formats()

for kind, formats in groups.items():
    print(kind, [item.name for item in formats])
```

Some selectors are semantic or family views, and clean-fixture coverage is not the same as production accuracy on blur, glare, damage, perspective, or poor quiet zones. See the [compatibility notes](https://github.com/flh-raouf/codarascan/blob/main/docs/COMPATIBILITY.md) and [0.1.0 format evidence](https://github.com/flh-raouf/codarascan/blob/main/benchmarks/results/FORMAT_STATUS_0.1.0.md).

## Inputs and results

### Image inputs

`scan_image` accepts:

- `str` and `pathlib.Path` image paths;
- encoded `bytes`, `bytearray`, and `memoryview` values;
- Pillow images;
- 2D `uint8` grayscale NumPy arrays;
- 3-channel BGR and 4-channel BGRA `uint8` NumPy arrays.

EXIF orientation is applied to encoded and Pillow inputs. NumPy arrays are analyzed exactly as supplied and copied to isolate the scan from caller mutation.

### PDF inputs

`scan_document` and `iter_document` accept PDF paths or encoded PDF bytes. Version 0.1 supports PDFs only through the document API; password-protected documents do not yet have a password parameter.

### Geometry and payloads

Every symbol includes:

- `status` and `kind`;
- engine-specific `confidence` and contributing `sources`;
- `quad`: four pixel-space points in top-left, top-right, bottom-right,
  bottom-left order;
- `normalized_quad`: the same four points normalized by image width and height;
- analyzed `image_width` and `image_height`.

A `DecodedSymbolResult` additionally includes `format`, `text`, its `value`
alias, and exact `raw_bytes`.

### Deterministic JSON

```python
from codarascan import scan_image, to_json

result = scan_image("label.png", mode="robust")
print(to_json(result, indent=2))
```

`to_json` emits UTF-8, sorted keys, finite JSON numbers, and Base64-encoded raw bytes. JSON schemas and golden examples ship in `codarascan/schemas` for consumer contract tests.

## Command-line interface

The `codarascan` CLI uses the same engines and result contract as the Python API.

```bash
# Scan an image and emit one JSON result
codarascan image shipping-label.png --mode fast --symbols all --json

# Restrict decoding to QR Code, Data Matrix, and Code 128
codarascan image mixed-label.png --formats qr-code,data-matrix,code-128 --json

# Return barcode locations without decoding payloads
codarascan image page.png --no-decode --json

# Scan selected PDF pages in parallel and stream NDJSON
codarascan document archive.pdf --pages 1-4,7 --workers auto --ndjson
```

Human-readable output is the default. `--json` emits one shared-schema result; `--ndjson` streams page objects and a final document summary. Standard output is reserved for results, while warnings and debug-output locations use standard error.

| Exit code | Meaning |
|---:|---|
| `0` | Scan completed and found symbols |
| `1` | Scan completed successfully with no symbols |
| `2` | Invalid usage or input |
| `3` | Processing failure |
| `4` | Partial document result |

Run `codarascan image --help` or `codarascan document --help` for every option.

## Platform support

Official 0.1.0 wheels target:

| Operating system | Architectures | Python |
|---|---|---|
| Linux glibc / manylinux | x86_64, ARM64 | CPython 3.11–3.14 |
| macOS | Intel x86_64, Apple Silicon ARM64 | CPython 3.11–3.14 |
| Windows | x86_64 | CPython 3.11–3.14 |

The native Tessera extension is included in supported wheels. If it cannot
load, CodaraScan emits one warning and continues with the slower Python
reference backend. Set `CODARASCAN_FORCE_PYTHON=1` to exercise that path.
Result metadata records the selected backend.

PyPy, CPython 3.10 and older, Alpine/musl, 32-bit systems, and Windows ARM64 are outside the initial support contract. See the full [compatibility matrix](https://github.com/flh-raouf/codarascan/blob/main/docs/COMPATIBILITY.md).

## Offline operation, privacy, and resources

Import, warm-up, image scanning, PDF rendering, CLI, and worker operations make no network requests and emit no telemetry. Default scans leave no persistent crops, overlays, rendered pages, or payload files.

Inputs, decoded payloads, diagnostics, and explicit debug output may still be sensitive. Applications should apply their own access control and retention policy.

CodaraScan intentionally imposes no hidden limits on input bytes, dimensions, pages, workers, CPU, memory, or result count. Large images, 300-DPI PDF rendering, broad format searches, diagnostics, and high worker counts can exhaust resources. Production callers own quotas, isolation, timeouts, admission control, and cancellation.

## Frequently asked questions

### How do I detect barcodes in a PDF with Python?

Install CodaraScan, create a `Scanner`, and call `scanner.scan_document("file.pdf")`. Use `pages=[...]` to select pages,
`workers="auto"` for bounded parallel analysis, or `iter_document` to stream a large PDF without accumulating all page results.

### Can CodaraScan find multiple barcodes in one image or document page?

Yes. Each image or page result contains a `symbols` tuple with zero, one, or multiple barcode detections. Each detection has its own status, kind, confidence, and quadrilateral.

### Can it return barcode coordinates without reading the value?

Yes. Set `decode=False` in Python or pass `--no-decode` to the CLI. CodaraScan returns localization results with pixel and normalized quadrilaterals and does not expose payload attributes.

### Is CodaraScan an offline barcode scanner?

Yes. Barcode detection, decoding, PDF rendering, the CLI, and the worker all run locally with no telemetry or network calls. No cloud account or API key is required.

### Is CodaraScan a universal barcode reader?

It is a multi-format barcode detection layer with 40 selectable 1D and 2D formats in version 0.1.0. No honest scanner can guarantee every barcode under every capture condition, so CodaraScan publishes the exact catalog, evidence, and known limitations instead of making an unlimited compatibility claim.


### Which mode should I use?

Begin with `mode="fast"` for lower latency. Evaluate `mode="robust"` when difficult inputs justify stronger recovery and extra computation. Benchmark both on your actual documents.

### Is CodaraScan open source and usable commercially?

Yes. CodaraScan is released under the permissive Apache License 2.0. Review the [license](https://github.com/flh-raouf/codarascan/blob/main/LICENSE), [notice](https://github.com/flh-raouf/codarascan/blob/main/NOTICE), and [third-party notices](https://github.com/flh-raouf/codarascan/blob/main/THIRD_PARTY_NOTICES.md) for the complete terms.

## Documentation

- [Public API contract](https://github.com/flh-raouf/codarascan/blob/main/docs/API.md)
- [Compatibility and barcode format status](https://github.com/flh-raouf/codarascan/blob/main/docs/COMPATIBILITY.md)
- [Known limitations](https://github.com/flh-raouf/codarascan/blob/main/docs/KNOWN_LIMITATIONS.md)
- [Worker protocol](https://github.com/flh-raouf/codarascan/blob/main/docs/WORKER_PROTOCOL.md)
- [Release policy](https://github.com/flh-raouf/codarascan/blob/main/docs/RELEASE.md)
- [Changelog](https://github.com/flh-raouf/codarascan/blob/main/CHANGELOG.md)
- [Migration notes](https://github.com/flh-raouf/codarascan/blob/main/MIGRATION.md)

The private `codarascan _worker` command provides a persistent, versioned, length-prefixed JSON protocol for backend adapters without opening a network port. It is not a public network service or a pre-1.0 stability promise.

## Development and contributing

Contributions, reproducible bug reports, and representative barcode samples are welcome. Read [CONTRIBUTING.md](https://github.com/flh-raouf/codarascan/blob/main/CONTRIBUTING.md) before opening a pull request.

```bash
git clone --recurse-submodules https://github.com/flh-raouf/codarascan.git
cd codarascan
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/ruff check src/codarascan tests
.venv/bin/mypy
.venv/bin/python -m pytest
.venv/bin/python -m build
```

Research history remains under `archive/` and `benchmarks/`; it is not installed as runtime package data.

## Contact

CodaraScan is maintained by Abderraouf FELLAHI. For questions and bug reports,
use [GitHub Issues](https://github.com/flh-raouf/codarascan/issues) or email
[ma_fellahi@esi.dz](mailto:ma_fellahi@esi.dz).

## License

CodaraScan is open-source software licensed under [Apache-2.0](https://github.com/flh-raouf/codarascan/blob/main/LICENSE). Third-party provenance is recorded in [THIRD_PARTY_NOTICES.md](https://github.com/flh-raouf/codarascan/blob/main/THIRD_PARTY_NOTICES.md).
