# Structure-tensor fast linear localizer

This is a standalone, experimental fast tier for **1D barcode localization**.
The production engine does not modify or import Pipeline 7. The research-only
`benchmark.py` can load Pipeline 7 when an explicit comparison is requested.

The implementation uses a fused native C++ sparse-tile structure tensor
proposer followed by native-resolution physical verification. It performs:

- no barcode decoding;
- no ZXing calls;
- no learned-model inference;
- no GPU work;
- no payload guessing.

It is intended for users who know that their documents contain ordinary,
reasonably large linear barcodes and prefer substantially lower latency than
Pipeline 7.

## Current verdict

The experiment succeeded as a fast document tier, but not as a universal
replacement for Pipeline 7 on unknown image distributions.

On prepared private evaluation document pages:

| Engine | Linear locations | Unmatched predictions | Median engine latency | 8-worker queue throughput |
|---|---:|---:|---:|---:|
| Pipeline 7 / OpenCV | 41/41 | 0 | 22.9 ms/page | about 7.2 ms/page |
| Structure tensor / document | **41/41** | **0** | **8.7 ms/page** | **2.70 ms/page** |

The queue measurements use concurrent prepared pages and exclude loading,
grayscale conversion, and detector resizing. On this workload the new path is
about 2.6 times faster in median engine latency. Pipeline 7 still returns
tighter boxes under strict IoU (0.805 F1 versus 0.659), so the two engines are
equal in containment recall here, not in every geometry metric.

Use Pipeline 7 when the input distribution is unknown or tight geometry matters
most. Use this pipeline when low latency is the priority and the input contract
is controlled.

## Build

Run from this folder:

```bash
cd '/path/to/private-project/code/barcode-detection/8-structure-tensor-linear-localization'
../venv/bin/python setup.py build_ext --inplace
```

The resulting `_sttg_native` extension is intentionally ignored by Git and
must be rebuilt for each deployment platform/Python version.

## Run

PDF input, automatic document profile, JSON only:

```bash
../venv/bin/python run.py \
  '../notebooks/data/pdfs/private evaluation document.pdf' \
  --output 'output/quality-fast' \
  --profile auto \
  --workers 8 \
  --overwrite
```

Add overlays without including artifact generation in engine timing:

```bash
../venv/bin/python run.py \
  '../notebooks/data/pdfs/private evaluation document.pdf' \
  --output 'output/quality-visual' \
  --profile auto \
  --workers 8 \
  --overlays \
  --overwrite
```

Image or image-directory input:

```bash
../venv/bin/python run.py \
  '/path/to/images' \
  --output 'output/photo-run' \
  --profile general \
  --workers 8 \
  --overlays \
  --overwrite
```

`--profile auto` selects `document` for PDFs and `general` for images.
For an ablation of the original 37/41 document path, add
`--no-periodic-rescue`.

## Operating profiles

| Profile | Tile | Native transition gate | Refinement policy | Intended domain |
|---|---:|---:|---|---|
| `document` | 7 px | 24 | Strict weak-parent retention, periodic-band split, verified fragment join | Scanned documents, precision first |
| `general` | 12 px | 12 | Conservative fragment join; periodic rescue disabled | Product/parcel photographs |

The document profile's stricter transition gate is important. A smaller tile
recovers more document barcodes, but without the 24-transition verification
threshold it also admits short text-like patterns on difficult negative pages.

## Timing contract

The service boundary is:

```python
prepared = prepare_page(image, config)  # load/gray/resize boundary
result = locate_prepared(prepared, config)  # measured engine
```

The CLI prepares every selected page before starting the engine stopwatch. Its
JSON output reports these separately:

- `extraction_seconds`;
- `preparation_wall_seconds` and per-page preparation;
- `engine_wall_seconds`, engine latency, and queue throughput;
- `artifact_wall_seconds`;
- `overall_wall_seconds`.

This separation matches a future queue where another service can load and
resize inputs before handing them to the localization engine. It does not hide
the costs: they remain present in the same report.

For very large PDFs, a production service should prepare bounded batches rather
than retain every full-resolution page at once. The CLI intentionally preloads
the selected set to make the engine-only timing boundary exact.

## Architecture

```text
prepared 1600-pixel grayscale overview
        |
        v
fused C++ Scharr + sparse tile tensor statistics
        |
        v
adaptive energy threshold + local context coherence
        |
        v
orientation-aware strong/weak component growth
        |
        v
cheap three-profile periodicity rejection
        |
        v
strictly retain rare oversized weak parents
        |
        v
split native-pixel periodic row bands inside those parents
        |
        v
map candidate to untouched source pixels
        |
        v
native-resolution multi-scanline physical verification
        |
        v
shared edge-direction verification rejects curved periodic texture
        |
        v
conservative fragment joining + final oriented boxes
```

The native kernel releases Python's GIL, allowing independent prepared pages to
run concurrently. Eight workers are the default because throughput generally
plateaus around 8–21 workers on the tested Apple M4 while individual task
latency and resource contention continue to rise.

## Programmatic use

```python
import cv2

from pipeline import locate_prepared, prepare_page, profile_config

image = cv2.imread("/path/page.png", cv2.IMREAD_GRAYSCALE)
config = profile_config("document")
prepared = prepare_page(image, config)
result = locate_prepared(prepared, config)

for detection in result.detections:
    print(detection.quad, detection.confidence)
```

## Validation

Containment-aware localization results for frozen profiles:

| Dataset | Profile | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| private evaluation document | document | 1.000 | **1.000** | **1.000** |
| private corpus A holdout | document | 1.000 | 1.000 | 1.000 |
| Easy ordinary documents | general | 1.000 | 0.904 | 0.949 |
| InventBar test fold | general | 0.988 | 0.923 | 0.955 |
| ParcelBar test fold | general | 0.908 | 0.889 | 0.898 |
| Kaggle mixed test folds | general | 0.694 | 0.767 | 0.729 |
| DEAL single-test | general | 0.858 | 0.877 | 0.867 |

These are localization metrics, not payload-decoding results. Some public data
and internal document references had already been studied in earlier pipeline
work, so they are regression/holdout evidence rather than a claim of a pristine
unseen scientific test. A new customer-document holdout should be frozen before
future threshold tuning.

The periodic refinement was also checked against two larger document
regressions after its Quality thresholds were frozen:

| Dataset | Baseline | Final | Change in unmatched predictions |
|---|---:|---:|---:|
| private corpus A full, 315 pages / 33 labels | 29 TP, 2 unmatched | 29 TP, 0 unmatched | -2 |
| Separate document confirmation, 42 images / 51 labels | 43 TP, 0 unmatched | 45 TP, 0 unmatched | 0 |

The general profile remains byte-for-byte equivalent in measured output on the
160-image Easy regression: 104/118 matches and zero unmatched predictions both
before and after these document-only changes.

## Files

- `pipeline.py` — public API, native verification, fragment handling.
- `native/sttg_native.cpp` — fused proposal kernel.
- `run.py` — PDF/image CLI and separated timing.
- `benchmark.py` — neutral manifest benchmark, Pipeline 7 and ZXing baselines.
- `RESEARCH_REPORT.md` — methods, ablations, measurements, and limitations.
- `tests/test_pipeline.py` — smoke and regression tests.
