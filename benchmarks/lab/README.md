# Barcode pipeline tournament

This directory is the neutral evaluation layer used to compare the existing
pipelines, literature reproductions, learned detectors, and external readers.
It is intentionally not named `5.2-*`: that folder is reserved for the design
that wins the benchmark and regression gates.

The paper-by-paper review, dataset provenance, leakage warnings and current
results are in [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md).

The evaluator keeps four questions separate:

1. Was every visible symbol localized?
2. How precise was the returned polygon?
3. Was the payload decoded exactly, with the correct symbology?
4. What latency and memory cost produced that result?

It reports strict polygon IoU and a containment-aware metric. The latter is
useful when a tight oriented prediction is compared with a loose axis-aligned
annotation, but it never replaces strict IoU.

## Canonical files

`dataset.json` uses `barcode-benchmark-dataset-v1`. Each image has a stable ID,
a source path, dimensions, source/split metadata, and zero or more objects.

`predictions.json` uses `barcode-benchmark-predictions-v1`. Each image contains
the predictions returned by one frozen pipeline configuration.

Both schemas store polygons in source-image pixel coordinates. Payloads are
strings and are never coerced to numbers, because leading zeros are meaningful.
Strict payload accuracy keeps those strings untouched. A separate semantic
metric recognizes only the standards-defined equivalence between UPC-A and an
EAN-13 value with number-system zero.

## External benchmark data

Large third-party datasets are intentionally not committed to this repository.
Their provenance is tracked in [`datasets.json`](datasets.json), including the
source page, license, pinned version or revision where available, local artifact
path, and SHA-256 checksum. The registry is the single place to look when a
benchmark input needs to be restored.

List the registered sources and local artifact paths:

```bash
python3 benchmarks/lab/fetch_data.py list
```

Verify any artifacts that are currently present:

```bash
python3 benchmarks/lab/fetch_data.py verify
```

Fetch and extract a dataset when it is needed:

```bash
python3 benchmarks/lab/fetch_data.py fetch deal-kaist --extract
python3 benchmarks/lab/fetch_data.py fetch kaggle-barcode-and-qr-v8 --extract
```

Kaggle sources require the Kaggle CLI and configured credentials. The QR-DN1.0
record currently requires a manual download from Mendeley; save the downloaded
archive at the registered path and run `verify`. UniqueData is retained only as
a locally reproducible reference: its registry entry records the non-commercial,
no-derivatives terms, so it must not be redistributed from this project.

## Commands

Import a Pascal VOC image directory:

```bash
venv/bin/python benchmarks/lab/tournament.py import-pascal-voc \
  /path/to/images-and-xml \
  --output benchmarks/lab/manifests/example.json \
  --dataset-name example
```

Import BarBeR/VGG VIA annotations:

```bash
venv/bin/python benchmarks/lab/tournament.py import-vgg \
  /path/to/dataset/images \
  /path/to/Annotations/VIA/*.json \
  --output benchmarks/lab/manifests/barber.json \
  --dataset-name barber
```

Import YOLO-normalized box annotations such as InventBar/ParcelBar:

```bash
venv/bin/python benchmarks/lab/tournament.py import-yolo /path/to/ParcelBar \
  --output parcelbar.json --dataset-name ParcelBar \
  --kind 1d --symbology CODE_128
```

Normalize an existing project 5/5.1/6/7 `detections.json`:

```bash
venv/bin/python benchmarks/lab/tournament.py adapt-current \
  benchmarks/lab/manifests/example.json \
  /path/to/detections.json \
  --output benchmarks/lab/runs/project-5.1.json \
  --pipeline project-5.1
```

Evaluate:

```bash
venv/bin/python benchmarks/lab/tournament.py evaluate \
  benchmarks/lab/manifests/example.json \
  benchmarks/lab/runs/project-5.1.json \
  --output benchmarks/lab/reports/project-5.1.json
```

Run an independent competitor:

```bash
venv/bin/python benchmarks/lab/run_competitor.py \
  benchmarks/lab/manifests/example.json \
  --method structure-tensor \
  --output benchmarks/lab/runs/structure-tensor.json
```

Run the evidence-selected adaptive consensus decoder with the published
reference locator:

```bash
venv/bin/python benchmarks/lab/run_competitor.py dataset.json \
  --method bafalo-published-decode-adaptive-consensus \
  --output adaptive-consensus.json
```

Apply an application-declared symbology family without changing localization:

```bash
venv/bin/python benchmarks/lab/filter_symbologies.py adaptive-consensus.json \
  --allow EAN_13,UPC_A \
  --output adaptive-consensus-retail.json
```

Train the clean compact locator. Generated scenes are created in memory:

```bash
venv/bin/python benchmarks/lab/train_tiny_locator.py \
  development.json inventbar.json parcelbar.json deal-rec-train.json \
  --synthetic-samples 1500 --epochs 10 --size 320 --device mps \
  --output tiny-barcode-locator.pt
```

Create deterministic development/test subsets without moving image files:

```bash
venv/bin/python benchmarks/lab/tournament.py subset dataset.json \
  --folds 5 --include-folds 0 \
  --output development.json

venv/bin/python benchmarks/lab/tournament.py subset dataset.json \
  --folds 5 --include-folds 1,2,3,4 \
  --output test.json
```

Run tests:

```bash
venv/bin/python -m unittest discover -s benchmarks/lab/tests -v
```

## Evaluation invariants

- Dataset and prediction image IDs must match; silent positional truncation is
  forbidden.
- A ground-truth object and prediction can be matched at most once.
- Strict and containment-aware matching are performed independently.
- Undecodable ground-truth regions count toward localization but not payload
  accuracy.
- A wrong non-empty payload is counted separately from an unresolved decode.
- Strict exact and semantic GTIN-equivalent payload counts remain separate.
- A decoded prediction with no matched ground truth is a false decode.
- Metrics are emitted globally and by source, kind, symbology, and size.
