# Barcode pipeline run commands

This file provides copy-paste commands for running every pipeline against the
real `private evaluation document.pdf`.

The current evidence-selected implementation is
[`5.2-evidence-guided-coarse-to-fine-pipeline`](5.2-evidence-guided-coarse-to-fine-pipeline/README.md).
Its complete paper review, dataset audit, neutral benchmark, and
reference-versus-deployable results are in
[`benchmark-lab/RESEARCH_REPORT.md`](benchmark-lab/RESEARCH_REPORT.md).

Run all commands from the repository root:

```bash
cd /path/to/private-project/code/barcode-detection
```

They use the existing repository virtual environment:

```text
venv/bin/python
```

Projects 2, 3, and 4 use `numbered_project_aliases.py` automatically. This is
required because their original Python package names are not the same as their
new numbered folder names.

## 1. Deterministic barcode locator

Classical localization, structural verification, optional ZXing decoding,
deskewed crops, overlays, JSON, and rejected-candidate diagnostics:

```bash
venv/bin/python '1-deterministic_barcode_locator/barcode_locator.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '1-deterministic_barcode_locator/output/quality-dossier' \
  --dpi 200 \
  --angle-step 15 \
  --include-rejected
```

For a slower high-recall review run, add `--high-recall`.

## 2. Hybrid barcode pipeline

Candidate-driven localization and decoding. This command explicitly selects
ZXing so it does not require a Dynamsoft license:

```bash
venv/bin/python '2-hybrid_barcode_pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '2-hybrid_barcode_pipeline/output/quality-dossier' \
  --engine zxing \
  --angle-step 0
```

To omit rectified crops, add `--no-crops`.

## 3. ZXing-only barcode pipeline

Open-source ZXing-C++ decoding with deterministic 1D localization, advanced
linear retries, and matrix recovery:

```bash
venv/bin/python '3-zxing_only_barcode_pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '3-zxing_only_barcode_pipeline/output/quality-dossier' \
  --angle-step 0
```

To omit rectified crops, add `--no-crops`.

## 4. Format-aware 2D-first ZXing pipeline

Runs the Data Matrix, QR Code, Code 39, and Code 128 paths:

```bash
venv/bin/python '4-zxing_2d_barcode_pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '4-zxing_2d_barcode_pipeline/output/quality-dossier' \
  --formats all \
  --overwrite
```

Use `--formats 2d` for only Data Matrix and QR Code, or `--formats 1d` for
only Code 39 and Code 128. To omit crops, add `--no-crops`.

## 5. Optimized decoding pipeline

Accuracy-gated decoding with batch PDF extraction and page-level concurrency.
Four workers are the validated setting for the complete workload on this
10-core MacBook Air:

```bash
venv/bin/python '5-optimized pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '5-optimized pipeline/output/quality-dossier' \
  --formats all \
  --workers 4 \
  --overwrite
```

For the fastest run without visual artifacts:

```bash
venv/bin/python '5-optimized pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '5-optimized pipeline/output/quality-dossier-fast' \
  --formats all \
  --workers 4 \
  --no-crops \
  --no-overlays \
  --overwrite
```

## 5.1 Adaptive generalization pipeline

Project 5 plus scale-aware recovery for small and variably sized uploads:

```bash
venv/bin/python '5.1-adaptive-generalization-pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '5.1-adaptive-generalization-pipeline/output/quality-dossier' \
  --formats all \
  --workers 4 \
  --overwrite
```

For JSON-only benchmarking, add `--no-crops --no-overlays`. Use two to four
workers for mixed collections of small images.

## 5.2 Evidence-guided coarse-to-fine pipeline

The research-selected clean-room pipeline combines a compact learned locator,
source-resolution crop consensus, an explicit unresolved state, and
decode-gated QR restoration:

```bash
venv/bin/python \
  '5.2-evidence-guided-coarse-to-fine-pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output \
  '5.2-evidence-guided-coarse-to-fine-pipeline/output/quality-dossier' \
  --device mps \
  --overwrite
```

## 5.3 Fast extraction pipeline

Opt-in bounded extraction for known clear, large, dark-on-light symbols. It
performs one valid-only ZXing-C++ page pass with no proposal or recovery stages:

```bash
venv/bin/python '5.3-fast-extraction-pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '5.3-fast-extraction-pipeline/output/quality-fast' \
  --kinds all \
  --workers 4 \
  --overwrite
```

Declare known formats whenever possible, for example
`--formats EAN13,UPCA`. Use project 5.1 instead for unknown, small, damaged,
colored, inverted, or photographed symbols.

Use `--formats EAN13,UPCA` only when the application really has that declared
symbology constraint. Add `--decoded-only --no-overlays` for payload-only
JSON benchmarking.

## 6. Localization-only pipeline

Locates linear and 2D barcodes without attempting to decode payloads:

```bash
venv/bin/python '6-localization-only pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '6-localization-only pipeline/output/quality-dossier' \
  --kinds all \
  --workers 4 \
  --overlays \
  --overwrite
```

Remove `--overlays` for JSON-only execution. Use `--kinds linear` when only
1D localization is required.

## 7. CPU coarse-to-fine localization

The latest optimized localization-only pipeline. It uses coarse proposals,
native-pixel verification, and the fast-negative cascade for empty pages:

```bash
venv/bin/python '7-cpu-coarse-to-fine-localization/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '7-cpu-coarse-to-fine-localization/output/quality-dossier' \
  --kinds all \
  --workers 4 \
  --overlays \
  --overwrite
```

Fastest JSON-only complete-localization run:

```bash
venv/bin/python '7-cpu-coarse-to-fine-localization/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '7-cpu-coarse-to-fine-localization/output/quality-dossier-fast' \
  --kinds all \
  --workers 4 \
  --overwrite
```

Use `--kinds linear --workers 21` only when the two 2D symbols are deliberately
out of scope. For complete localization, four workers are faster than 21 on
the tested machine because the matrix and QR stages compete for cache and
memory bandwidth.

## Running selected pages

Most pipelines support `--pages`. For example:

```bash
venv/bin/python '4-zxing_2d_barcode_pipeline/run.py' \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output '4-zxing_2d_barcode_pipeline/output/difficult-pages' \
  --formats all \
  --pages 8,16,19,21 \
  --overwrite
```

Projects 2 and 3 do not have an `--overwrite` option. When repeating those
runs, use a new output directory or remove the previous output directory first.
