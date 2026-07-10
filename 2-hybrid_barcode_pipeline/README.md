# Hybrid barcode pipeline

This folder contains the production-oriented successor to the experimental
notebooks. It does not use a known barcode count and does not depend on the
YOLO baseline.

## Pipeline

1. Extract the dominant embedded page raster from a PDF when possible.
2. Run a cheap whole-page decoder pass for semantic anchors.
3. Generate oriented 1-D proposals with the existing deterministic locator.
4. Perspective-rectify proposals from the original page pixels.
5. Decode each proposal with a bounded preprocessing portfolio.
6. Optionally recover Data Matrix/QR symbols with a regional 2-D sweep.
7. Merge decoded and unresolved proposals into JSON, overlays, and crops.

An orange box means that a barcode-like structure was localized but no payload
was accepted. Green boxes are decoded 1-D symbols; magenta boxes are decoded
matrix symbols.

## Install

The existing repository `venv` already contains the required libraries. For a
fresh environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r hybrid_barcode_pipeline/requirements.txt
```

Poppler commands `pdfinfo`, `pdfimages`, and `pdftoppm` must be available.

## Decoder modes

ZXing-C++ is always available and is the open-source decoder. Dynamsoft is
optional but strongly recommended for undersampled Code 128 symbols such as
the small barcode on dossier page 16.

Set the license through the environment; do not commit it:

```bash
export DYNAMSOFT_LICENSE_KEY='your-license-key'
```

`--engine auto` uses Dynamsoft when a license is available and otherwise runs
ZXing-only. `--engine dynamsoft` fails immediately if Dynamsoft cannot start.

## Run

Full dossier:

```bash
venv/bin/python hybrid_barcode_pipeline/run.py \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --output hybrid_barcode_pipeline/output/dossier \
  --engine auto
```

Regression pages only:

```bash
venv/bin/python hybrid_barcode_pipeline/run.py \
  'notebooks/data/pdfs/private evaluation document.pdf' \
  --pages 8,16,19,21 \
  --output hybrid_barcode_pipeline/output/regression \
  --engine auto
```

Open-source-only run:

```bash
venv/bin/python hybrid_barcode_pipeline/run.py INPUT.pdf \
  --engine zxing \
  --no-matrix-sweep
```

## Outputs

```text
output/
  detections.json
  pages/       native extracted rasters or rendered fallbacks
  overlays/    decoded and unresolved boxes
  crops/       rectified candidate crops
```

The JSON includes per-page stage timings, decoder provenance, every attempted
candidate recipe, and structural metrics for unresolved 1-D proposals.

## Performance controls

- Full-page deskew fallbacks are disabled by default. Use `--angle-step 15`
  only when a new document family needs the slower rotated proposal pass.
- `--no-matrix-sweep` disables regional matrix recovery.
- `--decoded-only` removes unresolved review candidates from final outputs.
- `--pages` is useful for regression tests and diagnostics.
- `--minimum-linear-length` defaults to 100 native pixels to reject short
  document text fragments while keeping the dossier's smallest true 1-D code.

The regional sweep is matrix-only at the acceptance stage. Linear results
found in a tile are intentionally ignored because clipped tiles can produce
plausible but incorrect short payloads.
