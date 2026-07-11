# ZXing 2D barcode pipeline

This is a separate, format-aware implementation. It does not modify the
existing `zxing_only_barcode_pipeline`.

The default `2d` mode uses:

1. native embedded PDF raster extraction;
2. QR/Data Matrix whole-page reads with ZXing scale search enabled;
3. LocalAverage and GlobalHistogram binarization;
4. bounded recovery around decoder-reported checksum/format error positions;
5. distinct decoded, unresolved-matrix, and review result tiers;
6. atomic, self-validating output directories with relative artifact paths.

## Run the synthetic 2D dossier

```bash
venv/bin/python zxing_2d_barcode_pipeline/run.py \
  'output/pdf/Synthetic Quality Dossier - 50 scanned 2D barcode pages.pdf' \
  --output zxing_2d_barcode_pipeline/output/synthetic-2d-dossier \
  --formats 2d \
  --overwrite
```

## Validate exact payloads and artifacts

```bash
venv/bin/python zxing_2d_barcode_pipeline/regression.py \
  zxing_2d_barcode_pipeline/output/synthetic-2d-dossier/detections.json \
  --expected-csv 'output/pdf/Synthetic Quality Dossier - 2D barcode ground truth.csv' \
  --require-no-unresolved \
  --require-no-review
```

## Format routing

- `--formats 2d` - QR/Data Matrix only; the 1D proposal path is skipped.
- `--formats 1d` - Code 39/Code 128 path only.
- `--formats all` - run the matrix path and the separate linear path.

Undecoded linear candidates never enter primary results. They are either
rejected by physical bar-density guards or written to `review_candidates`.

`--residual-proposals` enables the slower high-recall 2D fallback. It searches
only outside decoded/error-anchored regions, applies spatial proposal quotas,
and emits an unresolved matrix only after conservative structure verification.

## Output contract

```text
output/run/
  detections.json
  pages/
  overlays/
  crops/
    decoded/
    unresolved/
    review/
```

All paths in `detections.json` are relative to the run root. A run is built in
a temporary sibling directory, checked for missing/orphan artifacts, and then
renamed into place. Existing output is replaced only with `--overwrite`.
