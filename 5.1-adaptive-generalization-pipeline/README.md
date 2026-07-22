# 5.1 Adaptive generalization pipeline

This project extends `5-optimized pipeline` without modifying that validated
predecessor. It targets the generalization failures observed on small web
uploads and differently scaled documents while retaining the original Quality
Dossier route.

See [REGRESSION_REPORT.md](REGRESSION_REPORT.md) for the exact dossier,
synthetic-suite, scale-stress, and 1,696-object external benchmark results.

## What changed

- Bounded 1.5x, 2x, and 3x scale-space searches for small inputs only.
- Decoder coordinates are mapped back to original-image pixels.
- Linear candidates receive a narrow-module width and run-periodicity audit.
- The original 100 px length rule remains the normal gate.
- A candidate below 100 px can unlock decoding through stricter physical
  evidence, but is not exposed as an undecoded box unless format-aware evidence
  confirms it.
- Matrix recovery windows scale from candidate geometry instead of assuming a
  particular document DPI.
- Residual 2D proposals accept grayscale images and upscale small images.
- Generic residual texture can trigger decoding but is never presented as an
  unresolved barcode unless a format-aware decoder supplied an error anchor.
- Image-directory inputs are copied into the atomic output work tree, fixing a
  project-5 serialization bug.
- JSON diagnostics include the scale plan, image contrast/sharpness indicators,
  matrix attempt scales, and module evidence.
- Dense barcode sheets are detected from page-level proposal evidence. Very
  wide OpenCV regions are rectified, split only at persistent inter-symbol
  quiet zones, and every child is independently verified and decoded.
- The dense splitter requires at least twelve page proposals. This prevents an
  internal gap inside one isolated barcode from fragmenting a valid symbol.
- Colour uploads retain chromatic contrast instead of forcing every recovery
  crop through one luminance image. Valid colour-distance views still pass
  through ZXing and its normal checksum validation.
- Dense repeated EAN-13 sheets can reuse a checksum-valid ZXing anchor only
  after each unresolved box independently fits all 95 modules, beats every
  one-digit checksum-valid neighbour, and agrees across multiple scan bands.
- Uploaded images also receive OpenCV QR finder-pattern geometry at bounded
  scales. Unlike generic square texture, this is format-aware evidence from
  three QR finder patterns, so a geometrically valid result may be retained as
  an unresolved QR location even when no payload can be recovered.

Upscaling does not recreate missing information. Symbols below roughly one
pixel per narrow 1D element or two pixels per 2D module can remain impossible.

## Dense catalogue-image regression

The 623 x 492 `Barcodes 1D Images.jpeg` example previously produced one QR
decode and fourteen coarse, sometimes multi-symbol review boxes. The revised
pipeline produces the same valid QR decode plus twenty-one tighter linear
locations. Code 93/Code 11, Code 128/MSI, and the three-symbol Codabar/Code
2-of-5/Plessey row are no longer merged.

The linear samples remain undecoded because the downloaded catalogue JPEG has
been rescaled to approximately one pixel per narrow element; isolated ZXing
tests also return no valid read. They are therefore reported honestly as
review candidates rather than invented payloads.

## Coloured, undersampled EAN-13 regression

The 508 x 294 `1D Colored Barcodes.png` example contains twelve renderings of
one EAN-13 symbol. Its 95-module bar field occupies only about 86--88 source
pixels, so several one-pixel bars have already merged. The final pipeline:

- localizes all 12 symbols;
- decodes 1 directly with ZXing;
- recovers 3 additional copies through checksum-gated repeated-template fits;
- retains 8 as explicitly resolution-limited review candidates.

The remaining eight are intentionally not assigned the visible human-readable
number: for those raster samples, a different checksum-valid EAN-13 pattern can
fit almost as well. Upscaling or colour separation cannot restore that lost
information. Codara now reports `Located` separately from `Decoded`, so this is
shown as 12/12 localization rather than a misleading 1/12 detection result.

## Run

From the `barcode-detection` repository root:

```bash
venv/bin/python '5.1-adaptive-generalization-pipeline/run.py' \
  'notebooks/data/pdfs/Quality Dossier.pdf' \
  --output '5.1-adaptive-generalization-pipeline/output/my-run' \
  --formats all \
  --workers 4 \
  --overwrite
```

Fast JSON-only run:

```bash
venv/bin/python '5.1-adaptive-generalization-pipeline/run.py' \
  'notebooks/data/pdfs/Quality Dossier.pdf' \
  --output '5.1-adaptive-generalization-pipeline/output/my-fast-run' \
  --formats all \
  --workers 4 \
  --no-crops \
  --no-overlays \
  --overwrite
```

Run an image directory:

```bash
venv/bin/python '5.1-adaptive-generalization-pipeline/run.py' \
  '/path/to/images' \
  --output '5.1-adaptive-generalization-pipeline/output/images' \
  --formats all \
  --workers 2 \
  --no-crops \
  --no-overlays \
  --overwrite
```

Single image files are also accepted directly:

```bash
venv/bin/python '5.1-adaptive-generalization-pipeline/run.py' \
  '/path/to/image.jpeg' \
  --output '5.1-adaptive-generalization-pipeline/output/single-image' \
  --formats all \
  --workers 1 \
  --overwrite
```

Use two to four workers for the adaptive small-image path. Upscaled OpenCV and
matrix maps are memory-bandwidth intensive, so more workers can be slower.

For an A/B comparison, disable the new behavior:

```bash
# Add both flags:
--no-adaptive-generalization --no-residual-proposals
```

## Regression commands

Quality Dossier exact payload gate:

```bash
venv/bin/python '5-optimized pipeline/regression.py' \
  '5.1-adaptive-generalization-pipeline/output/my-fast-run/detections.json' \
  --expected '3-zxing_only_barcode_pipeline/dossier_expected.json' \
  --require-exact \
  --require-no-unresolved \
  --require-no-review
```

Pascal VOC localization benchmark:

```bash
venv/bin/python '5.1-adaptive-generalization-pipeline/evaluate_pascal_voc.py' \
  '/path/to/dataset' \
  '5.1-adaptive-generalization-pipeline/output/images/detections.json'
```

The permissive metric accepts IoU >= 0.5 or intersection over the smaller
polygon >= 0.5. The evaluator also reports strict IoU recall, class breakdown,
size bins, and unassigned predictions.
