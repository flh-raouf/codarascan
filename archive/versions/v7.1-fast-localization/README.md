# 7.1 fast localization pipeline

This is the opt-in speed tier for users who know their documents contain
clear, adequately sized barcodes. It localizes only:

- no ZXing;
- no payload decoding;
- no native-resolution proposal search;
- no restoration or recovery;
- one 31,424-parameter grayscale segmentation model;
- one 256-pixel overview;
- oriented boxes mapped back to source coordinates.

Pipeline 7 remains the conservative default for dense tables, small symbols,
and workflows where false locations are expensive.

## Run

From the barcode-detection repository root:

```bash
venv/bin/python 'archive/versions/v7.1-fast-localization/run.py' \
  'archive/experiments/notebooks/data/pdfs/private-evaluation-document.pdf' \
  --output 'archive/versions/v7.1-fast-localization/output/my-run' \
  --profile fast \
  --workers 1 \
  --overwrite
```

Add `--overlays` for visual QA. Overlay writing is measured separately from
localization.

The input can be one image, an image directory, or a PDF. For PDFs, the runner
prefers the largest embedded page raster and renders only pages without a
suitable native raster.

The output does not claim a decoded payload or symbology. Its `kind` field is
only a geometric routing estimate (`linear` or `matrix_2d`); a rectangular
stacked 2D symbol can therefore be routed as linear until a decoder identifies
its actual format.

## Intended envelope

Use this engine when:

- symbols are clear, dark-on-light, and reasonably large;
- a miss is acceptable in exchange for low latency;
- pages are not dominated by dense ruled tables or barcode-like text;
- difficult tiny 2D symbols are not expected.

Use pipeline 7 instead when:

- symbols have a long side below 100 source pixels;
- pages contain dense tables or repeated narrow text;
- high precision matters more than latency;
- a single missed separator would be costly.

## Frozen holdout results

The selected `fast-locator-v3` checkpoint uses a 256-pixel canvas, a 0.65
mask threshold, and rejects connected components smaller than 60 overview
pixels. The larger component gate improved every frozen holdout and is the
default after the VN Lot 6 investigation.

| Holdout | Precision | Recall | F1 | Median model + postprocess |
|---|---:|---:|---:|---:|
| Clear arbitrary-angle pages, 360 symbols | 97.19% | 96.11% | 96.65% | 1.71 ms |
| Kaggle mixed test folds, 1,324 symbols | 83.14% | 63.67% | 72.11% | 1.22 ms |
| InventBar disjoint fold, 91 symbols | 76.07% | 97.80% | 85.58% | 1.43 ms |
| ParcelBar disjoint fold, 199 symbols | 60.20% | 88.94% | 71.81% | 1.60 ms |

On the mixed Kaggle holdout, containment recall by source long side was:

- under 32 px: 3.03%;
- 32–59 px: 15.50%;
- 60–99 px: 58.00%;
- 100–199 px: 85.10%;
- 200–399 px: 95.05%;
- at least 400 px: 98.80%.

This is why the engine is labeled for large/easy symbols rather than presented
as a universal replacement.

## VN Lot 6 and scanned forms

The neural fast-photo profile must not be used for VN Lot 6. Stamps, signatures,
Arabic text, and document borders create a severe domain shift. Increasing the
model canvas improved recall but multiplied false regions; ZXing valid-only,
published BaFaLo, tiling, and generic hard-negative training also failed to
produce a safe universal result.

For VN-style batches known to contain only 1-D barcodes, use pipeline 7's
verified linear branch:

```bash
venv/bin/python 'archive/versions/v7-coarse-to-fine/run.py' \
  '/path/to/private-corpus-a.pdf' \
  --output 'archive/versions/v7.1-fast-localization/output/vn-linear' \
  --kinds linear \
  --workers 4 \
  --overwrite
```

On VN Lot 6 it returned all 33 pipeline-7 linear locations, zero additional
regions, 21 ms/page sequential latency, and 10 ms/page four-worker throughput.
It intentionally skipped all eight Data Matrix regions. Use full pipeline 7
when those 2-D symbols matter.

## Why ZXing is not used

ZXing error geometry was tested on the same frozen sets. It was lower-recall
and slower for this localization-only task. Adding it to the selected model
improved Kaggle F1 by only 0.37 points while making easy-page median latency
roughly 15 times larger and slightly reducing easy-page precision.

The result does not imply that ZXing is slow in general. It remains the selected
engine for fast payload extraction. It simply does more work than this tiny
localizer and is not the best localization-only component.

## Codara engine

Codara registers the same checkpoint as:

```text
fast-localizer
```

Codara labels it **Fast photo localizer** and also exposes
`p7-linear-fast` as **Fast linear localizer** for scanned 1-D forms. The
existing `p7-localizer` remains the mixed-format default.

## Reproduction

- `train.py` performs controlled manifest-supervised fine-tuning.
- `teacher_to_manifest.py` creates development-only pipeline-7 distillation
  manifests.
- `run_benchmark.py` emits the neutral benchmark prediction schema.
- `candidates.py` contains the ZXing, model, union, and structural-verification
  tournament candidates.

See [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) for the complete
selection logic and rejected branches.
