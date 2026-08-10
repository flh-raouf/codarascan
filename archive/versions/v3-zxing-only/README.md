# ZXing-only barcode pipeline

This is the open-source production path. It uses ZXing-C++, OpenCV, NumPy,
Pillow, and Poppler only. It never initializes or calls Dynamsoft and requires
no license key.

The fast path handles normal symbols. Expensive recovery runs only for an
unresolved 1-D proposal or a high-scoring matrix proposal:

- native PDF raster extraction;
- whole-page ZXing anchors;
- deterministic oriented 1-D proposals;
- bounded normal crop portfolio;
- geometry correction, anisotropic scaling, subpixel phase, and scanline
  consensus for unresolved 1-D codes;
- two-directional-gradient matrix proposals;
- context-aware Data Matrix/QR crop recovery;
- decoded/unresolved overlays, crops, JSON provenance, and stage timings.

The dossier regression manifest is used only by `regression.py`. The pipeline
does not read expected counts, page numbers, payloads, or coordinates.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r zxing_only_barcode_pipeline/requirements.txt
```

Poppler commands `pdfinfo`, `pdfimages`, and `pdftoppm` must be installed.

## Run

```bash
venv/bin/python zxing_only_barcode_pipeline/run.py \
  'notebooks/data/pdfs/private-evaluation-document.pdf' \
  --output zxing_only_barcode_pipeline/output/dossier
```

No environment variable or license is needed.

Useful controls:

```text
--pages 8,16,19,21       run selected PDF pages
--deep-matrix-recovery   add slower matrix context variants
--angle-step 15          enable slower full-page proposal rotations
--decoded-only           omit unresolved visual review candidates
--no-matrix-recovery     disable the matrix proposal/recovery stage
```

## Verify the dossier regression

```bash
venv/bin/python zxing_only_barcode_pipeline/regression.py \
  zxing_only_barcode_pipeline/output/dossier/detections.json
```

The expected manifest checks exact `(format, payload)` multisets per page. A
symbol with a valid checksum but the wrong payload fails the regression.

## Output

```text
output/dossier/
  detections.json
  pages/
  overlays/
  crops/
```

The JSON records the exact fast or recovery recipe that produced every result.
An unresolved proposal is retained rather than assigned a guessed payload.

