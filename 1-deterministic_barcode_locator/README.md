# Deterministic barcode locator

This is a classical computer-vision locator for scanned manufacturing dossiers.
It contains **no trained detector, no cloud service, and no commercial SDK**.
Its primary output is a rotated quadrilateral around the bars, plus an overlay
and JSON manifest. Decoding is an optional confirmation step, never a reason to
discard a visually convincing barcode.

## What the dossier actually contains

`Quality Dossier.pdf` is a 21-page A3 scan. Each page is essentially one
approximately 200-ppi JPEG raster rather than a vector PDF, so rendering it at
1200 dpi would only interpolate pixels. The useful input is its native raster.

The dossier is a useful stress test:

- Code 39 vehicle/header labels and Code 128 component labels;
- 0 to many codes per page;
- labels independently rotated at about 10, 20, 30, 45, 90, and 180 degrees;
- yellow stickers, overlapping labels, dense forms, and large tables;
- a decodable Data Matrix on page 19, and a damaged/obscured matrix-like label
  on page 21;
- extremely faint bleed-through stripe groups on page 1 that cannot be
  responsibly claimed as readable codes from this raster.

The native document size matters. The clean small header code is only about
356 x 31 pixels; the page-19 Data Matrix is about 63 x 64 pixels. Never let a
library silently reduce an A3 page to 512 pixels before detection.

## The approach

```text
native PDF raster
  ├─ ZXing-C++ whole-page reads (high-confidence 1-D and 2-D anchors)
  ├─ OpenCV directional-coherence detector at several document-specific scales
  ├─ optional coarse deskew passes (0..90 degrees, preserving the full canvas)
  └─ optional independent structure-tensor proposals
             ↓
oriented-polygon containment/NMS
             ↓
local perspective rectification
             ↓
parallel-edge persistence + scanline consensus + alternating-run verification
             ↓
optional ZXing-C++ read of the deskewed crop
             ↓
oriented quads, overlays, crops, and JSON
```

### Why it is rotation-tolerant

A 1-D barcode is a compact field of many strong, nearly parallel edges. In a
local structure tensor

```text
J = [ G(Ix²)  G(IxIy) ]
    [ G(IxIy) G(Iy²)  ]
```

the anisotropy `(lambda1 - lambda2) / (lambda1 + lambda2)` is high for those
parallel edges regardless of their direction. Its principal direction supplies
the barcode's orientation. That is the physical cue used here; it is not a
learned feature.

The verifier then deskews each candidate and checks facts that dense document
text does not share:

1. horizontal (scan-direction) gradient energy dominates;
2. many edge columns persist through most of the bar height;
3. transition positions agree across several independent scanlines;
4. a median scanline contains enough alternating dark/light runs;
5. the proposal is materially longer than it is tall.

This is why the detector can reject table rules and headings such as
`ID | Libelle`, which otherwise look stripe-like at a glance.

## Install

From this folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies are all local/open source:

- `opencv-contrib-python-headless` - classical `BarcodeDetector`; its default
  constructor is non-ML.
- `zxing-cpp` - Apache-2.0 reader used only to add confidence and extract text.
- Poppler's `pdftoppm` - required only when the input is a PDF.

## Run it on this dossier

```bash
.venv/bin/python barcode_locator.py \
  ../notebooks/data/pdfs/Quality\ Dossier.pdf \
  --output output/dossier-run \
  --dpi 200 \
  --angle-step 15 \
  --include-rejected
```

## Visual page-6 teaching notebook

`page_06_visual_walkthrough.ipynb` is an executed, presentation-ready walkthrough
of every page-6 stage: PDF rendering, whole-page ZXing, all 16 real OpenCV
detector calls, gradient/coherence visualizations, the optional tensor route,
deduplication, rectification, every structural metric and acceptance gate, local
decoding, final overlays, crops, and JSON output.

Every major visual is preceded by a beginner-level reading guide explaining its
panels, axes, colors, thresholds, the page-6 observation, and why it matters to
the next pipeline decision.

Install the requirements and open it from this directory:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab page_06_visual_walkthrough.ipynb
```

To execute a clean copy non-interactively:

```bash
.venv/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  page_06_visual_walkthrough.ipynb \
  --ExecutePreprocessor.timeout=900
```

The notebook also writes 20 high-resolution slide images under
`output/notebook-page-06/figures/`. Its paired
`page_06_visual_walkthrough.py` file is the Jupytext source used to maintain the
notebook.

For the most aggressive recall experiment, use a denser fallback sweep and
keep the overlay/diagnostics under human review:

```bash
.venv/bin/python barcode_locator.py INPUT.pdf \
  --output output/high-recall \
  --angle-step 5 \
  --high-recall \
  --include-rejected
```

`--high-recall` intentionally permits uncorroborated structure-tensor regions,
so it can surface faint/damaged candidates but may produce extra review boxes.
The normal mode is tuned to minimize false positives in manufacturing forms.

## Output contract

```text
output/
  detections.json      machine-readable positions and decoder results
  overlays/page-XX.png visual result, green=decoded, amber=visual-only
  crops/page-XX-...    locally deskewed crops for a later decoding stage
  diagnostics.json     rejected proposals when --include-rejected is used
```

Each JSON detection preserves all four `quad` points in page pixel coordinates,
plus an `aabb` for consumers that only support axis-aligned boxes. Use the quad
to crop/rectify before sending a symbol to a decoder.

## Practical operating guidance

- For new vector-born PDFs, render at 300-400 dpi first. For this dossier,
  native 200 dpi is the information limit.
- Keep `--angle-step 15` for ordinary processing. Use 5 degrees only on
  difficult batches; it adds compute and interpolation but helps very faint
  oblique stickers.
- Do not make a decoder result a hard filter. A valid barcode can be visible
  but not decodable because of blur, scan resolution, damage, or a missing
  quiet zone.
- Conversely, do not trust raw edge density or a Hough-line count. Tables and
  printed text have both. Require the rectified persistence/consensus checks.
- A damaged 2-D code can be impossible to decode or localize reliably from a
  200-ppi scan. The deterministic remedy is a better source scan or a native
  embedded image, not a trained detector.

## Research and implementation references

- [OpenCV's classical 1-D barcode explanation](https://opencv.org/blog/recognizing-one-dimensional-barcode-using-opencv/)
- [OpenCV `BarcodeDetector` API and scale/downsampling controls](https://docs.opencv.org/4.x/javadoc/org/opencv/objdetect/BarcodeDetector.html)
- [OpenCV detector implementation](https://codebrowser.dev/opencv/opencv/modules/objdetect/src/barcode_detector/bardetect.cpp.html)
- [Structure-matrix 1-D/2-D localization paper](https://vs.inf.ethz.ch/publ/papers/soeroesg-icassp2014-gpulocalization.pdf)
- [Dominant-gradient-orientation barcode localization](https://www.researchgate.net/publication/265906880_Vision-Based_Localization_and_Scanning_of_1D_UPC_and_EAN_Barcodes_with_Relaxed_Pitch_Roll_and_Yaw_Camera_Alignment_Constraints)
- [ZXing-C++ formats, API, and Apache-2.0 licence](https://github.com/zxing-cpp/zxing-cpp)
