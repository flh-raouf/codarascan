# 5.3 fast extraction pipeline

This is the opt-in speed tier for documents whose barcodes are known to be
large, clear, dark-on-light, and normally oriented. It complements project 5.1;
it does not replace it.

The selected engine performs exactly one valid-only ZXing-C++ scan over each
prepared page raster:

```text
grayscale page
    ↓
ZXing-C++ valid-only read
    - rotation enabled
    - internal downscale pyramid disabled
    - inverted-symbol search disabled
    - LocalAverage binarizer
    ↓
short-ITF safety filter and duplicate removal
    ↓
decoded results
```

It deliberately contains no:

- OpenCV proposal search;
- error-location promotion;
- candidate confirmation;
- crop enhancement or upscaling;
- restoration model;
- checksum-constrained rescue;
- fallback to project 5.1.

A miss remains a miss. This is what makes the engine's cost bounded and what
keeps the user's speed choice truthful.

## When to use it

Use 5.3 when the user knows that:

- symbols are large and clean;
- the print is dark on a light background;
- pages have ordinary document orientation (0 or ±90 degrees);
- blur, perspective, damage, and undersampling are not expected;
- a missed symbol is preferable to silently paying the adaptive pipeline cost.

Use project 5.1 for unknown documents, small symbols, colored/inverted codes,
camera photographs, severe rotation or perspective, damage, dense layouts, and
maximum recall.

## Measured support envelope

The frozen easy-document holdout contains 200 document-like pages, 360 large
symbols, 25 negative pages, broad 1D/2D symbologies, and rotations. On the
ordinary-orientation subset:

| Engine | Exact payloads | Wrong reads | Median page latency |
|---|---:|---:|---:|
| Selected 5.3 one-pass engine | **123/132 (93.18%)** | **0** | **28.63 ms** |
| Clean tiny learned locator + crop decoder | 122/132 (92.42%) | 0 | ~44–49 ms |

On the full arbitrary-angle holdout, 5.3 reaches 302/360 (83.89%) with three
wrong image-level reads. This is why the production description promises
ordinary document rotations, not unrestricted oblique symbols.

On the real 21-page Quality Dossier, the selected engine reads 28/43 payloads.
Project 5.1 reads 43/43. That is the intended product distinction: 5.3 is fast
for easy symbols; 5.1 remains the quality default.

The packaged CLI, using four page workers on 300-DPI JPEG rasters, processed the
21 pages in 0.534 seconds of decoding wall time (25.4 ms/page throughput).
PDF rasterization took another 4.33 seconds and is reported separately. Codara
already prepares page images before invoking an extraction engine.

See [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) for the full candidate
tournament and real-data guardrails.

## Run

From the `barcode-detection` repository root:

```bash
venv/bin/python 'archive/versions/v5.3-fast-extraction/run.py' \
  'archive/experiments/notebooks/data/pdfs/Quality Dossier.pdf' \
  --output 'archive/versions/v5.3-fast-extraction/output/quality-fast' \
  --kinds all \
  --workers 4 \
  --overwrite
```

Generate visual overlays outside the measured processing time:

```bash
# Add:
--overlays
```

When the application knows the expected formats, declare them:

```bash
venv/bin/python 'archive/versions/v5.3-fast-extraction/run.py' \
  INPUT.pdf \
  --output OUTPUT \
  --formats EAN13,UPCA \
  --workers 4 \
  --overwrite
```

Format restriction is strongly recommended. On the real DEAL benchmark,
restricting the same fast pass to retail formats preserved its 1,332 exact
reads, reduced extraneous/wrong image-level reads from 41 to 15, and reduced
median latency from 22.24 to 14.07 ms.

## Codara integration

The engine is registered as:

```text
fast-direct-extractor
```

It appears in Codara as **Fast extractor**. The existing
`adaptive-extractor` remains the default and recommended engine.

The former Codara application adapter was a compatibility-only integration
snapshot and is not part of the installable source tree. The standalone
implementation above is the reproducible reference for this version.

## Files

- `fast_direct.py` — selected pure OpenCV/ZXing engine;
- `run.py` — image, directory, and PDF CLI;
- `benchmark.py` — fixed-manifest candidate comparison;
- `make_easy_benchmark.py` — deterministic support-envelope generator;
- `model.py`, `train.py`, `pipeline.py` — clean learned candidate retained for
  reproducibility, not selected for deployment.

## Sources

- [ZXing-C++](https://github.com/zxing-cpp/zxing-cpp) is Apache-2.0,
  thread-safe, and exposes the rotation, downscale, invert, binarizer, and
  format controls used here.
- [BaFaLo](https://federicobolelli.it/media/publications/pdfs/2025caip.pdf)
  demonstrates the potential of tiny CPU segmentation, but the released
  implementation/checkpoint has licensing and benchmark-overlap constraints.
- [Smart Inference](https://arxiv.org/abs/2004.06297) supports format/checksum
  constraints and bounded test-time evidence as practical accuracy controls.
