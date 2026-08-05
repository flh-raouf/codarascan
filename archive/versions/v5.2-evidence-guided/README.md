# 5.2 Evidence-guided coarse-to-fine pipeline

This is the controlled-provenance, clean-room result of the paper review and neutral
tournament in [`benchmarks/lab`](../../../benchmarks/lab/RESEARCH_REPORT.md). It does
not import the AGPL BaFaLo implementation or the license-unclear reference
checkpoints.

## Architecture

1. ZXing scans the untouched full-resolution image.
2. A locally authored 424 KB two-class network proposes 1D and 2D regions from
   a 320-pixel overview.
3. Every proposal maps back to the untouched source. Two differently padded
   source-resolution crops are decoded first.
4. Matching payloads take the fast path. Missing or discordant evidence
   escalates to a bounded grayscale/CLAHE/Otsu consensus. Ties remain
   unresolved.
5. Only unresolved QR proposals enter a locally trained 124 KB QR restoration
   network, and a restoration is accepted only if ZXing validates it.
6. The default `accuracy` mode also fuses the proven local 5.1 classical
   Code 39/128, Data Matrix, and QR rescue routes. `--mode balanced` omits
   that slower document-oriented safety net.

The default output contains both decoded and unresolved locations. This is
intentional: localization, decoding, and ambiguity are different outcomes.

## Run

```bash
venv/bin/python \
  'archive/versions/v5.2-evidence-guided/run.py' \
  '/path/to/image-or-directory-or.pdf' \
  --output 'archive/versions/v5.2-evidence-guided/output/my-run' \
  --device mps \
  --overwrite
```

For a retail workflow that truly expects only EAN-13/UPC-A, declare that
constraint. On the external DEAL and Quick Browser tests this preserved
recovery while reducing wrong-family decodes:

```bash
venv/bin/python \
  'archive/versions/v5.2-evidence-guided/run.py' \
  '/path/to/images' \
  --formats EAN13,UPCA \
  --output '/path/to/output' \
  --decoded-only \
  --no-overlays \
  --overwrite
```

Output is written to `detections.json`; optional page overlays use green for
decoded symbols and amber for unresolved locations. PDFs require
`pdftoppm` (Poppler).

Use `--mode balanced` for high-throughput camera streams. The default
`accuracy` mode is the research recommendation for documents and mixed input.

## Evidence boundaries

- The final locator is deployment-safe and independent, but it is not the
  accuracy champion. On the frozen mixed Kaggle split its containment-aware
  F1 is 0.637, versus 0.779 for each of the two public reference checkpoints.
  Its strength is high external DEAL single-code recall (0.950), small size,
  and controlled provenance.
- The public reference cascade reaches 0.829 containment F1, but is excluded
  from this folder because BaFaLo is AGPL and the other checkpoint's
  deployment license was not established.
- The tiny QR restorer reaches 31.20% exact decoding on QR-DN, versus 22.49%
  for the authors' ADNet LENet and 20.67% for the classical restoration
  ensemble.
- With the truthful EAN-13/UPC-A constraint, final accuracy mode reaches
  83.10% semantic recovery on the 2,000-image external DEAL test, with 0.90%
  wrong payload and 48.5 ms median latency. The stronger reference detector
  reaches 87.80%/0.65%, so detector licensing or retraining remains the main
  upgrade opportunity.
- On the disjoint 2,498-label Quick Browser real test, the same configuration
  reaches 86.67% semantic recovery, 0.36% wrong payload, 99.59% conditional
  semantic precision, and 25.3 ms median.
- On the 21-page private evaluation document integration test, accuracy mode recovers
  42/43 expected payload instances with no unexpected payload. The remaining
  Data Matrix is reported as unresolved.
- Commercial SDKs were not available under a current license in this
  workspace. Historical vendor numbers are kept separate in the research
  report.

The model checkpoints contain only locally authored architectures and locally
trained weights. Third-party runtime licenses still need the normal deployment
review: PyTorch (BSD-style), OpenCV (Apache 2.0), and ZXing-C++ (Apache 2.0).
