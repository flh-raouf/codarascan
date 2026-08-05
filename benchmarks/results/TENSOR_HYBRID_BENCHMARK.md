# Recommended fast mixed engines

Codara exposes two recommended fast compositions:

- `tensor-adaptive-extractor`: Tensor 1-D proposals decoded by ZXing plus the
  latest classical QR/Data Matrix extractor.
- `tensor-p7-localizer`: Tensor 1-D geometry plus the latest classical
  QR/Data Matrix geometry adapter.

Their IDs are retained for API compatibility, but neither composition invokes
the Adaptive extractor or Pipeline 7. Guarded v3 is the selected product Base
for both extraction and separation; the older adaptive and Pipeline 7 engines
remain available only for deliberate internal comparisons.

## Architecture

The shared architecture is:

```text
prepared grayscale page
  |-> 1-D: native structure-tensor proposals
  |         -> native-resolution candidate ZXing (Extraction only)
  |
  `-> 2-D: complementary prepared-scale ZXing
            -> decoder-error geometry
            -> physically verified morphology rescue
```

The recommended Extraction engine does not execute:

- whole-page linear ZXing;
- OpenCV `BarcodeDetector` linear proposals;
- the Adaptive extractor's legacy linear branch.

The Separation composition does not execute Pipeline 7. Its 2-D adapter keeps
ZXing internally as the validation gate, then discards payload and symbology
before returning geometry. Removing that gate was rejected because generic
square texture and document furniture would again become visible false
positives.

## Accuracy

Ground-truth scoring uses containment-aware one-to-one geometry for
localization and exact payload strings for extraction.

| Dataset | Adaptive Base exact | Recommended extract exact | P7 Base localized | Recommended separation localized |
|---|---:|---:|---:|---:|
| private evaluation document | 43/43 | 43/43 | 43/43 | 43/43 |
| Balanced synthetic | 121/122 | 120/122 | 121/122 | 120/122 |
| Severe stress | 136/217 | 124/217 | 155/217 | 129/217 |

The recommended extractor produced zero wrong payloads on all three
ground-truth runs. It retained five unresolved Tensor regions and one unmatched
unresolved region on the severe stress set; neither is presented as a decoded
payload. That set is intentionally outside the fast tier's intended envelope:
it contains small, blurred, damaged, and heavily transformed symbols.

## Timing

Measurements were made on the development Apple Silicon machine after engine
warm-up. Image loading and Tensor resize/preparation are outside engine
compute, matching Codara's prepared-input queue contract. Batch wall time
includes preparation.

### Current mixed engine compute

These are sequential prepared-page measurements after warm-up. File loading and
shared resize/preparation remain separately visible in diagnostics.

| Dataset | Extraction median | Separation median |
|---|---:|---:|
| private evaluation document, 21 pages | 28.5 ms/page | 27.8 ms/page |
| Balanced, 50 pages | 21.3 ms/page | 21.2 ms/page |
| Severe stress, 60 pages | 23.9 ms/page | 23.3 ms/page |

The 1-D-only Tensor localizer remains approximately 6–9 ms median. The mixed
figures are higher because they also execute validated QR/Data Matrix work and,
for Extraction, candidate-level ZXing decoding.

## Selection guidance

Choose a recommended fast composition for ordinary documents with clear,
reasonably sized symbols when latency is the priority. Choose the Base engines
for small, noisy, damaged, unreadable, or highly transformed barcodes.

The standalone specialists are intentionally separate:

- Tensor fast localizer: Latest fast 1-D Separation engine.
- Fast classical 2D extractor/localizer: Latest fast 2-D engines.
- A raw Tensor localizer is not listed under Extraction because it cannot
  produce payloads; Tensor + classical extractor is its truthful extraction
  counterpart.
