# Fast extraction engine investigation

## Decision

Ship the one-pass valid-only ZXing-C++ engine as the explicit fast tier. Keep
project 5.1 as Codara's default quality tier.

The investigation did not select a learned model. A tiny model can localize in
under one millisecond, but localization is not extraction: crop preparation,
decoding, ambiguity control, and false-proposal handling determine the final
latency and safety. The independently trained model failed those real-data
gates.

## Product contract

The fast engine is not an automatic pre-pass in front of 5.1. Such a cascade
would make the advertised fast choice dishonest whenever it silently escalated.
The user selects one of two contracts:

```text
Adaptive extractor (default)
  unknown/difficult pages → maximum practical recovery

Fast extractor (opt-in)
  known easy pages → one bounded decoder pass; misses remain misses
```

This also prevents the dangerous policy “fast result found, therefore the page
is complete.” Finding one easy barcode does not prove that another difficult
barcode is absent.

## Candidate tournament

### 1. Direct ZXing-C++ variants

The following controls were ablated independently:

- external page downscaling;
- ZXing's internal downscale search;
- inverted-symbol search;
- rotation search;
- LocalAverage versus other binarizers;
- all-readable versus declared format families.

Findings:

- External resize to a 1,600-pixel long side was faster but lost too many large
  holdout reads. It was rejected.
- Disabling rotation caused a severe recall loss. Rotation remains enabled.
- Disabling ZXing's internal downscale pyramid preserved the intended
  large-symbol support subset and removed work.
- Disabling inverted-symbol search preserved that subset and removed work.
- Declaring formats substantially reduces both runtime and wrong-family reads.

The resulting pass is:

```python
zxingcpp.read_barcodes(
    page_gray,
    formats=declared_or_all,
    try_rotate=True,
    try_downscale=False,
    try_invert=False,
    binarizer=zxingcpp.Binarizer.LocalAverage,
    return_errors=False,
)
```

Only valid decoded payloads are returned. ZXing error positions are not
promoted because that would unlock proposal/recovery behavior.

### 2. OpenCV as a direct reader or conditional fallback

OpenCV's barcode and QR readers were evaluated alone, after ZXing failure, and
as an always-on union.

On a 200-image DEAL sample:

| Policy | Exact | Wrong | Median |
|---|---:|---:|---:|
| OpenCV alone | 116/200 | 3 | 18.36 ms |
| ZXing, then OpenCV on failure | 147/200 | 4 | 18.54 ms |
| Always union both | 148/200 | 3 | 34.24 ms |

The conditional reader recovered some payloads but worsened safety and tail
cost. It was rejected for the bounded fast tier.

### 3. Published BaFaLo locator

BaFaLo's released checkpoint localizes in approximately 3.4 ms on this Mac and
is the strongest technical reference. Combined with adaptive crop consensus,
existing neutral reports show:

| Dataset | Exact | Wrong | Median |
|---|---:|---:|---:|
| DEAL, 2,000 images | 1,752 (87.60%) | 25 (1.25%) | 14.27 ms |
| QuickBrowser, 2,498 images | 2,102 (84.15%) | 20 (0.80%) | 25.76 ms |

It was not selected for Codara production because:

- the released repository is AGPL-3.0;
- the checkpoint was trained on BarBeR sources that overlap benchmark families;
- using it would make provenance and proprietary deployment difficult;
- localization still needs a decoder and ambiguity policy.

It remains the best licensed/provenance-cleared architecture to reproduce in a
future training program.

### 4. Independently trained tiny segmentation model

To test that direction fairly, a clean model was authored and trained from
scratch:

- 31,424 parameters;
- one grayscale 256-pixel overview;
- separate linear and 2D heatmaps;
- synthetic Code 39/128, EAN-8/13, QR, Data Matrix, PDF417, and Aztec scenes;
- blur, perspective, rotation, contrast, and multi-code composition;
- native-resolution rectified crop decoding;
- bounded LocalAverage, FixedThreshold, CLAHE, and unresolved-linear upscale
  stages.

Raw inference is approximately 0.95 ms. End-to-end results were:

| Dataset | Exact | Wrong | Median |
|---|---:|---:|---:|
| Easy arbitrary-angle holdout | 339/360 (94.17%) | 0 | 48.52 ms |
| DEAL real | 1,258/2,000 (62.90%) | 57 (2.85%) | 57.43 ms |
| QuickBrowser real | 1,908/2,498 (76.38%) | 35 (1.40%) | 21.39 ms |

The result is decisive: fast model inference did not produce a fast, safe
extractor. Synthetic-to-real proposal errors caused expensive crop escalation
and wrong-family reads. The model is retained as a negative result and training
baseline, but it is not registered in Codara.

### 5. Restoration, enhancement, and consensus

The previous literature tournament already proved that restoration and
multi-view consensus improve difficult-code recovery. They are intentionally
excluded here because they are exactly the work this speed tier promises not to
perform:

- more crops;
- more binarizers;
- upscaling;
- learned QR restoration;
- agreement/rejection bookkeeping.

Those techniques remain in 5.1/5.2 where they belong.

## Evaluation results

### Frozen support envelope

The easy holdout was generated with a fixed seed before final selection:

- 200 document-like pages;
- 360 exact payloads;
- 25 negative pages;
- broad linear and matrix families;
- large dark-on-light renderings;
- ordinary and oblique rotations.

For pages containing only 0 or ±90-degree symbols:

| Candidate | Exact | Wrong | Median | Verdict |
|---|---:|---:|---:|---|
| One-pass ZXing fast engine | **123/132 (93.18%)** | **0** | **28.63 ms** | selected |
| Clean learned locator/crops | 122/132 (92.42%) | 0 | ~44–49 ms | rejected |

Across arbitrary angles, one-pass ZXing reaches 302/360 (83.89%) with three
wrong image-level reads. The clean model reaches 339/360 but is slower and fails
the real-data safety gate.

### Real guardrails

The broad real datasets are intentionally outside the “known easy document”
promise. They demonstrate why 5.3 must be opt-in:

| Engine/configuration | DEAL | QuickBrowser |
|---|---:|---:|
| Selected one-pass, all formats | 1,332/2,000, 41 extraneous/wrong | 1,869/2,498, 12 extraneous/wrong |
| Selected one-pass, retail formats | same exact, 15 extraneous/wrong | same exact, 11 extraneous/wrong |
| Existing direct ZXing with extra searches | 1,420/2,000 | 1,926/2,498 |
| Published BaFaLo adaptive cascade | 1,752/2,000 | 2,102/2,498 |

The first two rows use image-level set scoring, which counts every additional
payload on an image as wrong. Older neutral reports use one-to-one
localization-aware matching, so their wrong counts are not directly comparable.
The exact-recovery denominators are comparable.

### Quality Dossier

| Engine | Exact |
|---|---:|
| Project 5.1 adaptive quality tier | **43/43** |
| Project 5.3 fast tier | 28/43 |

This is not a regression because 5.3 does not replace 5.1. It gives users with
simple pages an explicit way to avoid recovery work.

## Safety decisions

- 5.1 remains the default.
- The fast tier never automatically falls back.
- Only valid ZXing results are returned.
- ITF payloads shorter than six characters are rejected.
- Known formats should be declared whenever possible.
- An empty result means “not decoded by the fast contract,” not “the page
  contains no barcode.”
- The UI description names the easy-page assumptions and directs difficult
  documents to the Adaptive extractor.

## Implementation

Standalone selected engine:

- [fast_direct.py](fast_direct.py)
- [run.py](run.py)

Codara adapter and registry:

- [extraction_fast.py](../../../src/barcode_detection/integrations/codara/engines/extraction_fast.py)
- [engine registry](../../../src/barcode_detection/integrations/codara/engines/registry.py)

Tests pin:

- fast engine availability;
- adaptive engine remains default;
- valid-only single-pass diagnostics;
- all-format fallback for direct callers;
- format and ROI restrictions;
- engine capability separation.

The backend suite passes 100 tests and 17 subtests.
