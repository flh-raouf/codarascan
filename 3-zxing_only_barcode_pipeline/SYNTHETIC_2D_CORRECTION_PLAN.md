# Synthetic 2D dossier - detector audit and correction plan

## Scope

This is a read-only audit of the current run in:

`zxing_only_barcode_pipeline/output/syntehtic-2d-dossier`

It is compared against:

`output/pdf/Synthetic private evaluation document - 2D barcode ground truth.csv`

No Python source was changed during this audit.

## Executive conclusion

The current run is much closer to correct than the overlays initially suggest:

- Ground truth: 86 symbols - 56 Data Matrix and 30 QR Code.
- Correct decoded results: 82.
- Incorrect decoded payloads: 0.
- Genuine misses: 4 Data Matrix symbols, on pages 11, 15, 22, and 33.
- Unresolved false localization candidates: 2, both on page 31.
- Decoded-only precision: 100%.
- Decoded recall: 95.35%.
- If unresolved candidates are treated as detections, localization precision is 97.62%, recall is 95.35%, and F1 is 96.47%.

The dominant recall problem is not a lack of image information. ZXing's matrix
downscale search is disabled in the current pipeline. A read-only experiment
with matrix-only downscale search recovered 85/86 symbols. A bounded local retry
around ZXing's invalid candidate position recovered page 11 as well, reaching
86/86 with no unexpected payloads.

The page 31 false positives come from the 1D locator running on a 2D-only test.
It mistakes long table columns for unresolved linear barcodes. These are not
decoder hallucinations: the real QR code is decoded correctly and the two extra
regions have no payload.

## Baseline evidence

### Timing

Current 50-page processing time, excluding extraction: 52.83 seconds.

| Stage | Total seconds | Contribution |
|---|---:|---:|
| Whole-page ZXing | 1.82 | 3.5% |
| 1D localization | 28.06 | 53.1% |
| 1D candidate decoding | 1.40 | 2.7% |
| Matrix proposal recovery | 17.44 | 33.0% |

The matrix recovery stage merged with 66 already decoded anchors but added zero
new unique decodes. The 1D path added only the two page 31 false candidates.

### Matrix downscale experiment

The experiment used the existing extracted page images and did not modify the
pipeline:

| Matrix whole-page portfolio | Correct unique decodes | Unexpected | Time |
|---|---:|---:|---:|
| LocalAverage, downscale enabled | 84/86 | 0 | 1.27 s |
| LocalAverage + GlobalHistogram, downscale enabled | 85/86 | 0 | 1.76 s |
| Add FixedThreshold | 85/86 | 0 | 2.16 s |

FixedThreshold added cost and invalid candidates but no recall. A local 500 px
context retry, 1.5x enlargement, and downscale-enabled matrix decode around
invalid ZXing positions recovered the final page 11 symbol. The complete staged
experiment reached 86/86 with zero unexpected values in 1.77 seconds, excluding
PDF extraction and artifact writing.

### Existing deep recovery option

`--deep-matrix-recovery` was tested on pages 11, 15, 22, and 33. It recovered
page 15 only. Pages 11, 22, and 33 still failed. Enabling the current deep mode
globally is therefore not the right primary correction.

## Failure analysis by page

| Page | Symbol | Visual condition | Current failure | Read-only recovery result |
|---:|---|---|---|---|
| 11 | Data Matrix, about 156 x 156 px | Low contrast, blur, 13-degree rotation | No output result. Downscale-enabled ZXing localizes it but reports `ChecksumError`. | A context crop enlarged 1.5-2x and decoded with matrix downscale enabled returns the exact payload. |
| 15 | Data Matrix, about 180 px, near -90 degrees | Good contrast but vertical and close to form structure | Normal pass finds the QR code only. The relevant vision proposal is outside the normal top-14 region set. | GlobalHistogram with matrix downscale enabled decodes it on the whole page. Existing deep mode also recovers it, but less efficiently. |
| 22 | Data Matrix, about 96 x 96 px | Small, faded, vertical, close to the right form edge | The correct matrix proposal is present around rank 8, but the decoder sampling fails. | LocalAverage with matrix downscale enabled decodes it directly. |
| 33 | Data Matrix, about 181 px, 21 degrees | Faded/blurred; another Data Matrix on the page is easy | The correct proposal is around rank 17 and is excluded by the top-14 limit. | LocalAverage with matrix downscale enabled decodes it directly. |

These are sampling and proposal-priority failures, not corrupted ground truth.

## Page 31 false-positive analysis

Page 31 contains one real QR code. It is decoded correctly. Two additional
unresolved candidates cover long sections of the measurement table.

| Candidate | AABB | Structural score | Dark fraction | Transitions | Scanline agreement |
|---|---|---:|---:|---:|---:|
| First table column region | 326 x 1114 px | 0.6240 | 0.0234 | 26 | 1.0000 |
| Second table column region | 321 x 836 px | 0.6174 | 0.0252 | 20 | 0.9859 |

The current threshold is 0.52. Long grid lines produce excellent persistence
and scanline agreement, so the total score passes. However, only 2.3-2.5% of
the rectified signal is dark. That is table-line sparsity, not a barcode bar
field. `dark_fraction` contributes only 3% to the score and is not a hard
acceptance condition. Both candidates then receive the expensive advanced 1D
decode portfolio and remain unresolved.

Do not solve this by only increasing the global score threshold. That would
hide this example but can remove faint true barcodes. Use format routing and
physical rejection features instead.

## Output-folder integrity problems

The result directory is not a clean snapshot:

- `detections.json` contains 84 current results, but the crop directory contains
  88 files.
- Four crops are stale 1D barcode crops from an earlier run: page 11 barcode 1,
  page 15 barcode 2, page 22 barcode 1, and page 33 barcode 2.
- All `source_image` and `overlay` paths inside the JSON point to
  `syntehtic-dossier`, not the current `syntehtic-2d-dossier` location. This is
  consistent with moving/renaming the result folder after the run.
- The folder name itself misspells `synthetic` as `syntehtic`.

This does not cause the four decoder misses, but it makes manual review and
automated regression unreliable. A crop's presence cannot currently be treated
as proof that it belongs to the current manifest.

## Root causes in the current design

### 1. Matrix scale search is disabled

Whole-page and context matrix reads use `try_downscale=False`. Three misses are
recovered by enabling it on a matrix-only pass, and the fourth is recovered by
using it on a local context crop.

### 2. The matrix proposal ranking favors fragments

The current response is based on bidirectional Scharr energy. Each connected
component is scored by its mean response. Small, sharp text fragments and table
intersections can therefore outrank a complete faded matrix symbol.

- Forty-five of fifty pages hit the maximum of 14 proposals.
- The run processed 694 matrix proposals.
- Page 15's relevant region is around rank 20.
- Page 33's relevant region is around rank 17.
- Fixed 70 px center suppression is not scaled to symbol size.
- Already decoded regions are not suppressed before proposal recovery.
- Sixty-six context recoveries merely duplicate whole-page results.

### 3. A 2D-only workload runs the full 1D proposal stack

The 1D locator consumes more than half of the processing time and creates the
two page 31 false candidates. The pipeline needs an explicit format mode rather
than always running every localization path.

### 4. Undecodable matrix proposals disappear

The matrix proposal stage emits a result only after a valid decode. It has no
`unresolved_matrix` output. Page 11 can be localized as an invalid Data Matrix
with a checksum error, but the current output records nothing. This conflates
"not localized" with "localized but not decoded."

### 5. Primary results and review candidates are mixed

The page 31 candidates are labeled `unresolved`, but they are stored and drawn
alongside accepted decoded results. Consumers can reasonably interpret them as
detected barcodes. Review proposals need a separate output tier.

## Recommended target pipeline

### Stage A - input and run integrity

1. Extract the native raster as today.
2. Record input SHA-256, page dimensions, command/configuration, code version,
   and run ID.
3. Write into a new temporary run directory and atomically rename it only after
   validation succeeds.
4. Refuse a non-empty output directory unless an explicit overwrite option is
   provided.
5. Store artifact paths relative to the run root.

### Stage B - format-aware fast decoding

Add an explicit format policy: `2d`, `1d`, or `all`.

For `2d`:

1. Run a matrix-only ZXing whole-page pass with rotation and inversion enabled.
2. Enable downscale search.
3. Use LocalAverage, then GlobalHistogram only for still-unresolved content.
4. Merge by payload, format, and polygon overlap.
5. Do not run the 1D locator.

For `all`, run the 1D path independently after the matrix path. Never let a
linear proposal consume matrix proposal budget.

### Stage C - error-guided matrix recovery

Use `return_errors=True` during the matrix fast pass.

1. Accept valid checksum-protected decodes immediately.
2. Preserve positions for `ChecksumError` and `FormatError` results as
   high-priority matrix candidates.
3. Around each invalid position, create one bounded context crop with adequate
   quiet-zone padding.
4. Perspective-rectify if the decoder supplied a quadrilateral.
5. Try 1.5x and then 2x enlargement with downscale enabled, using LocalAverage
   and GlobalHistogram.
6. Stop on the first valid payload.

This path directly addresses page 11 without scanning hundreds of arbitrary
windows.

### Stage D - residual matrix localization

Run visual matrix proposals only in regions not explained by decoded or invalid
ZXing anchors.

Improve proposal quality before increasing the proposal count:

1. Suppress expanded decoded/invalid anchor regions.
2. Cluster multi-scale fragments that belong to the same physical square.
3. Replace mean-response ranking with a score that includes:
   - bidirectional energy balance;
   - square or near-square geometry;
   - occupied area rather than a tiny hot fragment;
   - cell/grid periodicity;
   - contrast relative to local paper;
   - Data Matrix border or QR finder-pattern evidence.
4. Use size-relative non-maximum suppression instead of a fixed 70 px radius.
5. Reserve proposal slots spatially, for example global top candidates plus a
   small quota per page tile, so one dense table area cannot consume the budget.
6. Derive a candidate quadrilateral, not only a center point.

### Stage E - independent matrix verification

Decoding must not be the only way to say a matrix symbol was localized.

For a rectified candidate:

- Data Matrix: look for two adjacent solid borders and alternating timing
  patterns on the opposite borders.
- QR Code: look for the three finder patterns and their expected geometry.
- Both: check square module periodicity, bidirectional transition density, quiet
  zone where available, and consistent cell pitch.

If geometry is strong but decoding still fails, emit `unresolved_matrix` with a
candidate confidence and error evidence. Do not invent a payload.

### Stage F - linear false-positive control

In `2d` mode, the linear path is disabled, which removes page 31 immediately.

In `1d` or `all` mode:

1. Keep decoded 1D symbols as accepted results.
2. Put undecoded linear proposals in a review-only collection by default.
3. Add physical guards for undecoded proposals:
   - minimum dark/ink occupancy in the central bar band;
   - minimum transition density after trimming quiet margins;
   - reject very large white-run gaps relative to median run width;
   - require transitions to occupy a broad, barcode-like portion of the long
     axis;
   - measure table-grid continuity and intersections in surrounding context.
4. Validate thresholds on the original real dossier and a negative set of
   tables before adopting them. Page 31's 2.3-2.5% dark fraction provides a
   strong negative example, but it should not be the only calibration sample.

### Stage G - result contract

Split outputs into three explicit tiers:

1. `decoded`: checksum/format-valid payload and position.
2. `localized_unresolved_matrix`: independently verified 2D position with no
   payload.
3. `review_candidates`: low-confidence or unresolved 1D/2D proposals.

Use different overlay colors and summaries. Do not count review candidates as
detected barcodes in the primary precision metric.

Correct the diagnostics as part of this work. The current
`accepted_linear_proposals` value is `len(located)`, which can include decoded
matrix anchors and is therefore not a true linear count.

## Benchmark and regression plan

### Ground truth

The current CSV is sufficient for exact payload recall, but not for evaluating
undecoded localization. Its `source_x` and `source_y` are pre-composition sheet
coordinates and it does not contain the final page-space quadrilateral.

For future synthetic dossiers, record:

- final page-space quadrilateral;
- page-space AABB;
- symbol dimensions in pixels;
- module size estimate;
- format, payload, rotation, and difficulty;
- all transforms applied during page composition.

Use one-to-one polygon matching for localization evaluation, then exact format
and payload matching for decoding evaluation.

### Required regression sets

1. This 50-page 2D dossier.
2. The 50-page synthetic 1D dossier.
3. The original real private evaluation document.
4. Negative table/form pages with no barcode.
5. Held-out noisy 2D pages not produced by the same generator settings.

### Acceptance gates

- Synthetic 2D dossier: 86/86 correct payloads, zero incorrect payloads, zero
  primary false positives.
- Page 31: one primary QR result; table columns may not appear in primary
  results.
- Pages 11, 15, 22, and 33: all expected Data Matrix payloads recovered.
- QR: 30/30; Data Matrix: 56/56; hard subset: 8/8.
- Output directory: no orphan crops or overlays; artifact counts agree with the
  manifest; all paths resolve from the run root.
- 1D and real-dossier regressions: no recall loss compared with their current
  accepted baselines.
- 2D decode stage performance target on this machine: under 5 seconds for 50
  native 200 dpi pages, excluding PDF extraction and image/artifact writing.

## Implementation order

1. **P0 - Output integrity and evaluator.** Clean/atomic run directories,
   relative paths, run manifest, exact payload evaluation, and orphan checks.
2. **P1 - Format routing.** Add `2d`/`1d`/`all`; skip 1D proposals in 2D mode.
3. **P1 - Matrix scale search.** Downscale-enabled matrix pass with
   LocalAverage and conditional GlobalHistogram.
4. **P1 - Error-guided retry.** Local context recovery around invalid ZXing
   matrix positions.
5. **P2 - Result tiers.** Separate decoded, verified-unresolved matrix, and
   review candidates.
6. **P2 - Linear physical guards.** Dark occupancy, run-spacing, transition
   coverage, and table-context rejection for undecoded linear candidates.
7. **P3 - Proposal redesign.** Decoded-region suppression, fragment clustering,
   size-aware NMS, spatial quotas, and finder/border evidence.
8. **P3 - Held-out regression and performance tuning.** Tune only after the
   evaluator and negative set exist.

## Changes that should not be used as the main fix

- Do not only raise `minimum_score`; page 31 exposes a feature-design problem.
- Do not only increase `matrix_proposal_limit`; it increases duplicate work and
  still ranks fragments above faded symbols.
- Do not enable the existing deep mode on every page; it recovered only one of
  four misses in the targeted audit.
- Do not apply every thresholding and sharpening variant globally; use invalid
  decoder positions and residual proposals to bound recovery.
- Do not discard all undecoded candidates; verified unresolved matrix locations
  are useful, but they must be separated from accepted decoded results.
