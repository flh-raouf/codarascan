# Regression report

Date: 2026-07-22

All measurements below use the repository virtual environment and the final
source in this folder. Dataset archives can change; the downloaded Kaggle
revision contained 952 images and 1,696 annotated objects.

## private evaluation document

| Measure | Result |
|---|---:|
| Pages | 21 |
| Exact payloads | 43/43 |
| Unexpected payloads | 0 |
| Unresolved boxes | 0 |
| Review boxes | 0 |

The Codara backend copy and the standalone 5.1 pipeline both passed the exact
payload regression. Page 21 was visually inspected and retains all nine tight,
oriented boxes, including the Data Matrix.

## Kaggle Barcode and QR

Matching is one-to-one. The permissive metric requires IoU >= 0.5 or
intersection-over-smaller >= 0.5. Strict matching requires IoU >= 0.5.

| Measure | Project 5 behavior | 5.1 adaptive | Change |
|---|---:|---:|---:|
| Annotated objects | 1,696 | 1,696 | - |
| Permissive matches | 875 | 932 | +57 |
| Permissive recall | 51.59% | 54.95% | +3.36 points |
| Strict IoU matches | 642 | 593 | -49 |
| Predictions | 1,216 | 1,373 | +157 |
| Unassigned predictions | 341 | 441 | +100 |

The final conservative change keeps short undecoded linear candidates hidden;
they may unlock recovery but cannot become user-visible review boxes. The 5.1
gain consists of eight additional linear objects and 49 additional QR objects.

The strict-IoU metric decreases because 5.1 deliberately retains tighter
oriented barcode geometry while the Pascal VOC ground truth contains loose,
axis-aligned rectangles. The containment-aware metric is therefore the primary
localization measure for Codara. The strict result is still reported because it
shows that 5.1 is not universally better under every box-scoring convention.
The 100 additional unassigned predictions are not automatically false
positives: manual inspection found real symbols missing from some annotations,
but the dataset does not provide enough information to classify every one.

### Recall by annotated long side

| Long side | Project 5 | 5.1 adaptive | Change |
|---|---:|---:|---:|
| Under 60 px | 16/543 | 30/543 | +14 |
| 60-99 px | 100/205 | 112/205 | +12 |
| 100-199 px | 174/250 | 185/250 | +11 |
| 200-399 px | 321/388 | 338/388 | +17 |
| 400 px or larger | 264/310 | 267/310 | +3 |

The largest relative weakness remains the 543-object under-60-pixel bin: only
30 are localized. This is the strongest evidence that the result must not be
described as universal. Many QR samples in that bin contain too few source
pixels for their module grid, and additional upscaling cannot recreate it.

## Hugging Face dataset suitability

The supplied `amaye15/Barcode` dataset was audited but is not a localization
benchmark: its published schema contains an image and one image-level class
label selected from `Barcode`, `Invoice`, `Object`, `Receipt`, and
`Non-Object`, but no bounding box or polygon field. It therefore cannot measure
IoU or object-level localization recall. A downloaded 20-image positive sample
also exposed noisy labels, including ordinary application and phone UI images
without a visible barcode. The dataset page reports 65,005 rows and 27.7 GB, so
downloading the complete archive would add substantial cost without creating
valid localization ground truth. It remains useful only after a representative
subset is manually annotated and label-audited.

## Synthetic suites

| Suite | Result |
|---|---:|
| Synthetic 1D | 86/86 exact, zero extras/review |
| Synthetic 2D | 86/86 exact, zero extras/review |
| Balanced mixed | 121/122 localized, zero unmatched boxes |
| Extreme mixed | 163/217 localized, zero unmatched boxes; 136 exact payloads |

The extreme set retains project 5's existing Code 39 ambiguity:
`MX-08-I4-5311` instead of ground-truth `MX-08-04-5311`. Code 39 does not
require a checksum, so the decoder cannot prove which character is correct.
The 5.1 pipeline did not introduce this behavior.

## Controlled small-image checks

- At roughly one-third dossier resolution, exact decodes increased from 21 to
  24. This does not mean every missing symbol became recoverable; many modules
  had already lost source information.
- On a 416 x 416 Kaggle image containing an annotated 40 px QR code, the legacy
  app path returned no location. The adaptive app path created a checksum-error
  QR anchor at 3x, mapped it to the original 21.7 x 18 px region, and retained
  it as a structurally verified unresolved QR candidate.
- The same API response includes input dimensions, contrast, sharpness, scales,
  and recovery-stage diagnostics.

## Dense barcode catalogue check

The supplied 623 x 492 barcode-symbology chart exposed a different failure:
OpenCV grouped nearby symbols into very wide row-level candidates. Before the
dense-layout refinement it returned one decoded QR code and fourteen coarse
linear review boxes. The final implementation returns the QR decode and
twenty-one tighter linear review locations, splitting the three confirmed
multi-symbol regions at their quiet zones.

The splitter activates only when at least twelve independent proposals establish
a dense barcode sheet. An initial generic version split a valid page-17 symbol;
the page-context requirement eliminated that regression. The private evaluation document
returned to 43/43 exact with no review boxes, and the synthetic 1D suite remained
86/86 exact.

## Coloured EAN-13 sheet check

The supplied 508 x 294 coloured sheet contains twelve copies of one EAN-13
symbol. All twelve are localized. One decodes directly, three more pass the
new checksum-constrained 95-module template verifier, and eight remain honest
resolution-limited review candidates. Their bar fields occupy roughly 86--88
pixels for 95 modules, so several distinct checksum-valid patterns remain
plausible after rasterization. The pipeline deliberately refuses to copy the
human-readable digits into those eight results.

## Known boundaries

- Upscaling changes sampling behavior but cannot recreate missing modules.
- The strongest fallback validation remains Code 39, Code 128, QR Code, and
  Data Matrix. Other ZXing formats need symbology-specific benchmark coverage.
- Adaptive small-image processing is slower than the project-5 native route.
  Two to four workers are recommended; oversubscription is counterproductive.
- Direct-part marking, severe glare, curvature, and sub-pixel modules remain
  outside a universal accuracy guarantee.
