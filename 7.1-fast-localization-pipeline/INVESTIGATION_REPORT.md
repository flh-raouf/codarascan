# Fast localization investigation

Date: 2026-07-28

## Outcome

The selected speed engine is an independently authored 31,424-parameter
segmentation model operating on a 256-pixel grayscale overview. It returns
oriented 1D/2D regions and never decodes.

It does not replace pipeline 7:

- `fast-localizer` is the opt-in low-latency tier for clear, large symbols;
- `p7-localizer` remains the high-precision tier for dense documents, small
  codes, and conservative separation.

After the private-evaluation-corpus audit, the selected model also rejects components smaller
than 60 overview pixels. This raised frozen-holdout F1 from 95.99% to 96.65%
on clear pages, 70.27% to 72.11% on Kaggle, 83.18% to 85.58% on InventBar,
and 67.04% to 71.81% on ParcelBar.

## Competitors

### ZXing geometry

Two modes were tested:

- valid reads only;
- valid reads plus damaged/error geometry.

The error-geometry mode was the stronger localization candidate, but on frozen
holdouts it remained below the selected model:

| Holdout | ZXing P/R/F1 | Selected model P/R/F1 | ZXing median | Model median |
|---|---:|---:|---:|---:|
| Clear/easy | 93.03 / 85.28 / 88.99 | 95.59 / 96.39 / 95.99 | 23.48 ms | 1.71 ms |
| Kaggle test | 82.30 / 49.17 / 61.56 | 78.04 / 63.90 / 70.27 | 3.35 ms | 1.22 ms |
| InventBar test | 82.95 / 80.22 / 81.56 | 72.36 / 97.80 / 83.18 | 6.17 ms | 1.43 ms |
| ParcelBar test | 56.07 / 30.15 / 39.22 | 53.61 / 89.45 / 67.04 | 16.81 ms | 1.60 ms |

Evidence-preserving union added only 31 Kaggle matches, increased false
predictions by 44, and moved F1 from 70.27% to 70.64%. On easy pages it reduced
precision and increased median latency to 25.35 ms. ZXing was therefore
rejected for this localization-only speed tier.

ZXing-C++ remains valuable: its implementation is Apache-2.0, thread-safe, and
supports broad 1D/2D reading. It is the correct choice when a payload is wanted,
as demonstrated by Codara's separate fast extraction engine.

Primary source: [ZXing-C++](https://github.com/zxing-cpp/zxing-cpp).

### Published BaFaLo checkpoint

BaFaLo remains the strongest technical reference. The published checkpoint
measured about 3.4 ms here and reached:

- Kaggle test: 91.89% precision, 67.60% recall, 77.89% F1;
- InventBar: 90.24% precision, 100% recall, 94.87% F1;
- ParcelBar: 90.32% precision, 66.91% recall, 76.87% F1.

It was not selected for Codara because its released repository is AGPL-3.0 and
the checkpoint was trained on BarBeR sources overlapping several benchmark
families. It remains the target accuracy level for future clean training.

Primary sources: [BaFaLo paper](https://federicobolelli.it/media/publications/pdfs/2025caip.pdf),
[BarBeR benchmark paper](https://federicobolelli.it/media/publications/pdfs/2025eaai.pdf).

### Pipeline 7

Pipeline 7 remains much more conservative:

- Quality, VN, balanced, and stress document tests: no unmatched locations;
- full Kaggle v8: 74.3% precision, 35.0% recall, 47.6% F1;
- approximately 97 ms/image on Kaggle;
- approximately 160 ms sequential average on the dense internal document
  corpus.

The selected model's Kaggle test result is stronger in both recall and F1 at
roughly 1–3 ms, but it cannot reproduce pipeline 7's zero-false-positive
document behavior.

### Classical and library alternatives

- Quirc is exceptionally small and separates QR identification from decoding,
  but it is QR-only and cannot cover Codara's mixed symbologies.
  [Quirc source](https://github.com/dlbeer/quirc)
- ZBar provides mature 1D/QR scanning but is a decoder-oriented LGPL library,
  not a universal high-recall localization model.
  [ZBar API](https://zbar.sourceforge.net/api/)
- Classical texture localization remains useful and interpretable, including
  the Microsoft real-time system, but pipeline 7 already represents the
  strongest tuned classical path in this workspace.
  [Microsoft Research](https://www.microsoft.com/en-us/research/publication/automatic-real-time-barcode-localization-complex-scenes/)

## Model development

### Version 1: synthetic-only

The initial clean model had excellent raw inference speed but a substantial
synthetic-to-real precision gap.

### Version 2: frozen development fine-tuning

Fine-tuning used:

- generated easy-development pages;
- Kaggle v8 fold 0 only.

The selected 256/0.65 point improved broad development precision materially
without opening Kaggle folds 1–4.

### Version 3: broader disjoint training

Version 3 added deterministic training folds from InventBar and ParcelBar.
Their fold 4 images remained disjoint test data. Threshold and canvas stayed
frozen at 256/0.65.

This is the selected checkpoint.

### Pipeline-7 distillation experiment

Pipeline 7 generated high-precision labels on development folds of Quality,
balanced, stress, synthetic 1D, and synthetic 2D documents. The student was
fine-tuned to reproduce those regions and empty pages.

At 320–640 pixels this improved aggregate document recall, but visual QA on
held-out dense dossier pages revealed large table/text false regions. The
document model was rejected from production. Distillation remains promising,
but a future model needs explicit hard-negative instance mining and
boundary-aware training rather than only page-mask imitation.

This conclusion agrees with current localization-distillation literature:
student accuracy depends on transferring location structure, not merely output
confidence. [Localization Distillation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Localization_Distillation_for_Object_Detection_CVPR_2022_paper.pdf)

## Architecture rationale

The selected model uses:

- one grayscale input;
- three early strided convolutions;
- a low-resolution context branch;
- one high/low feature fusion;
- pixel shuffle to produce linear and matrix masks.

The architecture minimizes high-resolution activation maps because CPU latency
is determined by memory movement and operator implementation, not parameter
count alone. This matches the CPU-detector literature's warning that FLOPs do
not directly predict real latency.

Primary source: [RefineDetLite, CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Chen_RefineDetLite_A_Lightweight_One-Stage_Object_Detection_Framework_for_CPU-Only_Devices_CVPRW_2020_paper.html).

## Deployment recommendation

Expose an explicit choice:

- Fast localizer — clear, large barcodes; lowest latency; misses and some false
  regions are accepted trade-offs.
- Coarse-to-fine localizer — small/dense/difficult documents; higher latency;
  conservative native-pixel evidence.

Do not silently fall back from the fast engine to pipeline 7. A hidden fallback
would destroy the latency contract the user selected.

## private-evaluation-corpus audit

The original 256/0.65/14 fast model returned 512 regions on the 315-page VN
Lot 6 PDF. Full pipeline 7 returned 41 regions: 33 linear and eight matrix.
Only six model regions overlapped pipeline-7 evidence.

The following alternatives were rejected:

- increasing the canvas to 320-640 pixels raised VN recall but multiplied
  document-text false regions;
- four-tile inference took roughly 4.2 ms but produced hundreds of regions;
- valid-only ZXing produced no holdout locations at roughly 32 ms median;
- ZXing error geometry found 1/9 with 14 false regions;
- published BaFaLo found 2/9 with 14 false regions;
- native structural verification reduced false regions but had pathological
  p95 latency;
- a VN-only student reached 7/9 with three false regions on its temporal
  holdout but catastrophically forgot InventBar and ParcelBar;
- mixed-domain distillation preserved public holdouts but did not generalize
  to unseen VN barcode styles.

The production-safe VN tier is therefore pipeline 7 with `kinds=linear`. On
the complete PDF it returned all 33 linear teacher locations, no extra regions,
21 ms/page sequential latency, and 10 ms/page four-worker throughput. This is
registered in Codara as `p7-linear-fast`. It is explicitly blind to the eight
Data Matrix symbols; mixed-format VN processing must continue using full
pipeline 7.
