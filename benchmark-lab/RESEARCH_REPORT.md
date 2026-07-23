# Barcode localization and decoding: evidence review and tournament

Status: final selection report, 2026-07-23. Numerical claims marked **local
reproduction** were measured in this repository. Paper numbers are the
authors' reported results and are not assumed to be directly comparable.

## Executive conclusion

There is no defensible reason to replace the whole pipeline with any one
paper. The strongest design is a synthesis:

1. run a compact 1D/2D proposal model on a low-resolution image;
2. route empty, tiny, or uncertain cases to a higher-resolution detector;
3. map proposals back to the untouched source image and decode oriented,
   padded crops;
4. use symbology constraints, check digits/error correction, multi-view
   agreement, and an explicit ambiguous result rather than accepting the first
   syntactically valid string;
5. invoke learned restoration only on unresolved degraded QR crops.

This is the common useful core of Quenum's coarse-to-fine system,
Szentandrási's hierarchical pre-detector, Wachenfeld's full-resolution
recognizer, BLaDE's constrained probabilistic decoder, and QR-DN restoration.
The 2025 BarBeR benchmark independently supports the compact learned detector
choice.

What should be tried:

- **Yes:** compact learned localization, low-res-to-full-res routing,
  detector-gated crops, synthetic multi-scale training, QR restoration,
  checksum-constrained consensus and ambiguity rejection.
- **As references, not final dependencies:** BaFaLo and the public Piero
  YOLOv8s checkpoint. They are valuable empirical controls, but BaFaLo's code
  is AGPL-3.0 and its checkpoint was trained on BarBeR data that overlaps some
  evaluation sets.
- **No as primary methods:** hand-tuned structure tensors, Hough+MLP
  localization, classical QR line-grid recovery, or a fixed 13-digit
  end-to-end recognizer. Their ideas remain useful, but modern learned
  localization generalizes better.

## What was actually read

Full text or an author/open copy was obtained for the following:

- Vezzali et al., *State-of-the-art review and benchmarking of barcode
  localization methods* (2025), the [author PDF](https://federicobolelli.it/media/publications/pdfs/2025eaai.pdf).
- Quenum, Wang, Zakhor, *Fast, Accurate Barcode Detection in Ultra
  High-Resolution Images* (2021), [arXiv](https://arxiv.org/abs/2102.06868).
- Do and Kim, *Quick Browser: A Unified Model to Detect and Read Simple
  Object in Real-time* (2021), linked from the [maintainer's Zenodo record](https://zenodo.org/records/13586402).
- Do et al., *Smart Inference for Multidigit Convolutional Neural Network
  based Barcode Decoding* (2020/2021), [author arXiv version](https://arxiv.org/abs/2004.06297).
- Kamnardsiri et al., *A Deep Learning Method for Barcode Recognition for
  Inventory Management* (2022), [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9696533/).
- Zamberletti et al., *Neural Image Restoration for Decoding 1-D Barcodes
  Using Common Camera Phones* (2010).
- Zamberletti, Gallo, Albertini, *Robust Angle Invariant 1D Barcode Detection*
  (2013).
- Szentandrási, Herout, Dubská, *Fast Detection and Recognition of QR Codes
  in High-Resolution Images* (2012/2013).
- Dubská, Herout, Havel, the PClines/Hough QR localization and rectification
  paper associated with the Dubská datasets.
- Bodnár, Grósz, Tóth, *Efficient visual code localization with neural
  networks* (2018).
- Sörös and Flörkemeier, *Blur-resistant joint 1D and 2D barcode localization
  for smartphones* (2013).
- Tekin and Coughlan, *BLaDE: Barcode Localization and Decoding Engine*
  (2012 author technical report), plus the open precursor
  [*An Algorithm Enabling Blind Users to Find and Read Barcodes*](https://pmc.ncbi.nlm.nih.gov/articles/PMC2898214/).
- Wachenfeld, Terlunen, Jiang, *Robust Recognition of 1-D Barcodes Using
  Camera Phones*; an indexed [open manuscript](https://citeseerx.ist.psu.edu/document?doi=01242aaa4a606cee520a0f74e05b4e94d3a4b5a3&repid=rep1&type=pdf)
  supplied the complete method and results.
- Zhu et al., the VR panoramic QR detector paper.

The SmartEngines SPIE paper was not openly retrievable. Its conference
abstract/program and dataset metadata were examined, but claims that require
the missing body are not treated as established. QR-DN1.0 is a dataset record,
not a detector paper; its complete paired data and protocol were examined at
[Mendeley Data](https://data.mendeley.com/datasets/t2bdr663ms/2).

## Paper-by-paper analysis

### 1. Wachenfeld, Terlunen and Jiang

The method is deliberately structural and symbology-specific. It searches
scanlines for UPC-A/EAN-13/ISBN-13 guard and bar patterns, estimates module
width, normalizes the signal, and tests the fixed digit structure. Its most
important result is resolution sensitivity: reported recognition rises from
90.5% at 640×480 to 99.2% at 2592×1944 on 1,055 Muenster images, with
near-100% in the authors' daily-use experiment.

The paper is strong evidence against resizing the only decoding input. It is
not evidence for a universal locator: the supported family is narrow, the
later literature notes limited rotation tolerance, and its test device/domain
are old.

**Decision:** retain full-resolution decoding and structural validation; do
not reimplement this as the universal front end.

### 2. Zamberletti et al. 2010: neural restoration

A tiny 5-5 sliding-window MLP restores blurred 1D barcode imagery before a
standard decoder. It was trained from only 14 sharp/degraded pairs. On the
no-autofocus set the paper reports recall 0.70 versus 0.04 for the contemporary
ZXing baseline; stepping the window every five pixels gives 0.64 recall at
37.5 ms rather than 156.6 ms.

The enduring contribution is not that old MLP. It is the paired,
decoder-validated restoration experiment: visual similarity is irrelevant
unless the restored symbol decodes correctly.

**Decision:** try restoration only as a fallback and score exact payloads.
That design was reproduced on QR-DN1.0 below.

### 3. Zamberletti, Gallo and Albertini 2013

The pipeline applies Canny edges, tiled Hough transforms, a neural classifier
over angle histograms, and segment grouping. The paper reports orientation
accuracy around 0.83–0.86 and decoding improvements from 0.73 to 0.81 on
Muenster and 0.61 to 0.82 on rotated imagery, at roughly 270 ms in its
implementation.

The standardized 2025 BarBeR reproduction is much less favorable:
localization F1 is 0.278 at 640 pixels. The method has numerous coupled
thresholds and its old timing is not attractive.

**Decision:** do not pursue as a production locator.

### 4. Sörös and Flörkemeier

This elegant classical method uses structure-tensor eigenvalues: one dominant
orientation signals linear bars; two strong perpendicular directions signal a
matrix code. With a 7-pixel derivative neighborhood and 30-pixel aggregation,
the paper reports 82.5% on sharp Muenster images but 29.8% when blurred; QR
results are 81% on its own data and 42.6% on Dubská. Reported execution was
about 73 ms on a PC, 380 ms on a phone, or 118 ms with mobile GPU help.

Its assumptions fail with very small/large symbols, multiple competing
textures, or blur. Our three-scale clean-room structure-tensor implementation
had containment recall 0.822 but precision about 0.019, F1 0.037: the texture
signal is useful for proposals, not sufficient for acceptance.

**Decision:** reject as a stand-alone detector; possibly use its anisotropy as
an auxiliary feature or hard-negative generator.

### 5. Szentandrási, Herout and Dubská

This is an early coarse-to-fine success. Tiled HOG and a quadtree-like
hierarchy cheaply reject most high-resolution regions before the expensive QR
line detector and ZBar decoder. On 251 codes the paper reports 92.8%
localization and 70.1% decoding, versus 74.8% decoding for direct ZBar; under
blur, localization/decoding fall to 80.4%/41.2%. Reported runtime is 54 ms for
1080p.

**Decision:** use the routing principle, not the HOG classifier.

### 6. Dubská, Herout and Havel

The method maps image lines into a parallel-coordinate representation
(PClines), detects the QR grid/finder geometry in Hough-like space, estimates
the projective transform, and samples modules. On the authors' planar QR data,
97.4% of cases have at least 95% correct modules versus 60.9% for the then
current ZXing, with a reported 14.6 ms runtime.

Its failure modes are small codes, non-planar surfaces and strong structured
backgrounds. Modern decoders have also changed substantially since the
baseline.

**Decision:** do not make it the default. Projective rectification remains a
useful rescue operation when finder/grid confidence is high.

### 7. Bodnár, Grósz and Tóth

The proposed detector classifies circularly sampled pixel blocks, including a
DCT-domain variant designed to avoid full JPEG decompression. A deep
rectifier network uses three 1,000-unit hidden layers. The reported Muenster
F-measure is 0.8865 and Jaccard 0.8204.

The compressed-domain idea is genuinely interesting for bandwidth- or
decode-bound servers. The patch classifier is scale-specific and a modern
fully convolutional segmentation head is a cleaner generalization.

**Decision:** revisit only if JPEG decode cost dominates the target deployment.

### 8. Quenum, Wang and Zakhor

The first stage reduces an ultra-high-resolution image to 256×256 and proposes
regions; source-resolution crops are then segmented at 400×400 by Y-Net, whose
parallel regular, dilated and pyramid-pooling branches exploit barcode
texture at several scales. The authors generate 100,000 low-resolution and
100,000 UHR synthetic scenes with multiple symbologies.

Reported synthetic COCO mAP is 0.937 at 16 ms on a V100, versus 0.882/40.5 ms
for YOLOv4 and 0.466/94.8 ms for Mask R-CNN. On Muenster and Arte-Lab, reported
detection recall is 1.0. The paper honestly notes that overlapping symbols are
merged. Its real industrial UHR corpus is private, and synthetic evaluation is
the largest validity risk.

**Decision:** this is the most useful architectural paper. Use proposal/crop
routing, dilated multi-scale features and synthetic scenes, but validate on
unseen real datasets.

### 9. Do and Kim: Quick Browser

Quick Browser shares a backbone between a YOLO-like box head and 13 digit
classification heads, with the EAN-13 checksum enforcing the final sequence.
The ResNet version reports 94.38% mAP, 86.9% recognition and 127 ms; MobileNet
reports 91.62%, 75.35% and 49 ms on the authors' 2,000-image test. The 2021
paper reports historical commercial comparisons: Dynamsoft 76.3% recognition
at 375 ms, Cognex 66.75% at 115 ms, Inlite 50.55% at 218 ms, and then-current
ZXing 26.9% at 53 ms. These are not current SDK results.

Recognition exceeds 90% once barcode area is above roughly 2% of the image.
This directly supports the local router's use of relative detected area.

The main limitation is severe: it is an EAN-13 recognizer, not a general
barcode/2D-code pipeline. The public Zenodo record also warns of up to 25
dataset label errors; our checksum audit found 2 invalid EAN-13 labels in the
2,500-image `real` archive and excluded them from payload scoring.

**Decision:** use shared localization/recognition and checksum constraints as
ideas. Do not adopt a fixed 13-head decoder.

### 10. Kamnardsiri et al.

The paper compares EfficientDet, Faster R-CNN, RetinaNet, YOLOv5 and YOLOX on
InventBar (527 images) and ParcelBar (844), using rotation augmentation and a
416-pixel input. YOLOv5 has the best reported mAP@[.5:.95]: 0.873 on InventBar
and 0.918 on ParcelBar; Faster R-CNN gives 0.827/0.854.

The work is useful because ParcelBar contains dense logistics labels and
InventBar is modern high-resolution mobile imagery. It does not establish
cross-domain generalization: models are trained and tested within each small
dataset, decoding is not the central metric, and tables emphasized as timing
include training rather than a clean deployment inference comparison.

**Decision:** use both datasets, rotated augmentation and a compact detector;
do not infer universal performance from the in-domain mAP.

### 11. Do et al.: Smart Inference

This paper makes the checksum idea operational for a 13-head EAN-13 CNN.
Instead of taking each digit's top class, it ranks positions by the gap between
their first and second probabilities, substitutes alternatives at the least
certain positions, and tests checksum-valid combinations. It also evaluates
two rotation test-time-augmentation policies: stop at the first valid result,
or collect/vote across views.

The training curriculum is unusually substantial: 250,000 synthetic samples
cover darkness, occlusion, perspective, cylindrical warp, blur, noise and
combinations, followed by 20,000 augmentations of 500 real images. The held-out
2,000-image real test is drawn from a 2,500-image collection spanning
Muenster, Arte-Lab and 1,037 newly captured supermarket images. The authors
report ResNet50 accuracy improving from 93.35% to 95.85% with the best smart
inference setting. More alternative substitutions eventually reduce accuracy
and increase errors. Full voting does not consistently beat the fast-track
policy. A distilled MobileNetV2 reaches 90.85% and 34.2 ms on Jetson Nano.

This is strong independent support for three choices made in our experiments:
synthetic-first curriculum, checksum-aware uncertainty search, and conditional
augmentation rather than exhaustive voting on every input. It is still
EAN-family/fixed-length specific and the paper acknowledges false valid
predictions.

**Decision:** yes. The local adaptive two-view decoder is the library-decoder
analogue: accept agreement on easy crops, escalate uncertain cases, and retain
an ambiguous state.

### 12. QR-DN1.0

This is paired real degradation data: QR images are embedded into 30 color
hosts, photographed under varying cameras/screens/light, then recovered by
three extraction procedures. It provides 6,750 degraded 512×512 images and a
clean target for each, split into three official 750-image test groups.

The source metadata is internally inconsistent about some training counts,
so the files rather than prose define the local manifest. It is also a
watermark-extraction domain, not ordinary photographed labels.

**Decision:** excellent for a controlled restoration ablation; insufficient as
the only QR test.

### 13. SmartEngines synthetic data

The accessible abstract describes 51,480 synthetic 512×512 samples covering
Codabar, Code 39/93/128, EAN-13, UPC-A and UPC-E, with geometry, noise,
brightness and illumination augmentation and a ZXing comparison. The linked
download is dead and the full SPIE text was inaccessible.

**Decision:** reproduce the broad synthetic curriculum, but do not quote
unverifiable performance or depend on the lost archive.

### 14. BLaDE / Tekin and Coughlan

BLaDE first finds regions with many roughly parallel edges and enough
orientation entropy, then scans across the inferred bar direction. Its
UPC-A decoder is the most valuable part: guard and symbol boundaries form a
Bayesian deformable template; Viterbi inference incorporates digit evidence,
geometry and the checksum as state. It rejects a result if the optimum is too
ambiguous or too inconsistent with independent digit evidence.

This addresses a production risk ignored by most localization papers: a valid
but wrong payload can be worse than no result.

**Decision:** implement the policy—symbology-conditioned candidates,
multi-view evidence, checksum constraints and rejection. A complete new
Viterbi decoder for every symbology is not currently justified over mature
decoding libraries.

### 15. VR panoramic QR detection

The paper trains VGG16 Faster R-CNN on a 2,466-image aggregate (1,742 train,
724 test), resizing the shortest side to 600 and using rotation/flip
augmentation. It reports mAP 0.886 and mAR 0.811, ahead of its classical and
older YOLO baselines.

It supplies no persuasive decoding or deployment-speed evaluation, uses
horizontal boxes for heavily rotated codes, and its named panorama dataset is
now lost.

**Decision:** no. A current compact detector with oriented crop decoding is a
better-controlled experiment.

### 16. The 2025 BarBeR review and benchmark

This is the most comparable published evidence. BarBeR standardizes 8,748
images / 9,818 objects: 8,062 linear and 1,756 matrix codes, with 1,722 marked
undecodable. At 640 pixels and IoU 0.5, classical linear F1 values reported
include Gallo 0.533, Sörös 0.658, Zamberletti 0.278, Yun 0.757 and Zharkov
0.823. Reported learned F1 is approximately 0.987–0.993. For 2D, Sörös is
0.140, Zharkov 0.804 and the learned models about 0.975–0.985.

In a joint two-class task, YOLO nano reaches mAP50 0.973 and
mAP50:95 0.906; YOLO medium 0.982/0.920; RT-DETR 0.981/0.922. At 320 pixels,
YOLO nano remains 0.961/0.866, but small-object AP falls to 0.627. The practical
message is that compact low-resolution inference is an excellent default but
requires a small-code escape route.

**Decision:** use this as the external anchor and report overlap/leakage
whenever a BarBeR-trained checkpoint is tested on a constituent dataset.

## Current literature beyond the linked repository

The repository is a valuable dataset index, but it necessarily misses newer
work. A separate search through the current literature found four relevant
additions.

### EgoQR (2024)

[*EgoQR*](https://arxiv.org/abs/2410.05497) is unusually close to the
architecture selected here: a thumbnail detector proposes regions, untouched
full-resolution crops are tried under multiple preprocessing and
super-resolution hypotheses, and failures are preserved rather than replaced
by guessed strings. On the authors' private/internal 528-image, 697-code set,
they report 94.4% detection, 70.82% conditional decoding and 66.86%
end-to-end success. Their table reports 17% for ZXing, 42% for ZBar/qreader,
44% for WeChat, 50% for Dynamsoft, and 66% for their system; super-resolution
adds roughly two absolute points.

The dataset and implementation are not public, so these are not reproducible
controls. The work nevertheless provides independent, recent validation of
thumbnail-to-full-resolution routing and decode-gated restoration.

**Decision:** adopt the architecture, not the untestable numerical claim.

### ADNet (2025)

[*ADNet*](https://arxiv.org/abs/2510.12098) proposes edge-guided QR motion
deblurring. The authors released a QR-specialized LENet checkpoint. We made a
checkpoint-compatible evaluator because the upstream package is not directly
usable under the current Python environment.

**Local reproduction:** on the complete 2,250-image QR-DN test, ADNet produced
506 exact payloads (22.49%), four wrong payloads, and 78.2 ms median latency.
Our 124 KB paired U-Net produced 702 exact (31.20%), the same four wrong, and
22.3 ms median. ADNet is also narrowly specialized to synthetic motion blur,
and the repository has no declared license.

**Decision:** do not ship it. Keep edge-aware training as a possible future
loss ablation.

### UAV and drone barcode papers (2025–2026)

The 2025 [Scientific Reports UAV study](https://www.nature.com/articles/s41598-025-29720-w)
reports YOLOv8 mAP of 92.4% followed by OpenCV decoding, but its evaluation is
simulation-heavy and does not establish a new general barcode architecture.

The 2026 [DroneviewBar study](https://www.sciencedirect.com/science/article/pii/S2667305326000232)
is more useful operationally. It tests separate 1D and 2D drone-view models on
small, controlled warehouse collections (about 68 images in each collection,
with roughly 397/395 annotated instances). YOLO is preferred for 1D and Faster
R-CNN for 2D. Average reported decoding is 96.7% for 1D and 81.3% for 2D. The
published dataset short link timed out during this review and no mirror was
found.

**Decision:** these support separate 1D/2D routing and an image-quality/rescan
policy. Their small controlled domains do not displace BarBeR or the external
tests used here.

## Dataset acquisition and integrity

Acquired and normalized locally:

| Dataset | Local scope | Purpose | Important caveat |
|---|---:|---|---|
| Kaggle Barcode and QR v8 | 952 images / 1,696 objects | mixed 1D/2D localization | random hash fold 0 is development; folds 1–4 frozen |
| InventBar | 527 / 527 | modern 1D localization | used for training the clean prototype |
| ParcelBar | 844 / 1,088 | dense Code 128 localization | used for training the clean prototype |
| DEAL single test | 2,000 | EAN-13 localization + exact decode | external archives MD5-verified |
| DEAL recognition | 1,200 | permitted train/validation source | never mixed into the 2,000-image test |
| DEAL EAN-8 | 90 images, 88 checksum-valid labels | exact EAN-8 | two invalid labels excluded from payload denominator |
| DEAL multi | 20 images | multi-instance localization | no payload truth |
| Quick Browser `real` | 2,500 images, 2,498 valid labels | image-level EAN-13 decode | 2 invalid labels excluded; geometry not fabricated |
| QR-DN1.0 | official 2,250-image test | degraded QR exact decode | watermark-extraction domain |
| UniqueData sample | 10 checksum-valid retail images | small independent decode smoke test | too small for ranking |

DEAL archive MD5 values matched the Zenodo record. The QR-DN archive was kept
with its clean/degraded pairing and official source groups (`One`, `Quad`,
`Voted`).

## Neutral evaluation protocol

- Polygon matching uses maximum-cardinality assignment, not greedy matching.
- Strict localization requires IoU ≥ 0.5.
- A second containment-aware metric accepts IoU ≥ 0.5 or
  intersection-over-smaller ≥ 0.5, because a decoder's narrow scan band can be
  operationally valid while having low IoU against a full printed-code box.
- Payload scoring separates exact, GTIN-equivalent, wrong, unresolved,
  not-localized and false-extra decodes.
- UPC-A and the number-system-zero EAN-13 representation are equivalent only
  in the semantic metric; strict string accuracy is preserved.
- Latency is measured per image after model construction. Median and p95 are
  reported because decoder tails are large.
- Thresholds are selected on the Kaggle development fold only. Frozen test
  results are not used for tuning.

## Reproduced results

### Localization

| Method / set | Strict F1 | Containment F1 | Recall | Median |
|---|---:|---:|---:|---:|
| Existing project 5.1 / Kaggle full | — | 0.607 | 0.550 | ~548 ms |
| Direct ZXing+OpenCV / Kaggle full | — | 0.647 | 0.540 | 12.7 ms |
| Clean 424 KB locator / frozen Kaggle test | 0.516 | 0.637 | 0.723 | 58.8 ms* |
| Clean 424 KB locator / DEAL single test | 0.531 | 0.676 | **0.950** | 22.0 ms |
| Clean 424 KB locator / DEAL multi | 0.258 | 0.581 | 0.468 | 17.3 ms |
| Piero YOLOv8s / frozen Kaggle test | 0.668 | 0.779 | 0.740 | 85.2 ms |
| Published BaFaLo / frozen Kaggle test | 0.694 | 0.779 | 0.676 | 3.4 ms |
| BaFaLo→YOLO uncertainty cascade / frozen test | 0.722 | 0.826 | 0.767 | 3.4 ms |
| BaFaLo 320→448→YOLO cascade / frozen test | — | **0.829** | **0.798** | 9.6 ms |
| Published BaFaLo / DEAL single test | 0.942 | 0.952 | 0.979 | 3.6 ms |
| Piero YOLOv8s / DEAL single test | 0.867 | 0.887 | 0.980 | 54.0 ms |
| Published BaFaLo / DEAL multi | 0.641 | 0.733 | 0.623 | 4.8 ms |
| Piero YOLOv8s / DEAL multi | 0.688 | **0.792** | **0.792** | 61.2 ms |

The published BaFaLo checkpoint's DEAL score is not a clean generalization
claim because DEAL is incorporated into BarBeR. The frozen Kaggle split also
cannot eliminate pretraining overlap. These runs establish runtime and an
upper reference, not deployable ownership.

The first clean-room 424 KB segmentation model reached only 0.532 containment
F1 on development. Continued real training and 1,500 on-the-fly synthetic
multi-code scenes per epoch raised development containment F1 to 0.683 at the
preselected 320-pixel/0.60 setting. Frozen-test F1 is 0.637. The 58.8 ms frozen
median was measured while another MPS benchmark was running and is marked
with `*`; the uncontended CPU development median was 22.1 ms.

This locator is still substantially below the public reference models, and
its dense multi-code weakness is explicit. It is selected only for the
deployment-safe folder because its architecture and weights are locally
authored, it has high external single-code recall, and no acceptable license
was established for the stronger controls. Applications that accept AGPL or
obtain a properly licensed detector should replace only this front end.

### Exact decoding

| Method / set | Exact | Semantic | Wrong | Median |
|---|---:|---:|---:|---:|
| ZXing direct / QR-DN test | 16.53% | same | 0.13% | 5.6 ms |
| Classical restoration / QR-DN test | 20.67% | same | 0.40% | 41.7 ms |
| Paired tiny QR restorer / QR-DN test | **31.20%** | same | 0.18% | 22.3 ms |
| ADNet LENet / QR-DN test | 22.49% | same | 0.18% | 78.2 ms |
| ZXing direct / Quick Browser real | 77.10% | same | 0.32% | 5.5 ms |
| Detector-gated crop / Quick Browser real | 82.95% | 83.03% | 1.32% | 28.0 ms |
| Adaptive consensus / Quick Browser real | 84.15% | 84.31% | 0.80% | 25.8 ms |
| Adaptive + declared EAN/UPC / Quick Browser | 84.15% | **84.31%** | **0.48%** | 25.8 ms |
| Final 5.2 accuracy + declared EAN/UPC / Quick Browser | **86.51%** | **86.67%** | **0.36%** | **25.3 ms** |
| ZXing direct / DEAL single test | 71.00% | 71.00% | 0.65% | 19.6 ms |
| Final 5.2 accuracy + declared EAN/UPC / DEAL | 82.80% | **83.10%** | 0.90% | 48.5 ms |
| First-valid detector crop / DEAL single test | 85.95% | 86.30% | 1.95% | 22.1 ms |
| Adaptive two-view consensus / DEAL single test | 87.60% | **87.80%** | 1.25% | **14.3 ms** |
| Adaptive consensus + declared EAN/UPC family / DEAL | 87.60% | **87.80%** | **0.65%** | **14.3 ms** |
| Pruned exhaustive consensus / DEAL single test | 87.95% | 88.15% | 0.90% | 125.2 ms |
| Full exhaustive consensus / DEAL single test | **88.65%** | **88.90%** | 0.95% | 196.1 ms |
| Direct ZXing / DEAL EAN-8 | 88.64% | same | 0% | 73.3 ms |
| Detector-gated crops / 10-image retail sample | 100% | same | 0% | sample too small |

The QR restorer is a 124 KB paired U-Net trained only on the official QR-DN
training pairs. It recovers all 372 directly correct test decodes plus 330 new
ones. By source, exact recovery is 7.87% (`One`), 45.33% (`Quad`) and 40.40%
(`Voted`). Classical and learned restoration have a 719-image oracle union,
only 17 above the learned model; the learned fallback captures nearly all
useful classical wins at lower median latency.

The 1D crop gain is substantial, but the wrong-decode increase is unacceptable
without a confidence policy. On the DEAL fast-crop run, UPC-A values equivalent
to zero-prefixed EAN-13 explain part of the strict difference; others include
wrong symbology and wrong valid EAN strings.

The adaptive consensus is the practical Pareto point. It first asks ZXing to
decode two differently padded source-resolution crops. A matching payload is
accepted; absent or discordant evidence escalates to a pruned
padding/CLAHE/Otsu/upscale ensemble. In this run 1,459 of 2,112 detector
proposals take the cheap agreement path and 653 escalate. Compared with the
first-valid crop decoder, it adds 35 semantic recoveries, removes 14 wrong
payloads, and lowers median latency because easy symbols avoid unnecessary
OpenCV/restoration passes. The exhaustive modes remain appropriate when a
lower wrong-decode rate matters more than latency.

When the application truthfully declares that only EAN-13/UPC-A is expected,
rejecting valid payloads from other symbologies preserves all 1,756 semantic
successes and reduces wrong payloads from 25 to 13 (0.65%), equal to direct
ZXing's wrong count while recovering 336 additional products. This is an
explicit configuration, not a benchmark-specific hidden heuristic: a general
deployment keeps `all`, while a retail workflow should state its allowed
family.

The final controlled-provenance pipeline improves further on the disjoint
Quick Browser real set: 2,165/2,498 semantic successes (86.67%), nine wrong
payloads (0.36%), conditional semantic precision 99.59%, and 25.3 ms median.
Its DEAL result is lower (83.10% semantic), exposing genuine dataset/domain
variance rather than a single flattering aggregate. The most defensible
deployment claim is therefore the per-dataset table, not “best ever” as one
unqualified number.

## Commercial-reader comparison

The Quick Browser paper is the only reproducible source here that ran the same
test against Dynamsoft, Cognex and Inlite, but those 2020-era versions are
historical, not a present-day ranking. A current Dynamsoft web demo was opened
for a public difficult-image sample; automated file upload was unstable and
the target browser session closed. Current commercial SDKs require a trial or
paid license key, and none was present in the workspace. No current commercial
accuracy number is therefore invented.

The reader landscape audit was broader than the benchmark table:

- current, documented commercial candidates include
  [Dynamsoft Barcode Reader](https://www.dynamsoft.com/barcode-reader/docs/core/introduction/),
  [Cognex Mobile Barcode SDK](https://www.cognex.com/blogs/machine-vision/cognex-mobile-barcode-scanner-app-and-sdk),
  and [Inlite ClearImage/Barcode Reader CLI](https://docs.inliteresearch.com/);
- other procurement candidates worth a licensed bake-off include Scandit,
  Zebra, Google ML Kit, Apple Vision, LEADTOOLS, Accusoft, VintaSoft,
  Scanbot, and enterprise fixed readers from Cognex/Datalogic/Zebra;
- open engines include ZXing/ZXing-C++, ZBar, OpenCV, BoofCV and quirc;
- Batoo and JJIL appear in old Android/Java reader discussions and historical
  comparisons, but no maintained official distribution, reproducible current
  benchmark, or suitable drop-in package was found. They are named here
  explicitly rather than silently conflated with current commercial SDKs.

For a fair procurement evaluation, export a fixed 200–500-image difficult
subset and require every vendor to return payload, symbology, polygon,
confidence, error status and elapsed time under the same hardware and
symbology configuration.

## Final 5.2 selection

The selected controlled-provenance implementation is
[`5.2-evidence-guided-coarse-to-fine-pipeline`](../5.2-evidence-guided-coarse-to-fine-pipeline/README.md).
It combines:

1. the locally authored 424 KB 1D/2D overview locator at the
   development-selected 320-pixel/0.60 operating point;
2. untouched source-resolution crops;
3. two-padding agreement with bounded consensus escalation and explicit
   ambiguity;
4. a truthful optional symbology allowlist;
5. the locally trained 124 KB QR restorer only after ordinary crop decoding
   fails;
6. in default accuracy mode, the proven local 5.1 Code 39/128, Data Matrix,
   and QR rescue routes as a complementary classical safety net.

All five selection gates were evaluated. The exact-decoding and restoration
gates pass decisively. The mixed/dense localization gate passes only as a
known baseline, not as parity with the reference cascade: deployment
provenance forced a measurable accuracy tradeoff. The JSON contract separates
`decoded` from `localized` and records decode rejection evidence. The final
folder contains no BaFaLo, Ultralytics, Piero, or ADNet code/weights.

On the real 21-page private evaluation document integration test, the native-raster PDF
path recovered 42/43 expected payload instances with no unexpected payload,
plus one honest unresolved Data Matrix location. A former 5.1 regression
artifact contains a decode for that last symbol, but the current decoder
cannot reproduce it even after a bounded crop/scale/binarization search; 5.2
does not copy a stale expected string into the result.

The overall recommendation is therefore conditional:

- **Ship 5.2 now** when controlled provenance, offline operation, and a
  modular fallback matter most.
- **License or train a stronger detector** before claiming best-in-class
  mixed/dense localization. The measured target to beat is the reference
  cascade's 0.829 containment F1.
- **Keep the selected consensus/restoration back end** even if the locator is
  replaced; those components won their direct exact-payload comparisons.
