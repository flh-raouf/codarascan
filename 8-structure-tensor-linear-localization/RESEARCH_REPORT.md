# Research report: a faster CPU structure-tensor localizer

## Executive conclusion

A purpose-built sparse structure-tensor localizer can replace OpenCV
`BarcodeDetector.detectMulti()` in a **fast linear-only tier**.

The best implementation is not the literal dense paper algorithm. It is a
Sparse Tensor Tile Graph (STTG):

1. Fused native Scharr derivatives and tile tensor accumulation.
2. Page-adaptive energy thresholding.
3. Local context coherence.
4. Orientation-aware strong/weak region growth.
5. Cheap periodic-profile rejection before Python sees a proposal.
6. Strict retention of rare oversized weak components.
7. Native-pixel periodic-band refinement inside those components.
8. Pipeline 7's source-resolution physical verification.
9. Conservative, reverified joining of split barcode fragments.

On identically prepared private evaluation document pages it is approximately 3.9 times
faster than Pipeline 7's OpenCV proposal engine. The refined document profile
now reaches 41/41 containment matches with zero unmatched predictions. Pipeline
7 still has tighter strict-IoU geometry, and the general profile still has
documented weaknesses on difficult mixed images. This remains a latency-first
service tier, not a universal replacement for Pipeline 7 or the Adaptive
Extractor.

## Scope and invariants

The research question was deliberately narrow:

> Can a clean-room CPU structure-tensor implementation localize ordinary 1D
> barcodes substantially faster than Pipeline 7, with only a small accuracy
> loss?

Hard constraints:

- Pipeline 7 remains untouched.
- 1D localization only.
- No payload decoding.
- No trained model or BaFaLo checkpoint.
- No GPU.
- No user-visible payload.
- Loading, grayscale conversion, and resize are outside engine timing for every
  compared method.
- False positives and misses are both reported.

## Research basis

### Sörös and Flörkemeier

The 2013 method derives separate edge and corner saliency from a local structure
matrix. For 1D codes, one eigen-direction should dominate because bars create
many parallel edges. A larger box filter connects local evidence and suppresses
isolated clutter. The paper also uses saturation as a rejection signal.

Source:
<https://vs.inf.ethz.ch/publ/papers/soeroesg-mum2013-localization.pdf>

The 2014 follow-up maps these operations to GPU shaders and demonstrates that
the saliency calculation is highly parallel. Its most reusable CPU lesson is
not the GPU itself: separable/fused computation and keeping intermediate data
compact matter more than language choice.

Source:
<https://vs.inf.ethz.ch/publ/papers/soeroesg-icassp2014-gpulocalization.pdf>

### OpenCV BarcodeDetector source

Source inspection showed that Pipeline 7 was already using a structure-tensor
family detector indirectly. OpenCV:

- computes Scharr X/Y derivatives;
- thresholds gradient magnitude;
- builds integral maps for edge occupancy and tensor terms;
- tests local coherence near 0.9 and edge occupancy near 42%;
- performs orientation-aware region growing;
- applies morphology and rotated-rectangle fitting;
- repeats analysis at configured detector scales.

Source:
<https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/barcode_detector/bardetect.cpp>

This changed the engineering question. Reimplementing the same algorithm in
Python could not win. The opportunity was to implement only the operations
needed by the fast use case, fuse them in one native pass, and use a sparse
tile graph instead of full integral maps plus repeated detector scales.

### Other relevant directions

- Gallo and Manduchi use directional gradient evidence for fast 1D
  localization, but their method assumes one approximately horizontal barcode.
  <https://vision.soe.ucsc.edu/node/313>
- Szentandrási et al. use low-cost tiled gradient evidence before precise
  high-resolution QR work. This supports the coarse-tile architecture even
  though the target here is 1D.
  <https://www.fit.vut.cz/research/result/c91506/.en>
- A universal semantic-segmentation detector reports real-time CPU behavior and
  strong localization, but it introduces training/model provenance and was not
  necessary to answer this CPU deterministic experiment.
  <https://arxiv.org/abs/1906.06281>

## What was measured before implementation

Profiling Pipeline 7 showed that `detectMulti()` dominated the linear path.
Native source-resolution verification was a small fraction of the total. This
meant:

- optimizing JSON, Python loops, or deduplication could not produce a major
  gain;
- a faster proposal generator was required;
- keeping the existing verifier was attractive because it protected precision
  cheaply.

## Experiments and rejected assumptions

### 1. Literal dense structure-tensor maps

The first reproduction computed multi-scale dense tensor responses. It achieved
very high proposal recall, but emitted tens of thousands of text/table
components and was too slow. This established an important distinction:

> Structure-tensor evidence is a good barcode signal; naïve grouping is not a
> detector.

Rejected as a deployable design.

### 2. Pure Python tiled tensor

Pooling tensor values into tiles reduced candidate volume, but the Python and
NumPy orchestration remained slower than the desired engine budget. It was
retained as a readable research backend only.

Rejected as the production backend.

### 3. Exact OpenCV-like integral-image clone

An integral-image approach worked, but reproduced several costs of
`detectMulti()`:

- multiple full-size floating-point maps;
- repeated memory passes;
- scale analysis that did not contribute accepted results;
- generic grouping for use cases outside the fast contract.

Rejected in favor of directly accumulating tensor values into tiles.

### 4. Direct ZXing localization

ZXing was benchmarked at the prepared 1600-pixel boundary, not dismissed by
assumption.

| Development domain | Containment F1 | Median engine time |
|---|---:|---:|
| Document validation | 0.385 | 13.1 ms |
| Easy documents | 0.605 | 12.3 ms |
| InventBar | 0.900 | 5.9 ms |
| ParcelBar | 0.301 | 14.8 ms |
| Kaggle development | 0.578 | 3.7 ms |

ZXing is excellent when it successfully decodes, but it is not a complete
localizer. A conditional ZXing union improved some photo datasets but hurt the
simple-document precision/latency contract. It also changes the semantics from
pure localization to decode-dependent geometry.

Rejected from the default. It remains a possible application-specific route
when valid payloads are valuable independently.

### 5. One profile for every domain

A 12-pixel tile was best for broad photographs and multi-code scenes. It missed
too many narrow document bars. A 7-pixel tile recovered document codes, but its
extra sensitivity admitted two text-like false positives in private corpus A.

The clean solution was two frozen profiles:

- `general`: tile 12, normal 12-transition native gate;
- `document`: tile 7, stricter 24-transition native gate.

Rejected the one-profile assumption.

### 6. Aggressive fragment merging

Tensor components can split one barcode at a quiet internal gap. A wide merge
fixed the two observed VN fragments, but naïvely applying it to ParcelBar
merged neighboring independent codes.

The final policy:

- every fragment must pass native verification independently;
- directions must agree within five degrees;
- heights and centerlines must agree;
- wide gaps require independent barcode-like bridge evidence;
- the merged crop must pass the full verifier again;
- the general profile keeps a shorter maximum gap.

This fixed document splits without preserving the unsafe global merge.

### 7. Multi-scale tensor passes

Adding more tile sizes raised compute and duplicate proposals. The final
profiles use one domain-appropriate tile size. The retained native-resolution
verification supplies the precision that a larger proposal pyramid was trying
to recover.

Rejected from the default.

## Final native architecture

### Fused derivative and tile accumulation

The C++ kernel walks the prepared grayscale image and computes 3x3 Scharr
derivatives. Instead of materializing full `Gx`, `Gy`, magnitude, and integral
images, it accumulates directly into tile cells:

```text
Jxx = sum(Gx²)
Jyy = sum(Gy²)
Jxy = sum(Gx Gy)
```

It also tracks edge occupancy and gradient polarity.

### Tensor coherence

For each tile:

```text
trace = Jxx + Jyy
delta = sqrt((Jxx - Jyy)² + 4 Jxy²)
coherence = delta / trace
theta = 0.5 atan2(2 Jxy, Jxx - Jyy)
```

A barcode tile should have substantial edge energy and a dominant orientation.
Text may have local orientation, but usually does not sustain the same
periodicity and context over a barcode-shaped component.

### Adaptive strong/weak evidence

A percentile of nonzero tile energy supplies a page-adaptive floor. Strong
seeds require high occupancy, local coherence, and neighborhood coherence.
Weaker cells may join only when their energy and orientation agree with nearby
evidence.

This is a hysteresis design: weak barcode sections can survive without allowing
all weak text edges to become seeds.

### Orientation-aware sparse graph

Tiles are assigned to overlapping orientation bins. Adjacent compatible cells
are unioned into components. The component receives:

- weighted orientation;
- density;
- seed count;
- occupancy/coherence;
- polarity balance;
- oriented extent.

No dense response image or generic full-resolution morphology is required.

### Native profile rejection

Before a candidate crosses the Python boundary, three scan profiles test:

- contrast;
- black/white transition count;
- transition rate;
- agreement between profiles.

This reduced the private evaluation document proposal median from roughly 78 ms to about
9.5 ms in the 2300-pixel research configuration because expensive
source-resolution verification no longer processed thousands of text
components.

### Source-resolution verification

Candidates are mapped back to untouched source pixels. The verifier checks:

- bar-edge orientation;
- edge persistence through the candidate height;
- black/white transition count and density;
- agreement across scanlines;
- aspect and physical extent.

The document profile raises the minimum transition count to 24. Uncertain
patterns are rejected rather than returned as a barcode.

### Oversized-component periodic refinement

The initial frozen document profile missed four Quality symbols. Diagnostic
inspection showed that none required another full-page detector:

- three small header barcodes were already inside compact tensor components
  merged with their printed digits;
- two nearby, parallel barcodes on one page were already inside one broad
  component.

The refinement therefore operates only inside a rare, oversized tensor parent:

1. The native proposer may retain a low-transition component only when aspect,
   orientation, contrast, scanline agreement, and component-size bounds all
   pass strict thresholds.
2. Retention is not acceptance. A weak parent can never become a final
   detection directly.
3. The parent is rectified once at native resolution.
4. Every short-axis row is tested for repeated transitions across the long
   axis; isolated one-pixel flips are suppressed and only contiguous periodic
   bands are considered.
5. Each mapped child must independently pass the unchanged 24-transition
   source-resolution verifier.
6. Compact failed parents contribute only their strongest periodic child.
   Broad merged parents are replaced only when at least two children verify.

This is a targeted split of existing evidence, not a new detector pass. The
general profile keeps both weak-parent retention and periodic splitting
disabled.

## Benchmark protocol

Localization was scored with the neutral benchmark harness:

- one-to-one matching;
- strict polygon IoU and containment-aware matching;
- precision, recall, and F1;
- false positives visible;
- prepared-input engine time separated from load/resize;
- linear ground truth only.

Development and holdout partitions used stable image IDs where available.
private evaluation document and private document corpus document references used validated Pipeline 7/5.1
geometry. Public datasets include independent annotations, but several had
already been inspected during earlier research. Results should therefore be
read as strong regression evidence, not a pristine academic blind test.

## Frozen accuracy results

| Dataset | Profile | Ground truth | TP | FP | Precision | Recall | F1 | Pipeline 7 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| private evaluation document | document | 41 | 41 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| private corpus A holdout | document | 8 | 8 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| Easy ordinary documents | general | 52 | 47 | 0 | 1.000 | 0.904 | 0.949 | 0.896 |
| InventBar test fold | general | 91 | 84 | 1 | 0.988 | 0.923 | 0.955 | 0.850 |
| ParcelBar test fold | general | 199 | 177 | 18 | 0.908 | 0.889 | 0.898 | 0.902 |
| Kaggle mixed test folds | general | 408 | 313 | 138 | 0.694 | 0.767 | 0.729 | 0.578 |
| DEAL single-test | general | 2000 | 1753 | 291 | 0.858 | 0.877 | 0.867 | not rerun |

The new method is not merely faster on easy documents. It exceeds Pipeline 7
F1 on Easy, InventBar, and Kaggle, is nearly tied on ParcelBar, and now ties
Pipeline 7's containment result on Quality and the frozen VN subset. This does
not mean the geometry is identical: on Quality, strict-IoU F1 is 0.659 for the
refined tensor path and 0.805 for Pipeline 7.

The Kaggle precision is not suitable for a precision-critical production
contract. That set contains small, mixed, and visually difficult codes and
demonstrates why this fast tier must not replace the exhaustive localizer.

## Controlled speed comparison

The following original test loaded and prepared all 21 private evaluation document images
before either engine stopwatch. Seven randomized rounds were run per worker
count on the same Apple M4 environment. These rows describe the initial 37/41
profile and are retained as experimental history.

| Workers | STTG throughput | Pipeline 7 throughput | Speedup |
|---:|---:|---:|---:|
| 1 | 6.51 ms/page | 24.83 ms/page | 3.82x |
| 2 | 3.48 ms/page | 13.90 ms/page | 3.99x |
| 4 | 2.09 ms/page | 8.32 ms/page | 3.97x |
| 8 | 1.93 ms/page | 7.18 ms/page | 3.73x |
| 10 | 1.94 ms/page | 7.06 ms/page | 3.63x |
| 21 | 1.88 ms/page | 7.22 ms/page | 3.85x |

Individual engine task latency rises under heavy concurrency even when total
throughput improves. Eight workers is the default compromise. The original
1.9 ms/page queue throughput was close to, but did not meet, the 1 ms goal.

On the document profile, median stage costs were:

| Stage | private evaluation document median |
|---|---:|
| Fused native proposal | 5.31 ms |
| Native-pixel verification | 0.82 ms |
| Complete engine | 7.38 ms |

The complete-engine median differs from the 6.51 ms controlled wall
measurement because one summarizes per-image stage records from a separate
benchmark run and the other summarizes repeated whole-dataset wall time.

After periodic refinement was frozen, five serial Quality runs produced a
median-of-medians of 5.83 ms/page. The matching Pipeline 7/OpenCV benchmark
measured 22.91 ms/page, a 3.9x median-latency advantage. Five eight-worker CLI
runs measured 2.276--2.472 ms/page engine throughput, with a median of
2.320 ms/page. The refinement therefore closes the four-miss gap at a small
mean-tail cost while leaving the typical per-page engine cost essentially
unchanged.

## Periodic-refinement ablation and confirmation

| Quality configuration | Matches | Unmatched | Containment F1 |
|---|---:|---:|---:|
| Original document profile | 37/41 | 0 | 0.949 |
| Weak-parent retention only | 37/41 | 0 | 0.949 |
| Periodic split without weak-parent retention | 39/41 | 0 | 0.975 |
| Both stages | **41/41** | **0** | **1.000** |

Thresholds were frozen after the Quality analysis. Subsequent checks were:

| Confirmation corpus | Original | Final | Unmatched change |
|---|---:|---:|---:|
| private corpus A full, 315 pages / 33 labels | 29 TP, 2 unmatched | 29 TP, 0 unmatched | -2 |
| Separate document set, 42 images / 51 labels | 43 TP, 0 unmatched | 45 TP, 0 unmatched | 0 |
| Easy general profile, 160 images / 118 labels | 104 TP, 0 unmatched | 104 TP, 0 unmatched | 0 |

An initial refinement version did add one VN false positive because a retained
low-transition parent could pass the final verifier directly. Making retention
strictly refinement-only removed it. This negative result is important: weak
coarse evidence is useful for selecting where to inspect, but is not sufficient
final localization evidence.

## False-positive investigation

The private corpus A test was important because the tile-7 document profile returned
patterns on pages with no barcode. Overlay inspection separated three failure
classes:

1. True barcode fragments: two boxes covered different halves of the same
   barcode. Conservative fragment joining fixed these.
2. Text-like patterns: short repeated strokes passed the low transition gate.
   The document profile's 24-transition requirement removed both.
3. Curved periodic texture: fingerprint ridges on page 127 passed transition
   count and tolerant scanline agreement. A native-pixel shared edge-direction
   test now measures the weighted 80th-percentile angular deviation of strong
   edges after rectification. The document threshold is 18 degrees. The two
   remaining VN false positives measured at least 24.7 degrees, while all 70
   matched Quality/VN detections measured no more than 11.8 degrees.

The original frozen VN subset remains 8/8 with zero extra boxes. On the larger
315-page VN manifest, the final profile retains 29/33 teacher matches and
reduces unmatched predictions from two to zero. Synthetic curved-ridge and
table/text tests were also added to the unit suite.

## Implementation and provenance

- `native/sttg_native.cpp` is a clean-room implementation written for this
  experiment.
- It does not copy or load BaFaLo code or weights.
- It links only against the Python/NumPy ABI and the C++ standard library.
- OpenCV remains in the Python verification/artifact path, but
  `BarcodeDetector.detectMulti()` is not used by the production STTG backend.
- The extension releases the GIL around native compute.
- The compiled binary is not committed.

## Deployment recommendation

Keep three distinct responsibilities:

```text
ordinary 1D + latency priority
    -> Structure-tensor fast localizer

unknown/difficult localization
    -> Pipeline 7 coarse-to-fine localizer

payload extraction / maximum quality
    -> Adaptive Extractor (5.1-derived production path)
```

Do not silently fall back from the fast localizer to Pipeline 7 inside the same
latency measurement. Routing should be explicit at the service layer so users
know which accuracy/latency contract they selected.

## Remaining work before production

1. Freeze a genuinely unseen customer-document holdout.
2. Validate the native build on production Linux/ARM hardware.
3. Measure peak memory and sustained thermal behavior under the real queue.
4. Add service-level backpressure and bounded preparation batches.
5. Decide whether low-confidence fast results should be returned empty or
   explicitly escalated to Pipeline 7.
6. Keep 2D codes out of this engine; adding them would require a separate corner
   tensor/finder-pattern design and would change its latency contract.
