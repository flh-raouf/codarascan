# Pipeline selection benchmark

Date: 2026-07-28

This benchmark compares the current source of pipelines 5, 5.1, 5.2, and 7
for choosing a Codara production path.

## Test environment

- Apple M4, 10 CPU cores, 16 GB RAM
- Python 3.13.7
- OpenCV 5.0.0
- Visual artifacts disabled
- Pipelines 5, 5.1, and 7 use one page worker
- Pipeline 5.2 uses MPS and its recommended `accuracy` mode for document
  tests. Its model is shared and processing is serialized by design.
- The Kaggle test uses pipeline 5.2 `balanced` mode to measure its independent
  learned pipeline without the serial 5.1 document rescue.

Times are measured sequential processing latency per page with PDF extraction
and artifact writing excluded. No pages overlap. OpenCV may still use its
native threads inside one page, which is also how Codara analyzes an individual
page after shared page preparation.

Every timing cell below comes from a fresh one-worker run of the current source
on 2026-07-28. The displayed values are the measured processing total divided
by the page count and rounded to 0.1 milliseconds; they are not estimates
derived from batch throughput.

Pipeline 5's standalone launcher currently cannot serialize an image-directory
input. Its 33%-resolution cell therefore uses the 5.1 wrapper with both
`--no-adaptive-generalization` and `--no-residual-proposals`, which executes
the unchanged pipeline-5 page algorithm while avoiding that launcher bug.

## Results

In each document cell, "found" means one-to-one containment-aware
localization, "false detections" means unmatched output locations, and "wrong
payload" means a non-empty but incorrect decoded value.

| Pipeline | Quality Dossier, native (43 barcodes) | Quality at 33% resolution (43 barcodes) | VN LOT 06, audited reference (41 barcodes) | Balanced mixed (122 barcodes) | Extreme mixed (217 barcodes) | Kaggle v8 localization (1,696 barcodes) |
|---|---|---|---|---|---|---|
| **5** | **43/43 payloads decoded correctly**; 43 found; 0 false detections; **167.9 ms/page** | 25/43 payloads decoded correctly; 32 found; 0 false detections; **113.9 ms/page** | 7 payloads decoded; 37/41 found; 62 false detections; **127.0 ms/page** | 120/122 payloads decoded correctly; 121 found; 0 false detections; **136.2 ms/page** | 136/217 payloads decoded correctly; 1 wrong payload; 164 found; 1 false detection; **192.4 ms/page** | Precision 72.0%; recall 51.6%; F1 0.601; **305.8 ms/image** |
| **5.1** | **43/43 payloads decoded correctly**; 43 found; 0 false detections; **233.8 ms/page** | **26/43 payloads decoded correctly**; **33 found**; 0 false detections; **402.1 ms/page** | 7 payloads decoded; 37/41 found; 62 false detections; **185.1 ms/page** | **121/122 payloads decoded correctly**; 121 found; 0 false detections; **151.6 ms/page** | 136/217 payloads decoded correctly; 1 wrong payload; **165 found**; 1 false detection; **211.1 ms/page** | Precision 67.9%; recall 55.0%; F1 0.607; **500.8 ms/image** |
| **5.2** | 42/43 payloads decoded correctly; 43 found; **25 false detections**; **640.7 ms/page** | 26/43 payloads decoded correctly; 1 wrong payload; 32 found; **28 false detections**; **521.3 ms/page** | 6 payloads decoded; 7/41 found; **710 false detections**; **616.9 ms/page** | 121/122 payloads decoded correctly; 121 found; **14 false detections**; **463.3 ms/page** | **137/217 payloads decoded correctly**; 1 wrong payload; 146 found; 12 false detections; **527.9 ms/page** | **Precision 64.4%; recall 72.0%; F1 0.680**; **87.6 ms/image** in `balanced` mode |
| **7** | 43/43 found; 0 false detections; **121.9 ms/page**; no payload decoding | 17/43 found; 1 false detection; **60.3 ms/page**; no payload decoding | **41/41 found**; 0 false detections; **101.2 ms/page**; no payload decoding | 121/122 found; 0 false detections; **180.2 ms/page**; no payload decoding | 155/217 found; **0 false detections**; **164.6 ms/page**; no payload decoding | **Precision 74.3%**; recall 35.0%; F1 0.476; **97.0 ms/image**; no payload decoding |

The exact measured processing totals behind the rounded per-page values were:

| Dataset | Pages/images | Pipeline 5 | Pipeline 5.1 | Pipeline 5.2 | Pipeline 7 |
|---|---:|---:|---:|---:|---:|
| Quality Dossier, native | 21 | 3.5251 s | 4.9095 s | 13.455278 s | 2.560503 s |
| Quality at 33% resolution | 21 | 2.3925 s | 8.4450 s | 10.947492 s | 1.266767 s |
| VN LOT 06 | 315 | 40.0067 s | 58.2963 s | 194.322621 s | 31.865107 s |
| Balanced mixed | 50 | 6.8095 s | 7.5803 s | 23.164950 s | 9.009351 s |
| Extreme mixed | 60 | 11.5447 s | 12.6670 s | 31.672819 s | 9.874294 s |
| Kaggle v8 | 952 | 291.1076 s | 476.7283 s | 83.369864 s | 92.387241 s |

Across the 131-page Quality, balanced, and stress reference corpus used by
Codara's pipeline-7 label, the newly measured sequential average is **163.7
ms/page**. This directly supports the app's rounded `~160 ms latency` value.
The equivalent sequential corpus averages are 167.0 ms/page for pipeline 5,
192.0 ms/page for pipeline 5.1, and 521.3 ms/page for pipeline 5.2.

Kaggle v8 does not provide payload ground truth, so that column measures
localization only. Pipeline 5.2 uses `balanced` mode in that column; its
document `accuracy` mode is represented by the first five columns.

VN LOT 06 does not have an independent exhaustive annotation file. Its 41
reference locations are the strict, visually reviewed pipeline-7 result: 33
linear and 8 physically verified Data Matrix-like regions. The VN decoded
count is therefore reported, but not claimed as exact payload accuracy.

## What the table means

### Pipeline 5

Best speed/accuracy compromise for native-resolution document extraction.
It exactly matches 5.1 on Quality and VN, is materially faster on every shared
document test, and gives up only one exact decode on the balanced set.

Its main weakness is unresolved/review noise on VN: 62 of its 99 returned
locations do not overlap the strict audited reference.

### Pipeline 5.1

Best decoding choice when Codara must support small, rescaled, colored, or
unusual web uploads in addition to PDFs. It recovers one additional payload
on the 33% dossier and one additional Data Matrix on the balanced set.

The gain is small on the tested high-resolution documents, while its adaptive
scale-space paths make it 1.1x to 3.5x slower than pipeline 5 depending on the
dataset.

### Pipeline 5.2

Not suitable as Codara's general document extractor in its current form.
Accuracy mode is roughly 2.7 to 4.9 times slower than pipeline 5 on these
document tests and emits many learned unresolved regions, especially 710 false
detections on VN LOT 06.

Its independent `balanced` mode has the strongest Kaggle localization recall
and F1. It remains useful for a specialized camera/retail experiment, not as
the default scanned-dossier path.

### Pipeline 7

Best choice when only reliable barcode location is required. It is the only
pipeline with zero false detections on native Quality, VN, balanced, and stress
tests. It is the fastest sequential document path on Quality, low-resolution
Quality, VN, and stress; pipeline 5 is faster on the dense balanced set.

It cannot extract payloads by design and loses substantial recall after severe
downscaling, so it cannot replace a decoding pipeline for Extraction.

## Recommendation

1. **Choose pipeline 5 for the normal Codara Extraction default** if production
   input is primarily native-resolution scanned PDFs. It offers the strongest
   practical speed/accuracy balance.
2. **Choose pipeline 5.1 instead** only if small and variably scaled image
   uploads are an explicit product requirement and the extra recovery is worth
   the latency.
3. **Use pipeline 7 for Separation/detection-only workflows**, where payloads
   are unnecessary and false locations are more costly than missed extreme
   symbols.
4. **Do not choose pipeline 5.2 as the general default** in its current form.
   Keep it only as an experimental specialized locator.

If one single pipeline must be selected for every Extraction request, the
evidence favors **pipeline 5**. If one pipeline must cover both ordinary PDFs
and low-resolution user uploads without routing, select **pipeline 5.1**.

## Reproduction configurations

Pipelines 5 and 5.1:

```bash
--formats all --workers 1 --no-crops --no-overlays
```

Pipeline 5.2 document tests:

```bash
--device mps --mode accuracy --no-overlays
```

Pipeline 5.2 Kaggle generalization test:

```bash
--device mps --mode balanced --no-overlays
```

Pipeline 7:

```bash
--kinds all --workers 1
```
