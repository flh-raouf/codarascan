# Deterministic 2D localization: research and benchmark

## Why the previous 2D branch failed

The old implementation used bidirectional gradient energy both to propose and
accept generic `matrix_2d` regions. That signal is useful for finding compact
two-dimensional texture, but it is not barcode-specific. Dense text, table
intersections, stamps, and mechanical drawings can produce the same response.

On `private corpus A.pdf`, the old 2D mode returned 140 regions on 100 pages. Visual
inspection showed that most were ordinary document texture. Pipeline 5.1
decoded six real Data Matrix symbols on pages 5, 45, 75, 94, 152, and 226.

The corrected architecture separates proposal from proof:

```text
bidirectional-energy proposal
          |
          +-- local QR finder verification
          |
          `-- Data Matrix border recovery
                    |
                    `-- ECC 200 L-finder + alternating timing borders
                              + module-grid consistency
```

Directional energy can now nominate a region, but it can never create a final
2D result on its own.

## Data Matrix verification

For each energy proposal the implementation:

1. Builds several local border hypotheses using Otsu/adaptive thresholding,
   contours, and small morphological closings.
2. Rectifies each hypothesis to a normalized square.
3. Samples valid ECC 200 square and rectangular symbol grids.
4. Requires two adjacent solid borders and two opposite alternating borders.
5. Checks module purity and interior occupancy.
6. Accepts only a sufficiently strong combined physical-structure score.

This follows the defining Data Matrix geometry instead of treating arbitrary
two-directional texture as a code.

## QR verification

The normal full-page OpenCV QR detector remains decode-free. A local recovery
path handles difficult QR symbols found by the generic proposal stage. It
requires:

- a plausible square quadrilateral;
- geometric agreement with the original proposal;
- nested QR finder contours; and
- for damaged symbols, abundant independent `1:1:3:1:1` finder-pattern
  scanline evidence.

## Safe CPU optimizations

The strict physical checks are unchanged. The optimized implementation removes
duplicate work around them:

1. Valid Data Matrix timing patterns are precomputed once and reused.
2. Geometrically identical proposals are removed before any grid scoring.
3. Every border hypothesis is rectified once. Otsu is evaluated for all
   hypotheses, while the more expensive adaptive threshold is generated only
   for the two strongest Otsu geometries.
4. QR verification creates threshold views lazily: grayscale first, then Otsu,
   then adaptive thresholding only if the previous view did not prove a finder.
5. On large sparse pages, a conservative 700-pixel matrix screen and 900-pixel
   QR-finder screen may skip the 1400-pixel full scan. Any evidence or
   uncertainty escalates to the original strict path. In explicit 2-D mode a
   final hidden linear-evidence check protects mixed barcode pages; its result
   is never emitted.

Several faster-looking changes were rejected:

- A direct 700-pixel final detector lost one private evaluation document symbol, two
  synthetic 2-D symbols, two balanced symbols, and 17 stress symbols.
- Capping border hypotheses at nine lost the difficult Data Matrix on Quality
  Dossier page 21; the validated limit remains 12.
- A permissive QR nested-contour prefilter lost a degraded VN QR and nine
  Kaggle QR results. The final code uses lossless lazy evaluation instead.

## Primary research used

- Rybakova, Limonova, and Bezmaternykh, *Improving Data Matrix mobile
  recognition via fast Hough transform and adaptive grid extractors* (2025):
  candidate extraction followed by line pairing, L-finder recovery,
  quadrilateral refinement, and timing-grid analysis.
  <https://doi.org/10.18287/COJ1804>
- Huang et al., *Data Matrix Code Location Based on Finder Pattern Detection
  and Bar Code Border Fitting* (2012): morphological candidate extraction is
  followed by explicit L-finder and dashed-border fitting.
  <https://doi.org/10.1155/2012/515296>
- Karrach et al., *Data Matrix Code Localization and Recognition on Automotive
  Parts* (2021): comparison of deterministic Data Matrix localization methods,
  all centered on the L-shaped finder pattern; adaptive thresholding is
  particularly useful on uneven scans.
  <https://doi.org/10.3390/jimaging7090163>
- Sun et al., *Research on QR Image Code Recognition System Based on Artificial
  Intelligence Algorithm* (2021): documents the QR finder pattern's
  `1:1:3:1:1` run-length geometry.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8321072/>

## Validation results

### private corpus A, 315 pages

| Mode | Before | After |
|---|---:|---:|
| Linear | 33 regions | 33 regions |
| 2D | 140 regions on 100 pages | 8 structurally verified regions on 8 pages |
| Linear regression | — | 33 matched, 0 missing, 0 added |

The eight retained 2D regions are six clean Data Matrix symbols and two
strongly Data Matrix-like degraded marks on pages 56 and 311. Those two marks
are not decodable by pipeline 5.1, so they are intentionally reported as
high-confidence localization candidates rather than decoded payloads.

The 1D branch was not modified. Its measured processing wall time remained
approximately 3.16 seconds for 315 pages with four workers, or 10.0 ms/page
throughput.

Before these safe optimizations, the strict corrected 2D branch had a repeated
median of 64.7 ms/page with four workers. The final implementation has a
repeated median of 54.6 ms/page, retaining the same eight regions. That is a
15.5% throughput improvement. The old permissive 33.3 ms/page branch was
faster only because it emitted unverified document texture.

### Other VN lots

| Dossier | Strict baseline | Optimized | Result |
|---|---:|---:|---:|
| private corpus A, 315 pages | 64.7 ms/page | 54.6 ms/page | 8 = 8 |
| private corpus B, 178 pages | 74.2 ms/page | 65.7 ms/page | 5 = 5 |
| private corpus C, 342 pages | 73.6 ms/page | 68.1 ms/page | 7 = 7 |

These are sparse dossiers, where the conservative negative gate avoids the
full scan on many pages.

### Existing regression families

| Dataset | Result after correction |
|---|---:|
| private evaluation document | 43/43, exact geometry |
| Synthetic 2D | 86/86, exact geometry |
| Balanced mixed | 121/121 strict-reference locations, exact geometry |
| Extreme mixed | 155/155 strict-reference locations, exact geometry |
| Kaggle barcode + QR v8, 952 images | 799/799 strict-reference locations |
| Low-resolution QRDN sample, 225 images | 24/24 strict-reference locations |
| Low-resolution amaye sample, 20 images | 3/3 strict-reference locations |

On the extreme set, two old generic 2D proposals are now rejected because
neither exhibits a valid observable Data Matrix finder/timing lattice. Their
scores overlap the VN text/stamp false-positive population, so accepting them
would reintroduce the original defect. One additional true linear location is
retained.

The gate is workload-sensitive. A dense synthetic 2-D set, where almost every
page must escalate, increased from 92.1 to 102.7 ms/page with four workers.
Use `--no-empty-page-gate` for a known dense 2-D-only batch. The normal `all`
mode and sparse VN workloads benefit because linear evidence is reused and
negative pages can terminate early.

## Reproduction

Run only the frozen 1D branch:

```bash
venv/bin/python '7-cpu-coarse-to-fine-localization/run.py' \
  '/path/to/private-project/docs/vn pour raouf/private corpus A/private corpus A.pdf' \
  --output '/tmp/private-evaluation-corpus-p7-linear' \
  --kinds linear \
  --workers 4 \
  --overwrite
```

Run only the corrected 2D branch:

```bash
venv/bin/python '7-cpu-coarse-to-fine-localization/run.py' \
  '/path/to/private-project/docs/vn pour raouf/private corpus A/private corpus A.pdf' \
  --output '/tmp/private-evaluation-corpus-p7-2d' \
  --kinds 2d \
  --workers 4 \
  --overlays \
  --overwrite
```

Overlay generation is reported separately and is not included in localization
time.
