# Synthetic Mixed Barcode Stress Dossier - baseline

## Dataset

- Pages: 60 A3-style native 200 dpi scans.
- Symbols: 217.
- Formats: 67 Code 128, 49 Code 39, 59 Data Matrix, 42 QR Code.
- Guaranteed zero-code pages: 5, 12, 19, 27, 36, 44, 53, 60.
- Dense pages: page 10 (12), 20 (9), 30 (14), 40 (11), 50 (13), 58 (16).
- Difficulty: 59 easy, 86 medium, 54 hard, 18 extreme.

## Baseline engine

Run with `zxing_2d_barcode_pipeline` in `--formats all` mode. Crops were
disabled for the baseline run; native pages, overlays, and `detections.json`
were retained.

### Exact decoding

- Decoded result records: 136.
- Exact page/format/payload matches: 133.
- Ground-truth symbols not exactly decoded: 84.
- Strict payload precision: 97.8%.
- Strict payload recall: 61.3%.
- Unexpected result records: 3.
  - Two are duplicate reads of a real Code 39 symbol.
  - One is a Code 39 substitution (`0` read as `I`); Code 39 has no mandatory
    checksum, so the payload can remain syntactically valid.

### Localization

One-to-one polygon matching at overlap >= 0.30 across decoded results,
unresolved matrices, and review candidates:

- Correctly localized symbols: 163/217.
- Localization recall: 75.1%.
- Localization precision: 98.2%.
- Matched by tier: 134 primary results, 8 unresolved matrices, 21 review
  candidates.
- Unmatched/duplicate candidate records: 3.
- All eight zero-code pages produced zero primary, unresolved, and review
  results.

### Exact payload recall by format

| Format | Decoded | Total | Recall |
|---|---:|---:|---:|
| Code 128 | 41 | 67 | 61.2% |
| Code 39 | 30 | 49 | 61.2% |
| Data Matrix | 32 | 59 | 54.2% |
| QR Code | 30 | 42 | 71.4% |

### Exact payload recall by difficulty

| Difficulty | Decoded | Total | Recall |
|---|---:|---:|---:|
| Easy | 40 | 59 | 67.8% |
| Medium | 60 | 86 | 69.8% |
| Hard | 29 | 54 | 53.7% |
| Extreme | 4 | 18 | 22.2% |

### Exact payload recall by scenario

| Scenario | Decoded | Total | Recall |
|---|---:|---:|---:|
| Standard | 58 | 74 | 78.4% |
| Low contrast | 22 | 27 | 81.5% |
| Edge | 20 | 26 | 76.9% |
| Oblique | 12 | 17 | 70.6% |
| Overlapped | 4 | 6 | 66.7% |
| Clustered | 5 | 17 | 29.4% |
| Small | 12 | 50 | 24.0% |

## Interpretation

This dossier is intentionally a stress benchmark, not a clean regression set.
The earlier synthetic dossiers remain the correctness regressions. This mixed
set is intended for measuring improvements in small-symbol recovery, clustered
candidate separation, error-guided decoding, and localization without creating
false positives on zero-code forms.
