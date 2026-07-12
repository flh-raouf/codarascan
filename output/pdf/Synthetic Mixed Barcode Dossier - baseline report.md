# Synthetic Mixed Barcode Dossier — baseline report

## Purpose

This is the balanced replacement for the earlier extreme stress dossier. It keeps realistic scan degradation and layout variation, but the barcodes are generally larger, given usable quiet zones, and not routinely placed over one another.

## Dataset composition

- 50 scanned-image PDF pages: 44 portrait and 6 landscape
- 122 ground-truth symbols
- Formats: 25 Code 39, 38 Code 128, 38 Data Matrix, and 21 QR Code
- Five intentionally empty pages: 7, 16, 28, 39, and 48
- Most non-empty pages contain 1–4 symbols
- Busy but readable pages: page 10 (8 symbols), page 25 (7), page 38 (9), and page 46 (10)
- Difficulty labels: 59 easy, 53 medium, and 10 hard
- Moderate scan noise on 95 symbols and heavier scan noise on 27 symbols
- Includes skew, mild blur, compression artifacts, uneven illumination, scan streaks, edge placements, rotation, and several nearby-code layouts

## Baseline command

```bash
venv/bin/python zxing_2d_barcode_pipeline/run.py \
  "output/pdf/Synthetic Mixed Barcode Dossier - 50 balanced scanned pages.pdf" \
  --output zxing_2d_barcode_pipeline/output/balanced-mixed-dossier \
  --formats all --overwrite --no-crops
```

## Exact ground-truth comparison

| Measure | Result |
|---|---:|
| Exact payload + format + page matches | 120 / 122 (98.36%) |
| Decoded false positives | 0 |
| Symbols localized in any output tier | 121 / 122 (99.18%) |
| Unmatched localization outputs | 0 |
| Empty pages incorrectly flagged | 0 / 5 |

### Recall by format

| Format | Exact matches | Recall |
|---|---:|---:|
| Code 39 | 25 / 25 | 100.0% |
| Code 128 | 37 / 38 | 97.4% |
| Data Matrix | 37 / 38 | 97.4% |
| QR Code | 21 / 21 | 100.0% |

### Recall by difficulty

| Difficulty | Exact matches | Recall |
|---|---:|---:|
| Easy | 59 / 59 | 100.0% |
| Medium | 53 / 53 | 100.0% |
| Hard | 8 / 10 | 80.0% |

## Remaining challenging cases

1. Page 25: a hard, nearby Data Matrix was localized into the unresolved-matrix tier but not decoded.
2. Page 46: a hard Code 128 was not localized or decoded. This is one of ten symbols on a busy landscape page.

These two cases preserve useful headroom without making the entire dossier artificially hostile.

## Integrity checks

- Strict PDF parse: passed
- PDF pages: 50
- Embedded page images: 50
- Ground-truth CSV rows: 122
- Ground-truth JSON symbols: 122
- PDF SHA-256: `395921ae21c850cf4e0c0e6616fdc2d36991be4b7088ba4000c41666abe61bc4`
