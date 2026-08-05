# Tensor candidate decoding budget

This experiment isolates the ZXing work after the Tensor fast localizer. The
production default remains the original exhaustive policy; `compact-8` is
available only through the internal `tensor_decode_policy` engine option while
the result is reviewed.

## Why the original tail was expensive

An unresolved Tensor proposal could enter:

- seven padded/scaled crop reads;
- four additional angle reads;
- 84 advanced angle/phase/binarizer combinations; and
- eight scanline-consensus reads.

That is up to 103 ZXing calls for one candidate. Most ordinary symbols exit on
the first read, but false or genuinely unreadable candidates pay the complete
failure portfolio.

## Compact portfolio

The evidence-selected replacement has an eight-call ceiling:

1. native rectified crop;
2. 1.5x cubic crop;
3. 2x cubic crop;
4. wide 3x cubic crop;
5. wide CLAHE 3x crop;
6. one +6 degree binary correction;
7. one -2 degree nearest/Otsu correction; and
8. one median scanline reconstruction.

ZXing's internal rotation and inversion search are disabled because Tensor has
already rectified an ordinary dark-on-light candidate. The last three views
were retained because each had a unique verified win; the remaining failure
portfolio was removed.

## Candidate-decoder benchmark

Image preparation and Tensor localization were shared between policies.
Ground-truth scoring used one-to-one containment-aware geometry and exact
payload strings.

| Dataset | Exhaustive ZXing | Compact-8 ZXing | Speedup | Result comparison |
|---|---:|---:|---:|---|
| private evaluation document, 41 linear candidates | 50.3 ms / 82 calls | 33.2 ms / 67 calls | 1.51x | 41 exact, 0 wrong under both |
| Balanced, 61 candidates | 25.5 ms / 61 calls | 22.4 ms / 61 calls | 1.14x | 60 exact, 1 wrong under both |
| Severe stress, 63 candidates | 775.3 ms / 614 calls | 108.4 ms / 128 calls | 7.15x | 55 exact, 2 wrong, 5 unresolved under both |
| private corpus A, 29 candidates | 3348.9 ms / 2980 calls | 350.3 ms / 232 calls | 9.56x | Same 1 decoded and 28 unresolved |

The balanced wrong result is inherited and identical under both policies; the
compact policy introduced no new decoded values or wrong payloads.

The naïve one-, two-, and three-call policies were rejected. On the Quality
Dossier they recovered only 33, 34, and 36 of 41 linear payloads respectively.

## Complete private corpus A engine benchmark

Three paired concurrent runs used the same 315 normalized pages.

### Linear-only

| Measurement | Exhaustive median | Compact-8 median | Change |
|---|---:|---:|---:|
| User-visible wall time | 5.244 s | 4.400 s | 16.1% faster |
| Summed engine compute | 10.677 s | 3.660 s | 65.7% lower |
| Summed candidate ZXing | 7.815 s | 0.821 s | 89.5% lower |
| Semantic output | 1 decoded, 28 review | Identical | No change |

Preparation remained about 30.3 summed page-seconds and overlapped across
workers. It is now the dominant linear-only wall-time cost.

### All symbols

| Measurement | Exhaustive median | Compact-8 median | Change |
|---|---:|---:|---:|
| User-visible wall time | 15.377 s | 15.102 s | 1.8% faster |
| Summed engine compute | 87.819 s | 85.090 s | 3.1% lower |
| Summed candidate ZXing | 6.368 s | 0.728 s | 88.6% lower |
| Semantic output | 7 decoded, 28 review, 127 regions | Identical | No change |

In all-symbol mode, Adaptive QR/Data Matrix work dominates the run. Compact
linear decoding is therefore valid but cannot materially solve the independent
2-D bottleneck.

## Reproduction

This is a historical Codara integration benchmark. Its original benchmark
script and application runtime are not part of this repository; the migrated
Tessera/Mosaic runtime is exercised through the public registry instead.

```bash
# Run from the corresponding Codara backend checkout. The benchmark script
# and its runtime dependencies are not included in this repository.
cd /path/to/codara-app/apps/backend
.venv/bin/python benchmarks/tensor_decode_budget.py \
  --datasets quality,balanced,stress,vn \
  --output /tmp/tensor-decode-budget.json
```
