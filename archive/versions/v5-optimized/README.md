# 5 Optimized barcode pipeline

Version 5 is the throughput-optimized successor to [v4](../v4-zxing-2d/).
It preserves the ZXing/OpenCV recovery cascade while reducing repeated work:

- one proposal-only OpenCV pass;
- grayscale buffers;
- batch PDF extraction; and
- page-level concurrency.

The archived implementation supports Code 39, Code 128, Data Matrix, and QR
Code. It is frozen historical code; current active engines live under
`src/barcode_detection/engines/`.

## Entry points

- [`pipeline.py`](pipeline.py) — implementation and CLI
- [`run.py`](run.py) — executable wrapper
- [`regression.py`](regression.py) — historical regression checks

From the repository root, a run can be started with:

```bash
python archive/versions/v5-optimized/run.py INPUT \
  --output archive/versions/v5-optimized/output/run \
  --overwrite
```

The output directory is generated and ignored. The repository history did not
contain a README for this version; this file is a retrospective archive index,
not a recovered source report.
