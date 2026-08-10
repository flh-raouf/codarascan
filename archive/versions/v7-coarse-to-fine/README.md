# 7 CPU coarse-to-fine localization

Version 7 is a CPU-only localization pipeline. It finds barcode-like regions
but does not decode payloads. Linear symbols are proposed with reduced-scale
OpenCV directional coherence and verified against native pixels. QR candidates
must show finder-pattern geometry, while Data Matrix candidates must pass an
ECC 200 finder, timing-border, and module-grid proof. No GPU is required.

This is frozen historical code. The later [v7.1 fast-localization](../v7.1-fast-localization/)
and [v8 structure-tensor](../v8-structure-tensor/) experiments evolved the same
coarse-to-fine direction.

## Entry points

- [`pipeline.py`](pipeline.py) — implementation and CLI
- [`run.py`](run.py) — executable wrapper
- [`regression.py`](regression.py) — historical regression checks
- [`2D_RESEARCH_AND_BENCHMARK.md`](2D_RESEARCH_AND_BENCHMARK.md) — research and benchmark notes

From the repository root, run it against an input PDF or image:

```bash
python archive/versions/v7-coarse-to-fine/run.py /path/to/input.pdf \
  --output /tmp/barcode-detection-v7-run \
  --overwrite
```

The example keeps generated output outside the repository so the archive stays
focused on source and historical evidence.
