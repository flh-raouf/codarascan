# Shared engine runtime

This directory contains the repository-owned implementation shared by Tessera
and Mosaic:

- `recovery/` — deterministic localization, ZXing recovery, matrix recovery,
  and the adaptive runtime;
- `classical_*.py` — QR/Data Matrix proposal and validation stages;
- `tensor_*.py` — structure-tensor localization and candidate decoding;
- `linear_recovery.py` and `review_gate.py` — physical review gates; and
- `native/` — optional C++ acceleration for the structure-tensor proposal stage.

The product adapters under [`../tessera/`](../tessera/README.md) and
[`../mosaic/`](../mosaic/README.md) compose these modules. They import only
`codarascan` modules plus the declared NumPy, OpenCV, and ZXing-C++
dependencies.

The native extension is optional. The Python implementation remains available
when `_sttg_native` has not been built; see [`native/setup.py`](native/setup.py)
when the optimized extension is needed. From the repository root, the optional
build is:

```bash
python src/codarascan/engines/common/native/setup.py build_ext --inplace
```
