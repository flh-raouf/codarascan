# Repository checks

The default test suite verifies repository boundaries, model metadata, and the
local Tessera/Mosaic registry. Component tests for historical versions remain
beside their version; benchmark tests remain under `benchmarks/lab/tests` and
are run explicitly with the optional benchmark dependencies.

```bash
python -m pytest
python -m pytest benchmarks/lab/tests
python -m pytest archive/versions/v5.2-evidence-guided/tests
```
