# Benchmarks

This is the active evaluation area, separate from both production adapters and historical experiments.

- [`lab/`](lab/README.md) contains the neutral benchmark harness and unit tests.
- [`datasets/`](datasets/) contains committed synthetic fixtures and manifests.
- [`results/`](results/) contains committed benchmark notes and budget reports.

Local manifests, external datasets, run outputs, and generated reports remain ignored. Reproducible commands should be written relative to `benchmarks/` and should identify their input provenance.

The benchmark lab's external dataset registry is [`lab/datasets.json`](lab/datasets.json). It records source pages, licensing notes, pinned versions, and checksums for restoring large inputs without storing them in Git. Use [`lab/fetch_data.py`](lab/fetch_data.py) to list, verify, fetch, and extract registered inputs.
