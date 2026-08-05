# Official versions

The numbered folders are preserved in chronological order with repository-safe names. Their original numeric identity is kept in each folder name and in this table.

| Version | Historical folder | Focus |
| --- | --- | --- |
| 1 | [`v1-deterministic-locator`](v1-deterministic-locator/) | Deterministic barcode localization |
| 2 | [`v2-hybrid-pipeline`](v2-hybrid-pipeline/) | Hybrid localization and decoding |
| 3 | [`v3-zxing-only`](v3-zxing-only/) | ZXing-only barcode pipeline |
| 4 | [`v4-zxing-2d`](v4-zxing-2d/) | ZXing 2-D barcode pipeline |
| 5 | [`v5-optimized`](v5-optimized/) | Optimized pipeline |
| 5.1 | [`v5.1-adaptive-generalization`](v5.1-adaptive-generalization/) | Adaptive generalization |
| 5.2 | [`v5.2-evidence-guided`](v5.2-evidence-guided/) | Evidence-guided coarse-to-fine pipeline |
| 5.3 | [`v5.3-fast-extraction`](v5.3-fast-extraction/) | Fast extraction experiments |
| 6 | [`v6-localization-only`](v6-localization-only/) | Localization-only pipeline |
| 7 | [`v7-coarse-to-fine`](v7-coarse-to-fine/) | CPU coarse-to-fine localization |
| 7.1 | [`v7.1-fast-localization`](v7.1-fast-localization/) | Fast localization |
| 8 | [`v8-structure-tensor`](v8-structure-tensor/) | Structure-tensor linear localization |

The versions are baselines, not a second active package. When a historical script needs a dependency alias, use the root `numbered_project_aliases.py` compatibility helper or the version's own README.
