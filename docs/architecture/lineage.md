# Engine lineage

The project history is a progression, not a collection of unrelated folders:

```text
bin/ and notebooks/
        ↓ exploratory experiments
1 … 8 numbered releases
        ↓ validated ideas copied into the Codara application
Codara engine adapters
        ↓ product naming and composition
Tessera (recommended fast tier) + Mosaic (robust base tier)
```

## Current mapping

| Product line | Capability | Adapter snapshot | Role |
| --- | --- | --- | --- |
| Tessera | Detection | `src/barcode_detection/integrations/codara/engines/detection_tensor_p7.py` | Recommended |
| Tessera | Extraction | `src/barcode_detection/integrations/codara/engines/extraction_tensor_adaptive.py` | Recommended |
| Mosaic | Detection | `src/barcode_detection/integrations/codara/engines/detection_guarded_v3.py` | Base/fallback |
| Mosaic | Extraction | `src/barcode_detection/integrations/codara/engines/extraction_guarded_v3.py` | Base/fallback |
| Shared contract | Detection/extraction result types | `src/barcode_detection/core/contracts.py` | Stable core |

The codebase spells the fast product line **Tessera**. That spelling is preserved as the canonical name even when earlier conversation or notes use “Teserra.”

## What is and is not in this repository

The adapters are copied from the Codara development line so their behavior and provenance can be reviewed here. Their imports still resolve against Codara-side packages such as `pipeline` and `localization`; the Codara checkout is not present in this repository. A future migration should move the selected implementation modules into `src/barcode_detection/engines/tessera` and `src/barcode_detection/engines/mosaic` only after their dependencies and tests have been made local.

The old numbered implementations remain under [`archive/versions`](../../archive/versions/README.md). They are valuable baselines and reproducibility references, but they are not the active product registry.
