# Tessera

Tessera is the recommended fast product line. Its current adapter snapshot combines tensor-based 1-D proposals with the validated classical 2-D route.

Source mapping:

- Detection: `src/barcode_detection/integrations/codara/engines/detection_tensor_p7.py`
- Extraction: `src/barcode_detection/integrations/codara/engines/extraction_tensor_adaptive.py`

Move implementation code here only when the Codara-side `pipeline` and `localization` dependencies are part of the local package and covered by repository tests.
