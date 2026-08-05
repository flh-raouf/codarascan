# Piero YOLOv8s model provenance

This experiment uses the external Hugging Face baseline:

- Repository: `Piero2411/YOLOV8s-Barcode-Detection`
- File: `YOLOV8s_Barcode_Detection.pt`
- Acquisition: [`run.py`](run.py) downloads it with
  `huggingface_hub.hf_hub_download` when the local copy is absent.
- Expected local path: `models/YOLOV8s_Barcode_Detection.pt`

The downloaded weight is intentionally kept outside normal Git history because
it is a large external baseline asset. The experiment code, notebook, install
instructions, and this provenance record remain in the repository so the
baseline can be recreated when needed.

## Local integrity check

SHA-256 of the copy previously present in this archive:

```text
316ded312281da5d4de06c36c66fdc682bd1c2052689008237baf22eb8e4f5ed  YOLOV8s_Barcode_Detection.pt
```

The checksum identifies the local artifact; it does not establish ownership or
redistribution rights. Check the upstream repository's current license and
terms before copying the weight outside this experiment.
