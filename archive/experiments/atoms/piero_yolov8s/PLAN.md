# Piero YOLOv8s Plan

## Goal

Find barcode-like regions in scanned dossier pages, then hand the crops to a
decoder later. Detection quality comes first; decoding is intentionally out of
scope for this pass.

## Baseline

Run the model on full rendered pages:

```bash
./venv/bin/python piero_yolov8s/run.py
```

Defaults:

```text
pages: 16,20,21
dpi: 250
imgsz: 1280
conf: 0.10
iou: 0.45
device: cpu
```

Why `conf=0.10`: the first local check found the page 16 header VIN barcode at
`0.148` confidence, while pages 20 and 21 kept the same detection counts as
`conf=0.20`.

## Evaluation

Use the debug images, not only `detections.json`.

Pass criteria:

```text
page 16: finds the header VIN barcode and the visible pasted/body labels
page 20: finds header barcodes without text/table false-positive flooding
page 21: finds most pasted barcode labels
```

## Escalation

Only add complexity in this order:

1. Increase `imgsz` to `1600` if small labels are missed.
2. Try `conf=0.20` or `0.35` if false positives appear.
3. Add tiled inference only if larger `imgsz` still misses small labels.
4. Add rotated-box refinement only if crops are too loose for decoding.
5. Add decoding after detection boxes are stable.

Skipped for now: tiling, barcode decoding, fine-tuning, ONNX export, and
rotated quadrilateral output.
