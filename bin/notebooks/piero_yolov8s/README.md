# Piero YOLOv8s Barcode Detector

Self-contained baseline for testing `Piero2411/YOLOV8s-Barcode-Detection` on
the dossier PDF.

## Install

```bash
python3 -m pip install -r piero_yolov8s/requirements.txt
```

`pdftoppm` must also be available. On macOS:

```bash
brew install poppler
```

## Run

Default test pages are `16,20,21`.

```bash
python3 piero_yolov8s/run.py
```

Or open this notebook in Jupyter:

```text
piero_yolov8s/run_in_jupyter.ipynb
```

Equivalent explicit command:

```bash
python3 piero_yolov8s/run.py notebooks/data/pdfs/Quality\ Dossier.pdf \
  --pages 16,20,21 \
  --dpi 250 \
  --imgsz 1280 \
  --conf 0.10 \
  --save-crops
```

Outputs are written to:

```text
piero_yolov8s/output/
  pages/
  debug/
  crops/
  detections.json
```

The model weights are downloaded once to:

```text
piero_yolov8s/models/YOLOV8s_Barcode_Detection.pt
```

## Baseline Plan

This implementation intentionally does only the first useful pass:

1. download the Piero weights
2. render selected PDF pages
3. run full-page YOLO inference
4. write debug images, crops, and JSON

Evaluate the debug images before adding tiling, decoding, fine-tuning, or
rotated-box refinement.

Suggested comparison matrix:

```text
pages: 16,20,21
imgsz: 1280
conf: 0.10, 0.20, 0.35
```

`conf=0.10` is the default because it catches the low-confidence header VIN
barcode on page 16 without adding extra detections on pages 20 or 21 in the
first baseline run.

If small labels are still missed at `imgsz=1280`, try:

```bash
python3 piero_yolov8s/run.py --imgsz 1600 --force-render
```
