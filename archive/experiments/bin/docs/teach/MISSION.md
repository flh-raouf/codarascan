# Mission: Deterministic Barcode Localization

## Why
Learn how to locate every barcode-like symbol in scanned industrial automotive quality documents, then draw reliable bounding boxes so later code can decode only the right cropped regions.

## Success looks like
- Explain the difference between 1D barcodes, QR codes, and other 2D codes.
- Inspect a dossier page and predict which visual features help or hurt deterministic detection.
- Build and tune an OpenCV-based detector for rotated, multi-instance barcode regions.
- Evaluate false positives, false negatives, and decoding readiness on `Quality Dossier.pdf`.

## Constraints
- Prefer deterministic computer vision before training a model.
- Ground lessons in the real scanned A3 dossier pages in this workspace.
- Treat decoding as the second milestone; localization is the first hard problem.

## Out of scope
- Training YOLO or another detector unless classical detection has clear measured failure.
- Building a production pipeline before the detection principles are understood.
