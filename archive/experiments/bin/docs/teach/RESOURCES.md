# Barcode Localization Resources

## Knowledge

- [GS1 US: Types of Barcodes](https://www.gs1us.org/upcs-barcodes-prefixes/barcode-types)
  High-trust overview of 1D and 2D barcode families. Use for: vocabulary, symbologies, and barcode vs QR code distinctions.
- [OpenCV: Recognizing one-dimensional barcode using OpenCV](https://opencv.org/recognizing-one-dimensional-barcode-using-opencv/)
  Explains OpenCV's 1D barcode algorithm: localization, directional coherence, binarization, decoding, and why the pipeline is split.
- [OpenCV docs: `cv::barcode::BarcodeDetector`](https://docs.opencv.org/4.x/dc/df7/classcv_1_1barcode_1_1BarcodeDetector.html)
  API reference for detecting and decoding 1D barcodes, including quadrangle outputs and detector tuning knobs.
- [OpenCV docs: `cv::QRCodeDetector`](https://docs.opencv.org/4.x/de/dc3/classcv_1_1QRCodeDetector.html)
  API reference for QR detection and decoding, including multi-code detection and the QR finder-pattern tolerance knobs.
- [ZXing: multi-format barcode processing library](https://github.com/zxing/zxing)
  Established open-source decoder supporting many 1D and 2D formats. Use for: understanding decoder coverage and later crop validation.
- [pyzbar on PyPI](https://pypi.org/project/pyzbar/)
  Python wrapper around ZBar that returns decoded data plus rectangles/polygons. Use for: fast decoder experiments on cropped regions.
- [Barcode Detection with Morphological Operations and Clustering](https://publicatio.bibl.u-szeged.hu/8602/2/barcode_morph_02.pdf)
  Classical detection paper focused on morphology and clustering. Use for: deterministic localization ideas before deep learning.
- [State-of-the-art review and benchmarking of barcode localization methods](https://www.sciencedirect.com/science/article/pii/S0952197625002593)
  Open-access survey and benchmark paper. Use for: comparing classical and learned approaches without guessing.
- [BarBeR Barcode Benchmark Repository](https://ditto.ing.unimore.it/barber/)
  Dataset and benchmark companion with 1D and 2D barcode annotations. Use for: later benchmarking ideas and annotation formats.

## Wisdom (Communities)

- [OpenCV Forum](https://forum.opencv.org/)
  Practical community for debugging image processing pipelines and OpenCV API behavior.
- [ZXing GitHub Issues](https://github.com/zxing/zxing/issues)
  Useful for decoder edge cases and supported-format behavior, especially once cropped regions are ready.

## Gaps

- Need a dossier-specific ground-truth file listing expected barcode boxes per page.
- Need to identify the exact 1D symbologies used in the automotive pages after decoding representative clean crops.
