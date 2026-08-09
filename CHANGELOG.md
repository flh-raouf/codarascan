# Changelog

All notable changes are recorded here. CodaraScan follows semantic versioning
after 1.0; during 0.x, breaking changes are identified explicitly.

## 0.1.0 - Unreleased alpha

- Renamed the distribution and import package to `codarascan`.
- Added immutable, reusable Tessera/Mosaic `Scanner` APIs and cached one-off calls.
- Added image adapters, EXIF orientation, normalized ROI, full-image quads,
  explicit statuses, raw-byte preservation, metadata, diagnostics, typed errors,
  deterministic JSON, and published schemas.
- Exposed the concrete installed ZXing-C++ readable catalog: 27 linear and 13
  matrix selections, with clean generated or Apache-2.0 fixtures.
- Added lazy native Tessera loading with an exact once-per-process warning and
  tested Python-reference fallback.
- Replaced every installed Poppler runtime path with pypdfium2 native-raster
  reuse or 300-DPI PDFium rendering.
- Added ordered document scanning/streaming, one-based page selection, bounded
  workers, cancellation cleanup, and raise/collect error policies.
- Added human, JSON, and NDJSON CLI output plus a private framed-JSON worker.
- Added Apache-2.0 licensing, notices, typed-package metadata, compatibility and
  release documentation, CI/release workflows, and artifact tests.

Known validation gaps are listed in `docs/KNOWN_LIMITATIONS.md`; 1.0 production
gates are not claimed by this alpha.
