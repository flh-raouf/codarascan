# Public API contract

## Scanner configuration

`Scanner(mode="fast", symbols="all", formats=None, decode=True)` is immutable,
hashable, lazily initialized, reusable, and safe to share across threads.

- `mode`: `fast` selects Tessera; `robust` selects Mosaic. There is no `auto`
  mode or cross-engine fallback.
- `symbols`: `linear`, `2d`, or `all`.
- `formats`: optional canonical names or accepted aliases. Contradictions with
  `symbols` and unknown names fail at construction.
- `decode`: a strict boolean. False produces geometry-only public results.

`warm()` performs the same idempotent initialization used on first scan.

## Image calls

`scan_image(image, *, roi=None, diagnostics=False)` accepts:

- `str` or `Path` image paths;
- encoded image `bytes`, `bytearray`, or `memoryview`;
- Pillow images;
- 2-D `uint8` grayscale NumPy arrays;
- 3-channel BGR or 4-channel BGRA `uint8` arrays.

File-like streams, dtype conversion, RGB array guessing, and a color-space
parameter are not part of 0.1. Corrupt images and ambiguous arrays raise
`InvalidImageError`. The package snapshots pixels before scanning. EXIF
orientation is applied to encoded/Pillow inputs; arrays contain no metadata.

ROI is normalized `(x, y, width, height)` against the orientation-corrected
image. Values are finite and in `[0,1]`, width/height are positive, and the ROI
may not leave the image. Returned geometry remains in full-image coordinates.

## Document calls

`scan_document(pdf, *, pages=None, workers=1, on_error="raise", roi=None,
diagnostics=False)` accepts PDF paths or encoded PDF bytes and returns a
`DocumentResult`. `iter_document` accepts the same options and yields
`PageResult` objects without accumulating them.

Page numbering is one-based. `pages=None` means natural document order;
otherwise the exact requested order is preserved. Empty, duplicate, noninteger,
or out-of-range selections fail before scanning. `workers` is a positive
integer or `auto`; explicit integers are bounded only by selected page count,
while `auto` uses `min(os.cpu_count(), selected_pages)`.

PDFium access and rendering are serialized. Sole, axis-aligned, unrotated
full-page images are reused at native resolution. Other pages render at 300
DPI. Only owned pixel snapshots cross into analysis threads. At most `workers`
pages are in flight.

`on_error="raise"` is fail-fast. `collect` omits failed pages from iteration,
records typed entries in `DocumentStream.errors`/`DocumentResult.errors`, and
sets `complete=False`. Closing a stream stops scheduling, cancels work not yet
started, and releases PDF resources.

## Geometry and results

Pixel coordinates use an origin at the orientation-corrected image's top-left,
with x increasing right and y increasing down. Quads contain top-left,
top-right, bottom-right, and bottom-left corners in clockwise, non-self-
intersecting order. `normalized_quad` divides x by image width and y by image
height. There is no stable bounding-box field.

`DecodedSymbolResult` adds `text`, `value`, `raw_bytes`, and canonical `format`
to the base geometry/status contract. `SymbolResult` used for localization-only,
unresolved, and review states deliberately has none of those attributes.

Confidence is finite in `[0,1]` but is an engine-specific evidence/ranking
score, not a calibrated probability. Use status as the primary semantic field.

All image, page, and document results include elapsed time plus package,
engine, decoder, backend, and mode metadata. Diagnostics are opt-in and are not
a stable long-term field contract.

## Errors and serialization

Expected failures derive from `CodaraScanError` and carry a stable `code`, safe
message, and structured context. The public hierarchy covers configuration,
unsupported formats, invalid images, documents/rendering, page processing,
native loading, serialization/protocol, and internal engine-contract failures.

`to_json` uses UTF-8 text, sorted keys, strict finite JSON numbers, and Base64
raw bytes. `result.schema.json`, `stream.schema.json`, and `worker.schema.json`
ship as package data alongside decoded, localized, and stream golden examples.
Consumers should load them with `importlib.resources.files("codarascan")` rather
than assuming a filesystem installation layout.
