# Migration to CodaraScan 0.1

The former development distribution `barcode-detection` and import package
`barcode_detection` are replaced by `codarascan`. There is no compatibility
shim in 0.1.0.

```python
# Before: private engine access
from barcode_detection.engines import get_engine

# After: public package contract
from codarascan import Scanner

scanner = Scanner(mode="fast", symbols="all", decode=True)
result = scanner.scan_image("page.png")
```

Application code should not import `codarascan.engines`, recovery modules, or
the native extension directly. Use `Scanner`, public result/error types,
`supported_formats`, serialization helpers, CLI output, or the framed worker.

Poppler is no longer a runtime requirement. PDF callers should use
`scan_document` or `iter_document`; page numbers are one-based, duplicates are
invalid, and output follows requested order. `workers` now defaults explicitly
to 1.

Localization-only (`decode=False`) results intentionally have no `text`,
`value`, `raw_bytes`, or `format` attributes. Consumers must branch on status
or result type rather than expect null payload fields.
