# Compatibility and format status

## Interpreter and platform matrix

CodaraScan 0.2.0 declares CPython 3.11, 3.12, 3.13, and 3.14. The official
wheel target matrix is:

| Platform | Architectures | Wheel expectation |
|---|---|---|
| Linux glibc/manylinux | x86_64, ARM64 | Native Tessera extension included |
| macOS | Intel x86_64, Apple Silicon ARM64 | Native Tessera extension included |
| Windows | x86_64 | Native Tessera extension included |

PyPy, CPython 3.10 and older, Alpine/musl, 32-bit systems, and Windows ARM64 are
outside the initial support contract. A platform becomes release-verified only
after its clean wheel job and artifact smoke tests pass; source-checkout success
alone is not an official support claim.

Runtime dependencies are bounded in `pyproject.toml`: NumPy 2.x, headless
OpenCV contrib 4.10-5.x, zxing-cpp 3.x, Pillow 10-12, and pypdfium2 4.30-5.x.

## Selectable formats

The catalog is computed from ZXing-C++ `AllReadable`, excluding only `Other
barcode`, which is a reserved catch-all identifier rather than a concrete
symbology reader. Writer-only EAN-2 and EAN-5 supplements are not exposed as
independent readable filters.

Linear selections (27): Codabar, Code 128, Code 32, Code 39 and its standard/
extended views, Code 93, DataBar variants, DX Film Edge, EAN/UPC, EAN-13,
EAN-8, ISBN, ITF/ITF-14, PZN, Telepen variants, UPC-A, and UPC-E.

Matrix selections (13): Aztec/Aztec Code/Aztec Rune, PDF417/Compact PDF417/
MicroPDF417, QR Code/Model 1/Model 2/Micro QR/rMQR, Data Matrix, and MaxiCode.

Some ZXing selectors are semantic or family views. ISBN routes through EAN-13
and applies the 978/979 condition. Code 39 Standard, ITF-14, Aztec Code,
Compact PDF417, Aztec Rune, and QR model selectors may be reported by ZXing as
their canonical decoded family. The returned `format` reflects the valid
decoded format, except the explicit ISBN semantic result.

The 0.1.0 clean-fixture matrix is in
`benchmarks/results/FORMAT_STATUS_0.1.0.md`. It is evidence of smoke routing,
not a production accuracy claim.
