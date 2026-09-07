# Third-party notices

CodaraScan depends on or interoperates with the projects below. Runtime wheels
retain their own license metadata and license files; CodaraScan does not vendor
their source. The tested-version column records the local 0.1.0 release audit,
while the declared range remains authoritative for dependency resolution.

| Project | Declared range; locally audited version | Role and source | License/provenance | Modifications, obligations, and compatibility |
|---|---|---|---|---|
| NumPy | `>=2,<3`; 2.4.6 | Array processing and native build headers; https://numpy.org/ | Wheel metadata: BSD-3-Clause plus identified permissive bundled components (0BSD, MIT, Zlib, CC0-1.0) | External dependency, unmodified. Preserve the dependency's notices; all listed terms permit redistribution alongside Apache-2.0. |
| OpenCV / `opencv-contrib-python-headless` | `>=4.10,<6`; 5.0.0.93 | Image processing and robust localization; https://opencv.org/ | Wheel metadata: Apache-2.0; the wheel carries its own third-party notices | External dependency, unmodified. Preserve packaged notices; Apache-2.0 is redistribution-compatible. |
| ZXing-C++ / `zxing-cpp` | `>=3,<4`; 3.1.1 | Barcode decoding, writing, and format definitions; https://github.com/zxing-cpp/zxing-cpp | Package metadata and upstream source: Apache-2.0 | External runtime dependency, unmodified. One upstream QR Model 1 image is used only from the pinned test submodule with traceable path/revision; it is excluded from wheel and sdist artifacts. Preserve attribution and license. |
| Pillow | `>=10,<13`; 12.3.0 | Encoded-image loading and EXIF orientation; https://python-pillow.org/ | Package metadata: MIT-CMU | External dependency, unmodified. Retain its permission notice; permissive terms are compatible with Apache-2.0 distribution. |
| PDFium / `pypdfium2` | `>=4.30,<6`; 5.12.1 | Local PDF rendering; https://github.com/pypdfium2-team/pypdfium2 | Package metadata: BSD-3-Clause and Apache-2.0, with dependency license materials in its distribution | External dependency, unmodified. Preserve the dependency's bundled notices and licenses; the declared terms are compatible with Apache-2.0 distribution. |
| Tessera native extension | repository source at `src/codarascan/engines/common/native/sttg_native.cpp`; 0.1.0 | Structure-tensor acceleration maintained in this repository | Authored project source, Apache-2.0 | Modified as part of the CodaraScan package/native-loader boundary. Distributed with SPDX identification, LICENSE, and NOTICE. |
| Research-derived Tessera/Mosaic/Panorama Python modules | repository history under `src/codarascan/engines`; 0.2.0 | Canonical engine implementation maintained in this repository | Authored project source, Apache-2.0 | Reorganized and adapted behind the public Scanner contract. No external corpus or confidential document is bundled. Distributed under the project LICENSE and NOTICE. |

The `archive/`, benchmark datasets, reports, pinned third-party checkout, and
ignored research material are not installed as package data. `tools/inspect_artifacts.py`
checks the definitive wheel and sdist for license materials, native source,
forbidden paths, sensitive content, and machine-local paths. The resolved
artifact hash and declared dependency inventory are generated for every release;
dependency package license files remain the source of truth for their complete
transitive attribution.
