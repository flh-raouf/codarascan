# Exploratory experiments

This directory contains the research work that preceded the official numbered
versions. It is organized by the size of the experiment rather than by product
status:

```text
archive/experiments/
├── atoms/                         isolated tool and model probes
│   ├── api-dynamsoft.ipynb        Dynamsoft decoder probe
│   ├── dbr.ipynb                  DBR decoder probe
│   ├── opencv.ipynb               OpenCV detector probe
│   ├── pyzbar.ipynb               pyzbar probe
│   ├── quagga.ipynb               Quagga probe
│   ├── zxing-cpp.ipynb            ZXing-C++ probe
│   └── piero_yolov8s/              YOLOv8s learned baseline
└── pipelines/                     end-to-end composition experiments
    ├── 1-no count.ipynb           decoder-first, no-count baseline
    ├── 2-opencv-at-end.ipynb      OpenCV proposals plus decode validation
    ├── 3-rotated tiles.ipynb      rotated/tiled recovery
    ├── 4-deterministic-decoder-pipeline.ipynb
    ├── 5-opencv.ipynb             OpenCV-only comparison
    └── dynamsoft-decoder-pipeline.ipynb
```

The progression is:

```text
atoms → pipelines → official versions → current Tessera/Mosaic engines
```

`atoms/` answers narrow questions about individual decoders, libraries, and
learned baselines. `pipelines/` combines those findings into complete page
processing experiments and records how the architecture evolved before the
numbered releases.

These files are frozen research history. They may refer to local dossier data,
external packages, or machine-specific output paths that are not part of this
repository. They are references for comparison, not current APIs or production
entry points.
