#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import zxingcpp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from barcode_benchmark.io import DATASET_SCHEMA, write_json


VARIANTS = {
    "one": "extracted One",
    "quad": "extracted Quad",
    "voted": "extracted Voted",
}


def target_payload(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read target image: {path}")
    values = {
        str(result.text)
        for result in zxingcpp.read_barcodes(image)
        if result.valid and str(result.text)
    }
    if len(values) != 1:
        raise ValueError(f"expected one unique valid payload in {path}, got {values!r}")
    return next(iter(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import the QR-DN1.0 decode benchmark")
    parser.add_argument("root", type=Path, help="extracted QR-DN1.0 root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    arguments = parser.parse_args()
    root = arguments.root.expanduser().resolve()
    target_root = root / "target" / arguments.split
    payloads = {
        path.stem: target_payload(path)
        for path in sorted(target_root.glob("*.jpg"))
    }
    if not payloads:
        raise ValueError(f"no target images found under {target_root}")

    images = []
    for variant, folder in VARIANTS.items():
        source_root = root / folder / arguments.split
        source_paths = {path.stem: path for path in sorted(source_root.glob("*.jpg"))}
        if set(source_paths) != set(payloads):
            raise ValueError(
                f"{folder} IDs do not match target IDs: "
                f"missing={sorted(set(payloads) - set(source_paths))[:5]!r}"
            )
        for numeric_id, path in source_paths.items():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read image: {path}")
            height, width = image.shape[:2]
            images.append({
                "id": f"{variant}/{numeric_id}.jpg",
                "path": str(path.resolve()),
                "width": width,
                "height": height,
                "source_group": f"qrdn-{variant}",
                "split": arguments.split,
                "objects": [{
                    "id": "payload-0001",
                    "polygon": [
                        [0.0, 0.0],
                        [float(width), 0.0],
                        [float(width), float(height)],
                        [0.0, float(height)],
                    ],
                    "kind": "2d",
                    "symbology": "QR_CODE",
                    "payload": payloads[numeric_id],
                    "decodable": True,
                    "localization_evaluable": False,
                    "payload_scope": "image",
                    "attributes": {
                        "degradation_variant": variant,
                        "paired_target": str((target_root / f"{numeric_id}.jpg").resolve()),
                    },
                }],
            })

    value = {
        "schema_version": DATASET_SCHEMA,
        "dataset": {
            "name": f"qrdn-v1-{arguments.split}",
            "root": str(root),
            "annotation_format": "paired-clean-target",
            "source_url": "https://data.mendeley.com/datasets/t2bdr663ms/2",
            "doi": "10.17632/t2bdr663ms.2",
            "license": "CC BY 4.0",
            "archive_sha256": "1f175a62239646bd7d6b179245cb0970c03b179c2baf1a5e8e59ba0b156cdf61",
            "image_level_payload": True,
            "variants": list(VARIANTS),
        },
        "images": images,
    }
    write_json(arguments.output, value)
    print(f"{arguments.output.resolve()} ({len(images)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
