#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from barcode_benchmark.adapters import adapt_current_detections
from barcode_benchmark.evaluate import evaluate
from barcode_benchmark.importers import (
    import_cvat_image_xml,
    import_deal,
    import_image_payload_texts,
    import_pascal_voc,
    import_pascal_voc_separate,
    import_vgg,
    import_yolo,
)
from barcode_benchmark.io import load_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neutral barcode pipeline tournament")
    commands = parser.add_subparsers(dest="command_name", required=True)

    pascal = commands.add_parser("import-pascal-voc", help="create a dataset manifest")
    pascal.add_argument("root", type=Path)
    pascal.add_argument("--output", type=Path, required=True)
    pascal.add_argument("--dataset-name", required=True)
    pascal.add_argument("--source-group")
    pascal.add_argument("--split", default="test")

    pascal_separate = commands.add_parser(
        "import-pascal-voc-separate",
        help="create a manifest when VOC images and XML are in separate folders",
    )
    pascal_separate.add_argument("images_root", type=Path)
    pascal_separate.add_argument("annotations_root", type=Path)
    pascal_separate.add_argument("--output", type=Path, required=True)
    pascal_separate.add_argument("--dataset-name", required=True)
    pascal_separate.add_argument("--split", default="test")

    vgg = commands.add_parser("import-vgg", help="create a dataset manifest from VGG VIA JSON")
    vgg.add_argument("images_root", type=Path)
    vgg.add_argument("annotations", type=Path, nargs="+")
    vgg.add_argument("--output", type=Path, required=True)
    vgg.add_argument("--dataset-name", required=True)
    vgg.add_argument("--split", default="test")
    vgg.add_argument("--allow-missing-images", action="store_true")

    yolo = commands.add_parser("import-yolo", help="create a dataset manifest from YOLO text labels")
    yolo.add_argument("root", type=Path)
    yolo.add_argument("--output", type=Path, required=True)
    yolo.add_argument("--dataset-name", required=True)
    yolo.add_argument("--kind", choices=("1d", "2d"), required=True)
    yolo.add_argument("--symbology", default="unknown")
    yolo.add_argument("--split", default="test")

    cvat = commands.add_parser("import-cvat", help="create a dataset manifest from CVAT image XML")
    cvat.add_argument("images_root", type=Path)
    cvat.add_argument("annotations", type=Path)
    cvat.add_argument("--output", type=Path, required=True)
    cvat.add_argument("--dataset-name", required=True)
    cvat.add_argument("--symbology", default="unknown")
    cvat.add_argument("--payload-attribute", default="Code")
    cvat.add_argument("--split", default="test")
    cvat.add_argument("--validate-ean13-checksum", action="store_true")
    cvat.add_argument("--image-level-payload", action="store_true")

    deal = commands.add_parser("import-deal", help="create a manifest from DEAL labels")
    deal.add_argument("images_root", type=Path)
    deal.add_argument("annotations", type=Path)
    deal.add_argument("--output", type=Path, required=True)
    deal.add_argument("--dataset-name", default="DEAL single-test")
    deal.add_argument("--split", default="test")
    deal.add_argument("--payload-length", type=int, default=13)
    deal.add_argument("--symbology", default="EAN_13")

    payload_texts = commands.add_parser(
        "import-payload-texts",
        help="create a decode manifest from one text payload per image",
    )
    payload_texts.add_argument("images_root", type=Path)
    payload_texts.add_argument("labels_root", type=Path)
    payload_texts.add_argument("--output", type=Path, required=True)
    payload_texts.add_argument("--dataset-name", required=True)
    payload_texts.add_argument("--symbology", required=True)
    payload_texts.add_argument("--split", default="test")
    payload_texts.add_argument("--validate-ean13-checksum", action="store_true")

    adapt = commands.add_parser("adapt-current", help="normalize project 5/5.1/6/7 JSON")
    adapt.add_argument("dataset", type=Path)
    adapt.add_argument("detections", type=Path)
    adapt.add_argument("--output", type=Path, required=True)
    adapt.add_argument("--pipeline", required=True)
    adapt.add_argument("--run-command")

    score = commands.add_parser("evaluate", help="score normalized predictions")
    score.add_argument("dataset", type=Path)
    score.add_argument("predictions", type=Path)
    score.add_argument("--output", type=Path)
    score.add_argument("--iou-threshold", type=float, default=0.5)
    score.add_argument("--containment-threshold", type=float, default=0.5)

    subset = commands.add_parser("subset", help="create a deterministic image-fold subset")
    subset.add_argument("dataset", type=Path)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--folds", type=int, default=5)
    subset.add_argument("--include-folds", required=True, help="comma-separated zero-based fold indexes")
    subset.add_argument("--salt", default="barcode-tournament-v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command_name == "import-pascal-voc":
        value = import_pascal_voc(
            arguments.root,
            dataset_name=arguments.dataset_name,
            source_group=arguments.source_group,
            split=arguments.split,
        )
        write_json(arguments.output, value)
        print(arguments.output.expanduser().resolve())
        return 0
    if arguments.command_name == "import-pascal-voc-separate":
        value = import_pascal_voc_separate(
            arguments.images_root,
            arguments.annotations_root,
            dataset_name=arguments.dataset_name,
            split=arguments.split,
        )
        write_json(arguments.output, value)
        print(arguments.output.expanduser().resolve())
        return 0
    if arguments.command_name == "import-vgg":
        value = import_vgg(
            arguments.images_root,
            arguments.annotations,
            dataset_name=arguments.dataset_name,
            split=arguments.split,
            allow_missing_images=arguments.allow_missing_images,
        )
        write_json(arguments.output, value)
        print(arguments.output.expanduser().resolve())
        return 0
    if arguments.command_name == "import-yolo":
        value = import_yolo(
            arguments.root,
            dataset_name=arguments.dataset_name,
            kind=arguments.kind,
            symbology=arguments.symbology,
            split=arguments.split,
        )
        write_json(arguments.output, value)
        print(arguments.output.expanduser().resolve())
        return 0
    if arguments.command_name == "import-cvat":
        value = import_cvat_image_xml(
            arguments.images_root,
            arguments.annotations,
            dataset_name=arguments.dataset_name,
            symbology=arguments.symbology,
            payload_attribute=arguments.payload_attribute,
            split=arguments.split,
            validate_ean13_checksum=arguments.validate_ean13_checksum,
            image_level_payload=arguments.image_level_payload,
        )
        write_json(arguments.output, value)
        print(arguments.output.expanduser().resolve())
        return 0
    if arguments.command_name == "import-deal":
        value = import_deal(
            arguments.images_root,
            arguments.annotations,
            dataset_name=arguments.dataset_name,
            split=arguments.split,
            payload_length=arguments.payload_length,
            symbology=arguments.symbology,
        )
        write_json(arguments.output, value)
        print(arguments.output.expanduser().resolve())
        return 0
    if arguments.command_name == "import-payload-texts":
        value = import_image_payload_texts(
            arguments.images_root,
            arguments.labels_root,
            dataset_name=arguments.dataset_name,
            symbology=arguments.symbology,
            split=arguments.split,
            validate_ean13_checksum=arguments.validate_ean13_checksum,
        )
        write_json(arguments.output, value)
        print(arguments.output.expanduser().resolve())
        return 0
    if arguments.command_name == "adapt-current":
        value = adapt_current_detections(
            load_json(arguments.dataset),
            load_json(arguments.detections),
            pipeline=arguments.pipeline,
            command=arguments.run_command,
        )
        write_json(arguments.output, value)
        print(arguments.output.expanduser().resolve())
        return 0
    if arguments.command_name == "evaluate":
        value = evaluate(
            load_json(arguments.dataset),
            load_json(arguments.predictions),
            iou_threshold=arguments.iou_threshold,
            containment_threshold=arguments.containment_threshold,
        )
        if arguments.output:
            write_json(arguments.output, value)
            print(arguments.output.expanduser().resolve())
        else:
            print(json.dumps(value, indent=2, ensure_ascii=False))
        return 0
    if arguments.command_name == "subset":
        value = load_json(arguments.dataset)
        if arguments.folds < 2:
            raise ValueError("--folds must be at least 2")
        selected_folds = {
            int(item.strip())
            for item in arguments.include_folds.split(",")
            if item.strip()
        }
        if not selected_folds or min(selected_folds) < 0 or max(selected_folds) >= arguments.folds:
            raise ValueError("--include-folds contains an invalid fold index")
        selected = []
        fold_counts = {str(index): 0 for index in range(arguments.folds)}
        for image in value.get("images", []):
            digest = hashlib.sha256(f"{arguments.salt}:{image['id']}".encode("utf-8")).digest()
            fold = int.from_bytes(digest[:8], "big") % arguments.folds
            fold_counts[str(fold)] += 1
            if fold in selected_folds:
                selected.append(image)
        result = {
            **value,
            "dataset": {
                **value.get("dataset", {}),
                "parent_manifest": str(arguments.dataset.expanduser().resolve()),
                "partition": {
                    "method": "sha256-image-id",
                    "folds": arguments.folds,
                    "included_folds": sorted(selected_folds),
                    "salt": arguments.salt,
                    "full_fold_counts": fold_counts,
                },
            },
            "images": selected,
        }
        write_json(arguments.output, result)
        print(arguments.output.expanduser().resolve())
        return 0
    raise AssertionError(arguments.command_name)


if __name__ == "__main__":
    raise SystemExit(main())
