from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2

from .evaluate import canonical_kind, canonical_symbology
from .io import DATASET_SCHEMA, IMAGE_SUFFIXES, load_json


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _kind_from_label(label: str) -> str:
    normalized = canonical_symbology(label)
    if normalized in {"QR_CODE", "DATA_MATRIX", "AZTEC", "PDF417", "MAXICODE"}:
        return "2d"
    if normalized in {"BARCODE", "D"}:
        return "1d"
    if any(token in normalized for token in ("BARCODE", "EAN", "UPC", "CODE", "CODABAR", "ITF", "POST")):
        return "1d"
    return canonical_kind(label)


def import_pascal_voc(
    root: Path,
    *,
    dataset_name: str,
    source_group: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    images = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    output_images: list[dict[str, Any]] = []
    missing_annotations: list[str] = []
    for image_path in images:
        annotation_path = image_path.with_suffix(".xml")
        if not annotation_path.is_file():
            missing_annotations.append(str(image_path.relative_to(root)))
            continue
        annotation = ET.parse(annotation_path).getroot()
        actual_width, actual_height = _image_size(image_path)
        declared_size = annotation.find("size")
        if declared_size is not None:
            declared_width = int(float(declared_size.findtext("width", str(actual_width))))
            declared_height = int(float(declared_size.findtext("height", str(actual_height))))
            if (declared_width, declared_height) != (actual_width, actual_height):
                raise ValueError(
                    f"{annotation_path}: annotation size {(declared_width, declared_height)} "
                    f"does not match image {(actual_width, actual_height)}"
                )
        objects: list[dict[str, Any]] = []
        for object_index, item in enumerate(annotation.findall("object")):
            box = item.find("bndbox")
            if box is None:
                continue
            x1 = float(box.findtext("xmin", "0"))
            y1 = float(box.findtext("ymin", "0"))
            x2 = float(box.findtext("xmax", "0"))
            y2 = float(box.findtext("ymax", "0"))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{annotation_path}: object {object_index} has an invalid box")
            raw_label = (item.findtext("name", UNKNOWN) or UNKNOWN).strip()
            symbology = canonical_symbology(raw_label)
            if symbology in {"BARCODE", "D"}:
                symbology = UNKNOWN
            objects.append({
                "id": f"object-{object_index + 1:04d}",
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "kind": _kind_from_label(raw_label),
                "symbology": symbology,
                "payload": None,
                "decodable": False,
                "attributes": {"source_label": raw_label},
            })
        relative = image_path.relative_to(root).as_posix()
        output_images.append({
            "id": relative,
            "path": str(image_path),
            "width": actual_width,
            "height": actual_height,
            "source_group": source_group or dataset_name,
            "split": split,
            "objects": objects,
        })
    if not output_images:
        raise ValueError(f"{root}: no annotated images found")
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset": {
            "name": dataset_name,
            "root": str(root),
            "annotation_format": "pascal-voc",
            "missing_annotation_count": len(missing_annotations),
            "missing_annotation_examples": missing_annotations[:20],
        },
        "images": output_images,
    }


def import_pascal_voc_separate(
    images_root: Path,
    annotations_root: Path,
    *,
    dataset_name: str,
    split: str = "test",
) -> dict[str, Any]:
    """Import Pascal VOC when images and XML files are in separate folders."""
    images_root = images_root.expanduser().resolve()
    annotations_root = annotations_root.expanduser().resolve()
    images_by_stem = {
        path.stem: path
        for path in images_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    images: list[dict[str, Any]] = []
    for annotation_path in sorted(annotations_root.glob("*.xml")):
        image_path = images_by_stem.get(annotation_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"no image matching {annotation_path}")
        width, height = _image_size(image_path)
        root = ET.parse(annotation_path).getroot()
        objects = []
        for index, item in enumerate(root.findall("object"), start=1):
            box = item.find("bndbox")
            if box is None:
                continue
            x1 = float(box.findtext("xmin", "0"))
            y1 = float(box.findtext("ymin", "0"))
            x2 = float(box.findtext("xmax", "0"))
            y2 = float(box.findtext("ymax", "0"))
            label = item.findtext("name", UNKNOWN) or UNKNOWN
            objects.append({
                "id": f"object-{index:04d}",
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "kind": _kind_from_label(label),
                "symbology": canonical_symbology(label),
                "payload": None,
                "decodable": False,
                "attributes": {"source_label": label},
            })
        images.append({
            "id": image_path.name,
            "path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "source_group": dataset_name,
            "split": split,
            "objects": objects,
        })
    if not images:
        raise ValueError(f"{annotations_root}: no XML annotations")
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset": {
            "name": dataset_name,
            "root": str(images_root),
            "annotation_format": "pascal-voc-separate-folders",
            "annotation_root": str(annotations_root),
        },
        "images": images,
    }


UNKNOWN = "unknown"


def _vgg_points(shape: dict[str, Any], label: str) -> list[list[float]]:
    name = str(shape.get("name", "")).lower()
    if name == "polygon":
        points_x = shape.get("all_points_x")
        points_y = shape.get("all_points_y")
        if not isinstance(points_x, list) or not isinstance(points_y, list) or len(points_x) != len(points_y):
            raise ValueError(f"{label}: malformed polygon coordinates")
        if len(points_x) < 3:
            raise ValueError(f"{label}: polygon has fewer than three points")
        return [[float(x), float(y)] for x, y in zip(points_x, points_y)]
    if name == "rect":
        x = float(shape["x"])
        y = float(shape["y"])
        width = float(shape["width"])
        height = float(shape["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"{label}: rectangle has non-positive dimensions")
        return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    raise ValueError(f"{label}: unsupported VGG shape {name!r}")


def import_vgg(
    images_root: Path,
    annotations: list[Path],
    *,
    dataset_name: str,
    split: str = "test",
    allow_missing_images: bool = False,
) -> dict[str, Any]:
    images_root = images_root.expanduser().resolve()
    if not images_root.is_dir():
        raise ValueError(f"not a directory: {images_root}")
    image_paths = [
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    by_basename: dict[str, Path] = {}
    for path in image_paths:
        if path.name in by_basename:
            raise ValueError(
                f"duplicate image basename {path.name!r}: {by_basename[path.name]} and {path}"
            )
        by_basename[path.name] = path

    output_images: list[dict[str, Any]] = []
    used_filenames: set[str] = set()
    missing: list[str] = []
    annotation_sources: list[str] = []
    for annotation_path in annotations:
        annotation_path = annotation_path.expanduser().resolve()
        value = load_json(annotation_path)
        metadata = value.get("_via_img_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{annotation_path}: missing VGG '_via_img_metadata'")
        source_group = annotation_path.stem
        annotation_sources.append(str(annotation_path))
        for source_key, record in metadata.items():
            if not isinstance(record, dict):
                raise ValueError(f"{annotation_path}:{source_key}: expected an object")
            filename = Path(str(record.get("filename", ""))).name
            image_path = by_basename.get(filename)
            if image_path is None:
                missing.append(f"{source_group}/{filename}")
                continue
            if filename in used_filenames:
                raise ValueError(f"duplicate VGG annotation for image {filename!r}")
            used_filenames.add(filename)
            width, height = _image_size(image_path)
            regions = record.get("regions", [])
            if isinstance(regions, dict):
                regions = list(regions.values())
            if not isinstance(regions, list):
                raise ValueError(f"{annotation_path}:{filename}: 'regions' must be a list or object")
            objects: list[dict[str, Any]] = []
            for region_index, region in enumerate(regions):
                if not isinstance(region, dict):
                    raise ValueError(f"{annotation_path}:{filename}: region {region_index} is invalid")
                attributes = region.get("region_attributes", {})
                shape = region.get("shape_attributes", {})
                if not isinstance(attributes, dict) or not isinstance(shape, dict):
                    raise ValueError(f"{annotation_path}:{filename}: region {region_index} is malformed")
                raw_type = str(attributes.get("Type", UNKNOWN) or UNKNOWN)
                symbology = canonical_symbology(raw_type)
                kind = _kind_from_label(raw_type)
                raw_payload = attributes.get("String")
                object_payload = None if raw_payload in (None, "", "-1", -1) else str(raw_payload)
                raw_ppe = attributes.get("PPE")
                try:
                    ppe = float(raw_ppe)
                except (TypeError, ValueError):
                    ppe = -1.0
                objects.append({
                    "id": f"object-{region_index + 1:04d}",
                    "polygon": _vgg_points(
                        shape,
                        f"{annotation_path}:{filename}:region:{region_index}",
                    ),
                    "kind": kind,
                    "symbology": symbology,
                    "payload": object_payload,
                    "decodable": object_payload is not None,
                    "attributes": {
                        "source_type": raw_type,
                        "ppe": ppe if ppe >= 0 else None,
                        "vgg_region_attributes": attributes,
                    },
                })
            output_images.append({
                "id": f"{source_group}/{filename}",
                "path": str(image_path),
                "width": width,
                "height": height,
                "source_group": source_group,
                "split": split,
                "objects": objects,
            })
    if missing and not allow_missing_images:
        raise ValueError(
            f"{len(missing)} VGG annotations have no image under {images_root}; "
            f"examples: {missing[:10]!r}"
        )
    if not output_images:
        raise ValueError("no VGG-annotated images imported")
    output_images.sort(key=lambda item: item["id"])
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset": {
            "name": dataset_name,
            "root": str(images_root),
            "annotation_format": "vgg-via",
            "annotation_sources": annotation_sources,
            "missing_image_count": len(missing),
            "missing_image_examples": missing[:20],
        },
        "images": output_images,
    }


def import_yolo(
    root: Path,
    *,
    dataset_name: str,
    kind: str,
    symbology: str = UNKNOWN,
    split: str = "test",
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    normalized_kind = canonical_kind(kind)
    if normalized_kind == UNKNOWN:
        raise ValueError(f"unsupported barcode kind: {kind!r}")
    normalized_symbology = canonical_symbology(symbology)
    images = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    output_images: list[dict[str, Any]] = []
    missing_annotations: list[str] = []
    for image_path in images:
        width, height = _image_size(image_path)
        annotation_path = image_path.with_suffix(".txt")
        if not annotation_path.is_file():
            missing_annotations.append(str(image_path.relative_to(root)))
            continue
        objects: list[dict[str, Any]] = []
        for line_index, raw_line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{annotation_path}:{line_index + 1}: expected five YOLO fields")
            source_class = fields[0]
            try:
                center_x, center_y, box_width, box_height = map(float, fields[1:])
            except ValueError as error:
                raise ValueError(f"{annotation_path}:{line_index + 1}: invalid number") from error
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"{annotation_path}:{line_index + 1}: non-positive box")
            x1 = max(0.0, (center_x - box_width / 2.0) * width)
            y1 = max(0.0, (center_y - box_height / 2.0) * height)
            x2 = min(float(width), (center_x + box_width / 2.0) * width)
            y2 = min(float(height), (center_y + box_height / 2.0) * height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{annotation_path}:{line_index + 1}: box is outside image")
            objects.append({
                "id": f"object-{line_index + 1:04d}",
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "kind": normalized_kind,
                "symbology": normalized_symbology,
                "payload": None,
                "decodable": False,
                "attributes": {"source_class": source_class},
            })
        relative = image_path.relative_to(root).as_posix()
        output_images.append({
            "id": relative,
            "path": str(image_path),
            "width": width,
            "height": height,
            "source_group": dataset_name,
            "split": split,
            "objects": objects,
        })
    if not output_images:
        raise ValueError(f"{root}: no YOLO-annotated images found")
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset": {
            "name": dataset_name,
            "root": str(root),
            "annotation_format": "yolo-normalized-aabb",
            "missing_annotation_count": len(missing_annotations),
            "missing_annotation_examples": missing_annotations[:20],
        },
        "images": output_images,
    }


def _ean13_checksum_valid(value: str) -> bool:
    if len(value) != 13 or not value.isdigit():
        return False
    digits = [int(character) for character in value]
    expected = (10 - (sum(digits[:-1:2]) + 3 * sum(digits[1:-1:2])) % 10) % 10
    return digits[-1] == expected


def _ean8_checksum_valid(value: str) -> bool:
    if len(value) != 8 or not value.isdigit():
        return False
    digits = [int(character) for character in value]
    expected = (10 - (3 * sum(digits[:-1:2]) + sum(digits[1:-1:2])) % 10) % 10
    return digits[-1] == expected


def import_deal(
    images_root: Path,
    annotations: Path,
    *,
    dataset_name: str = "DEAL single-test",
    split: str = "test",
    payload_length: int = 13,
    symbology: str = "EAN_13",
) -> dict[str, Any]:
    """Import the official DEAL axis-aligned EAN-13 annotations.

    DEAL encodes the payload in the first 13 filename characters and stores
    one ``filename x1 y1 x2 y2 class`` record per image.
    """
    images_root = images_root.expanduser().resolve()
    annotations = annotations.expanduser().resolve()
    if not images_root.is_dir():
        raise ValueError(f"not a directory: {images_root}")
    if not annotations.is_file():
        raise ValueError(f"not a file: {annotations}")
    images: list[dict[str, Any]] = []
    invalid_payloads: list[str] = []
    for line_number, raw_line in enumerate(
        annotations.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError(f"{annotations}:{line_number}: expected six fields")
        filename, raw_x1, raw_y1, raw_x2, raw_y2, source_class = fields
        image_path = images_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"{annotations}:{line_number}: missing {image_path}")
        try:
            x1, y1, x2, y2 = map(float, (raw_x1, raw_y1, raw_x2, raw_y2))
        except ValueError as error:
            raise ValueError(f"{annotations}:{line_number}: invalid box") from error
        width, height = _image_size(image_path)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"{annotations}:{line_number}: box {(x1, y1, x2, y2)} "
                f"is outside image {(width, height)}"
            )
        payload = Path(filename).stem[:payload_length]
        payload_valid = (
            _ean13_checksum_valid(payload)
            if payload_length == 13
            else _ean8_checksum_valid(payload)
            if payload_length == 8
            else bool(payload)
        )
        if not payload_valid:
            invalid_payloads.append(filename)
        images.append({
            "id": filename,
            "path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "source_group": dataset_name,
            "split": split,
            "objects": [{
                "id": "object-0001",
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "kind": "1d",
                "symbology": canonical_symbology(symbology),
                "payload": payload if payload_valid else None,
                "decodable": payload_valid,
                "attributes": {
                    "source_class": source_class,
                    "payload_from_filename": True,
                },
            }],
        })
    if not images:
        raise ValueError(f"{annotations}: no records")
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset": {
            "name": dataset_name,
            "root": str(images_root),
            "annotation_format": "deal-aabb-filename-payload",
            "annotation_source": str(annotations),
            "source_url": "https://zenodo.org/records/13586402",
            "license": "CC BY 4.0",
            "invalid_payload_count": len(invalid_payloads),
            "invalid_payload_examples": invalid_payloads[:20],
        },
        "images": images,
    }


def import_image_payload_texts(
    images_root: Path,
    labels_root: Path,
    *,
    dataset_name: str,
    symbology: str,
    split: str = "test",
    validate_ean13_checksum: bool = False,
) -> dict[str, Any]:
    """Import one payload-only text file per image for decode evaluation."""
    images_root = images_root.expanduser().resolve()
    labels_root = labels_root.expanduser().resolve()
    image_by_stem = {
        path.stem: path
        for path in images_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    images: list[dict[str, Any]] = []
    invalid: list[str] = []
    for label_path in sorted(labels_root.glob("*.txt")):
        image_path = image_by_stem.get(label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"no image matching {label_path}")
        payload = label_path.read_text(encoding="utf-8").strip()
        valid = bool(payload)
        if validate_ean13_checksum:
            valid = _ean13_checksum_valid(payload)
        if not valid:
            invalid.append(label_path.name)
        width, height = _image_size(image_path)
        images.append({
            "id": image_path.name,
            "path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "source_group": dataset_name,
            "split": split,
            "objects": [{
                "id": "payload-0001",
                "polygon": [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
                "kind": _kind_from_label(symbology),
                "symbology": canonical_symbology(symbology),
                "payload": payload if valid else None,
                "decodable": valid,
                "localization_evaluable": False,
                "payload_scope": "image",
                "attributes": {"payload_label": str(label_path.resolve())},
            }],
        })
    if not images:
        raise ValueError(f"{labels_root}: no text labels")
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset": {
            "name": dataset_name,
            "root": str(images_root),
            "annotation_format": "one-image-one-payload-text",
            "image_level_payload": True,
            "invalid_payload_count": len(invalid),
            "invalid_payload_examples": invalid[:20],
        },
        "images": images,
    }


def import_cvat_image_xml(
    images_root: Path,
    annotations: Path,
    *,
    dataset_name: str,
    symbology: str = UNKNOWN,
    payload_attribute: str = "Code",
    split: str = "test",
    validate_ean13_checksum: bool = False,
    image_level_payload: bool = False,
) -> dict[str, Any]:
    images_root = images_root.expanduser().resolve()
    annotations = annotations.expanduser().resolve()
    if not images_root.is_dir():
        raise ValueError(f"not a directory: {images_root}")
    root = ET.parse(annotations).getroot()
    normalized_symbology = canonical_symbology(symbology)
    output_images: list[dict[str, Any]] = []
    invalid_payloads: list[dict[str, str]] = []
    for image_index, image_record in enumerate(root.findall("image")):
        source_name = Path(image_record.attrib["name"]).name
        image_path = images_root / source_name
        if not image_path.is_file():
            raise ValueError(f"{annotations}: image is missing: {image_path}")
        width, height = _image_size(image_path)
        declared = (
            int(image_record.attrib.get("width", width)),
            int(image_record.attrib.get("height", height)),
        )
        if declared != (width, height):
            raise ValueError(
                f"{annotations}:{source_name}: declared size {declared} "
                f"does not match image {(width, height)}"
            )
        objects: list[dict[str, Any]] = []
        for object_index, item in enumerate(image_record.findall("polygon")):
            points = []
            for raw_point in item.attrib["points"].split(";"):
                x, y = raw_point.split(",", 1)
                points.append([float(x), float(y)])
            attributes = {
                str(value.attrib.get("name", "")): (value.text or "")
                for value in item.findall("attribute")
            }
            object_payload = attributes.get(payload_attribute) or None
            checksum_valid = True
            if validate_ean13_checksum and object_payload is not None:
                checksum_valid = _ean13_checksum_valid(object_payload)
                if not checksum_valid:
                    invalid_payloads.append({
                        "image": source_name,
                        "payload": object_payload,
                    })
            objects.append({
                "id": f"object-{object_index + 1:04d}",
                "polygon": points,
                "kind": _kind_from_label(normalized_symbology),
                "symbology": normalized_symbology,
                "payload": object_payload,
                "decodable": object_payload is not None and checksum_valid,
                "localization_evaluable": not image_level_payload,
                "payload_scope": "image" if image_level_payload else "localized",
                "attributes": {
                    "source_label": item.attrib.get("label"),
                    "occluded": item.attrib.get("occluded") == "1",
                    "checksum_valid": checksum_valid,
                    "cvat_attributes": attributes,
                },
            })
        output_images.append({
            "id": source_name,
            "path": str(image_path),
            "width": width,
            "height": height,
            "source_group": dataset_name,
            "split": split,
            "objects": objects,
        })
    if not output_images:
        raise ValueError(f"{annotations}: no CVAT image annotations found")
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset": {
            "name": dataset_name,
            "root": str(images_root),
            "annotation_format": "cvat-image-xml",
            "annotation_source": str(annotations),
            "invalid_payload_count": len(invalid_payloads),
            "invalid_payload_examples": invalid_payloads[:20],
            "image_level_payload": image_level_payload,
        },
        "images": output_images,
    }
