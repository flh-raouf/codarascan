from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from barcode_benchmark.importers import (
    _ean13_checksum_valid,
    import_deal,
    import_vgg,
    import_yolo,
)


class DealImporterTests(unittest.TestCase):
    def test_imports_box_and_checksum_valid_filename_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cv2.imwrite(
                str(root / "4006381333931.jpg"),
                np.zeros((60, 80, 3), np.uint8),
            )
            annotations = root / "test.txt"
            annotations.write_text(
                "4006381333931.jpg 10 12 70 50 0\n",
                encoding="utf-8",
            )
            manifest = import_deal(root, annotations)
            item = manifest["images"][0]["objects"][0]
            self.assertEqual(item["payload"], "4006381333931")
            self.assertEqual(item["symbology"], "EAN_13")
            self.assertTrue(item["decodable"])
            self.assertEqual(item["polygon"], [[10.0, 12.0], [70.0, 12.0], [70.0, 50.0], [10.0, 50.0]])


class VggImporterTests(unittest.TestCase):
    def test_imports_polygon_payload_ppe_and_undecodable_rect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cv2.imwrite(str(root / "one.jpg"), np.zeros((60, 80, 3), np.uint8))
            annotation = {
                "_via_img_metadata": {
                    "one.jpg123": {
                        "filename": "one.jpg",
                        "regions": [
                            {
                                "shape_attributes": {
                                    "name": "polygon",
                                    "all_points_x": [1, 30, 30, 1],
                                    "all_points_y": [2, 2, 20, 20],
                                },
                                "region_attributes": {
                                    "Type": "C128",
                                    "PPE": "2.5",
                                    "String": "00123",
                                },
                            },
                            {
                                "shape_attributes": {
                                    "name": "rect",
                                    "x": 40,
                                    "y": 10,
                                    "width": 20,
                                    "height": 20,
                                },
                                "region_attributes": {
                                    "Type": "QR",
                                    "PPE": "-1",
                                    "String": "-1",
                                },
                            },
                        ],
                    }
                }
            }
            annotation_path = root / "source.json"
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            manifest = import_vgg(root, [annotation_path], dataset_name="unit")
            objects = manifest["images"][0]["objects"]
            self.assertEqual(objects[0]["payload"], "00123")
            self.assertEqual(objects[0]["symbology"], "CODE_128")
            self.assertEqual(objects[0]["attributes"]["ppe"], 2.5)
            self.assertEqual(objects[1]["kind"], "2d")
            self.assertIsNone(objects[1]["payload"])
            self.assertFalse(objects[1]["decodable"])
            self.assertIsNone(objects[1]["attributes"]["ppe"])

    def test_missing_images_fail_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation_path = root / "source.json"
            annotation_path.write_text(json.dumps({
                "_via_img_metadata": {
                    "missing": {"filename": "missing.jpg", "regions": []}
                }
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no image"):
                import_vgg(root, [annotation_path], dataset_name="unit")


class YoloImporterTests(unittest.TestCase):
    def test_ean13_checksum_validation(self) -> None:
        self.assertTrue(_ean13_checksum_valid("2274336000012"))
        self.assertFalse(_ean13_checksum_valid("2274336000013"))
        self.assertFalse(_ean13_checksum_valid("4603785f175695"))

    def test_converts_normalized_center_box_to_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cv2.imwrite(str(root / "one.jpg"), np.zeros((100, 200, 3), np.uint8))
            (root / "one.txt").write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
            manifest = import_yolo(
                root,
                dataset_name="unit",
                kind="1d",
                symbology="CODE128",
            )
            item = manifest["images"][0]["objects"][0]
            self.assertEqual(item["polygon"], [[60.0, 40.0], [140.0, 40.0], [140.0, 60.0], [60.0, 60.0]])
            self.assertEqual(item["symbology"], "CODE_128")


if __name__ == "__main__":
    unittest.main()
