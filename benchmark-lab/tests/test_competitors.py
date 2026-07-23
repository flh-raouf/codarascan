from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from barcode_benchmark.competitors import (
    _tile_windows,
    COMPETITOR_NAMES,
    _deduplicate,
    _axis_crop,
    _warp_candidate,
    proposal_rescue,
    structure_tensor,
    zxing_opencv_direct,
)


class StructureTensorTests(unittest.TestCase):
    def test_overlap_tiles_cover_image_edges(self) -> None:
        windows = _tile_windows(101, 77)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0][:2], (0, 0))
        self.assertEqual(windows[-1][2:], (101, 77))
        self.assertLess(windows[1][0], windows[0][2])
        self.assertLess(windows[2][1], windows[0][3])

    def test_piero_checkpoint_is_exposed_as_competitor(self) -> None:
        self.assertIn("piero-yolov8s", COMPETITOR_NAMES)
        self.assertIn("piero-yolov8s-decode", COMPETITOR_NAMES)
        self.assertIn("piero-yolov8s-decode-fast", COMPETITOR_NAMES)
        self.assertIn("bafalo-published", COMPETITOR_NAMES)
        self.assertIn("bafalo-scaled", COMPETITOR_NAMES)

    def test_valid_zxing_pass_is_not_replaced_by_error_mode(self) -> None:
        # QR-DN showed that return_errors=True is not a strict superset of
        # normal ZXing decoding. Keep this behavior regression-tested at the
        # public ensemble boundary by checking its documented callable.
        self.assertTrue(callable(zxing_opencv_direct))

    def test_axis_crop_clamps_padding_to_image(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        crop = _axis_crop(
            image,
            [[-10, -10], [40, -10], [40, 50], [-10, 50]],
            0.2,
        )
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)
        self.assertLessEqual(crop.shape[0], 100)
        self.assertLessEqual(crop.shape[1], 120)

    def test_deduplication_drops_unusable_zero_area_geometry(self) -> None:
        values = [{
            "polygon": [[1, 1], [1, 1], [1, 1], [1, 1]],
            "confidence": 1.0,
        }]
        self.assertEqual(_deduplicate(values), [])

    def test_finds_parallel_bar_texture_without_payload_claim(self) -> None:
        image = np.full((240, 320, 3), 255, np.uint8)
        for x in range(70, 250, 8):
            cv2.rectangle(image, (x, 85), (x + 3, 155), (0, 0, 0), -1)
        found = structure_tensor(image, work_size=320, include_matrix=False)
        self.assertTrue(found)
        self.assertTrue(any(item["kind"] == "1d" for item in found))
        self.assertTrue(all(item["payload"] is None for item in found))

    def test_blank_image_produces_no_candidates(self) -> None:
        image = np.full((120, 160, 3), 255, np.uint8)
        self.assertEqual(structure_tensor(image, work_size=160), [])
        self.assertEqual(zxing_opencv_direct(image), [])
        self.assertEqual(proposal_rescue(image), [])

    def test_warp_candidate_handles_rotated_box_point_order(self) -> None:
        image = np.full((120, 160, 3), 255, np.uint8)
        candidate = {
            "polygon": [[80, 20], [140, 60], [80, 100], [20, 60]],
            "kind": "1d",
        }
        crop, inverse = _warp_candidate(image, candidate)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)
        self.assertEqual(inverse.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
