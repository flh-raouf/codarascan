from __future__ import annotations

import sys
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from barcode_benchmark.fusion import fuse_predictions


class FusionTests(unittest.TestCase):
    def test_primary_geometry_is_preserved_when_payload_is_transferred(self) -> None:
        primary_polygon = [[0, 0], [100, 0], [100, 40], [0, 40]]
        primary = [{
            "polygon": primary_polygon,
            "payload": None,
            "confidence": 0.8,
            "sources": ["detector"],
        }]
        supplement = [{
            "polygon": [[10, 18], [90, 18], [90, 20], [10, 20]],
            "payload": "123",
            "symbology": "CODE_128",
            "confidence": 1.0,
            "sources": ["decoder"],
        }]
        result = fuse_predictions(primary, [[supplement[0]]])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["polygon"], primary_polygon)
        self.assertEqual(result[0]["payload"], "123")
        self.assertEqual(result[0]["sources"], ["detector", "decoder"])

    def test_unmatched_supplement_is_retained(self) -> None:
        primary = [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}]
        supplement = [{"polygon": [[30, 30], [40, 30], [40, 40], [30, 40]]}]
        self.assertEqual(len(fuse_predictions(primary, [supplement])), 2)


if __name__ == "__main__":
    unittest.main()
