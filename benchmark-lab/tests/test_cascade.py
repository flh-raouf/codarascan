from __future__ import annotations

import sys
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from barcode_benchmark.cascade import (
    conditional_fusion,
    relative_box_area,
    route_small_evidence,
)


def prediction(x1, y1, x2, y2, confidence=0.5):
    return {
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "confidence": confidence,
        "sources": ["test"],
    }


class CascadeTests(unittest.TestCase):
    def test_relative_box_area(self):
        self.assertEqual(relative_box_area(prediction(0, 0, 20, 10), 100, 100), 0.02)

    def test_empty_fast_result_routes(self):
        self.assertTrue(
            route_small_evidence(
                [], width=100, height=100, relative_area_threshold=0.02
            )
        )

    def test_threshold_is_strict(self):
        item = prediction(0, 0, 20, 10)
        self.assertFalse(
            route_small_evidence(
                [item], width=100, height=100, relative_area_threshold=0.02
            )
        )
        self.assertTrue(
            route_small_evidence(
                [item], width=100, height=100, relative_area_threshold=0.021
            )
        )

    def test_unrouted_result_does_not_add_fallback(self):
        fast = [prediction(0, 0, 40, 40)]
        fallback = [prediction(60, 60, 80, 80)]
        self.assertEqual(
            conditional_fusion(fast, fallback, routed=False),
            fast,
        )


if __name__ == "__main__":
    unittest.main()
