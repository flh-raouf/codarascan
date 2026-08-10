from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import (
    Detection,
    TensorConfig,
    _periodic_band_candidates,
    _accepted_linear_metrics,
    linear_metrics,
    locate_prepared,
    prepare_page,
    profile_config,
)


class TensorPipelineTests(unittest.TestCase):
    def test_blank_page_has_no_detection(self) -> None:
        page = np.full((720, 1280), 255, np.uint8)
        prepared = prepare_page(page)
        result = locate_prepared(prepared)
        self.assertEqual(result.detections, [])
        self.assertEqual(result.diagnostics["accepted_count"], 0)

    def test_parallel_bars_are_localized_without_payload(self) -> None:
        page = np.full((720, 1280), 255, np.uint8)
        for x in range(350, 930, 13):
            cv2.rectangle(page, (x, 250), (x + 6, 470), 0, -1)
        prepared = prepare_page(page)
        result = locate_prepared(prepared)
        self.assertTrue(result.detections)
        self.assertTrue(all(item.to_prediction()["payload"] is None for item in result.detections))
        self.assertIn(
            "proposal_profile_transitions",
            result.detections[0].metrics,
        )

    def test_legacy_native_research_backend_contract(self) -> None:
        page = np.full((720, 1280), 255, np.uint8)
        for x in range(350, 930, 13):
            cv2.rectangle(page, (x, 250), (x + 6, 470), 0, -1)
        config = replace(TensorConfig(), backend="native")
        result = locate_prepared(prepare_page(page, config), config)
        self.assertEqual(result.diagnostics["backend"], "native-sttg")

    def test_rotated_parallel_bars_remain_localizable(self) -> None:
        page = np.full((720, 1280), 255, np.uint8)
        patch = np.full((280, 620), 255, np.uint8)
        for x in range(25, 595, 13):
            cv2.rectangle(patch, (x, 30), (x + 6, 250), 0, -1)
        transform = cv2.getRotationMatrix2D((310, 140), 27.0, 1.0)
        page[220:500, 330:950] = cv2.warpAffine(
            patch,
            transform,
            (620, 280),
            borderValue=255,
        )
        result = locate_prepared(prepare_page(page))
        self.assertEqual(len(result.detections), 1)

    def test_dense_text_and_table_are_not_barcode_evidence(self) -> None:
        page = np.full((900, 1400), 255, np.uint8)
        for y in range(80, 850, 70):
            cv2.putText(
                page,
                "INVOICE 123456789 TOTAL 987654321",
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                0,
                2,
                cv2.LINE_AA,
            )
        for x in range(0, 1400, 140):
            cv2.line(page, (x, 0), (x, 900), 0, 1)
        for y in range(0, 900, 90):
            cv2.line(page, (0, y), (1400, y), 0, 1)
        result = locate_prepared(prepare_page(page))
        self.assertEqual(result.detections, [])

    def test_curving_parallel_texture_fails_shared_edge_direction(self) -> None:
        page = np.full((80, 300), 255, np.uint8)
        center_y = 40
        for index, base_x in enumerate(range(15, 285, 6)):
            slope = 0.4 * np.sin(index * 0.8)
            points = np.asarray(
                [
                    [base_x + slope * (10 - center_y), 10],
                    [base_x + 3 + slope * (10 - center_y), 10],
                    [base_x + 3 + slope * (70 - center_y), 70],
                    [base_x + slope * (70 - center_y), 70],
                ],
                np.int32,
            )
            cv2.fillConvexPoly(
                page,
                points,
                0,
                lineType=cv2.LINE_AA,
            )
        quad = np.asarray(
            [[5, 5], [295, 5], [295, 75], [5, 75]],
            np.float32,
        )
        metrics = linear_metrics(page, quad)
        self.assertGreaterEqual(metrics["transitions"], 24)
        self.assertGreaterEqual(metrics["scanline_agreement"], 0.50)
        self.assertGreater(
            metrics["edge_direction_p80_degrees"],
            profile_config(
                "document",
            ).maximum_edge_direction_deviation_degrees,
        )
        self.assertFalse(
            _accepted_linear_metrics(
                metrics,
                profile_config("document"),
            )
        )

    def test_preparation_is_outside_engine_timing(self) -> None:
        page = np.full((1200, 1800), 255, np.uint8)
        prepared = prepare_page(page, TensorConfig(work_size=640))
        result = locate_prepared(prepared, TensorConfig(work_size=640))
        self.assertGreater(prepared.preparation_seconds, 0.0)
        self.assertNotIn("preparation_seconds", result.timings)
        self.assertEqual(prepared.work_gray.shape[1], 640)

    def test_frozen_profiles_do_not_drift(self) -> None:
        document = profile_config("document")
        general = profile_config("general")
        self.assertEqual(document.aggregation_windows, (7,))
        self.assertEqual(document.fragment_join_max_gap_short, 5.0)
        self.assertEqual(document.minimum_native_transitions, 24.0)
        self.assertEqual(
            document.maximum_edge_direction_deviation_degrees,
            18.0,
        )
        self.assertTrue(document.retain_low_transition_rescue)
        self.assertTrue(document.split_periodic_bands)
        self.assertEqual(general.aggregation_windows, (12,))
        self.assertEqual(general.fragment_join_max_gap_short, 2.6)
        self.assertEqual(general.minimum_native_transitions, 12.0)
        self.assertEqual(
            general.maximum_edge_direction_deviation_degrees,
            90.0,
        )
        self.assertFalse(general.retain_low_transition_rescue)
        self.assertFalse(general.split_periodic_bands)

    def test_periodic_bands_split_two_barcodes_inside_one_parent(self) -> None:
        page = np.full((360, 760), 255, np.uint8)
        for top, bottom in ((70, 130), (220, 280)):
            for x in range(70, 690, 13):
                cv2.rectangle(page, (x, top), (x + 6, bottom), 0, -1)
        parent = Detection(
            quad=np.asarray(
                [[40, 30], [720, 30], [720, 320], [40, 320]],
                np.float32,
            ),
            confidence=1.0,
            source="unit-parent",
            metrics={"proposal_orientation_resultant": 1.0},
        )
        children = _periodic_band_candidates(
            page,
            parent,
            profile_config("document"),
        )
        self.assertEqual(len(children), 2)
        self.assertTrue(
            all(item.metrics["transitions"] >= 24 for item in children)
        )
        self.assertTrue(
            all(item.metrics["periodic_band_split"] == 1.0 for item in children)
        )


if __name__ == "__main__":
    unittest.main()
