from __future__ import annotations

import sys
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from barcode_benchmark.evaluate import (
    canonical_symbology,
    evaluate,
    payloads_are_gtin_equivalent,
)
from barcode_benchmark.geometry import maximum_cardinality_matches
from barcode_benchmark.io import DATASET_SCHEMA, PREDICTION_SCHEMA


def dataset(objects: list[dict]) -> dict:
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset": {"name": "test"},
        "images": [{
            "id": "image-1",
            "path": "/tmp/image.png",
            "width": 100,
            "height": 100,
            "source_group": "unit",
            "split": "test",
            "objects": objects,
        }],
    }


def predictions(items: list[dict]) -> dict:
    return {
        "schema_version": PREDICTION_SCHEMA,
        "run": {"pipeline": "unit"},
        "images": [{"id": "image-1", "predictions": items, "timing": {"latency_ms": 4.0}}],
    }


class EvaluationTests(unittest.TestCase):
    def test_image_level_payload_does_not_fabricate_localization(self) -> None:
        dataset_value = dataset(
            [{
                "polygon": [[0, 90], [100, 90], [100, 100], [0, 100]],
                "kind": "1d",
                "symbology": "EAN_13",
                "payload": "2274336000012",
                "decodable": True,
                "localization_evaluable": False,
                "payload_scope": "image",
            }]
        )
        prediction_value = predictions(
            [{
                "polygon": [[0, 0], [100, 0], [100, 80], [0, 80]],
                "kind": "1d",
                "symbology": "EAN_13",
                "payload": "2274336000012",
            }]
        )
        report = evaluate(dataset_value, prediction_value)
        self.assertEqual(report["localization"]["strict"]["ground_truth"], 0)
        self.assertEqual(report["decoding"]["exact_payload"], 1)
        self.assertEqual(report["decoding"]["false_decode"], 0)

    def test_common_symbology_aliases_are_canonical(self) -> None:
        self.assertEqual(canonical_symbology("QR"), "QR_CODE")
        self.assertEqual(canonical_symbology("C128"), "CODE_128")
        self.assertEqual(canonical_symbology("DATAMATRIX"), "DATA_MATRIX")

    def test_upca_is_semantically_equivalent_to_zero_prefixed_ean13(self) -> None:
        self.assertTrue(
            payloads_are_gtin_equivalent(
                "0012345678905",
                "EAN_13",
                "012345678905",
                "UPC_A",
            )
        )
        self.assertFalse(
            payloads_are_gtin_equivalent(
                "0012345678905",
                "EAN_13",
                "012345678905",
                "unknown",
            )
        )

    def test_gtin_equivalence_is_not_counted_as_wrong_decode(self) -> None:
        truth = {
            "polygon": [[10, 10], [90, 10], [90, 40], [10, 40]],
            "kind": "1d",
            "symbology": "EAN_13",
            "payload": "0012345678905",
            "decodable": True,
        }
        prediction = {
            "polygon": [[10, 10], [90, 10], [90, 40], [10, 40]],
            "kind": "1d",
            "symbology": "UPC_A",
            "payload": "012345678905",
        }
        report = evaluate(dataset([truth]), predictions([prediction]))
        self.assertEqual(report["decoding"]["exact_payload"], 0)
        self.assertEqual(report["decoding"]["gtin_equivalent_payload"], 1)
        self.assertEqual(report["decoding"]["semantic_payload"], 1)
        self.assertEqual(report["decoding"]["wrong_payload"], 0)
        self.assertEqual(report["decoding"]["end_to_end_semantic_rate"], 1.0)

    def test_image_level_gtin_equivalence_consumes_prediction(self) -> None:
        truth = {
            "polygon": [[0, 90], [100, 90], [100, 100], [0, 100]],
            "kind": "1d",
            "symbology": "EAN_13",
            "payload": "0012345678905",
            "decodable": True,
            "localization_evaluable": False,
            "payload_scope": "image",
        }
        candidate = {
            "polygon": [[0, 0], [100, 0], [100, 80], [0, 80]],
            "kind": "1d",
            "symbology": "UPC_A",
            "payload": "012345678905",
        }
        report = evaluate(dataset([truth]), predictions([candidate]))
        self.assertEqual(report["decoding"]["gtin_equivalent_payload"], 1)
        self.assertEqual(report["decoding"]["wrong_payload"], 0)
        self.assertEqual(report["decoding"]["false_decode"], 0)

    def test_tight_oriented_box_is_containment_match_but_not_strict_iou(self) -> None:
        truth = {
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "kind": "1d",
            "symbology": "CODE_128",
            "payload": "A001",
            "decodable": True,
        }
        prediction = {
            "polygon": [[20, 45], [80, 45], [80, 55], [20, 55]],
            "kind": "linear",
            "symbology": "Code 128",
            "payload": "A001",
            "confidence": 0.9,
        }
        report = evaluate(dataset([truth]), predictions([prediction]))
        self.assertEqual(report["localization"]["strict"]["true_positive"], 0)
        self.assertEqual(report["localization"]["containment_aware"]["true_positive"], 1)
        self.assertEqual(report["decoding"]["exact_payload"], 1)

    def test_wrong_and_false_decodes_are_not_hidden(self) -> None:
        truth = {
            "polygon": [[10, 10], [50, 10], [50, 30], [10, 30]],
            "kind": "1d",
            "symbology": "EAN_13",
            "payload": "0123456789012",
            "decodable": True,
        }
        items = [
            {
                "polygon": [[10, 10], [50, 10], [50, 30], [10, 30]],
                "kind": "1d",
                "symbology": "EAN_13",
                "payload": "1123456789012",
                "confidence": 0.99,
            },
            {
                "polygon": [[60, 60], [90, 60], [90, 90], [60, 90]],
                "kind": "2d",
                "symbology": "QR_CODE",
                "payload": "hallucinated",
                "confidence": 0.8,
            },
        ]
        report = evaluate(dataset([truth]), predictions(items))
        self.assertEqual(report["decoding"]["wrong_payload"], 1)
        self.assertEqual(report["decoding"]["false_decode"], 1)
        self.assertEqual(report["decoding"]["conditional_decode_precision"], 0.0)

    def test_undecodable_truth_only_affects_localization(self) -> None:
        truth = {
            "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]],
            "kind": "2d",
            "symbology": "QR_CODE",
            "payload": None,
            "decodable": False,
        }
        report = evaluate(dataset([truth]), predictions([]))
        self.assertEqual(report["localization"]["strict"]["false_negative"], 1)
        self.assertEqual(report["decoding"]["payload_evaluable"], 0)
        self.assertEqual(report["decoding"]["not_payload_evaluable"], 1)

    def test_refuses_image_id_mismatch(self) -> None:
        predicted = predictions([])
        predicted["images"][0]["id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "image id mismatch"):
            evaluate(dataset([]), predicted)

    def test_matching_maximizes_cardinality(self) -> None:
        import numpy as np

        scores = np.asarray([[0.9, 0.8], [0.7, 0.0]], dtype=float)
        eligible = scores > 0.5
        matches = maximum_cardinality_matches(scores, eligible)
        self.assertEqual(len(matches), 2)
        self.assertEqual({(truth, prediction) for truth, prediction, _ in matches}, {(0, 1), (1, 0)})


if __name__ == "__main__":
    unittest.main()
