from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any

import numpy as np

from .geometry import maximum_cardinality_matches, polygon, polygon_long_side, similarity, size_bin
from .io import (
    DATASET_SCHEMA,
    PREDICTION_SCHEMA,
    REPORT_SCHEMA,
    image_index,
    require_schema,
)


SIZE_ORDER = ("<32", "32-59", "60-99", "100-199", "200-399", ">=400")
UNKNOWN = "unknown"


def canonical_kind(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"1d", "linear", "barcode", "bar_code"}:
        return "1d"
    if text in {"2d", "matrix", "matrix_2d", "qrcode", "qr_code", "data_matrix", "datamatrix", "aztec", "pdf417"}:
        return "2d"
    return UNKNOWN


def canonical_symbology(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "QR": "QR_CODE",
        "QRCODE": "QR_CODE",
        "DM": "DATA_MATRIX",
        "DATAMATRIX": "DATA_MATRIX",
        "EAN13": "EAN_13",
        "EAN8": "EAN_8",
        "UPCA": "UPC_A",
        "UPCE": "UPC_E",
        "CODE39": "CODE_39",
        "CODE93": "CODE_93",
        "CODE128": "CODE_128",
        "C39": "CODE_39",
        "C93": "CODE_93",
        "C128": "CODE_128",
        "I2O5": "INTERLEAVED_2_OF_5",
    }
    return aliases.get(text, text or UNKNOWN)


def payload(value: Any) -> str | None:
    if value is None:
        return None
    # Never strip internal or leading whitespace automatically: some 2D
    # payloads legitimately contain it. Empty strings are unresolved.
    result = str(value)
    return result if result else None


def canonical_gtin(value: str | None, symbology: str) -> str | None:
    """Normalize only the standards-defined UPC-A/EAN-13 equivalence.

    A UPC-A symbol is the EAN-13 number-system-zero subset.  This deliberately
    does not strip arbitrary leading zeroes or normalize other symbologies.
    """
    if value is None or not value.isdigit():
        return None
    canonical = canonical_symbology(symbology)
    if canonical == "UPC_A" and len(value) == 12:
        return f"0{value}"
    if canonical == "EAN_13" and len(value) == 13:
        return value
    return None


def payloads_are_gtin_equivalent(
    truth_payload: str | None,
    truth_symbology: str,
    candidate_payload: str | None,
    candidate_symbology: str,
) -> bool:
    if truth_payload is None or candidate_payload is None:
        return False
    truth_gtin = canonical_gtin(truth_payload, truth_symbology)
    candidate_gtin = canonical_gtin(candidate_payload, candidate_symbology)
    return truth_gtin is not None and truth_gtin == candidate_gtin


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return round(2.0 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), percent)), 6)


def normalized_object(item: dict[str, Any], label: str) -> dict[str, Any]:
    points = polygon(item.get("polygon"), label)
    result = dict(item)
    result["_polygon"] = points
    result["_kind"] = canonical_kind(item.get("kind"))
    result["_symbology"] = canonical_symbology(item.get("symbology"))
    result["_payload"] = payload(item.get("payload"))
    result["_decodable"] = bool(item.get("decodable", result["_payload"] is not None))
    result["_localization_evaluable"] = bool(item.get("localization_evaluable", True))
    result["_payload_scope"] = str(item.get("payload_scope", "localized")).lower()
    result["_size_bin"] = size_bin(polygon_long_side(points))
    return result


def normalized_prediction(item: dict[str, Any], label: str) -> dict[str, Any]:
    result = normalized_object(item, label)
    confidence = item.get("confidence", 0.0)
    try:
        result["_confidence"] = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        result["_confidence"] = 0.0
    return result


def matching(
    truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    mode: str,
    threshold: float,
) -> tuple[list[tuple[int, int, float]], np.ndarray, np.ndarray]:
    scores = np.zeros((len(truths), len(predictions)), dtype=np.float64)
    containment = np.zeros_like(scores)
    for truth_index, truth in enumerate(truths):
        for prediction_index, prediction in enumerate(predictions):
            iou, ios = similarity(truth["_polygon"], prediction["_polygon"])
            scores[truth_index, prediction_index] = iou
            containment[truth_index, prediction_index] = ios
    if mode == "strict_iou":
        eligible = scores >= threshold
        selected_scores = scores
    elif mode == "containment":
        selected_scores = np.maximum(scores, containment)
        eligible = (scores >= threshold) | (containment >= threshold)
    else:
        raise ValueError(f"unknown matching mode: {mode}")
    return maximum_cardinality_matches(selected_scores, eligible), scores, containment


def add_localization(counter: Counter[str], truth_count: int, prediction_count: int, true_positive: int) -> None:
    counter["ground_truth"] += truth_count
    counter["predictions"] += prediction_count
    counter["true_positive"] += true_positive
    counter["false_negative"] += truth_count - true_positive
    counter["false_positive"] += prediction_count - true_positive


def summarize_localization(counter: Counter[str]) -> dict[str, Any]:
    precision = ratio(counter["true_positive"], counter["predictions"])
    recall = ratio(counter["true_positive"], counter["ground_truth"])
    return {
        **{key: int(counter[key]) for key in ("ground_truth", "predictions", "true_positive", "false_positive", "false_negative")},
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
    }


def evaluate(
    dataset: dict[str, Any],
    predictions: dict[str, Any],
    *,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.5,
) -> dict[str, Any]:
    require_schema(dataset, DATASET_SCHEMA, "dataset")
    require_schema(predictions, PREDICTION_SCHEMA, "predictions")
    truth_images = image_index(dataset, "dataset")
    predicted_images = image_index(predictions, "predictions")
    missing = sorted(set(truth_images) - set(predicted_images))
    unexpected = sorted(set(predicted_images) - set(truth_images))
    if missing or unexpected:
        raise ValueError(
            f"image id mismatch: missing_predictions={missing[:10]!r}, "
            f"unexpected_predictions={unexpected[:10]!r}"
        )

    strict_total: Counter[str] = Counter()
    permissive_total: Counter[str] = Counter()
    decode_total: Counter[str] = Counter()
    decode_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    strata: dict[str, dict[str, Counter[str]]] = {
        "source": defaultdict(Counter),
        "kind": defaultdict(Counter),
        "symbology": defaultdict(Counter),
        "size": defaultdict(Counter),
    }
    latencies: list[float] = []
    peak_memory: list[float] = []
    per_image: list[dict[str, Any]] = []

    for image_id, truth_image in truth_images.items():
        prediction_image = predicted_images[image_id]
        all_truths = [
            normalized_object(item, f"{image_id}:truth:{index}")
            for index, item in enumerate(truth_image.get("objects", []))
        ]
        truths = [item for item in all_truths if item["_localization_evaluable"]]
        image_payload_truths = [
            item for item in all_truths
            if not item["_localization_evaluable"] and item["_payload_scope"] == "image"
        ]
        predicted = [
            normalized_prediction(item, f"{image_id}:prediction:{index}")
            for index, item in enumerate(prediction_image.get("predictions", []))
        ]
        strict_matches, ious, containments = matching(truths, predicted, "strict_iou", iou_threshold)
        permissive_matches, _, _ = matching(truths, predicted, "containment", containment_threshold)
        add_localization(strict_total, len(truths), len(predicted), len(strict_matches))
        add_localization(permissive_total, len(truths), len(predicted), len(permissive_matches))

        strict_by_truth = {truth: prediction for truth, prediction, _ in strict_matches}
        permissive_by_truth = {truth: prediction for truth, prediction, _ in permissive_matches}
        permissive_prediction_ids = {prediction for _, prediction, _ in permissive_matches}

        source = str(truth_image.get("source_group") or dataset.get("dataset", {}).get("name") or UNKNOWN)
        for truth_index, truth in enumerate(truths):
            keys = {
                "source": source,
                "kind": truth["_kind"],
                "symbology": truth["_symbology"],
                "size": truth["_size_bin"],
            }
            for stratum, key in keys.items():
                counter = strata[stratum][key]
                counter["ground_truth"] += 1
                if truth_index in strict_by_truth:
                    counter["strict_true_positive"] += 1
                if truth_index in permissive_by_truth:
                    counter["permissive_true_positive"] += 1

            if not truth["_decodable"] or truth["_payload"] is None:
                decode_total["not_payload_evaluable"] += 1
                decode_by_source[source]["not_payload_evaluable"] += 1
                continue
            decode_total["payload_evaluable"] += 1
            decode_by_source[source]["payload_evaluable"] += 1
            prediction_index = permissive_by_truth.get(truth_index)
            if prediction_index is None:
                decode_total["not_localized"] += 1
                decode_by_source[source]["not_localized"] += 1
                continue
            candidate = predicted[prediction_index]
            candidate_payload = candidate["_payload"]
            if candidate_payload is None:
                decode_total["localized_unresolved"] += 1
                decode_by_source[source]["localized_unresolved"] += 1
            elif candidate_payload == truth["_payload"]:
                decode_total["exact_payload"] += 1
                decode_total["semantic_payload"] += 1
                decode_by_source[source]["exact_payload"] += 1
                decode_by_source[source]["semantic_payload"] += 1
                if (
                    truth["_symbology"] != UNKNOWN
                    and candidate["_symbology"] != UNKNOWN
                    and truth["_symbology"] == candidate["_symbology"]
                ):
                    decode_total["exact_payload_and_symbology"] += 1
            elif payloads_are_gtin_equivalent(
                truth["_payload"],
                truth["_symbology"],
                candidate_payload,
                candidate["_symbology"],
            ):
                decode_total["gtin_equivalent_payload"] += 1
                decode_total["semantic_payload"] += 1
                decode_by_source[source]["gtin_equivalent_payload"] += 1
                decode_by_source[source]["semantic_payload"] += 1
            else:
                decode_total["wrong_payload"] += 1
                decode_by_source[source]["wrong_payload"] += 1

        # Some public OCR-style barcode datasets supply an image-level code
        # string but annotate only the printed digits, not barcode geometry.
        # Score their payload without fabricating a localization match.
        available_payload_predictions = {
            index for index, candidate in enumerate(predicted)
            if candidate["_payload"] is not None
            and index not in permissive_prediction_ids
        }
        for truth in image_payload_truths:
            if not truth["_decodable"] or truth["_payload"] is None:
                decode_total["not_payload_evaluable"] += 1
                decode_by_source[source]["not_payload_evaluable"] += 1
                continue
            decode_total["payload_evaluable"] += 1
            decode_by_source[source]["payload_evaluable"] += 1
            exact_index = next(
                (
                    index for index in available_payload_predictions
                    if predicted[index]["_payload"] == truth["_payload"]
                ),
                None,
            )
            equivalent_index = next(
                (
                    index for index in available_payload_predictions
                    if payloads_are_gtin_equivalent(
                        truth["_payload"],
                        truth["_symbology"],
                        predicted[index]["_payload"],
                        predicted[index]["_symbology"],
                    )
                ),
                None,
            )
            if exact_index is not None:
                candidate = predicted[exact_index]
                available_payload_predictions.remove(exact_index)
                permissive_prediction_ids.add(exact_index)
                decode_total["exact_payload"] += 1
                decode_total["semantic_payload"] += 1
                decode_by_source[source]["exact_payload"] += 1
                decode_by_source[source]["semantic_payload"] += 1
                if (
                    truth["_symbology"] != UNKNOWN
                    and candidate["_symbology"] != UNKNOWN
                    and truth["_symbology"] == candidate["_symbology"]
                ):
                    decode_total["exact_payload_and_symbology"] += 1
            elif equivalent_index is not None:
                available_payload_predictions.remove(equivalent_index)
                permissive_prediction_ids.add(equivalent_index)
                decode_total["gtin_equivalent_payload"] += 1
                decode_total["semantic_payload"] += 1
                decode_by_source[source]["gtin_equivalent_payload"] += 1
                decode_by_source[source]["semantic_payload"] += 1
            elif available_payload_predictions:
                wrong_index = next(iter(available_payload_predictions))
                available_payload_predictions.remove(wrong_index)
                permissive_prediction_ids.add(wrong_index)
                decode_total["wrong_payload"] += 1
                decode_by_source[source]["wrong_payload"] += 1
            elif predicted:
                decode_total["localized_unresolved"] += 1
                decode_by_source[source]["localized_unresolved"] += 1
            else:
                decode_total["not_localized"] += 1
                decode_by_source[source]["not_localized"] += 1

        for prediction_index, candidate in enumerate(predicted):
            if prediction_index not in permissive_prediction_ids and candidate["_payload"] is not None:
                decode_total["false_decode"] += 1

        timing = prediction_image.get("timing", {})
        latency = timing.get("latency_ms") if isinstance(timing, dict) else None
        memory = timing.get("peak_rss_mb") if isinstance(timing, dict) else None
        if isinstance(latency, (int, float)) and math.isfinite(float(latency)) and float(latency) >= 0:
            latencies.append(float(latency))
        if isinstance(memory, (int, float)) and math.isfinite(float(memory)) and float(memory) >= 0:
            peak_memory.append(float(memory))

        per_image.append({
            "id": image_id,
            "ground_truth": len(truths),
            "image_level_payload_truths": len(image_payload_truths),
            "predictions": len(predicted),
            "strict_true_positive": len(strict_matches),
            "permissive_true_positive": len(permissive_matches),
            "strict_mean_iou": round(
                sum(ious[t, p] for t, p, _ in strict_matches) / len(strict_matches),
                6,
            ) if strict_matches else 0.0,
            "maximum_containment": round(float(containments.max()), 6) if containments.size else 0.0,
        })

    payload_evaluable = decode_total["payload_evaluable"]
    decoded_attempts = (
        decode_total["exact_payload"]
        + decode_total["gtin_equivalent_payload"]
        + decode_total["wrong_payload"]
    )
    report_strata: dict[str, Any] = {}
    for stratum, values in strata.items():
        ordered_keys = SIZE_ORDER if stratum == "size" else sorted(values)
        report_strata[stratum] = {}
        for key in ordered_keys:
            counter = values.get(key, Counter())
            ground_truth = counter["ground_truth"]
            report_strata[stratum][key] = {
                "ground_truth": ground_truth,
                "strict_recall": ratio(counter["strict_true_positive"], ground_truth),
                "containment_recall": ratio(counter["permissive_true_positive"], ground_truth),
            }

    return {
        "schema_version": REPORT_SCHEMA,
        "dataset": dataset.get("dataset", {}),
        "run": predictions.get("run", {}),
        "thresholds": {
            "strict_iou": iou_threshold,
            "containment_iou_or_intersection_over_smaller": containment_threshold,
        },
        "localization": {
            "strict": summarize_localization(strict_total),
            "containment_aware": summarize_localization(permissive_total),
        },
        "decoding": {
            **{key: int(decode_total[key]) for key in (
                "payload_evaluable",
                "not_payload_evaluable",
                "exact_payload",
                "exact_payload_and_symbology",
                "gtin_equivalent_payload",
                "semantic_payload",
                "wrong_payload",
                "localized_unresolved",
                "not_localized",
                "false_decode",
            )},
            "end_to_end_exact_rate": ratio(decode_total["exact_payload"], payload_evaluable),
            "end_to_end_semantic_rate": ratio(decode_total["semantic_payload"], payload_evaluable),
            "wrong_payload_rate": ratio(decode_total["wrong_payload"], payload_evaluable),
            "conditional_decode_precision": ratio(decode_total["exact_payload"], decoded_attempts),
            "conditional_semantic_precision": ratio(decode_total["semantic_payload"], decoded_attempts),
            "by_source": {
                key: {
                    **{name: int(counter[name]) for name in (
                        "payload_evaluable",
                        "not_payload_evaluable",
                        "exact_payload",
                        "gtin_equivalent_payload",
                        "semantic_payload",
                        "wrong_payload",
                        "localized_unresolved",
                        "not_localized",
                    )},
                    "end_to_end_exact_rate": ratio(
                        counter["exact_payload"],
                        counter["payload_evaluable"],
                    ),
                    "end_to_end_semantic_rate": ratio(
                        counter["semantic_payload"],
                        counter["payload_evaluable"],
                    ),
                    "wrong_payload_rate": ratio(
                        counter["wrong_payload"],
                        counter["payload_evaluable"],
                    ),
                }
                for key, counter in sorted(decode_by_source.items())
            },
        },
        "performance": {
            "timed_images": len(latencies),
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 6) if latencies else None,
                "median": round(median(latencies), 6) if latencies else None,
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "maximum": round(max(latencies), 6) if latencies else None,
            },
            "peak_rss_mb": {
                "samples": len(peak_memory),
                "p95": percentile(peak_memory, 95),
                "maximum": round(max(peak_memory), 6) if peak_memory else None,
            },
        },
        "strata": report_strata,
        "images": per_image,
    }
