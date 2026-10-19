"""Strict span-level evaluation for CMeEE predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from span_utils import entity_key


def _keys(entities: list[dict[str, Any]]) -> set[tuple[int, int, str]]:
    return {entity_key(entity) for entity in entities}


def _project_character_labels(record: dict[str, Any]) -> list[str]:
    """Project spans to one label per character using longest-span priority.

    CMeEE contains nested spans. A flat character metric needs one label at a
    position, so this deterministic projection keeps the longest span first.
    Entity-level scoring above still evaluates the original spans exactly.
    """
    text = record["text"]
    labels = ["O"] * len(text)
    entities = sorted(
        record.get("entities", []),
        key=lambda entity: (-len(entity["entity"]), entity["start_idx"], entity["end_idx"], entity["type"]),
    )
    occupied = [False] * len(text)
    for entity in entities:
        start, end, label = entity["start_idx"], entity["end_idx"], entity["type"]
        if any(occupied[start : end + 1]):
            continue
        for position in range(start, end + 1):
            occupied[position] = True
            labels[position] = label
    return labels


def evaluate_records(gold_records: list[dict[str, Any]], predicted_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate predictions aligned to gold records in the same order.

    A prediction is correct only when start, inclusive end, and label all match.
    """
    if len(gold_records) != len(predicted_records):
        raise ValueError("gold and prediction record counts differ")

    gold_count = 0
    predicted_count = 0
    true_positive = 0
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    token_counts: Counter[str] = Counter()
    token_total = 0
    token_correct = 0
    errors: list[dict[str, Any]] = []

    for record_index, (gold, predicted) in enumerate(zip(gold_records, predicted_records)):
        gold_keys = _keys(gold.get("entities", []))
        predicted_keys = _keys(predicted.get("entities", []))
        correct = gold_keys & predicted_keys
        missing = gold_keys - predicted_keys
        extra = predicted_keys - gold_keys
        gold_count += len(gold_keys)
        predicted_count += len(predicted_keys)
        true_positive += len(correct)

        for _, _, label in correct:
            by_label[label]["tp"] += 1
        for _, _, label in missing:
            by_label[label]["fn"] += 1
        for _, _, label in extra:
            by_label[label]["fp"] += 1

        gold_token_labels = _project_character_labels(gold)
        predicted_token_labels = _project_character_labels(predicted)
        for gold_label, predicted_label in zip(gold_token_labels, predicted_token_labels):
            token_total += 1
            token_correct += gold_label == predicted_label
            if gold_label != "O" and predicted_label != "O" and gold_label == predicted_label:
                token_counts[gold_label + ":tp"] += 1
            elif predicted_label != "O":
                token_counts[predicted_label + ":fp"] += 1
            if gold_label != "O" and gold_label != predicted_label:
                token_counts[gold_label + ":fn"] += 1

        if missing or extra:
            errors.append(
                {
                    "record_index": record_index,
                    "text": gold["text"],
                    "gold_entities": gold.get("entities", []),
                    "predicted_entities": predicted.get("entities", []),
                    "missing": sorted(missing),
                    "extra": sorted(extra),
                }
            )

    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_label: dict[str, dict[str, Any]] = {}
    for label, counts in sorted(by_label.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        label_precision = tp / (tp + fp) if tp + fp else 0.0
        label_recall = tp / (tp + fn) if tp + fn else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if label_precision + label_recall
            else 0.0
        )
        per_label[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    token_tp = sum(value for key, value in token_counts.items() if key.endswith(":tp"))
    token_fp = sum(value for key, value in token_counts.items() if key.endswith(":fp"))
    token_fn = sum(value for key, value in token_counts.items() if key.endswith(":fn"))
    token_precision = token_tp / (token_tp + token_fp) if token_tp + token_fp else 0.0
    token_recall = token_tp / (token_tp + token_fn) if token_tp + token_fn else 0.0
    token_f1 = (
        2 * token_precision * token_recall / (token_precision + token_recall)
        if token_precision + token_recall
        else 0.0
    )

    return {
        "metric": "strict_entity_micro",
        "gold_entities": gold_count,
        "predicted_entities": predicted_count,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "records_with_errors": len(errors),
        "token_projection": "longest_non_overlapping_span_per_character",
        "token_total": token_total,
        "token_accuracy": token_correct / token_total if token_total else 0.0,
        "token_entity_precision": token_precision,
        "token_entity_recall": token_recall,
        "token_entity_f1": token_f1,
        "errors": errors,
        "per_label": per_label,
    }

