"""
CMeEE 预测结果的严格实体跨度评估模块。
除官方标准实体级微平均 P/R/F1 外，额外提供两种字符维度指标：
token_*：最长非重叠投影，每个字符只保留一个标签
char_multi_label_*：保留全部实体（含嵌套），每个字符可绑定一组标签，按类别评估
"""

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


def _multi_label_char_sets(record: dict[str, Any]) -> list[set[str]]:
    """Keep every entity label per character, including nested entities.

    Unlike the longest-span projection, nothing is dropped: overlapping and
    nested entities all contribute their labels at the same positions.
    """
    text = record["text"]
    labels: list[set[str]] = [set() for _ in text]
    for entity in record.get("entities", []):
        start, end = entity["start_idx"], entity["end_idx"]
        for position in range(start, end + 1):
            labels[position].add(entity["type"])
    return labels


def _aggregate_token_counts(counts: Counter[str]) -> dict[str, Any]:
    """Turn per-label tp/fp/fn character counts into overall and per-label P/R/F1."""
    total_tp = sum(value for key, value in counts.items() if key.endswith(":tp"))
    total_fp = sum(value for key, value in counts.items() if key.endswith(":fp"))
    total_fn = sum(value for key, value in counts.items() if key.endswith(":fn"))
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_label: dict[str, dict[str, Any]] = {}
    for label in sorted({key.split(":", 1)[0] for key in counts}):
        tp, fp, fn = counts[label + ":tp"], counts[label + ":fp"], counts[label + ":fn"]
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
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_label": per_label,
    }


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
    multi_counts: Counter[str] = Counter()
    multi_total = 0
    multi_exact_correct = 0
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

        for gold_labels, predicted_labels in zip(_multi_label_char_sets(gold), _multi_label_char_sets(predicted)):
            multi_total += 1
            multi_exact_correct += gold_labels == predicted_labels
            for label in gold_labels | predicted_labels:
                if label in gold_labels and label in predicted_labels:
                    multi_counts[label + ":tp"] += 1
                elif label in predicted_labels:
                    multi_counts[label + ":fp"] += 1
                else:
                    multi_counts[label + ":fn"] += 1

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

    token_stats = _aggregate_token_counts(token_counts)
    multi_stats = _aggregate_token_counts(multi_counts)

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
        "token_entity_precision": token_stats["precision"],
        "token_entity_recall": token_stats["recall"],
        "token_entity_f1": token_stats["f1"],
        "token_per_label": token_stats["per_label"],
        "char_multi_label_projection": "none_all_entities_kept",
        "char_multi_label_total": multi_total,
        "char_multi_label_exact_accuracy": multi_exact_correct / multi_total if multi_total else 0.0,
        "char_multi_label_precision": multi_stats["precision"],
        "char_multi_label_recall": multi_stats["recall"],
        "char_multi_label_f1": multi_stats["f1"],
        "char_multi_label_per_label": multi_stats["per_label"],
        "errors": errors,
        "per_label": per_label,
    }

