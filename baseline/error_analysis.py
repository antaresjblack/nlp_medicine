"""Classify errors from a saved CMeEE prediction file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["start_idx"] <= right["end_idx"] and right["start_idx"] <= left["end_idx"]


def classify(gold: dict[str, Any], predicted: dict[str, Any]) -> list[dict[str, Any]]:
    gold_entities = gold.get("entities", [])
    predicted_entities = predicted.get("entities", [])
    errors: list[dict[str, Any]] = []
    matched_predicted: set[int] = set()
    matched_gold: set[int] = set()

    for gold_index, gold_entity in enumerate(gold_entities):
        for predicted_index, predicted_entity in enumerate(predicted_entities):
            if predicted_index in matched_predicted:
                continue
            same_span = (
                gold_entity["start_idx"] == predicted_entity["start_idx"]
                and gold_entity["end_idx"] == predicted_entity["end_idx"]
            )
            if same_span and gold_entity["type"] != predicted_entity["type"]:
                matched_gold.add(gold_index)
                matched_predicted.add(predicted_index)
                errors.append({"kind": "type_error", "gold": gold_entity, "predicted": predicted_entity})
                break
            if (
                gold_entity["type"] == predicted_entity["type"]
                and overlap(gold_entity, predicted_entity)
                and not same_span
            ):
                matched_gold.add(gold_index)
                matched_predicted.add(predicted_index)
                errors.append({"kind": "boundary_error", "gold": gold_entity, "predicted": predicted_entity})
                break

    for gold_index, gold_entity in enumerate(gold_entities):
        if gold_index not in matched_gold:
            errors.append({"kind": "missed_entity", "gold": gold_entity, "predicted": None})
    for predicted_index, predicted_entity in enumerate(predicted_entities):
        if predicted_index not in matched_predicted:
            errors.append({"kind": "spurious_entity", "gold": None, "predicted": predicted_entity})
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-kind", type=int, default=5)
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8-sig"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    if len(gold) != len(predictions):
        raise ValueError("gold and prediction record counts differ")

    all_errors: list[dict[str, Any]] = []
    for record_index, (gold_record, prediction_record) in enumerate(zip(gold, predictions)):
        for error in classify(gold_record, prediction_record):
            error["record_index"] = record_index
            error["text"] = gold_record["text"]
            all_errors.append(error)

    counts = Counter(error["kind"] for error in all_errors)
    examples = {
        kind: [error for error in all_errors if error["kind"] == kind][: args.examples_per_kind]
        for kind in sorted(counts)
    }
    result = {
        "gold_file": str(args.gold),
        "prediction_file": str(args.predictions),
        "error_counts": dict(sorted(counts.items())),
        "total_error_events": len(all_errors),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["error_counts"], ensure_ascii=False, indent=2))
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
