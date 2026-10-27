"""Score saved JSON/JSONL predictions with member 2's strict span metrics.

Examples:
    python src/eval/evaluate_predictions.py `
      --gold src/data/data_eval/eval_tune.jsonl `
      --predictions src/outputs/eval_split/predictions/tune_dictionary_longest_min_count_1_min_length_2_predictions.jsonl `
      --output src/outputs/eval_split/metrics/check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from metrics import evaluate_records


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON line") from error
        return records
    return json.loads(path.read_text(encoding="utf-8-sig"))


def align_predictions(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Align by id when both sides provide it; otherwise require the same order."""
    if all("id" in record for record in gold) and all("id" in record for record in predictions):
        by_id = {record["id"]: record for record in predictions}
        if len(by_id) != len(predictions):
            raise ValueError("prediction ids are not unique")
        missing = [record["id"] for record in gold if record["id"] not in by_id]
        if missing:
            raise ValueError(f"predictions missing {len(missing)} ids, first: {missing[0]}")
        aligned = [by_id[record["id"]] for record in gold]
    else:
        if len(gold) != len(predictions):
            raise ValueError("gold and prediction record counts differ")
        aligned = predictions

    for gold_record, prediction in zip(gold, aligned):
        if "text" not in prediction:
            prediction["text"] = gold_record["text"]
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-errors", type=Path, default=None)
    args = parser.parse_args()

    gold = load_records(args.gold)
    predictions = align_predictions(gold, load_records(args.predictions))
    result = evaluate_records(gold, predictions)
    compact = {key: value for key, value in result.items() if key != "errors"}
    compact["gold_file"] = args.gold.as_posix()
    compact["prediction_file"] = args.predictions.as_posix()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_errors is not None:
        args.save_errors.parent.mkdir(parents=True, exist_ok=True)
        args.save_errors.write_text(
            json.dumps(result["errors"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
