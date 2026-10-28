"""Run the frozen handwritten-rule baseline on eval_tune or eval_test.

The rule implementation remains in baseline/pure_rule_baseline.py. This wrapper
only adapts the JSONL split contract, preserves record ids and writes artifacts
to the same eval_split layout as the other evaluated methods.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from metrics import evaluate_records
from pure_rule_baseline import RULES, predict_record
from span_utils import validate_record


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON line") from error
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"{path}:{line_number}: record must be an object with id")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate record ids")
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["tune", "test"], required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    gold_path = root / "data" / "data_eval" / f"eval_{args.split}.jsonl"
    output_dir = (args.output_dir or root / "outputs" / "eval_split").resolve()
    prediction_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    errors_dir = output_dir / "errors"
    stem = f"{args.split}_pure_rules"
    prediction_path = prediction_dir / f"{stem}_predictions.jsonl"
    metrics_path = metrics_dir / f"{stem}_metrics.json"
    errors_path = errors_dir / f"{stem}_errors.json"

    existing = [path for path in (prediction_path, metrics_path, errors_path) if path.exists()]
    if existing:
        raise SystemExit(f"Pure-rule artifacts already exist; refusing overwrite: {existing[0]}")
    if not gold_path.is_file():
        raise SystemExit(f"Gold split not found: {gold_path}")

    started = time.perf_counter()
    gold = load_jsonl(gold_path)
    predictions: list[dict[str, Any]] = []
    for record in gold:
        prediction = predict_record(record)
        prediction["id"] = record["id"]
        validate_record(prediction)
        predictions.append(prediction)

    result = evaluate_records(gold, predictions)
    elapsed = time.perf_counter() - started
    metrics = {key: value for key, value in result.items() if key != "errors"}
    metrics.update(
        {
            "method": "pure_handwritten_rules_split",
            "split": f"eval_{args.split}",
            "gold_file": gold_path.relative_to(root).as_posix(),
            "rule_count": len(RULES),
            "rule_names": [rule.name for rule in RULES],
            "runtime_seconds": elapsed,
        }
    )

    write_jsonl(prediction_path, predictions)
    write_json(metrics_path, metrics)
    write_json(errors_path, result["errors"])

    print(f"Method: pure_handwritten_rules_split")
    print(f"Split:  {args.split} ({len(gold)} records)")
    print(f"Rules:  {len(RULES)}")
    print(f"Entity precision: {metrics['precision']:.4f}")
    print(f"Entity recall:    {metrics['recall']:.4f}")
    print(f"Entity F1:        {metrics['f1']:.4f}")
    print(f"Predictions: {prediction_path}")
    print(f"Metrics:     {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
