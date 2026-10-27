"""Run member 2's dictionary baseline on the frozen eval split.

This is an integration wrapper for member 4. It imports the authoritative
implementation from ``baseline/`` and never rewrites member 2's outputs.

Examples:
    python src/eval/run_dictionary_split.py --method longest --min-length 2 --split tune
    python src/eval/run_dictionary_split.py --method longest --min-length 2 --split dev
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

from dictionary_baseline import build_dictionary, load_json, predict_exact, predict_longest
from metrics import evaluate_records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    return load_json(path)


def resolve_train_path(root: Path) -> Path:
    for candidate in (
        root / "data" / "data_cleaned" / "CMeEE_train_cleaned_final.json",
        root / "data_cleaned" / "CMeEE_train_cleaned_final.json",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("找不到 CMeEE_train_cleaned_final.json")


def resolve_gold_path(root: Path, split: str) -> Path:
    if split == "dev":
        candidates = (
            root / "data" / "data_raw" / "CMeEE_dev.json",
            root / "data_raw" / "CMeEE_dev.json",
        )
    else:
        candidates = (
            root / "data" / "data_eval" / f"eval_{split}.jsonl",
            root / "data_eval" / f"eval_{split}.jsonl",
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 {split} 的金标文件")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["exact", "longest"], default="longest")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--split", choices=["tune", "test", "dev"], default="tune")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.min_count < 1 or args.min_length < 1:
        raise ValueError("--min-count and --min-length must be at least 1")

    root = args.root.resolve()
    train_path = resolve_train_path(root)
    gold_path = resolve_gold_path(root, args.split)
    output_dir = (args.output_dir or root / "outputs" / "eval_split").resolve()
    prediction_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    error_dir = output_dir / "errors"
    for directory in (prediction_dir, metrics_dir, error_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train = load_json(train_path)
    gold = load_records(gold_path)
    started = time.perf_counter()
    dictionary = build_dictionary(train, min_count=args.min_count, min_length=args.min_length)
    predictor = predict_exact if args.method == "exact" else predict_longest
    predictions: list[dict[str, Any]] = []
    for record in gold:
        prediction = predictor(record, dictionary)
        if "id" in record:
            prediction["id"] = record["id"]
        predictions.append(prediction)

    metrics = evaluate_records(gold, predictions)
    elapsed = time.perf_counter() - started
    method = f"dictionary_{args.method}_min_count_{args.min_count}_min_length_{args.min_length}"
    metrics.update(
        {
            "method": method,
            "split": args.split,
            "train_file": train_path.relative_to(root).as_posix(),
            "gold_file": gold_path.relative_to(root).as_posix(),
            "dictionary_size": len(dictionary),
            "runtime_seconds": elapsed,
        }
    )

    stem = f"{args.split}_{method}"
    prediction_path = prediction_dir / f"{stem}_predictions.jsonl"
    metrics_path = metrics_dir / f"{stem}_metrics.json"
    errors_path = error_dir / f"{stem}_errors.json"
    prediction_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in predictions),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps({key: value for key, value in metrics.items() if key != "errors"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    errors_path.write_text(json.dumps(metrics["errors"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Method: {method}")
    print(f"Split:  {args.split} ({len(gold)} records)")
    print(f"Entity precision: {metrics['precision']:.4f}")
    print(f"Entity recall:    {metrics['recall']:.4f}")
    print(f"Entity F1:        {metrics['f1']:.4f}")
    print(f"Predictions: {prediction_path}")
    print(f"Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
