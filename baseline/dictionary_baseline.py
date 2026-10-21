"""
CMeEE 词典基线模型。
仓库根目录运行示例：
python baseline/dictionary_baseline.py --method exact
python baseline/dictionary_baseline.py --method longest
python baseline/dictionary_baseline.py --method longest --min-length 2 --eval-file data_raw/CMeEE_test.json

默认在验证集评估。如果评估文件没有标准标注（如 CMeEE_test.json），脚本只输出预测结果，不计算指标。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metrics import evaluate_records
from span_utils import canonical_entity, find_occurrences, resolve_label


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def build_dictionary(records: list[dict[str, Any]], min_count: int = 1, min_length: int = 1) -> dict[str, str]:
    """Build phrase -> majority label from training entities only."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for entity in record.get("entities", []):
            counts[entity["entity"]][entity["type"]] += 1
    return {
        phrase: resolve_label(label_counts)
        for phrase, label_counts in counts.items()
        if sum(label_counts.values()) >= min_count and len(phrase) >= min_length
    }


def predict_exact(record: dict[str, Any], dictionary: dict[str, str]) -> dict[str, Any]:
    """Return all dictionary matches, including overlapping spans."""
    text = record["text"]
    predictions: list[dict[str, Any]] = []
    for phrase, label in dictionary.items():
        for start, end in find_occurrences(text, phrase):
            predictions.append(canonical_entity(start, end, label, phrase))
    predictions.sort(key=lambda item: (item["start_idx"], item["end_idx"], item["type"]))
    return {"text": text, "entities": predictions}


def predict_longest(record: dict[str, Any], dictionary: dict[str, str]) -> dict[str, Any]:
    """Greedily keep the longest non-overlapping dictionary matches."""
    candidates = predict_exact(record, dictionary)["entities"]
    candidates.sort(key=lambda item: (-len(item["entity"]), item["start_idx"], item["end_idx"]))
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        overlaps = any(
            candidate["start_idx"] <= chosen["end_idx"]
            and chosen["start_idx"] <= candidate["end_idx"]
            for chosen in selected
        )
        if not overlaps:
            selected.append(candidate)
    selected.sort(key=lambda item: (item["start_idx"], item["end_idx"], item["type"]))
    return {"text": record["text"], "entities": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["exact", "longest"], default="longest")
    parser.add_argument("--min-count", type=int, default=1, help="keep phrases seen at least this many times")
    parser.add_argument("--min-length", type=int, default=1, help="keep phrases at least this many characters long")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="file to predict/evaluate (default: data_raw/CMeEE_dev.json; use CMeEE_test.json for final predictions)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = root / "data_cleaned" / "CMeEE_train_cleaned_final.json"
    eval_path = (args.eval_file or root / "data_raw" / "CMeEE_dev.json").resolve()
    train = load_json(train_path)
    eval_records = load_json(eval_path)
    started = time.perf_counter()
    if args.min_count < 1 or args.min_length < 1:
        raise ValueError("--min-count and --min-length must be at least 1")
    dictionary = build_dictionary(train, min_count=args.min_count, min_length=args.min_length)
    predictor = predict_exact if args.method == "exact" else predict_longest
    predictions = [predictor(record, dictionary) for record in eval_records]
    elapsed = time.perf_counter() - started

    eval_tag = "" if eval_path.name == "CMeEE_dev.json" else "_" + eval_path.stem.removeprefix("CMeEE_")
    stem = f"dictionary_{args.method}_min_count_{args.min_count}_min_length_{args.min_length}{eval_tag}"
    prediction_path = output_dir / f"{stem}_predictions.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Method: dictionary_{args.method}_min_count_{args.min_count}_min_length_{args.min_length}")
    print(f"Dictionary entries: {len(dictionary)}")
    if not any(record.get("entities") for record in eval_records):
        print(f"Eval file {eval_path.name} has no gold entities -> predict-only mode, no metrics computed.")
        print(f"Predictions: {prediction_path}")
        return

    metrics = evaluate_records(eval_records, predictions)
    metrics["method"] = f"dictionary_{args.method}_min_count_{args.min_count}_min_length_{args.min_length}"
    metrics["train_file"] = str(train_path)
    metrics["eval_file"] = str(eval_path)
    metrics["dictionary_size"] = len(dictionary)
    metrics["runtime_seconds"] = elapsed

    metrics_path = output_dir / f"{stem}_metrics.json"
    errors_path = output_dir / f"{stem}_errors.json"
    compact_metrics = {key: value for key, value in metrics.items() if key != "errors"}
    metrics_path.write_text(json.dumps(compact_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_path.write_text(json.dumps(metrics["errors"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Entity precision: {metrics['precision']:.4f}")
    print(f"Entity recall:    {metrics['recall']:.4f}")
    print(f"Entity F1:        {metrics['f1']:.4f}")
    print(f"Predictions: {prediction_path}")
    print(f"Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
