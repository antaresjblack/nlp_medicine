"""Re-evaluate saved prediction files with the shared metrics.

Single mode writes a new compact metrics file:

    python baseline/evaluate_saved.py \
        --gold data_raw/CMeEE_dev.json \
        --predictions outputs/dictionary_longest_min_count_1_min_length_2_predictions.json \
        --output outputs/dictionary_best_metrics_recomputed.json

Batch mode refreshes every ``*_metrics.json`` in a directory in place after a
metrics-code change, keeping each file's method metadata:

    python baseline/evaluate_saved.py --gold data_raw/CMeEE_dev.json --batch-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import evaluate_records


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def refresh_batch(gold: list[dict], directory: Path) -> None:
    """Recompute and rewrite every metrics file that has a matching prediction file."""
    for metrics_path in sorted(directory.glob("*_metrics.json")):
        predictions_name = metrics_path.name.replace("_metrics.json", "_predictions.json")
        predictions_path = metrics_path.with_name(predictions_name)
        if not predictions_path.exists():
            print(f"skip {metrics_path.name}: no matching predictions file")
            continue
        predictions = load_json(predictions_path)
        result = evaluate_records(gold, predictions)
        compact = {key: value for key, value in result.items() if key != "errors"}
        old = load_json(metrics_path)
        merged = {key: value for key, value in old.items() if key not in compact}
        merged.update(compact)
        metrics_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"updated {metrics_path.name}: entity_f1={compact['f1']:.4f}, "
            f"char_multi_label_f1={compact['char_multi_label_f1']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None, help="single prediction file to re-evaluate")
    parser.add_argument("--output", type=Path, default=None, help="where to write the recomputed metrics (single mode)")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="refresh every *_metrics.json in this directory in place, keeping method metadata",
    )
    args = parser.parse_args()

    gold = load_json(args.gold)

    if args.batch_dir is not None:
        refresh_batch(gold, args.batch_dir)
        return

    if args.predictions is None or args.output is None:
        parser.error("--predictions and --output are required unless --batch-dir is given")

    predictions = load_json(args.predictions)
    result = evaluate_records(gold, predictions)
    compact = {key: value for key, value in result.items() if key != "errors"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
