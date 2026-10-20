"""Grid search over LinearSVC hyperparameters for the character-level baseline.

Reuses the feature pipeline of small_model.py: the shared feature matrix and
dev matrices are built once, then every (C, class_weight) combination fits its
own classifier. Results go to outputs/grid/ so the main experiment summary
table is not flooded with tuning runs.

Run from the repository root:

    python baseline/small_model_grid_search.py

This is a dev-set tuning experiment: the winner is selected on
data_raw/CMeEE_dev.json and must be reported as such.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from metrics import evaluate_records
from small_model import char_features, decode_bio, labels_from_entities, load_json, sparse_int32


def row_from_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one summary row from a saved grid metrics file."""
    return {
        "C": data["C"],
        "class_weight": data["class_weight"],
        "entity_precision": data["precision"],
        "entity_recall": data["recall"],
        "entity_f1": data["f1"],
        "token_f1": data["token_entity_f1"],
        "char_multi_label_f1": data["char_multi_label_f1"],
        "predicted_entities": data["predicted_entities"],
        "runtime_seconds": data["runtime_seconds"],
    }


def write_grid_summary(
    grid_dir: Path,
    summary_rows: list[dict[str, Any]],
    c_values: list[float],
    class_weights: list[Any],
    seed: int,
) -> None:
    """Sort grid rows, write the JSON/Markdown summaries and print the winner."""
    rows = sorted(
        summary_rows,
        key=lambda row: (-float(row.get("entity_f1") or -1.0), -float(row.get("token_f1") or -1.0)),
    )
    completed = [row for row in rows if "skipped" not in row]
    summary = {
        "grid": {
            "C_values": c_values,
            "class_weight_values": [None if value is None else value for value in class_weights],
            "seed": seed,
            "note": "dev 集调参实验；最终配置选 entity_f1 最高的组合，报告中须说明为开发集调参",
        },
        "rows": rows,
        "best": completed[0],
    }
    (grid_dir / "small_model_grid_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Small model grid search (dev)", ""]
    lines.append(
        "| C | class_weight | Entity P | Entity R | Entity F1 | Token F1 | Char-multi F1 | Predicted | Runtime (s) |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if "skipped" in row:
            lines.append(
                f"| {row['C']} | {row['class_weight']} | skipped | skipped | skipped | skipped | skipped | skipped | {row['skipped']} |"
            )
            continue
        lines.append(
            f"| {row['C']} | {row['class_weight']} | {row['entity_precision']:.4f} | {row['entity_recall']:.4f} | "
            f"{row['entity_f1']:.4f} | {row['token_f1']:.4f} | {row['char_multi_label_f1']:.4f} | "
            f"{row['predicted_entities']} | {row['runtime_seconds']:.1f} |"
        )
    (grid_dir / "small_model_grid_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nBest combination:")
    print(json.dumps(completed[0], ensure_ascii=False, indent=2))
    print(f"\nGrid outputs: {grid_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--C-values", type=str, default="0.01,0.1,1.0,10.0", help="comma-separated C candidates")
    parser.add_argument("--class-weights", type=str, default="balanced,none", help="comma-separated class_weight candidates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="rebuild the grid summary from existing metrics files without fitting",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    grid_dir = output_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    train_path = root / "data_cleaned" / "CMeEE_train_cleaned_final.json"
    dev_path = root / "data_raw" / "CMeEE_dev.json"

    c_values = [float(value) for value in args.C_values.split(",")]
    class_weights = [None if value.lower() == "none" else value for value in args.class_weights.split(",")]

    if args.summarize_only:
        rows: list[dict[str, Any]] = []
        for C, class_weight in itertools.product(c_values, class_weights):
            weight_name = "none" if class_weight is None else class_weight
            metrics_path = grid_dir / f"small_model_grid_C{C}_cw_{weight_name}_metrics.json"
            if metrics_path.exists():
                rows.append(row_from_metrics(json.loads(metrics_path.read_text(encoding="utf-8"))))
            else:
                rows.append({"C": C, "class_weight": weight_name, "skipped": "not run (see README tuning note)"})
        write_grid_summary(grid_dir, rows, c_values, class_weights, args.seed)
        return

    train = load_json(train_path)
    dev = load_json(dev_path)

    dictionary_phrases = {entity["entity"] for record in train for entity in record.get("entities", [])}
    dictionary_chars = {char for phrase in dictionary_phrases for char in phrase}

    print("Building shared features ...", flush=True)
    started = time.perf_counter()
    train_features: list[dict[str, str]] = []
    train_labels: list[str] = []
    for record in train:
        text = record["text"]
        train_features.extend(char_features(text, index, dictionary_chars) for index in range(len(text)))
        train_labels.extend(labels_from_entities(text, record.get("entities", [])))

    vectorizer = DictVectorizer()
    x_train = sparse_int32(vectorizer.fit_transform(train_features))

    dev_matrices: list[Any] = []
    for record in dev:
        text = record["text"]
        features = [char_features(text, index, dictionary_chars) for index in range(len(text))]
        dev_matrices.append(sparse_int32(vectorizer.transform(features)))
    print(f"Features ready in {time.perf_counter() - started:.1f}s", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for C, class_weight in itertools.product(c_values, class_weights):
        combo_started = time.perf_counter()
        weight_name = "none" if class_weight is None else class_weight
        stem = f"small_model_grid_C{C}_cw_{weight_name}"
        print(f"Fitting C={C}, class_weight={weight_name} ...", flush=True)
        classifier = LinearSVC(C=C, class_weight=class_weight, random_state=args.seed)
        classifier.fit(x_train, train_labels)

        predictions: list[dict[str, Any]] = []
        for record, matrix in zip(dev, dev_matrices):
            text = record["text"]
            labels = classifier.predict(matrix).tolist()
            predictions.append({"text": text, "entities": decode_bio(text, labels)})

        metrics = evaluate_records(dev, predictions)
        compact = {key: value for key, value in metrics.items() if key != "errors"}
        compact.update(
            {
                "method": f"character_linear_svm_grid_C{C}_class_weight_{weight_name}",
                "seed": args.seed,
                "C": C,
                "class_weight": weight_name,
                "train_file": str(train_path),
                "eval_file": str(dev_path),
                "projection": "longest_non_overlapping_gold_entities",
                "runtime_seconds": time.perf_counter() - combo_started,
            }
        )
        (grid_dir / f"{stem}_metrics.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "C": C,
                "class_weight": weight_name,
                "entity_precision": metrics["precision"],
                "entity_recall": metrics["recall"],
                "entity_f1": metrics["f1"],
                "token_f1": metrics["token_entity_f1"],
                "char_multi_label_f1": metrics["char_multi_label_f1"],
                "predicted_entities": metrics["predicted_entities"],
                "runtime_seconds": compact["runtime_seconds"],
            }
        )
        print(
            f"  entity F1={metrics['f1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"pred={metrics['predicted_entities']}), {time.perf_counter() - combo_started:.0f}s",
            flush=True,
        )

    write_grid_summary(grid_dir, summary_rows, c_values, class_weights, args.seed)


if __name__ == "__main__":
    main()
