"""
字典跨度特征消融实验。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from dict_span_features import DictSpanAnnotator
from metrics import evaluate_records
from small_model import char_features, decode_bio, labels_from_entities, load_json, sparse_int32

FEATURE_GROUPS: tuple[str, ...] = (
    "in_dict_span",
    "dict_boundary",
    "dict_label",
    "dict_span_len",
    "dict_freq",
    "label_conflict",
)


def zero_columns(matrix: Any, names: np.ndarray, drop_prefixes: set[str]) -> Any:
    """Return a copy of ``matrix`` with the columns of dropped features zeroed.

    Implemented by filtering COO entries instead of fancy column assignment,
    which avoids scipy index-dtype and efficiency issues.
    """
    if not drop_prefixes:
        return matrix
    mask = np.zeros(len(names), dtype=bool)
    for prefix in drop_prefixes:
        mask |= np.array([name.startswith(prefix) for name in names], dtype=bool)
    coo = matrix.tocoo()
    keep = ~mask[coo.col]
    filtered = coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=matrix.shape)
    return sparse_int32(filtered.tocsr())


def build_features(
    records: list[dict[str, Any]],
    dictionary_chars: set[str],
    annotator: DictSpanAnnotator | None,
) -> tuple[list[dict[str, str]], list[int]]:
    """Return per-character feature dicts and each record's character count."""
    features: list[dict[str, str]] = []
    lengths: list[int] = []
    for record in records:
        text = record["text"]
        lengths.append(len(text))
        spans = annotator.annotate(text) if annotator is not None else None
        features.extend(
            char_features(text, index, dictionary_chars, spans[index] if spans is not None else None)
            for index in range(len(text))
        )
    return features, lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--class-weight", type=str, default="balanced", help="'none' or 'balanced'")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    ablation_dir = output_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    train_path = root / "data_cleaned" / "CMeEE_train_cleaned_final.json"
    dev_path = root / "data_raw" / "CMeEE_dev.json"
    train = load_json(train_path)
    dev = load_json(dev_path)
    class_weight = None if args.class_weight.lower() == "none" else args.class_weight

    dictionary_phrases = {entity["entity"] for record in train for entity in record.get("entities", [])}
    dictionary_chars = {char for phrase in dictionary_phrases for char in phrase}
    annotator = DictSpanAnnotator(train)

    print("Building shared features ...", flush=True)
    started = time.perf_counter()
    train_features, _ = build_features(train, dictionary_chars, annotator)
    train_labels: list[str] = []
    for record in train:
        text = record["text"]
        train_labels.extend(labels_from_entities(text, record.get("entities", [])))
    dev_features, dev_lengths = build_features(dev, dictionary_chars, annotator)

    vectorizer = DictVectorizer()
    x_train_full = sparse_int32(vectorizer.fit_transform(train_features))
    x_dev_full = sparse_int32(vectorizer.transform(dev_features))
    names = vectorizer.get_feature_names_out()
    print(f"Features ready in {time.perf_counter() - started:.1f}s", flush=True)

    configs: list[tuple[str, set[str]]] = [("full", set())]
    configs.extend((f"remove_{group}", {group}) for group in FEATURE_GROUPS)
    configs.append(("remove_all", set(FEATURE_GROUPS)))

    rows: list[dict[str, Any]] = []
    for config_name, drop in configs:
        run_started = time.perf_counter()
        print(f"Training {config_name} ...", flush=True)
        x_train = zero_columns(x_train_full, names, drop)
        x_dev = zero_columns(x_dev_full, names, drop)
        classifier = LinearSVC(C=args.C, class_weight=class_weight, random_state=args.seed)
        classifier.fit(x_train, train_labels)

        flat_labels = classifier.predict(x_dev).tolist()
        predictions: list[dict[str, Any]] = []
        offset = 0
        for record, length in zip(dev, dev_lengths):
            text = record["text"]
            predictions.append({"text": text, "entities": decode_bio(text, flat_labels[offset : offset + length])})
            offset += length

        metrics = evaluate_records(dev, predictions)
        compact = {key: value for key, value in metrics.items() if key != "errors"}
        compact.update(
            {
                "method": f"character_linear_svm_ablation_{config_name}",
                "removed_feature_groups": sorted(drop),
                "seed": args.seed,
                "C": args.C,
                "class_weight": class_weight,
                "train_file": str(train_path),
                "eval_file": str(dev_path),
                "projection": "longest_non_overlapping_gold_entities",
                "runtime_seconds": time.perf_counter() - run_started,
            }
        )
        (ablation_dir / f"small_model_ablation_{config_name}_metrics.json").write_text(
            json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "config": config_name,
                "removed": sorted(drop),
                "entity_precision": metrics["precision"],
                "entity_recall": metrics["recall"],
                "entity_f1": metrics["f1"],
                "token_f1": metrics["token_entity_f1"],
                "char_multi_label_f1": metrics["char_multi_label_f1"],
                "runtime_seconds": compact["runtime_seconds"],
            }
        )
        print(
            f"  {config_name}: entity F1={metrics['f1']:.4f} "
            f"(P={metrics['precision']:.4f}, R={metrics['recall']:.4f}), "
            f"{time.perf_counter() - run_started:.0f}s",
            flush=True,
        )

    full_f1 = next(row["entity_f1"] for row in rows if row["config"] == "full")
    for row in rows:
        row["delta_f1"] = row["entity_f1"] - full_f1
    ordered = sorted(rows, key=lambda row: row["entity_f1"])

    report_json = {
        "hyperparameters": {"C": args.C, "class_weight": class_weight, "seed": args.seed},
        "note": "从最终配置出发逐组移除词典跨度特征，其余保持不变；delta_f1 为相对 full 的变化",
        "full_entity_f1": full_f1,
        "rows": ordered,
    }
    (ablation_dir / "ablation_report.json").write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 词典跨度特征消融实验报告（dev）",
        "",
        f"方法：从最终配置（C={args.C}、class_weight={class_weight}、seed={args.seed}，实体级 F1 {full_f1:.4f}）出发，"
        "每次移除一组特征，其余保持不变。ΔF1 = 移除后 F1 − 完整 F1（负数表示下降）。",
        "",
        "| 移除的特征 | Entity P | Entity R | Entity F1 | ΔF1 | Token F1 | 多标签字符 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        removed_text = "无（完整模型）" if row["config"] == "full" else ", ".join(row["removed"])
        lines.append(
            f"| {removed_text} | {row['entity_precision']:.4f} | {row['entity_recall']:.4f} | "
            f"{row['entity_f1']:.4f} | {row['delta_f1']:+.4f} | {row['token_f1']:.4f} | {row['char_multi_label_f1']:.4f} |"
        )
    single_removals = [row for row in ordered if len(row["removed"]) == 1]
    most_important = single_removals[0]
    least_important = single_removals[-1]
    remove_all = next(row for row in ordered if row["config"] == "remove_all")
    lines.extend(
        [
            "",
            "## 自动结论",
            "",
            f"- 贡献最大的单组特征：**{most_important['removed'][0]}**（移除后 F1 下降 {most_important['delta_f1']:+.4f}）",
            f"- 贡献最小的单组特征：**{least_important['removed'][0]}**（移除后 F1 变化 {least_important['delta_f1']:+.4f}）",
            f"- 全部跨度特征的总贡献：{remove_all['delta_f1']:+.4f}（从 {full_f1:.4f} 到 {remove_all['entity_f1']:.4f}）",
        ]
    )
    report_path = ablation_dir / "ablation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nAblation results (sorted by entity F1):")
    for row in ordered:
        removed_text = "none (full)" if row["config"] == "full" else ", ".join(row["removed"])
        print(f"  {row['config']:<24} F1={row['entity_f1']:.4f}  delta={row['delta_f1']:+.4f}  removed={removed_text}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
