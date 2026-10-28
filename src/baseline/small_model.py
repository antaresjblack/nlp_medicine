"""
一个轻量级的字符级线性命名实体识别基线模型。
CMeEE黄金数据集包含重叠/嵌套实体，而单个BIO序列在每个字符位置仅能表示一个标签。该基线采用了已记录的投影方案：训练时保留最长的非重叠黄金实体，预测阶段解码出一条BIO序列。原始的跨度数据保持不变，评估过程仍采用严格的跨度匹配规则。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

try:
    from .dict_span_features import DictSpanAnnotator
    from .metrics import evaluate_records
except ImportError:
    from dict_span_features import DictSpanAnnotator
    from metrics import evaluate_records


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def sparse_int32(matrix: Any) -> Any:
    """
    使用32位整数索引的稀疏矩阵，以避免scipy索引数据类型和效率问题。
    """
    matrix = matrix.tocsr()
    return csr_matrix(
        (matrix.data, matrix.indices.astype(np.int32), matrix.indptr.astype(np.int32)),
        shape=matrix.shape,
    )


def longest_non_overlapping_entities(record: dict[str, Any]) -> list[dict[str, Any]]:
    """将嵌套的金标准跨度映射到唯一确定的、仅保留最长跨度的层级中。"""
    entities = sorted(
        record.get("entities", []),
        key=lambda entity: (-len(entity["entity"]), entity["start_idx"], entity["end_idx"], entity["type"]),
    )
    selected: list[dict[str, Any]] = []
    for entity in entities:
        overlaps = any(
            entity["start_idx"] <= chosen["end_idx"]
            and chosen["start_idx"] <= entity["end_idx"]
            for chosen in selected
        )
        if not overlaps:
            selected.append(entity)
    return sorted(selected, key=lambda entity: (entity["start_idx"], entity["end_idx"]))


def char_features(
    text: str,
    index: int,
    dictionary_chars: set[str],
    span_features: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    为单个字符创建可解释的小型局部特征。
    `span_features` 可选择性携带该位置的字典跨度特征（参见 dict_span_features.DictSpanAnnotator）。
    """
    character = text[index]
    features = {
        "char": character,
        "prev1": text[index - 1] if index > 0 else "<BOS>",
        "next1": text[index + 1] if index + 1 < len(text) else "<EOS>",
        "prev2": text[index - 2] if index > 1 else "<BOS>",
        "next2": text[index + 2] if index + 2 < len(text) else "<EOS>",
        "is_digit": str(character.isdigit()),
        "is_alpha": str(character.isalpha()),
        "is_ascii": str(character.isascii()),
        "is_dictionary_char": str(character in dictionary_chars),
    }
    if span_features:
        features.update(span_features)
    return features


def labels_from_entities(text: str, entities: list[dict[str, Any]]) -> list[str]:
    labels = ["O"] * len(text)
    for entity in longest_non_overlapping_entities({"text": text, "entities": entities}):
        start, end, entity_type = entity["start_idx"], entity["end_idx"], entity["type"]
        for position in range(start, end + 1):
            labels[position] = f"B-{entity_type}" if position == start else f"I-{entity_type}"
    return labels


def decode_bio(text: str, labels: list[str]) -> list[dict[str, Any]]:
    """解码BIO标签，并以保守方式修复不合法的I-*类型转移。"""
    entities: list[dict[str, Any]] = []
    start: int | None = None
    current_type: str | None = None

    def close(end: int) -> None:
        nonlocal start, current_type
        if start is not None and current_type is not None:
            entities.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "type": current_type,
                    "entity": text[start : end + 1],
                }
            )
        start = None
        current_type = None

    for index, label in enumerate(labels):
        if label == "O" or "-" not in label:
            if start is not None:
                close(index - 1)
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or (prefix == "I" and entity_type != current_type):
            if start is not None:
                close(index - 1)
            start, current_type = index, entity_type
        elif prefix == "I" and start is None:
            start, current_type = index, entity_type
        else:
            if start is None:
                start, current_type = index, entity_type
    if start is not None:
        close(len(text) - 1)
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42, help="random seed for the LinearSVC")
    parser.add_argument("--C", type=float, default=0.1, help="LinearSVC regularization strength")
    parser.add_argument("--class-weight", type=str, default="balanced", help="LinearSVC class_weight ('none' or 'balanced')")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="file to predict/evaluate (default: data_raw/CMeEE_dev.json; use CMeEE_test.json for final predictions)",
    )
    parser.add_argument(
        "--dict-span-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add dictionary span features (DictSpanAnnotator) to each character",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = root / "data_cleaned" / "CMeEE_train_cleaned_final.json"
    eval_path = (args.eval_file or root / "data_raw" / "CMeEE_dev.json").resolve()
    train = load_json(train_path)
    eval_records = load_json(eval_path)
    dictionary_phrases = {entity["entity"] for record in train for entity in record.get("entities", [])}
    dictionary_chars = {char for phrase in dictionary_phrases for char in phrase}
    annotator = DictSpanAnnotator(train) if args.dict_span_features else None

    started = time.perf_counter()
    train_features: list[dict[str, str]] = []
    train_labels: list[str] = []
    for record in train:
        text = record["text"]
        spans = annotator.annotate(text) if annotator is not None else None
        train_features.extend(
            char_features(text, index, dictionary_chars, spans[index] if spans is not None else None)
            for index in range(len(text))
        )
        train_labels.extend(labels_from_entities(text, record.get("entities", [])))

    vectorizer = DictVectorizer()
    x_train = sparse_int32(vectorizer.fit_transform(train_features))
    class_weight = None if args.class_weight.lower() == "none" else args.class_weight
    classifier = LinearSVC(C=args.C, class_weight=class_weight, random_state=args.seed)
    classifier.fit(x_train, train_labels)

    predictions: list[dict[str, Any]] = []
    for record in eval_records:
        text = record["text"]
        spans = annotator.annotate(text) if annotator is not None else None
        features = [
            char_features(text, index, dictionary_chars, spans[index] if spans is not None else None)
            for index in range(len(text))
        ]
        labels = classifier.predict(sparse_int32(vectorizer.transform(features))).tolist()
        predictions.append({"text": text, "entities": decode_bio(text, labels)})

    eval_tag = "" if eval_path.name == "CMeEE_dev.json" else "_" + eval_path.stem.removeprefix("CMeEE_")
    stem = f"small_model{eval_tag}"
    prediction_path = output_dir / f"{stem}_predictions.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Method: character_linear_svm_longest_projection")
    print(f"LinearSVC: C={args.C}, class_weight={args.class_weight}, seed={args.seed}")
    print(f"Dictionary span features: {'on' if args.dict_span_features else 'off'}")
    print(f"Training characters: {len(train_labels)}")
    print(f"Feature count: {len(vectorizer.vocabulary_)}")
    if not any(record.get("entities") for record in eval_records):
        print(f"Eval file {eval_path.name} has no gold entities -> predict-only mode, no metrics computed.")
        print(f"Predictions: {prediction_path}")
        return

    full_metrics = evaluate_records(eval_records, predictions)
    metrics = {key: value for key, value in full_metrics.items() if key != "errors"}
    metrics.update(
        {
            "method": "character_linear_svm_longest_projection",
            "train_file": str(train_path),
            "eval_file": str(eval_path),
            "feature_count": len(vectorizer.vocabulary_),
            "train_characters": len(train_labels),
            "projection": "longest_non_overlapping_gold_entities",
            "seed": args.seed,
            "C": args.C,
            "class_weight": class_weight,
            "dict_span_features": args.dict_span_features,
            "dict_span_annotator": {
                "min_count": annotator.min_count,
                "min_length": annotator.min_length,
                "num_phrases": len(annotator.entries),
            }
            if annotator is not None
            else None,
            "runtime_seconds": time.perf_counter() - started,
        }
    )

    metrics_path = output_dir / f"{stem}_metrics.json"
    errors_path = output_dir / f"{stem}_errors.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_path.write_text(json.dumps(full_metrics["errors"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Entity precision: {metrics['precision']:.4f}")
    print(f"Entity recall:    {metrics['recall']:.4f}")
    print(f"Entity F1:        {metrics['f1']:.4f}")
    print(f"Predictions: {prediction_path}")
    print(f"Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
