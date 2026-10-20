"""A lightweight character-level linear NER baseline.

The CMeEE gold data contains overlapping/nested entities, while a single BIO
sequence can represent only one label at each character. This baseline uses a
documented projection: keep the longest non-overlapping gold entities for
training, then decode one BIO sequence at prediction time. The original span
data remains untouched and evaluation still uses strict spans.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from dict_span_features import DictSpanAnnotator
from metrics import evaluate_records


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def sparse_int32(matrix: Any) -> Any:
    """Make scipy sparse indices compatible with this scikit-learn build."""
    matrix = matrix.tocsr()
    matrix.indices = np.asarray(matrix.indices, dtype=np.int32)
    matrix.indptr = np.asarray(matrix.indptr, dtype=np.int32)
    return matrix


def longest_non_overlapping_entities(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Project nested gold spans to one deterministic, longest-only layer."""
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
    """Create small, explainable local features for one character.

    ``span_features`` optionally carries the dictionary span features for this
    position (see dict_span_features.DictSpanAnnotator).
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
    """Decode BIO labels and repair illegal I-* transitions conservatively."""
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
