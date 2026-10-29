"""Paired bootstrap for dictionary-fusion gain on the frozen eval_test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def entity_set(entities: list[dict[str, Any]]) -> set[tuple[int, int, str]]:
    return {
        (int(entity["start_idx"]), int(entity["end_idx"]), str(entity["type"]))
        for entity in entities
    }


def prediction_map(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {str(record["id"]): record.get("entities", []) for record in records}


def counts(gold: set[tuple[int, int, str]], predicted: set[tuple[int, int, str]]) -> tuple[int, int, int]:
    true_positive = len(gold & predicted)
    return true_positive, len(predicted) - true_positive, len(gold) - true_positive


def f1_scores(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def metric_summary(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def run_bootstrap(
    gold: list[dict[str, Any]],
    dict_predictions: dict[str, list[dict[str, Any]]],
    warmup_predictions: dict[str, list[dict[str, Any]]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be positive")
    record_count = len(gold)
    if len(dict_predictions) != record_count or len(warmup_predictions) != record_count:
        raise ValueError(
            "record count mismatch: "
            f"gold={record_count}, dict={len(dict_predictions)}, warmup={len(warmup_predictions)}"
        )

    arrays = {
        "dict_tp": np.zeros(record_count, dtype=np.int64),
        "dict_fp": np.zeros(record_count, dtype=np.int64),
        "dict_fn": np.zeros(record_count, dtype=np.int64),
        "warmup_tp": np.zeros(record_count, dtype=np.int64),
        "warmup_fp": np.zeros(record_count, dtype=np.int64),
        "warmup_fn": np.zeros(record_count, dtype=np.int64),
    }
    for index, record in enumerate(gold):
        record_id = str(record["id"])
        if record_id not in dict_predictions or record_id not in warmup_predictions:
            raise ValueError(f"missing prediction for record {record_id}")
        gold_set = entity_set(record.get("entities", []))
        dict_tp, dict_fp, dict_fn = counts(gold_set, entity_set(dict_predictions[record_id]))
        warm_tp, warm_fp, warm_fn = counts(gold_set, entity_set(warmup_predictions[record_id]))
        arrays["dict_tp"][index] = dict_tp
        arrays["dict_fp"][index] = dict_fp
        arrays["dict_fn"][index] = dict_fn
        arrays["warmup_tp"][index] = warm_tp
        arrays["warmup_fp"][index] = warm_fp
        arrays["warmup_fn"][index] = warm_fn

    dict_metrics = metric_summary(
        int(arrays["dict_tp"].sum()),
        int(arrays["dict_fp"].sum()),
        int(arrays["dict_fn"].sum()),
    )
    warmup_metrics = metric_summary(
        int(arrays["warmup_tp"].sum()),
        int(arrays["warmup_fp"].sum()),
        int(arrays["warmup_fn"].sum()),
    )

    rng = np.random.default_rng(seed)
    differences: list[np.ndarray] = []
    chunk_size = 500
    for start in range(0, samples, chunk_size):
        batch = min(chunk_size, samples - start)
        indices = rng.integers(0, record_count, size=(batch, record_count))
        dict_f1 = f1_scores(
            arrays["dict_tp"][indices].sum(axis=1),
            arrays["dict_fp"][indices].sum(axis=1),
            arrays["dict_fn"][indices].sum(axis=1),
        )
        warmup_f1 = f1_scores(
            arrays["warmup_tp"][indices].sum(axis=1),
            arrays["warmup_fp"][indices].sum(axis=1),
            arrays["warmup_fn"][indices].sum(axis=1),
        )
        differences.append(dict_f1 - warmup_f1)
    difference_samples = np.concatenate(differences)
    ci_low, ci_high = np.percentile(difference_samples, [2.5, 97.5])
    p_value = 2 * min(
        float((difference_samples <= 0).mean()),
        float((difference_samples >= 0).mean()),
    )

    return {
        "schema_version": 1,
        "split": "eval_test",
        "records": record_count,
        "bootstrap_samples": samples,
        "seed": seed,
        "dict": dict_metrics,
        "warmup_no_dictionary": warmup_metrics,
        "observed_f1_difference": float(dict_metrics["f1"] - warmup_metrics["f1"]),
        "ci95": [float(ci_low), float(ci_high)],
        "p_value_two_sided": p_value,
        "significant_at_0_05": bool(ci_low > 0 or ci_high < 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--dict-predictions", type=Path, required=True)
    parser.add_argument("--warmup-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_bootstrap(
        load_jsonl(args.gold),
        prediction_map(load_jsonl(args.dict_predictions)),
        prediction_map(load_jsonl(args.warmup_predictions)),
        args.bootstrap_samples,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"F1 difference={report['observed_f1_difference']:+.10f}; "
        f"95% CI=[{report['ci95'][0]:+.10f}, {report['ci95'][1]:+.10f}]; "
        f"p={report['p_value_two_sided']:.4f}"
    )
    print(f"Wrote bootstrap result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
