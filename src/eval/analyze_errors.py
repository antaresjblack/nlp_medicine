"""Classify strict entity errors into missed, extra, boundary and type cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CATEGORY_ORDER = ("missed", "false_positive", "boundary_error", "type_error")
CATEGORY_LABELS = {
    "missed": "完全漏检",
    "false_positive": "完全误报",
    "boundary_error": "边界错误",
    "type_error": "类型错误",
}
CATEGORY_DESCRIPTIONS = {
    "missed": "金标实体没有任何同标签的预测跨度与之重叠",
    "false_positive": "预测实体没有对应的金标类型或跨度",
    "boundary_error": "标签正确，但预测与金标的起止边界不同",
    "type_error": "起止跨度一致，但实体类型不同",
}
Entity = tuple[int, int, str]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    return load_json(path)


def entity_key(entity: dict[str, Any]) -> Entity:
    return int(entity["start_idx"]), int(entity["end_idx"]), str(entity["type"])


def to_entity(key: Entity, text: str) -> dict[str, Any]:
    start, end, label = key
    return {
        "start_idx": start,
        "end_idx": end,
        "type": label,
        "entity": text[start : end + 1],
    }


def overlaps(left: Entity, right: Entity) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def overlap_length(left: Entity, right: Entity) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)


def classify_record(
    gold: dict[str, Any], predicted: dict[str, Any], record_index: int
) -> list[dict[str, Any]]:
    text = gold["text"]
    gold_keys = {entity_key(entity) for entity in gold.get("entities", [])}
    predicted_keys = {entity_key(entity) for entity in predicted.get("entities", [])}
    correct = gold_keys & predicted_keys
    gold_errors = set(gold_keys - correct)
    predicted_errors = set(predicted_keys - correct)
    cases: list[dict[str, Any]] = []

    def add(category: str, gold_key: Entity | None, predicted_key: Entity | None) -> None:
        cases.append(
            {
                "category": category,
                "record_index": record_index,
                "id": gold.get("id", f"index-{record_index}"),
                "text": text,
                "gold": to_entity(gold_key, text) if gold_key else None,
                "predicted": to_entity(predicted_key, text) if predicted_key else None,
            }
        )

    # Same span with a different label is a type error.
    gold_by_span: dict[tuple[int, int], list[Entity]] = defaultdict(list)
    predicted_by_span: dict[tuple[int, int], list[Entity]] = defaultdict(list)
    for entity in gold_errors:
        gold_by_span[entity[:2]].append(entity)
    for entity in predicted_errors:
        predicted_by_span[entity[:2]].append(entity)
    for span in sorted(set(gold_by_span) & set(predicted_by_span)):
        for gold_entity, predicted_entity in zip(
            sorted(gold_by_span[span]), sorted(predicted_by_span[span])
        ):
            add("type_error", gold_entity, predicted_entity)
            gold_errors.remove(gold_entity)
            predicted_errors.remove(predicted_entity)

    # Same label and overlapping, but different boundaries.
    for gold_entity in sorted(gold_errors):
        candidates = [
            predicted_entity
            for predicted_entity in predicted_errors
            if predicted_entity[2] == gold_entity[2]
            and overlaps(gold_entity, predicted_entity)
        ]
        if not candidates:
            continue
        predicted_entity = max(
            candidates,
            key=lambda entity: (
                overlap_length(gold_entity, entity),
                -abs(gold_entity[0] - entity[0]),
                -abs(gold_entity[1] - entity[1]),
                -entity[0],
                -entity[1],
            ),
        )
        add("boundary_error", gold_entity, predicted_entity)
        gold_errors.remove(gold_entity)
        predicted_errors.remove(predicted_entity)

    for entity in sorted(gold_errors):
        add("missed", entity, None)
    for entity in sorted(predicted_errors):
        add("false_positive", None, entity)
    return cases


def analyze(errors: list[dict[str, Any]], gold_records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    counts = Counter({category: 0 for category in CATEGORY_ORDER})
    records_by_category: dict[str, set[int]] = {
        category: set() for category in CATEGORY_ORDER
    }
    examples: dict[str, list[dict[str, Any]]] = {
        category: [] for category in CATEGORY_ORDER
    }
    example_records: dict[str, set[int]] = {
        category: set() for category in CATEGORY_ORDER
    }
    missed_by_label: Counter[str] = Counter()
    false_positive_by_label: Counter[str] = Counter()
    boundary_by_label: Counter[str] = Counter()
    type_confusions: Counter[str] = Counter()

    for error in errors:
        record_index = int(error["record_index"])
        if not 0 <= record_index < len(gold_records):
            raise ValueError(f"Invalid error record index: {record_index}")
        gold = gold_records[record_index]
        predicted = {
            "id": gold.get("id", f"index-{record_index}"),
            "text": gold["text"],
            "entities": error.get("predicted_entities", []),
        }
        for case in classify_record(gold, predicted, record_index):
            category = case["category"]
            counts[category] += 1
            records_by_category[category].add(record_index)
            if category == "missed":
                missed_by_label[case["gold"]["type"]] += 1
            elif category == "false_positive":
                false_positive_by_label[case["predicted"]["type"]] += 1
            elif category == "boundary_error":
                boundary_by_label[case["gold"]["type"]] += 1
            else:
                type_confusions[
                    f"{case['gold']['type']}->{case['predicted']['type']}"
                ] += 1
            if (
                len(examples[category]) < 3
                and record_index not in example_records[category]
            ):
                examples[category].append(case)
                example_records[category].add(record_index)

    return {
        "schema_version": 1,
        "method": method,
        "error_records": len(errors),
        "categories": {
            category: {
                "label": CATEGORY_LABELS[category],
                "description": CATEGORY_DESCRIPTIONS[category],
                "count": counts[category],
                "affected_records": len(records_by_category[category]),
            }
            for category in CATEGORY_ORDER
        },
        "total_classified_errors": sum(counts.values()),
        "missed_by_label": dict(sorted(missed_by_label.items())),
        "false_positive_by_label": dict(sorted(false_positive_by_label.items())),
        "boundary_by_label": dict(sorted(boundary_by_label.items())),
        "type_confusions": dict(sorted(type_confusions.items())),
        "examples": examples,
    }


def entity_text(entity: dict[str, Any] | None) -> str:
    if entity is None:
        return "无"
    return f"{entity['entity']} / {entity['type']} [{entity['start_idx']},{entity['end_idx']}]"


def render_markdown(report: dict[str, Any], errors_path: Path) -> str:
    lines = [
        f"# 错误分析：{report['method']}",
        "",
        f"- 原始错误文件：`{errors_path.as_posix()}`",
        f"- 含错误的记录：{report['error_records']}",
        "",
        "| 类别 | 数量 | 涉及记录 | 定义 |",
        "|---|---:|---:|---|",
    ]
    for category in CATEGORY_ORDER:
        item = report["categories"][category]
        lines.append(
            f"| {item['label']} | {item['count']} | {item['affected_records']} | {item['description']} |"
        )

    if report["type_confusions"]:
        lines += ["", "## 类型混淆", "", "| 金标 → 预测 | 数量 |", "|---|---:|"]
        for confusion, count in report["type_confusions"].items():
            lines.append(f"| {confusion} | {count} |")

    for category in CATEGORY_ORDER:
        lines += ["", f"## {CATEGORY_LABELS[category]}样例"]
        category_examples = report["examples"][category]
        if not category_examples:
            lines.append("无")
            continue
        for number, example in enumerate(category_examples, 1):
            text = " ".join(example["text"].split())
            if len(text) > 120:
                text = text[:117] + "..."
            lines += [
                f"### {number}. {example['id']}",
                "",
                f"- 原文：{text}",
                f"- 金标：{entity_text(example['gold'])}",
                f"- 预测：{entity_text(example['predicted'])}",
            ]
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    errors = load_json(args.errors)
    gold_records = load_records(args.gold)
    report = analyze(errors, gold_records, args.method)
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report, args.errors), encoding="utf-8")
    print(f"Wrote error analysis: {args.output_json}")
    print(f"Wrote error report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
