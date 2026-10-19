"""Check and summarize the CMeEE files used by the baseline experiments.

This script is intentionally read-only: it never edits the dataset files.
Run it from the repository root with:

    python baseline/data_check.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_LABELS = {"dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep"}


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file and return its top-level value."""
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def check_file(path: Path, expect_gold: bool) -> dict[str, Any]:
    """Validate one CMeEE JSON file and return a compact summary."""
    data = load_json(path)
    errors: list[str] = []
    label_counts: Counter[str] = Counter()
    overlap_pair_count = 0
    records_with_overlap = 0
    entity_count = 0
    empty_entity_records = 0

    if not isinstance(data, list):
        return {"path": str(path), "records": 0, "entities": 0, "errors": ["top-level JSON value is not a list"]}

    for record_index, record in enumerate(data):
        if not isinstance(record, dict):
            errors.append(f"record {record_index}: not an object")
            continue

        text = record.get("text")
        entities = record.get("entities")
        if not isinstance(text, str):
            errors.append(f"record {record_index}: text is not a string")
            continue
        if not isinstance(entities, list):
            errors.append(f"record {record_index}: entities is not a list")
            continue
        if not entities:
            empty_entity_records += 1

        spans: list[tuple[int, int]] = []
        for entity_index, entity in enumerate(entities):
            entity_count += 1
            if not isinstance(entity, dict):
                errors.append(f"record {record_index}, entity {entity_index}: not an object")
                continue

            start = entity.get("start_idx")
            end = entity.get("end_idx")
            label = entity.get("type")
            entity_text = entity.get("entity")

            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"record {record_index}, entity {entity_index}: offsets are not integers")
                continue
            if label not in ALLOWED_LABELS:
                errors.append(f"record {record_index}, entity {entity_index}: unknown label {label!r}")
            else:
                label_counts[label] += 1
            if not isinstance(entity_text, str):
                errors.append(f"record {record_index}, entity {entity_index}: entity is not a string")
                continue
            if start < 0 or end < start or end >= len(text):
                errors.append(
                    f"record {record_index}, entity {entity_index}: offset out of range "
                    f"({start}, {end}) for text length {len(text)}"
                )
                continue
            if text[start : end + 1] != entity_text:
                errors.append(f"record {record_index}, entity {entity_index}: text slice mismatch")
            spans.append((start, end))

        record_overlap_pairs = 0
        for left_index, (left_start, left_end) in enumerate(spans):
            for right_start, right_end in spans[left_index + 1 :]:
                if left_start <= right_end and right_start <= left_end:
                    record_overlap_pairs += 1
        if record_overlap_pairs:
            records_with_overlap += 1
            overlap_pair_count += record_overlap_pairs

    if expect_gold and empty_entity_records:
        errors.append(
            f"file contains {empty_entity_records} records with empty entities; "
            "check whether this is an unlabeled test file"
        )

    return {
        "path": str(path),
        "records": len(data),
        "entities": entity_count,
        "empty_entity_records": empty_entity_records,
        "label_counts": dict(sorted(label_counts.items())),
        "records_with_overlap": records_with_overlap,
        "overlap_pairs": overlap_pair_count,
        "errors": errors,
    }


def render_report(summaries: list[dict[str, Any]]) -> str:
    """Render a human-readable report from file summaries."""
    lines = ["CMeEE baseline data check", "=" * 28, ""]
    for summary in summaries:
        lines.extend(
            [
                f"File: {summary['path']}",
                f"Records: {summary.get('records', 0)}",
                f"Entities: {summary.get('entities', 0)}",
                f"Empty-entity records: {summary.get('empty_entity_records', 0)}",
                f"Records with overlap: {summary.get('records_with_overlap', 0)}",
                f"Overlap pairs: {summary.get('overlap_pairs', 0)}",
                f"Label counts: {summary.get('label_counts', {})}",
                f"Errors: {len(summary.get('errors', []))}",
            ]
        )
        for error in summary.get("errors", [])[:20]:
            lines.append(f"  - {error}")
        if len(summary.get("errors", [])) > 20:
            lines.append(f"  - ... {len(summary['errors']) - 20} more errors")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of baseline/)",
    )
    parser.add_argument("--report", type=Path, default=None, help="optional path for the text report")
    args = parser.parse_args()

    root = args.root.resolve()
    files = [
        (root / "data_cleaned" / "CMeEE_train_cleaned_final.json", True),
        (root / "data_raw" / "CMeEE_dev.json", True),
        (root / "data_raw" / "CMeEE_test.json", False),
    ]

    summaries = []
    for path, expect_gold in files:
        if not path.exists():
            summaries.append({"path": str(path), "records": 0, "errors": ["file not found"]})
            continue
        summaries.append(check_file(path, expect_gold=expect_gold))

    report = render_report(summaries)
    print(report)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.report.resolve()}")

    has_errors = any(summary.get("errors") for summary in summaries)
    raise SystemExit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
