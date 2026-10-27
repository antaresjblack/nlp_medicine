"""Validate entity offsets without modifying source files."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "data_raw"
REPORT_FILE = ROOT / "docs" / "offset_validation_report.txt"
FILES = ["CMeEE_train.json", "CMeEE_dev.json", "CMeEE_test.json"]
METRICS = (
    "records", "invalid_record", "invalid_text", "invalid_entities_field",
    "empty_entities", "entities", "invalid_entity", "invalid_entity_text",
    "invalid_label", "invalid_offsets", "inclusive_out_of_bounds",
    "exclusive_out_of_bounds", "inclusive_match", "inclusive_mismatch",
    "exclusive_match", "exclusive_mismatch",
)


def validate_file(path):
    raw = path.read_bytes()
    records = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError(f"{path.name}: expected top-level JSON list")
    stats = Counter({key: 0 for key in METRICS})
    labels = Counter()
    examples = []
    anomaly_count = 0

    def issue(record_index, entity_index, reason, **details):
        nonlocal anomaly_count
        anomaly_count += 1
        if len(examples) < 20:
            examples.append(dict(record_index=record_index,
                                 entity_index=entity_index, reason=reason, **details))

    for record_index, record in enumerate(records):
        stats["records"] += 1
        if not isinstance(record, dict):
            stats["invalid_record"] += 1
            issue(record_index, None, "record is not a dictionary")
            continue
        text, entities = record.get("text"), record.get("entities")
        if not isinstance(text, str):
            stats["invalid_text"] += 1
            issue(record_index, None, "text is not a string")
        if not isinstance(entities, list):
            stats["invalid_entities_field"] += 1
            issue(record_index, None, "entities is not a list")
            continue
        if not entities:
            stats["empty_entities"] += 1
        for entity_index, entity in enumerate(entities):
            stats["entities"] += 1
            if not isinstance(entity, dict):
                stats["invalid_entity"] += 1
                issue(record_index, entity_index, "entity is not a dictionary")
                continue
            label = entity.get("type")
            if isinstance(label, str) and label:
                labels[label] += 1
            else:
                stats["invalid_label"] += 1
                issue(record_index, entity_index, "missing or invalid label")
            annotated = entity.get("entity")
            annotation_valid = isinstance(annotated, str) and bool(annotated)
            if not annotation_valid:
                stats["invalid_entity_text"] += 1
                issue(record_index, entity_index, "missing, empty or non-string entity text")
            start, end = entity.get("start_idx"), entity.get("end_idx")
            if type(start) is not int or type(end) is not int or start < 0 or end < start:
                stats["invalid_offsets"] += 1
                issue(record_index, entity_index, "invalid offsets", annotation=entity)
                continue
            if not isinstance(text, str):
                continue
            inclusive_valid = start < len(text) and end < len(text)
            exclusive_valid = start < end and end <= len(text)
            if not inclusive_valid:
                stats["inclusive_out_of_bounds"] += 1
                issue(record_index, entity_index, "inclusive offsets out of bounds",
                      text_length=len(text), annotation=entity, text=text)
            if not exclusive_valid:
                stats["exclusive_out_of_bounds"] += 1
            if annotation_valid and inclusive_valid:
                matches = text[start:end + 1] == annotated
                stats["inclusive_match" if matches else "inclusive_mismatch"] += 1
                if not matches:
                    issue(record_index, entity_index, "inclusive text mismatch",
                          annotation=entity, actual_slice=text[start:end + 1], text=text)
            if annotation_valid and exclusive_valid:
                matches = text[start:end] == annotated
                stats["exclusive_match" if matches else "exclusive_mismatch"] += 1

    digest = hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Source changed during validation: {path}")
    return stats, labels, examples, anomaly_count, digest


def main():
    lines = [
        "Entity offset validation (read-only)",
        "Index unit: Python Unicode characters; record/entity indices are zero-based.",
        "invalid_offsets: missing/non-integer (including bool), negative or reversed indices.",
        "inclusive_out_of_bounds: integer nonnegative ordered offsets outside text.",
        "exclusive_out_of_bounds: integer nonnegative ordered offsets failing 0 <= start < end <= len(text).",
        "Matches/mismatches count only entities with valid bounds and nonempty string annotations.",
        "Out-of-bounds entities are excluded from match/mismatch counts.",
        "Empty entity lists provide no evidence about index conventions or gold-label availability.",
        "Scope excludes duplicate, overlap/nesting and cross-split leakage checks.",
    ]
    total = Counter()
    for name in FILES:
        stats, labels, examples, anomaly_count, digest = validate_file(DATA_DIR / name)
        total.update(stats)
        lines.extend(["", "=" * 70, f"File: {name}", f"SHA256: {digest}"])
        lines.extend(f"{key}: {stats[key]}" for key in METRICS)
        lines.append("Entity label counts:")
        lines.extend(f"  {label}: {count}" for label, count in sorted(labels.items()))
        if not labels:
            lines.append("  (none)")
        lines.append(f"Anomaly events: {anomaly_count}; showing first {len(examples)}")
        lines.extend(json.dumps(example, ensure_ascii=False) for example in examples)
    lines.extend(["", "=" * 70, "Totals:"])
    lines.extend(f"{key}: {total[key]}" for key in METRICS)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved validation report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
