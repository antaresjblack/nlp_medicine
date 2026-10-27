"""Inspect the first three training records without changing source data."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "data_raw" / "CMeEE_train.json"
OUTPUT_FILE = ROOT / "docs" / "offset_samples.txt"


def main():
    records = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of training records")

    lines = [
        f"Source: {INPUT_FILE}",
        "Scope: first 3 training records only; no dataset-wide conclusion.",
        "Indices below use Python Unicode character positions, starting at zero.",
    ]
    for record_index, record in enumerate(records[:3]):
        lines.extend(["=" * 70, f"Record index: {record_index}"])
        if not isinstance(record, dict):
            lines.append("WARNING: record is not a dictionary")
            continue
        text = record.get("text")
        entities = record.get("entities")
        lines.append(f"Text: {text!r}")
        if not isinstance(text, str) or not isinstance(entities, list):
            lines.append("WARNING: text must be a string and entities a list")
            continue
        lines.extend([f"Entity count: {len(entities)}", ""])
        spans = []
        for entity_index, entity in enumerate(entities):
            lines.append(f"Entity #{entity_index}: {entity!r}")
            if not isinstance(entity, dict):
                lines.append("  WARNING: entity is not a dictionary")
                continue
            start_key = next((k for k in ("start_idx", "start_offset", "start") if k in entity), None)
            end_key = next((k for k in ("end_idx", "end_offset", "end") if k in entity), None)
            if start_key is None or end_key is None:
                lines.append("  WARNING: start/end fields not identified")
                continue
            start, end = entity[start_key], entity[end_key]
            entity_text = entity.get("entity")
            lines.extend([
                f"  start={start} ({start_key}), end={end} ({end_key})",
                f"  annotated entity text: {entity_text!r}",
            ])
            if not isinstance(entity_text, str) or not entity_text:
                lines.append("  WARNING: missing, empty or non-string entity text")
            if type(start) is int and type(end) is int:
                inclusive_valid = 0 <= start <= end < len(text)
                exclusive_valid = 0 <= start < end <= len(text)
                inclusive = text[start:end + 1]
                exclusive = text[start:end]
                lines.extend([
                    f"  inclusive-end bounds valid: {inclusive_valid}",
                    f"  exclusive-end bounds valid: {exclusive_valid}",
                    f"  inclusive-end slice: {inclusive!r}",
                    f"  exclusive-end slice: {exclusive!r}",
                    f"  inclusive matches annotation: {inclusive_valid and inclusive == entity_text}",
                    f"  exclusive matches annotation: {exclusive_valid and exclusive == entity_text}",
                ])
                spans.append((entity_index, start, end))
            else:
                lines.append("  WARNING: start/end must be integers (not booleans)")
            lines.append("")
        for convention, adjustment in [("inclusive-end", 1), ("exclusive-end", 0)]:
            pairs = []
            for i, (a, start_a, end_a) in enumerate(spans):
                for b, start_b, end_b in spans[i + 1:]:
                    if (0 <= start_a < end_a + adjustment <= len(text)
                            and 0 <= start_b < end_b + adjustment <= len(text)
                            and max(start_a, start_b) < min(end_a, end_b) + adjustment):
                        nested = ((start_a <= start_b and end_a >= end_b)
                                  or (start_b <= start_a and end_b >= end_a))
                        pairs.append((a, b, "containment/equal span" if nested else "crossing"))
            lines.append(f"Overlapping entity pairs assuming {convention}: {pairs!r}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved inspection report: {OUTPUT_FILE}")
    print("Please inspect the report before making any data changes.")


if __name__ == "__main__":
    main()
