"""Review exact-text duplicate annotations without modifying training data."""

import hashlib
import json
from collections import defaultdict

from src.scripts.validate_entity_offsets import DATA_DIR, ROOT

SOURCE = DATA_DIR / "CMeEE_train.json"
REPORT = ROOT / "docs" / "duplicate_annotation_review.txt"
ENTITY_FIELDS = ("start_idx", "end_idx", "type", "entity")


def annotation_signature(entities):
    """Sort entity tuples while retaining multiplicity; reject malformed input."""
    if not isinstance(entities, list):
        raise ValueError("entities must be a list")
    tuples = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict) or any(key not in entity for key in ENTITY_FIELDS):
            raise ValueError(f"entity {index}: expected dictionary with {ENTITY_FIELDS}")
        start, end, label, text = (entity[key] for key in ENTITY_FIELDS)
        if (type(start) is not int or type(end) is not int
                or not isinstance(label, str) or not isinstance(text, str)):
            raise ValueError(f"entity {index}: unexpected field types")
        # No offset validation or annotation correction here.
        tuples.append((start, end, label, text))
    return tuple(sorted(tuples))


def main():
    raw = SOURCE.read_bytes()
    before = hashlib.sha256(raw).hexdigest()
    records = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("Expected top-level JSON list")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"Record {index}: expected dictionary with string text")
        groups[record["text"]].append(index)

    duplicates = [(text, indices) for text, indices in groups.items() if len(indices) > 1]
    consistent = 0
    lines = [
        "CMeEE training exact-text duplicate annotation review",
        f"Source: {SOURCE}",
        f"records: {len(records)}",
        "Record indices are zero-based. Grouping uses exact text, without normalization.",
        "Comparison: sorted (start_idx, end_idx, type, entity) tuples; duplicate entities retained.",
        "Only entity ordering is ignored. Offsets and annotations are not changed.",
        "Inconsistent annotations require human review; no automatic error judgment.",
    ]
    for group_index, (text, indices) in enumerate(duplicates, start=1):
        signatures = []
        for index in indices:
            try:
                signatures.append(annotation_signature(records[index].get("entities")))
            except ValueError as exc:
                raise ValueError(f"Record {index}: {exc}") from exc
        same = all(signature == signatures[0] for signature in signatures[1:])
        consistent += int(same)
        lines.extend([
            "", "=" * 70, f"Group: {group_index}",
            f"Text: {json.dumps(text, ensure_ascii=False)}",
            f"record_indices: {indices}",
            "entity_counts: " + json.dumps({str(i): len(records[i]["entities"]) for i in indices}),
            f"annotations_consistent: {same}",
        ])
        if not same:
            for index, signature in zip(indices, signatures):
                lines.append(f"Record {index} sorted entity tuples:")
                lines.extend(json.dumps(entity, ensure_ascii=False) for entity in signature)
                if not signature:
                    lines.append("  (empty entity list)")
    after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Source SHA-256 changed during review; report not written")
    summary = [f"duplicate_text_groups: {len(duplicates)}",
               f"consistent_groups: {consistent}",
               f"inconsistent_groups: {len(duplicates) - consistent}"]
    lines.extend(["", "=" * 70, "Summary:", *summary,
                  f"SHA256 before: {before}", f"SHA256 after:  {after}",
                  "Source SHA256 unchanged: True",
                  "No cleaning, correction or BIO conversion performed."])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"Source SHA256: {after}")
    print(f"Saved review: {REPORT}")


if __name__ == "__main__":
    main()
