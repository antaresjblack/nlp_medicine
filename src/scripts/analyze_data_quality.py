"""Read-only second-stage quality checks. Python standard library only."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations

from src.scripts.validate_entity_offsets import DATA_DIR, ROOT, validate_file

LABELS = ("dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep")
DOCS = ROOT / "docs"


def inspect_records(records):
    stats = Counter({k: 0 for k in (
        "empty_entity_text", "unknown_labels", "overlap_pairs",
        "equal_span_pairs", "containment_pairs", "crossing_pairs",
        "records_with_overlap", "excluded_overlap_entities",
        "duplicate_text_groups", "duplicate_text_extra_records",
    )})
    examples = defaultdict(list)
    text_rows = defaultdict(list)

    def example(kind, value):
        if len(examples[kind]) < 10:
            examples[kind].append(value)

    for ri, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        text, entities = record.get("text"), record.get("entities")
        if isinstance(text, str):
            text_rows[text].append(ri)
        if not isinstance(entities, list):
            continue
        valid_spans = []
        for ei, entity in enumerate(entities):
            if not isinstance(entity, dict):
                stats["excluded_overlap_entities"] += 1
                continue
            ref = {"record_index": ri, "entity_index": ei, "annotation": entity}
            if entity.get("entity") == "":
                stats["empty_entity_text"] += 1
                example("empty_entity_text", ref)
            if entity.get("type") not in LABELS:
                stats["unknown_labels"] += 1
                example("unknown_labels", ref)
            start, end = entity.get("start_idx"), entity.get("end_idx")
            if (isinstance(text, str) and type(start) is int and type(end) is int
                    and 0 <= start <= end < len(text)):
                valid_spans.append((ei, start, end))
            else:
                stats["excluded_overlap_entities"] += 1
        overlapping = False
        for (a, sa, ea), (b, sb, eb) in combinations(valid_spans, 2):
            if max(sa, sb) > min(ea, eb):
                continue
            overlapping = True
            stats["overlap_pairs"] += 1
            if (sa, ea) == (sb, eb):
                kind = "equal_span_pairs"
            elif (sa <= sb and ea >= eb) or (sb <= sa and eb >= ea):
                kind = "containment_pairs"
            else:
                kind = "crossing_pairs"
            stats[kind] += 1
            example(kind, {"record_index": ri, "entity_indices": [a, b],
                           "entities": [entities[a], entities[b]], "text": text})
        stats["records_with_overlap"] += int(overlapping)
    for text, indices in text_rows.items():
        if len(indices) > 1:
            stats["duplicate_text_groups"] += 1
            stats["duplicate_text_extra_records"] += len(indices) - 1
            example("duplicate_text", {"record_indices": indices, "text": text})
    return stats, examples, text_rows


def main():
    paths = [DATA_DIR / f"CMeEE_{split}.json" for split in ("train", "dev", "test")]
    raw = {p.name: p.read_bytes() for p in paths}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    lines = [
        "CMeEE second-stage data quality report",
        "Indices: zero-based Python Unicode characters; end_idx is inclusive.",
        "Counts are observations, not automatic error judgments. Source data is never modified.",
        "Label denominator: ALL entity entries in the split, including anomalous entries.",
        "invalid_offsets and inclusive_out_of_bounds follow validate_entity_offsets.py.",
        "Text mismatch checks require valid inclusive bounds and nonempty string entity text.",
        "Overlaps: each unordered entity pair counted once within a record; inclusive spans intersect.",
        "Only valid in-bounds spans enter overlap checks; labels/text matches do not filter spans.",
        "Equal spans, proper containment and crossing overlaps are mutually exclusive categories.",
        "Overlap is a data feature, not by itself an annotation error.",
        "Duplicates use exact text equality without trimming or normalization; extra records = sum(n-1).",
        "Examples: first 10 per category; offset anomaly events: first 20 per file.",
    ]
    distribution = []
    text_maps = {}
    for split in ("train", "dev"):
        name = f"CMeEE_{split}.json"
        # Reuse the previously executed index-validation implementation.
        base, labels, offset_examples, events, digest = validate_file(DATA_DIR / name)
        if digest != hashes[name]:
            raise RuntimeError(f"Source changed while reading {name}")
        records = json.loads(raw[name].decode("utf-8-sig"))
        extra, examples, text_maps[split] = inspect_records(records)
        lines.extend(["", "=" * 70, f"File: {name}", f"SHA256: {digest}"])
        lines.extend(f"{key}: {value}" for key, value in base.items())
        lines.extend(f"{key}: {value}" for key, value in extra.items())
        lines.append("Label counts and percentages (all entities denominator):")
        for label in LABELS:
            count = labels[label]
            ratio = count / base["entities"] if base["entities"] else 0
            lines.append(f"  {label}: {count} ({ratio:.6%})")
            distribution.append([split, label, count, base["entities"], f"{ratio:.10f}", f"{ratio * 100:.6f}"])
        lines.append(f"Offset anomaly events: {events} (one entity may trigger multiple events)")
        lines.extend(json.dumps(e, ensure_ascii=False) for e in offset_examples)
        for kind, values in sorted(examples.items()):
            lines.append(f"Examples: {kind}")
            lines.extend(json.dumps(e, ensure_ascii=False) for e in values)

    shared = sorted(set(text_maps["train"]) & set(text_maps["dev"]))
    pair_count = sum(len(text_maps["train"][t]) * len(text_maps["dev"][t]) for t in shared)
    lines.extend(["", "Train/dev exact-text intersection:",
                  f"shared_unique_texts: {len(shared)}", f"cross_split_record_pairs: {pair_count}"])
    for text in shared[:10]:
        lines.append(json.dumps({"text": text, "train_indices": text_maps["train"][text],
                                 "dev_indices": text_maps["dev"][text]}, ensure_ascii=False))

    test = json.loads(raw["CMeEE_test.json"].decode("utf-8-sig"))
    if not isinstance(test, list):
        raise ValueError("Test must be a top-level JSON list")
    structures = Counter()
    field_types = defaultdict(Counter)
    empty = 0
    for row in test:
        if isinstance(row, dict):
            structures[tuple(sorted(row))] += 1
            for key, value in row.items():
                field_types[key][type(value).__name__] += 1
            empty += int(isinstance(row.get("entities"), list) and not row["entities"])
        else:
            structures[(f"NON_DICT:{type(row).__name__}",)] += 1
    lines.extend(["", "=" * 70, "File: CMeEE_test.json", f"SHA256: {hashes['CMeEE_test.json']}",
                  "top_level: list", f"records: {len(test)}", f"empty_entities: {empty}",
                  "Field structures: " + json.dumps([{"keys": k, "records": v} for k, v in structures.items()]),
                  "Field value types: " + json.dumps(dict(field_types)),
                  "Test empty lists are NOT gold negative labels. No test entity evaluation or label statistics performed."])
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.name]:
            raise RuntimeError(f"Source changed during analysis: {p.name}")
    lines.extend(["", "Source SHA256 unchanged before/after this run: all three files.",
                  "Pending: human review of anomalies, source/version/license confirmation, model format agreement.",
                  "Exact text comparison does not detect near-duplicates. No cleaning or BIO conversion performed."])
    DOCS.mkdir(exist_ok=True)
    report = DOCS / "data_quality_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = DOCS / "entity_label_distribution.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "count", "all_entities", "proportion", "percentage"])
        writer.writerows(distribution)
    print(f"Saved report: {report}")
    print(f"Saved label distribution: {csv_path}")


if __name__ == "__main__":
    main()
