"""Summarize the existing text review; never read or change raw training data."""

import csv
import hashlib
import json
import re
from collections import Counter
from itertools import combinations

from validate_entity_offsets import ROOT

SOURCE = ROOT / "docs" / "duplicate_annotation_review.txt"
OUTPUT = ROOT / "docs" / "duplicate_annotation_review.csv"
LOG = ROOT / "docs" / "duplicate_annotation_summary.log"
FIELDS = ["group_id", "record_indices", "annotations_consistent", "issue_category",
          "record_a_entities", "record_b_entities", "manual_review_status", "review_notes",
          "text", "entity_counts", "all_record_entities", "parse_status"]


def classify_pair(a, b):
    """Conservative structural hints; overlapping unmatched spans are only candidates."""
    left = list((Counter(map(tuple, a)) - Counter(map(tuple, b))).elements())
    right = list((Counter(map(tuple, b)) - Counter(map(tuple, a))).elements())
    flags = set()
    connected_a, connected_b = set(), set()
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            if max(x[0], y[0]) <= min(x[1], y[1]):
                connected_a.add(i)
                connected_b.add(j)
                if x[2] != y[2]:
                    flags.add("type_conflict")
                if x[:2] != y[:2]:
                    flags.add("boundary_conflict")
                if x[:3] == y[:3] and x[3] != y[3]:
                    flags.add("needs_manual_review")
    if len(connected_a) < len(left) or len(connected_b) < len(right):
        flags.add("possible_missing_entity")
    return flags


def parse_group(block):
    def field(pattern):
        match = re.search(pattern, block, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing field: {pattern}")
        return match.group(1)

    gid = int(field(r"^Group: (\d+)$"))
    indices = json.loads(field(r"^record_indices: (.+)$"))
    counts = json.loads(field(r"^entity_counts: (.+)$"))
    text = json.loads(field(r"^Text: (.+)$"))
    consistent = field(r"^annotations_consistent: (True|False)$") == "True"
    if (not isinstance(indices, list) or len(indices) < 2
            or any(type(i) is not int for i in indices) or len(set(indices)) != len(indices)):
        raise ValueError("Invalid record indices")
    if not isinstance(text, str) or set(counts) != {str(i) for i in indices}:
        raise ValueError("Invalid text/entity counts")
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("Invalid entity count")
    details = {}
    current = None
    for line in block.splitlines()[1:]:
        header = re.fullmatch(r"Record (\d+) sorted entity tuples:", line)
        if header:
            current = int(header.group(1))
            if current not in indices or current in details:
                raise ValueError("Unexpected/duplicate entity detail header")
            details[current] = []
        elif line.startswith("["):
            if current is None:
                raise ValueError("Entity tuple outside detail section")
            entity = json.loads(line)
            if (not isinstance(entity, list) or len(entity) != 4
                    or any(type(v) is not int for v in entity[:2])
                    or any(not isinstance(v, str) for v in entity[2:])):
                raise ValueError("Malformed entity tuple")
            details[current].append(entity)
        elif line.strip() == "(empty entity list)":
            if current is None or details[current]:
                raise ValueError("Unexpected empty entity marker")
        elif line.strip() and not re.match(
                r"^(Group:|Text:|record_indices:|entity_counts:|annotations_consistent:)", line):
            raise ValueError(f"Unrecognized report line: {line!r}")
    for i, entities in details.items():
        if len(entities) != counts[str(i)]:
            raise ValueError(f"Entity count mismatch: record {i}")
    if not consistent and set(details) != set(indices):
        raise ValueError("Inconsistent group lacks complete entity details")
    if not consistent and len({tuple(sorted(map(tuple, e))) for e in details.values()}) < 2:
        raise ValueError("Report says inconsistent but tuples agree")
    if consistent and details:
        raise ValueError("Unexpected details in consistent group; review parser assumptions")
    flags = set()
    if not consistent:
        for a, b in combinations(indices, 2):
            flags.update(classify_pair(details[a], details[b]))
    category = ("consistent_duplicate" if consistent else
                "needs_manual_review" if not flags else
                next(iter(flags)) if len(flags) == 1 else "multiple_conflicts")
    notes = ("Source report omits entity details for consistent groups; blank entity cells mean unavailable, not empty lists."
             if consistent else
             "Preliminary structural signals: " + ", ".join(sorted(flags)) +
             ". Overlapping unmatched spans are candidates, not confirmed correspondences; no annotation selected as correct.")
    return dict(zip(FIELDS, [gid, json.dumps(indices), str(consistent), category,
                            json.dumps(details[indices[0]], ensure_ascii=False) if indices[0] in details else "",
                            json.dumps(details[indices[1]], ensure_ascii=False) if indices[1] in details else "",
                            "pending", notes, text, json.dumps(counts),
                            json.dumps(details, ensure_ascii=False), "ok"]))


def main():
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    report = raw.decode("utf-8-sig").replace("\r\n", "\n")
    body = report.split("Summary:", 1)[0]
    blocks = re.findall(r"^Group: \d+\n.*?(?=^={70}$|\Z)", body, re.MULTILINE | re.DOTALL)
    rows, failures = [], []
    for block in blocks:
        try:
            rows.append(parse_group(block.rstrip()))
        except (ValueError, TypeError, KeyError) as exc:
            failures.append(f"{block.splitlines()[0]}: {exc}")
    header_count = len(re.findall(r"^Group:", body, re.MULTILINE))
    ids = [row["group_id"] for row in rows]
    inconsistent = sum(row["annotations_consistent"] == "False" for row in rows)
    checks = (len(rows) == 43 and inconsistent == 18 and header_count == 43
              and sorted(ids) == list(range(1, 44)))
    if len(blocks) != header_count:
        failures.append(f"Group headers: {header_count}; parsed blocks: {len(blocks)}")
    if not checks:
        failures.append("Expected 43 unique groups (IDs 1..43), 18 inconsistent; validation failed")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != digest:
        failures.append("Source report changed during processing")
    messages = ["Duplicate annotation summary run", f"Source: {SOURCE}",
                f"Source SHA256: {digest}", f"groups: {len(rows)}",
                f"consistent_groups: {len(rows) - inconsistent}", f"inconsistent_groups: {inconsistent}",
                "Issue categories: " + json.dumps(dict(sorted(Counter(r["issue_category"] for r in rows).items()))),
                f"parse_failures: {len(failures)}", *failures]
    if failures:
        messages.append("CSV not written; resolve logged failures first.")
        LOG.write_text("\n".join(messages) + "\n", encoding="utf-8")
        print("\n".join(messages))
        raise SystemExit(1)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with OUTPUT.open(encoding="utf-8-sig", newline="") as handle:
        saved = list(csv.DictReader(handle))
    if (len(saved) != 43 or sum(r["annotations_consistent"] == "False" for r in saved) != 18
            or any(r["manual_review_status"] != "pending" for r in saved)):
        raise RuntimeError("CSV read-back verification failed")
    messages.extend(["CSV read-back: PASS (43 rows, 18 inconsistent, all pending)",
                     "Source report SHA256 unchanged: True", f"Saved CSV: {OUTPUT}"])
    LOG.write_text("\n".join(messages) + "\n", encoding="utf-8")
    print("\n".join(messages))


if __name__ == "__main__":
    main()
