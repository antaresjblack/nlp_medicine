"""Dictionary span features for the character-level baseline.

The character-level LinearSVC only sees a four-character local window, so
entity-boundary decisions are made almost blindly. This module encodes the
dictionary's span-level knowledge as per-character features: for every
position, find the dictionary matches covering it and report the longest
covering match's label, boundary role, length and frequency buckets, plus a
label-conflict flag when several matches with different types overlap.

The dictionary is built from training entities only, with the same
min_count/min_length/majority-label convention as the final dictionary
baseline, so the two methods stay comparable. Matching uses an inverted index
on the first character: a text only checks phrases that could start at one of
its characters, instead of scanning the whole dictionary per text.

Feature values are strings so DictVectorizer can consume them directly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _bucket_len(length: int) -> str:
    if length <= 2:
        return "2"
    if length <= 4:
        return "3-4"
    if length <= 8:
        return "5-8"
    return "9+"


def _bucket_freq(freq: int) -> str:
    if freq <= 1:
        return "1"
    if freq <= 5:
        return "2-5"
    if freq <= 20:
        return "6-20"
    return "21+"


class DictSpanAnnotator:
    """Annotate each character of a text with dictionary span features.

    For every position the features describe the longest dictionary match
    covering it (ties prefer higher frequency, then the earlier start):

    - ``in_dict_span``: 0/1, whether the position is covered by a match.
    - ``dict_boundary``: S/B/I/E inside that match, O when not covered.
    - ``dict_label``: the match's entity type, or ``none``.
    - ``dict_span_len``: length bucket of the match (2 / 3-4 / 5-8 / 9+).
    - ``dict_freq``: training frequency bucket (1 / 2-5 / 6-20 / 21+).
    - ``label_conflict``: 0/1, whether several matches with different types
      cover the position.
    """

    def __init__(self, records: list[dict[str, Any]], min_count: int = 1, min_length: int = 2) -> None:
        self.min_count = min_count
        self.min_length = min_length
        label_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            for entity in record.get("entities", []):
                label_counts[entity["entity"]][entity["type"]] += 1

        self.entries: list[tuple[str, str, int]] = []  # (phrase, majority label, frequency)
        for phrase, counts in label_counts.items():
            total = sum(counts.values())
            if total < min_count or len(phrase) < min_length:
                continue
            label = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            self.entries.append((phrase, label, total))

        self.by_first_char: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for phrase, label, freq in self.entries:
            self.by_first_char[phrase[0]].append((phrase, label, freq))

    def annotate(self, text: str) -> list[dict[str, str]]:
        """Return one feature dict per character position of ``text``."""
        matches = self._find_matches(text)
        return self._per_position_features(text, matches)

    def _find_matches(self, text: str) -> list[tuple[int, int, str, int]]:
        """Return (start, inclusive end, label, frequency) for every match."""
        matches: list[tuple[int, int, str, int]] = []
        for start, char in enumerate(text):
            bucket = self.by_first_char.get(char)
            if not bucket:
                continue
            remaining = text[start:]
            for phrase, label, freq in bucket:
                if remaining.startswith(phrase):
                    matches.append((start, start + len(phrase) - 1, label, freq))
        return matches

    def _per_position_features(self, text: str, matches: list[tuple[int, int, str, int]]) -> list[dict[str, str]]:
        positions = len(text)
        best: list[tuple[int, int, int, str] | None] = [None] * positions  # (length, freq, start, label)
        ends: list[int] = [-1] * positions
        conflicts: list[set[str]] = [set() for _ in range(positions)]
        for start, end, label, freq in matches:
            length = end - start + 1
            for position in range(start, end + 1):
                current = best[position]
                if current is None or (length, freq, -start) > (current[0], current[1], -current[2]):
                    best[position] = (length, freq, start, label)
                    ends[position] = end
                conflicts[position].add(label)

        features: list[dict[str, str]] = []
        for position in range(positions):
            match = best[position]
            if match is None:
                features.append(
                    {
                        "in_dict_span": "0",
                        "dict_boundary": "O",
                        "dict_label": "none",
                        "dict_span_len": "0",
                        "dict_freq": "0",
                        "label_conflict": "0",
                    }
                )
                continue
            length, freq, start, label = match
            if length == 1:
                boundary = "S"
            elif position == start:
                boundary = "B"
            elif position == ends[position]:
                boundary = "E"
            else:
                boundary = "I"
            features.append(
                {
                    "in_dict_span": "1",
                    "dict_boundary": boundary,
                    "dict_label": label,
                    "dict_span_len": _bucket_len(length),
                    "dict_freq": _bucket_freq(freq),
                    "label_conflict": "1" if len(conflicts[position]) >= 2 else "0",
                }
            )
        return features


def main() -> None:
    """Build the annotator from the final train set and demo it on one dev record."""
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--demo-record", type=int, default=0, help="dev record index for the feature demo")
    args = parser.parse_args()

    root = args.root.resolve()
    train_path = root / "data_cleaned" / "CMeEE_train_cleaned_final.json"
    dev_path = root / "data_raw" / "CMeEE_dev.json"
    train = json.loads(train_path.read_text(encoding="utf-8-sig"))
    dev = json.loads(dev_path.read_text(encoding="utf-8-sig"))

    started = time.perf_counter()
    annotator = DictSpanAnnotator(train)
    print(f"Dictionary phrases: {len(annotator.entries)} (min_count={annotator.min_count}, min_length={annotator.min_length})")
    print(f"Annotator built in {time.perf_counter() - started:.1f}s")

    record = dev[args.demo_record]
    text = record["text"]
    started = time.perf_counter()
    features = annotator.annotate(text)
    print(f"\nDemo record {args.demo_record}: {text}")
    print("Gold entities:", [entity["entity"] + "/" + entity["type"] for entity in record["entities"]])
    for position, char in enumerate(text):
        span = features[position]
        if span["in_dict_span"] == "1":
            print(
                f"  {position:>3} {char} | boundary={span['dict_boundary']} label={span['dict_label']} "
                f"len={span['dict_span_len']} freq={span['dict_freq']} conflict={span['label_conflict']}"
            )

    started = time.perf_counter()
    for dev_record in dev:
        annotator.annotate(dev_record["text"])
    print(f"\nAnnotated all {len(dev)} dev records in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
