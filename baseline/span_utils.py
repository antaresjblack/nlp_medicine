"""Utilities for CMeEE span-format entities.

The shared project format uses inclusive character offsets:
text[start_idx:end_idx + 1] == entity.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def entity_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    """Return the strict-comparison key for one entity."""
    return (int(entity["start_idx"]), int(entity["end_idx"]), str(entity["type"]))


def canonical_entity(start: int, end: int, label: str, text: str) -> dict[str, Any]:
    """Build a project-compatible entity object."""
    return {"start_idx": start, "end_idx": end, "type": label, "entity": text}


def canonicalize_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort entities without changing their original span semantics."""
    return sorted(
        [dict(entity) for entity in entities],
        key=lambda entity: (
            int(entity["start_idx"]),
            int(entity["end_idx"]),
            str(entity["type"]),
        ),
    )


def validate_record(record: dict[str, Any]) -> None:
    """Raise ValueError if an entity does not match its source text."""
    text = record["text"]
    for index, entity in enumerate(record.get("entities", [])):
        start = entity["start_idx"]
        end = entity["end_idx"]
        if text[start : end + 1] != entity["entity"]:
            raise ValueError(f"entity {index} does not match its source text")


def find_occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    """Find all possibly overlapping occurrences of phrase in text."""
    if not phrase:
        return []
    positions: list[tuple[int, int]] = []
    start = 0
    while True:
        position = text.find(phrase, start)
        if position < 0:
            return positions
        positions.append((position, position + len(phrase) - 1))
        start = position + 1


def resolve_label(label_counts: Counter[str]) -> str:
    """Choose a deterministic majority label for an ambiguous dictionary entry."""
    return sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

