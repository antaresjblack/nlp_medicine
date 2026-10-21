"""
适用于CMeEE跨度格式实体的实用工具。
共享项目格式采用包含性字符偏移量：
text[起始索引:结束索引 + 1] == 实体。

"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def entity_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    """返回一个实体的严格比较键。"""
    return (int(entity["start_idx"]), int(entity["end_idx"]), str(entity["type"]))


def canonical_entity(start: int, end: int, label: str, text: str) -> dict[str, Any]:
    """构建一个与项目兼容的实体对象。"""
    return {"start_idx": start, "end_idx": end, "type": label, "entity": text}


def canonicalize_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """对实体进行排序，且不改变其原始的跨度语义。"""
    return sorted(
        [dict(entity) for entity in entities],
        key=lambda entity: (
            int(entity["start_idx"]),
            int(entity["end_idx"]),
            str(entity["type"]),
        ),
    )


def validate_record(record: dict[str, Any]) -> None:
    """如果实体与其源文本不匹配，则抛出ValueError异常。"""
    text = record["text"]
    for index, entity in enumerate(record.get("entities", [])):
        start = entity["start_idx"]
        end = entity["end_idx"]
        if text[start : end + 1] != entity["entity"]:
            raise ValueError(f"entity {index} does not match its source text")


def find_occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    """找出文本中该短语所有可能存在重叠的出现位置。"""
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
    """为存在歧义的词典条目选择一个确定性的主流标签。"""
    return sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

