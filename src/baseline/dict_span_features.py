"""
为字符级基线添加字典跨度特征。
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
    """
    给文本每个字符标注词典跨度特征。
    对每个字符位置，提取覆盖该位置的最长词典匹配（长度相同优先选频次更高，频次一样选起始位置靠前）：
    in_dict_span：0/1，该字符是否被词典匹配覆盖
    dict_boundary：匹配内部标记 S/B/I/E；未命中则 O
    dict_label：匹配对应的实体类型，无则none
    dict_span_len：匹配长度分箱（2 / 3-4 / 5-8 / 9+）
    dict_freq：词条训练集频次分箱（1 / 2-5 / 6-20 / 21+）
    label_conflict：0/1，是否有多个不同类型词条同时覆盖当前字符
    """

    def __init__(self, records: list[dict[str, Any]], min_count: int = 1, min_length: int = 2) -> None:
        self.min_count = min_count
        self.min_length = min_length
        label_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            for entity in record.get("entities", []):
                label_counts[entity["entity"]][entity["type"]] += 1
        self.label_counts = label_counts

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
        """
        为文本每个字符标注词典跨度特征。
        """
        matches = self._find_matches(text)
        return self._per_position_features(text, matches)

    def match_all(self, text: str) -> list[dict[str, Any]]:
        """
        返回文本中所有词典命中（供成员3把词典特征融入 BERT 训练）。

        与 annotate 的逐位置特征不同，这里不做任何取舍：重复位置、重叠、
        嵌套命中全部保留（不取最长匹配、不去重、不过滤否定语境）。每条命中
        包含：start_idx、end_idx（闭区间）、entity、多数标签 type、该词条在
        训练集中各类别的出现次数 type_counts、总频次 total_count。
        结果按 (start_idx, end_idx, type) 排序。
        """
        matches: list[dict[str, Any]] = []
        for start, char in enumerate(text):
            bucket = self.by_first_char.get(char)
            if not bucket:
                continue
            remaining = text[start:]
            for phrase, label, total in bucket:
                if remaining.startswith(phrase):
                    matches.append(
                        {
                            "start_idx": start,
                            "end_idx": start + len(phrase) - 1,
                            "entity": phrase,
                            "type": label,
                            "type_counts": dict(self.label_counts[phrase]),
                            "total_count": total,
                        }
                    )
        matches.sort(key=lambda match: (match["start_idx"], match["end_idx"], match["type"]))
        return matches

    def _find_matches(self, text: str) -> list[tuple[int, int, str, int]]:
        """
        输出所有匹配词条：起点、闭区间终点、标签、出现频次。
        """
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
    """
    为文本每个字符标注词典跨度特征。
    """
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--demo-record", type=int, default=0, help="dev record index for the feature demo")
    args = parser.parse_args()

    root = args.root.resolve()
    train_path = root / "data" / "data_cleaned" / "CMeEE_train_cleaned_final.json"
    dev_path = root / "data" / "data_raw" / "CMeEE_dev.json"
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
    print("\nmatch_all 原始命中（含重叠/嵌套，不取最长）：")
    for match in annotator.match_all(text):
        print(f"  [{match['start_idx']}, {match['end_idx']}] {match['entity']} -> {match['type']} "
              f"type_counts={match['type_counts']} total={match['total_count']}")
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
