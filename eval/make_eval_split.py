#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 dev 切出冻结的评估集（成员4 维护，可复现）。

    python eval/make_eval_split.py

读 data_raw/CMeEE_dev.json，按记录随机打乱（seed=42），切成：
  data_eval/eval_tune.jsonl   2,500 条，调参用
  data_eval/eval_test.jsonl   2,500 条，最终评估用

索引沿用原文件的含尾口径：text[start_idx:end_idx + 1] == entity。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
HALF = 2500
LABEL_ORDER = ["dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep"]


def load_dev(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise SystemExit(f"{path} 顶层不是列表")
    return data


def resolve_paths(root: Path) -> tuple[Path, Path]:
    """优先采用成员2基线的根目录布局，同时兼容旧的 data/ 布局。"""
    for candidate in (root / "data_raw" / "CMeEE_dev.json",
                      root / "data" / "data_raw" / "CMeEE_dev.json"):
        if candidate.exists():
            return candidate, root / "data_eval"
    raise SystemExit(
        "找不到 CMeEE_dev.json。期望位置：data_raw/CMeEE_dev.json "
        "或 data/data_raw/CMeEE_dev.json"
    )


def check_source(records: list[dict]) -> None:
    """切分前先确认输入是干净的，避免把坏数据带进考卷。"""
    problems = []
    for i, r in enumerate(records):
        text = r.get("text")
        if not isinstance(text, str) or not text:
            problems.append(f"第 {i} 条 text 缺失")
            continue
        for j, e in enumerate(r.get("entities", [])):
            s, en = e.get("start_idx"), e.get("end_idx")
            if not isinstance(s, int) or not isinstance(en, int):
                problems.append(f"第 {i} 条实体 {j} 索引不是整数")
                continue
            if not (0 <= s <= en < len(text)):
                problems.append(f"第 {i} 条实体 {j} 区间非法 [{s}, {en}]")
            elif text[s:en + 1] != e.get("entity"):
                problems.append(f"第 {i} 条实体 {j} 切片与 entity 字段不一致")
    if problems:
        print("[FATAL] dev 源文件有问题：", file=sys.stderr)
        for p in problems[:10]:
            print("  -", p, file=sys.stderr)
        raise SystemExit(2)


def make_records(records: list[dict], rows: list[int], prefix: str) -> list[dict]:
    out = []
    for n, i in enumerate(rows, 1):
        r = records[i]
        out.append({
            "id": f"{prefix}-{n:04d}",
            "text": r["text"],
            "entities": [
                {
                    "start_idx": e["start_idx"],
                    "end_idx": e["end_idx"],
                    "type": e["type"],
                    "entity": e["entity"],
                }
                for e in r.get("entities", [])
            ],
        })
    return out


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_stats(rows: list[dict]) -> tuple[int, Counter]:
    counter: Counter = Counter()
    for r in rows:
        for e in r["entities"]:
            counter[e["type"]] += 1
    return sum(counter.values()), counter


def render_readme(tune: list[dict], test: list[dict], hashes: dict, src_rel: str) -> str:
    n_tune, c_tune = label_stats(tune)
    n_test, c_test = label_stats(test)
    lines = [
        "# data_eval：冻结的评估集",
        "",
        "> 这两份文件是全组共用的考卷，生成后不再修改。",
        "> 改动会导致所有人的分数失去可比性。",
        "",
        "## 文件",
        "",
        "| 文件 | 用途 | 条数 | 实体数 |",
        "| --- | --- | ---: | ---: |",
        f"| `eval_tune.jsonl` | 调参：提示词、超参数、模型选择只能用这份 | {len(tune)} | {n_tune} |",
        f"| `eval_test.jsonl` | 最终评估：报告里的分数只能来自这份 | {len(test)} | {n_test} |",
        "",
        "原始 `CMeEE_test.json` 的 3,000 条实体列表全为空，是官方无答案测试集，",
        "不能用来计算 F1，本目录不使用它。",
        "",
        "## 怎么切出来的",
        "",
        f"- 来源：`{src_rel}`（5,000 条，有标注）",
        "- 随机种子：`42`",
        "- 切分单位：整条记录（不按实体切，避免同一句拆到两边）",
        "- 比例：打乱后取前 2,500 条为 tune，其余 2,500 条为 test",
        "- 索引口径：沿用原始含尾索引，`text[start_idx:end_idx + 1] == entity`",
        "",
        "复现命令：",
        "",
        "```powershell",
        "python src/eval/make_eval_split.py",
        "```",
        "",
        "## 校验值",
        "",
        "| 文件 | SHA-256 |",
        "| --- | --- |",
        f"| `eval_tune.jsonl` | `{hashes['eval_tune.jsonl']}` |",
        f"| `eval_test.jsonl` | `{hashes['eval_test.jsonl']}` |",
        "",
        "## 类别分布",
        "",
        "| 类型 | tune | 占比 | test | 占比 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for t in LABEL_ORDER:
        a, b = c_tune.get(t, 0), c_test.get(t, 0)
        lines.append(
            f"| {t} | {a} | {a / n_tune:.2%} | {b} | {b / n_test:.2%} |"
        )
    lines += [
        "",
        "## 使用规则",
        "",
        "1. `eval_tune` 可以反复跑、反复改参数，不会影响最终结论。",
        "2. `eval_test` 只在定稿时跑一次，跑完就写进报告。",
        "3. 预测文件命名：`outputs/predictions/{方法名}_tune_predictions.jsonl` 和",
        "   `outputs/predictions/{方法名}_test_predictions.jsonl`。",
        "   这里的 `test` 指本目录的 `eval_test.jsonl`，不是官方无答案的 `CMeEE_test.json`。",
        "4. 本目录任何文件都不允许改；确需重新切分，必须全组同意并重跑所有方法。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成冻结评估集")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的评估集")
    args = ap.parse_args()

    root = args.root
    src_path, out_dir = resolve_paths(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / "eval_tune.jsonl", out_dir / "eval_test.jsonl"]
    if any(p.exists() for p in targets) and not args.force:
        raise SystemExit("评估集已存在。确认要重新生成请加 --force；重切会导致所有人重跑。")

    dev = load_dev(src_path)
    print(f"读入 {src_path.name}：{len(dev)} 条")
    check_source(dev)

    idx = list(range(len(dev)))
    random.Random(SEED).shuffle(idx)
    tune = make_records(dev, idx[:HALF], "tune")
    test = make_records(dev, idx[HALF:], "test")

    if len(tune) + len(test) != len(dev):
        raise SystemExit("切分后条数对不上")
    if {r["text"] for r in tune} & {r["text"] for r in test}:
        raise SystemExit("两半之间存在重复文本，切分失败")

    hashes = {}
    for path, rows in ((targets[0], tune), (targets[1], test)):
        hashes[path.name] = write_jsonl(path, rows)
        n_ent, _ = label_stats(rows)
        print(f"写出 {path.relative_to(root)}：{len(rows)} 条，{n_ent} 个实体")

    readme = out_dir / "README.md"
    src_rel = src_path.relative_to(root).as_posix()
    readme.write_text(render_readme(tune, test, hashes, src_rel), encoding="utf-8")
    print(f"写出 {readme.relative_to(root)}")
    print()
    print("SHA-256")
    for name, h in hashes.items():
        print(f"  {name}  {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
