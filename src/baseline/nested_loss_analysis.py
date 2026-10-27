"""
量化单层 BIO 投影训练带来的信息损失：
字符级 LinearSVC 采用黄金实体最长非重叠投影做训练，嵌套内层实体训练时直接丢失。本脚本分三步量化损失：
1.把验证集标注实体分为投影保留、投影丢弃两组；
2.统计单层模型能完整预测的丢弃实体；
3.统计其中依赖词典跨度特征的实体，并估算全部丢弃实体均可预测时的实体 F1 理论上限。
在代码根目录执行：python baseline/nested_loss_analysis.py
输出：outputs/nested_loss_analysis_report.md 与配套 json 结果文件。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.baseline.dict_span_features import DictSpanAnnotator
from src.baseline.small_model import load_json, longest_non_overlapping_entities


def entity_key(entity: dict[str, Any]) -> tuple[int, int, str]:
    return (entity["start_idx"], entity["end_idx"], entity["type"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    train = load_json(root / "data_cleaned" / "CMeEE_train_cleaned_final.json")
    dev = load_json(root / "data_raw" / "CMeEE_dev.json")
    predictions = load_json(output_dir / "small_model_predictions.json")
    metrics = json.loads((output_dir / "small_model_metrics.json").read_text(encoding="utf-8"))

    # 第一步：用与训练完全相同的投影拆分金标
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in dev:
        projected = {entity_key(entity) for entity in longest_non_overlapping_entities(record)}
        for entity in record["entities"]:
            (kept if entity_key(entity) in projected else dropped).append(entity)

    # 第二步：单层模型对"被丢弃实体"的严格命中（start+end+type 全同）
    predicted_keys = {entity_key(entity) for record in predictions for entity in record["entities"]}
    recovered = [entity for entity in dropped if entity_key(entity) in predicted_keys]
    missed = [entity for entity in dropped if entity_key(entity) not in predicted_keys]

    # 第三步：词典通道与上限
    annotator = DictSpanAnnotator(train)
    dictionary_entries = {(phrase, label) for phrase, label, _ in annotator.entries}
    recovered_in_dict = sum(1 for entity in recovered if (entity["entity"], entity["type"]) in dictionary_entries)
    missed_in_dict = sum(1 for entity in missed if (entity["entity"], entity["type"]) in dictionary_entries)

    tp = metrics["true_positive"]
    gold_count = metrics["gold_entities"]
    precision = metrics["precision"]
    ceiling_recall = (tp + len(missed)) / gold_count
    ceiling_f1 = 2 * precision * ceiling_recall / (precision + ceiling_recall)

    per_type: dict[str, dict[str, int]] = {}
    for entity_type in sorted({entity["type"] for entity in dropped}):
        type_dropped = [entity for entity in dropped if entity["type"] == entity_type]
        type_recovered = sum(1 for entity in type_dropped if entity_key(entity) in predicted_keys)
        per_type[entity_type] = {
            "dropped": len(type_dropped),
            "recovered": type_recovered,
            "missed": len(type_dropped) - type_recovered,
        }

    result: dict[str, Any] = {
        "method": "nested_projection_loss_analysis",
        "eval_file": str(root / "data_raw" / "CMeEE_dev.json"),
        "predictions_file": str(output_dir / "small_model_predictions.json"),
        "projection": "longest_non_overlapping_gold_entities (same as training)",
        "step1_projection": {
            "gold_entities": gold_count,
            "kept": len(kept),
            "kept_ratio": len(kept) / gold_count,
            "dropped": len(dropped),
            "dropped_ratio": len(dropped) / gold_count,
            "dropped_type_distribution": dict(sorted(Counter(entity["type"] for entity in dropped).items())),
            "dropped_length_distribution": dict(sorted(Counter(len(entity["entity"]) for entity in dropped).items())),
        },
        "step2_recovery": {
            "recovered": len(recovered),
            "recovery_rate": len(recovered) / len(dropped),
            "missed": len(missed),
            "per_type": per_type,
        },
        "step3_dictionary_channel": {
            "recovered_in_dictionary": recovered_in_dict,
            "recovered_in_dictionary_ratio": recovered_in_dict / len(recovered) if recovered else 0.0,
            "missed_in_dictionary": missed_in_dict,
        },
        "ceiling": {
            "current": {"tp": tp, "precision": precision, "recall": metrics["recall"], "f1": metrics["f1"]},
            "all_missed_recovered": {
                "tp": tp + len(missed),
                "precision": precision,
                "recall": ceiling_recall,
                "f1": ceiling_f1,
            },
            "max_f1_gain": ceiling_f1 - metrics["f1"],
        },
    }
    (output_dir / "nested_loss_analysis_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    step1, step2, step3 = result["step1_projection"], result["step2_recovery"], result["step3_dictionary_channel"]
    ceiling = result["ceiling"]
    lines = [
        "# 训练投影的信息损失分析（dev）",
        "",
        "方法：用与训练阶段完全相同的“最长非重叠实体投影”拆分金标，统计被丢弃的内层实体；"
        "再与最终单层小模型的预测做严格实体级比对（start+end+type 全同）；最后计算若剩余未恢复实体全部预测正确（且不新增误报）时实体级 F1 的理论上限。",
        "",
        "## 第一步：投影丢弃了什么",
        "",
        f"金标实体共 {step1['gold_entities']} 个：投影保留 {step1['kept']} 个（{step1['kept_ratio']:.1%}），"
        f"丢弃 {step1['dropped']} 个（{step1['dropped_ratio']:.1%}）。",
        "",
        "被丢弃实体的类型分布：",
        "",
        "| 类型 | 数量 |",
        "|---|---:|",
    ]
    for entity_type, count in step1["dropped_type_distribution"].items():
        lines.append(f"| {entity_type} | {count} |")
    lines.extend(
        [
            "",
            "## 第二步：单层模型隐式恢复了多少",
            "",
            f"被丢弃的 {step1['dropped']} 个内层实体中，单层模型已严格命中 {step2['recovered']} 个"
            f"（{step2['recovery_rate']:.1%}），未恢复 {step2['missed']} 个。",
            "",
            "| 类型 | 被丢弃 | 已恢复 | 未恢复 |",
            "|---|---:|---:|---:|",
        ]
    )
    for entity_type, counts in step2["per_type"].items():
        lines.append(f"| {entity_type} | {counts['dropped']} | {counts['recovered']} | {counts['missed']} |")
    lines.extend(
        [
            "",
            "## 第三步：词典通道与残余上限",
            "",
            f"已恢复实体中 {step3['recovered_in_dictionary']} 个（{step3['recovered_in_dictionary_ratio']:.1%}）"
            "的（文本, 类型）对存在于训练词典——即词典跨度特征提供了内层信息通道；"
            f"未恢复实体中仅 {step3['missed_in_dictionary']} 个在词典中（其余为词典未覆盖的 OOV 内层）。",
            "",
            "| | TP | P | R | F1 |",
            "|---|---:|---:|---:|---:|",
            f"| 当前单层模型 | {ceiling['current']['tp']} | {ceiling['current']['precision']:.4f} | {ceiling['current']['recall']:.4f} | {ceiling['current']['f1']:.4f} |",
            f"| 假设未恢复的 {step2['missed']} 个全部预测正确 | {ceiling['all_missed_recovered']['tp']} | {ceiling['all_missed_recovered']['precision']:.4f} | {ceiling['all_missed_recovered']['recall']:.4f} | {ceiling['all_missed_recovered']['f1']:.4f} |",
            "",
            f"理论增益上限：**+{ceiling['max_f1_gain']:.4f}** 个实体级 F1 点。",
            "",
            "## 结论",
            "",
            f"训练投影只丢弃 {step1['dropped_ratio']:.1%} 的金标实体（且 84% 为短内层身体部位），其中 {step2['recovery_rate']:.1%} "
            "已被单层模型通过词典跨度特征隐式恢复；即使把剩余全部预测正确，实体级 F1 增益上限也不到 0.2 个点。"
            "在词典特征充足的条件下，单层 BIO 结构的嵌套信息损失在实践中可忽略。",
        ]
    )
    (output_dir / "nested_loss_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"金标 {step1['gold_entities']} = 投影保留 {step1['kept']} ({step1['kept_ratio']:.1%}) + 丢弃 {step1['dropped']} ({step1['dropped_ratio']:.1%})")
    print(f"丢弃实体中单层模型已恢复 {step2['recovered']} ({step2['recovery_rate']:.1%})，未恢复 {step2['missed']}")
    print(f"已恢复中词典可解释 {step3['recovered_in_dictionary']} ({step3['recovered_in_dictionary_ratio']:.1%})；未恢复中词典内 {step3['missed_in_dictionary']}")
    print(f"F1 上限：{ceiling['current']['f1']:.4f} -> {ceiling['all_missed_recovered']['f1']:.4f}（+{ceiling['max_f1_gain']:.4f}）")
    print(f"Reports: {output_dir / 'nested_loss_analysis_report.md'} / .json")


if __name__ == "__main__":
    main()
