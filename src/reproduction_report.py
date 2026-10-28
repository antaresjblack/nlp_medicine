"""Human-readable console report for the default reproduction flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LABEL_ORDER = ("dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def dataset_summary(path: Path) -> dict[str, int]:
    records = load_jsonl(path) if path.suffix.lower() == ".jsonl" else load_json(path)
    return {
        "records": len(records),
        "entities": sum(len(record.get("entities", [])) for record in records),
    }


def collect_data_scope() -> dict[str, dict[str, int]]:
    paths = {
        "train": ROOT / "data" / "data_cleaned" / "CMeEE_train_cleaned_final.json",
        "dev": ROOT / "data" / "data_raw" / "CMeEE_dev.json",
        "eval_tune": ROOT / "data" / "data_eval" / "eval_tune.jsonl",
        "eval_test": ROOT / "data" / "data_eval" / "eval_test.jsonl",
    }
    return {name: dataset_summary(path) for name, path in paths.items()}


def _summary_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(metrics.get(key, default))


def build_reproduction_report(
    summary: dict[str, Any], data_scope: dict[str, dict[str, int]]
) -> str:
    small_path = _summary_path(summary["small_model"]["metrics_file"])
    bert_path = _summary_path(summary["bert_execution"]["metrics_file"])
    bert_manifest_path = _summary_path(summary["bert_execution"]["manifest"])
    small = load_json(small_path)
    bert = load_json(bert_path)
    bert_manifest = load_json(bert_manifest_path)

    environment = summary["environment"]
    frozen = summary["frozen_small_model_config"]
    selected = frozen["selected_candidate"]
    bert_display = summary.get("bert_variant", {}).get("label", "BERT")
    small_f1 = _metric(small, "f1")
    bert_f1 = _metric(bert, "f1")
    delta_f1 = bert_f1 - small_f1
    relative = (delta_f1 / small_f1 * 100.0) if small_f1 else 0.0
    separator = "=" * 78

    lines = [
        separator,
        "CMeEE 医疗命名实体识别：实验复现报告",
        separator,
        f"状态             : {str(summary['status']).upper()}",
        f"代码提交         : {summary.get('repository_commit') or 'unknown'}",
        f"工作区           : dirty={summary.get('working_tree_dirty')}",
        f"运行时间         : {summary['started_at']}  ->  {summary['finished_at']}",
        f"耗时             : {summary['elapsed_seconds']:.1f} 秒",
        f"Python           : {environment['python']}",
        f"PyTorch          : {environment['torch']}",
        f"Transformers     : {environment['transformers']}",
        f"NumPy/SciPy/sklearn: {environment['numpy']} / {environment['scipy']} / "
        f"{environment['scikit_learn']}",
        f"计算设备         : {environment.get('cuda_device', 'CPU')}",
        "",
        "[1] 数据范围与完整性",
    ]
    hash_keys = {
        "train": "data/data_cleaned/CMeEE_train_cleaned_final.json",
        "dev": "data/data_raw/CMeEE_dev.json",
        "eval_tune": "data/data_eval/eval_tune.jsonl",
        "eval_test": "data/data_eval/eval_test.jsonl",
    }
    for split in ("train", "dev", "eval_tune", "eval_test"):
        scope = data_scope[split]
        digest = summary["data_hashes"][hash_keys[split]]
        lines.append(f"  {split:<9} {scope['records']:>5} 条 / {scope['entities']:>6} 实体")
        lines.append(f"    sha256={digest}")
    lines += [
        "  评估口径         严格实体级 Micro-F1：起点、终点和标签全部匹配才算正确",
        "  未使用           官方无标签 CMeEE_test；F1 仅来自冻结 eval_test",
    ]
    preparation = summary.get("data_preparation", {})
    if preparation.get("status") == "downloaded":
        lines.append(f"  数据包 SHA        {preparation.get('sha256')}")
    else:
        lines.append("  数据来源          项目内已有冻结数据")
    lines += [
        "",
        "[2] 冻结配置与模型范围",
        "  执行方式         LinearSVC 重新训练；BERT 使用固定 checkpoint 推理",
        f"  LinearSVC       C={selected['C']}, class_weight={selected['class_weight']}, "
        f"dict_span_features={selected['dict_span_features']}, seed={frozen['settings']['seed']}",
        f"  BERT             {bert_display}",
        f"  BERT checkpoint  {summary['bert_execution']['checkpoint']}",
        "  checkpoint SHA",
        f"    {summary['bert_execution']['checkpoint_sha256']}",
        f"  BERT 参数        threshold={bert_manifest['args']['threshold']}, "
        f"max_length={bert_manifest['args']['max_length']}, overlap={bert_manifest['args']['overlap']}",
        "",
        "[3] 冻结 eval_test 主结果",
    ]
    rows = (
        ("char LinearSVC + dict features", small),
        (bert_display, bert),
    )
    for name, metrics in rows:
        lines.append(f"  {name}")
        lines.append(
            f"    P={_metric(metrics, 'precision'):.6f} "
            f"R={_metric(metrics, 'recall'):.6f} "
            f"F1={_metric(metrics, 'f1'):.6f}"
        )
        lines.append(
            f"    Token-F1={_metric(metrics, 'token_entity_f1'):.6f} "
            f"CharML-F1={_metric(metrics, 'char_multi_label_f1'):.6f} "
            f"error-records={int(metrics.get('records_with_errors', 0))}"
        )
    lines += [
        f"  BERT - LinearSVC 绝对 F1 提升：{delta_f1:+.6f}",
        f"  BERT 相对 LinearSVC F1 提升：{relative:+.2f}%",
        "",
        "[4] 九类实体严格 F1",
        "  标签      LinearSVC      BERT       Delta",
    ]
    for label in LABEL_ORDER:
        small_label = _metric(small.get("per_label", {}).get(label, {}), "f1")
        bert_label = _metric(bert.get("per_label", {}).get(label, {}), "f1")
        lines.append(
            f"  {label:<8} {small_label:>10.6f} {bert_label:>10.6f} "
            f"{bert_label - small_label:>+10.6f}"
        )
    lines += [
        "",
        "[5] 本次输出",
        f"  输出目录         {_display_path(Path(summary['output_dir']))}",
        f"  控制台报告       {_display_path(_summary_path(summary.get('report_file', 'reproduction_report.txt')))}",
        f"  LinearSVC        {_display_path(small_path)}",
        f"  BERT             {_display_path(bert_path)}",
        f"  总摘要           {_display_path(Path(summary['output_dir']) / 'reproduction_summary.json')}",
        separator,
    ]
    return "\n".join(lines)
