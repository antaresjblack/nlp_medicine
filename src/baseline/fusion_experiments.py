"""融合实验：在 tune 上系统扫描 BERT × 词典 × 小模型 × 低分候选的组合。

纯预测结果级后处理：不训练、不调 API、不动 test。基准为 BERT 阈值 0 的
预测（tune F1 0.6628）。扫描以下融合家族，全部用 baseline/metrics.py 的
严格实体级指标评测：

    A 纯分数补丁      BERT + {低分候选 | score > t}
    B 词典∩分数       BERT + {低分候选 ∩ 词典命中 | score > t}
    C 小模型∩分数     BERT + {低分候选 ∩ 小模型预测 | score > t}
    D 词典∩小模型∩分数
    E 类型修正        词典同 span 不同类型（词频>=5）替换 BERT 的类型

在仓库根目录运行（约 1~2 分钟）：

    python src/baseline/fusion_experiments.py

输出：outputs/fusion/fusion_report.md/json 与各方案 metrics。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dict_span_features import DictSpanAnnotator
from dictionary_baseline import build_dictionary, predict_longest
from metrics import evaluate_records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def keys(record: dict[str, Any]) -> set[tuple[int, int, str]]:
    return {(e["start_idx"], e["end_idx"], e["type"]) for e in record.get("entities", [])}


def evaluate_fused(gold_list, id_order, fused_by_id) -> dict[str, Any]:
    return evaluate_records(
        [gold_list[i] for i in id_order],
        [fused_by_id[i] for i in id_order],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    fusion_dir = output_dir / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    gold = {r["id"]: r for r in load_jsonl(root / "data" / "data_eval" / "eval_tune.jsonl")}
    bert = {r["id"]: r for r in load_jsonl(root / "macbert_5ep_low_score_tune" / "positive_predictions.jsonl")}
    low_score = {r["id"]: r.get("candidates", []) for r in load_jsonl(root / "macbert_5ep_low_score_tune" / "low_score_candidates.jsonl")}
    svm_path = next((root / "outputs" / "eval_split" / "small_model" / "tune" / "runs").glob("*/candidates/span_c0.1_balanced_predictions.jsonl"))
    svm = {r["id"]: r for r in load_jsonl(svm_path)}
    ids = list(gold.keys())

    train = load_json(root / "data" / "data_cleaned" / "CMeEE_train_cleaned_final.json")
    dictionary = build_dictionary(train, min_count=1, min_length=2)
    freq = {(p, l): f for p, l, f in DictSpanAnnotator(train).entries}

    dict_keys = {i: keys(predict_longest(gold[i], dictionary)) for i in ids}
    svm_keys = {i: keys(svm[i]) for i in ids}
    bert_keys = {i: keys(bert[i]) for i in ids}

    baseline = {i: {"text": gold[i]["text"], "entities": list(bert[i]["entities"])} for i in ids}

    def apply_patches(score_t: float, require_dict: bool, require_svm: bool) -> dict[str, dict[str, Any]]:
        """BERT + 低分候选中满足条件的补丁（score>t，可选要求词典/小模型命中）。"""
        fused = {i: {"text": gold[i]["text"], "entities": list(bert[i]["entities"])} for i in ids}
        for i in ids:
            seen = bert_keys[i].copy()
            for c in low_score[i]:
                if c["score"] <= score_t:
                    continue
                k = (c["start_idx"], c["end_idx"], c["type"])
                if require_dict and k not in dict_keys[i]:
                    continue
                if require_svm and k not in svm_keys[i]:
                    continue
                if k in seen:
                    continue
                seen.add(k)
                fused[i]["entities"].append(
                    {"start_idx": k[0], "end_idx": k[1], "type": k[2], "entity": c["entity"]}
                )
        return fused

    def apply_type_fix() -> dict[str, dict[str, Any]]:
        """词典同 span 不同类型（词频>=5）时，用词典多数标签替换 BERT 类型。"""
        fused = {i: {"text": gold[i]["text"], "entities": list(bert[i]["entities"])} for i in ids}
        for i in ids:
            for e in fused[i]["entities"]:
                s, t, entity = e["start_idx"], e["end_idx"], e["type"]
                for (ds, de, dt) in dict_keys[i]:
                    if ds == s and de == t and dt != e["type"] and freq.get((e["entity"], dt), 0) >= 5:
                        e["type"] = dt
                        break
        return fused

    def apply_type_del() -> dict[str, dict[str, Any]]:
        """反向融合：删除与词典同 span 不同类型（词频>=5）的 BERT 实体。"""
        fused: dict[str, dict[str, Any]] = {}
        for i in ids:
            keep = []
            for e in bert[i]["entities"]:
                conflict = any(
                    ds == e["start_idx"]
                    and de == e["end_idx"]
                    and dt != e["type"]
                    and freq.get((e["entity"], dt), 0) >= 5
                    for (ds, de, dt) in dict_keys[i]
                )
                if not conflict:
                    keep.append(dict(e))
            fused[i] = {"text": gold[i]["text"], "entities": keep}
        return fused

    def apply_svm_type_replace() -> dict[str, dict[str, Any]]:
        """反向融合：小模型同 span 不同类型时，用小模型类型替换 BERT 类型。"""
        fused = {i: {"text": gold[i]["text"], "entities": list(bert[i]["entities"])} for i in ids}
        for i in ids:
            for e in fused[i]["entities"]:
                for (ss, se, st) in svm_keys[i]:
                    if ss == e["start_idx"] and se == e["end_idx"] and st != e["type"]:
                        e["type"] = st
                        break
        return fused

    def apply_boundary_align() -> dict[str, dict[str, Any]]:
        """反向融合：词典 span 与 BERT 实体重叠且同类型，且小模型也输出该词典 span
        时，把 BERT 实体对齐到词典的精确边界。"""
        fused: dict[str, dict[str, Any]] = {}
        for i in ids:
            out_ents: list[dict[str, Any]] = []
            for e in bert[i]["entities"]:
                replaced = False
                best = None
                for (ds, de, dt) in dict_keys[i]:
                    if dt != e["type"]:
                        continue
                    if e["start_idx"] <= de and ds <= e["end_idx"] and (ds, de) != (e["start_idx"], e["end_idx"]):
                        if (ds, de, dt) in svm_keys[i]:
                            if best is None or (de - ds) > (best[1] - best[0]):
                                best = (ds, de, dt)
                if best is not None:
                    ds, de, dt = best
                    out_ents.append(
                        {"start_idx": ds, "end_idx": de, "type": dt, "entity": gold[i]["text"][ds : de + 1]}
                    )
                    replaced = True
                if not replaced:
                    out_ents.append(dict(e))
            # 去重
            seen: set[tuple[int, int, str]] = set()
            dedup = []
            for e in out_ents:
                k = (e["start_idx"], e["end_idx"], e["type"])
                if k not in seen:
                    seen.add(k)
                    dedup.append(e)
            fused[i] = {"text": gold[i]["text"], "entities": dedup}
        return fused

    def combine_fixes() -> dict[str, dict[str, Any]]:
        """反向融合：类型修正 + 边界对齐同时应用。"""
        fixed = apply_type_fix()
        fused: dict[str, dict[str, Any]] = {}
        for i in ids:
            keep = []
            for e in fixed[i]["entities"]:
                # 边界对齐
                best = None
                for (ds, de, dt) in dict_keys[i]:
                    if dt != e["type"]:
                        continue
                    if e["start_idx"] <= de and ds <= e["end_idx"] and (ds, de) != (e["start_idx"], e["end_idx"]):
                        if (ds, de, dt) in svm_keys[i]:
                            if best is None or (de - ds) > (best[1] - best[0]):
                                best = (ds, de, dt)
                if best is not None:
                    ds, de, dt = best
                    keep.append({"start_idx": ds, "end_idx": de, "type": dt, "entity": gold[i]["text"][ds : de + 1]})
                else:
                    keep.append(dict(e))
            seen: set[tuple[int, int, str]] = set()
            dedup = []
            for e in keep:
                k = (e["start_idx"], e["end_idx"], e["type"])
                if k not in seen:
                    seen.add(k)
                    dedup.append(e)
            fused[i] = {"text": gold[i]["text"], "entities": dedup}
        return fused

    rows: list[dict[str, Any]] = []

    def record(name: str, fused: dict[str, dict[str, Any]]) -> None:
        m = evaluate_fused(gold, ids, fused)
        compact = {k: v for k, v in m.items() if k != "errors"}
        compact.update({"method": f"fusion_{name}", "eval_file": "data/data_eval/eval_tune.jsonl"})
        (fusion_dir / f"fusion_{name}_metrics.json").write_text(
            json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "name": name,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "predicted": m["predicted_entities"],
            }
        )

    record("baseline_bert_t0", baseline)
    for t in [-2.0, -1.0, -0.5, -0.3, -0.1, -0.05]:
        record(f"score_gt{t}", apply_patches(t, False, False))
    for t in [-1.0, -0.5, -0.3, -0.1]:
        record(f"dict_and_score_gt{t}", apply_patches(t, True, False))
        record(f"svm_and_score_gt{t}", apply_patches(t, False, True))
    for t in [-1.0, -0.5, -0.3]:
        record(f"dict_svm_and_score_gt{t}", apply_patches(t, True, True))
    record("type_fix_dict_majority", apply_type_fix())
    record("reverse_del_type_conflict", apply_type_del())
    record("reverse_svm_type_replace", apply_svm_type_replace())
    record("reverse_boundary_align", apply_boundary_align())
    record("reverse_combined_fixes", combine_fixes())

    base_f1 = next(r["f1"] for r in rows if r["name"] == "baseline_bert_t0")
    ordered = sorted(rows, key=lambda r: -r["f1"])
    for r in rows:
        r["delta_f1"] = r["f1"] - base_f1

    report = {
        "baseline_f1": base_f1,
        "eval_file": "data/data_eval/eval_tune.jsonl",
        "svm_predictions": str(svm_path),
        "note": "全部为 tune 结果级融合；最优方案若想作为最终配置，需按协议在 eval_test 上只跑一次",
        "rows": ordered,
        "best": ordered[0],
    }
    (fusion_dir / "fusion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# BERT 融合实验报告（tune）",
        "",
        f"基准：BERT 阈值 0，P={base_f1:.4f}（严格实体级 F1）。全部为预测结果级后处理。",
        "",
        "| 方案 | P | R | F1 | ΔF1 | 预测实体数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in ordered:
        lines.append(
            f"| {r['name']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} "
            f"| {r['delta_f1']:+.4f} | {r['predicted']} |"
        )
    (fusion_dir / "fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"基准 F1: {base_f1:.4f}")
    print(f"{'方案':<28} {'F1':>8} {'ΔF1':>8}")
    for r in ordered[:8]:
        print(f"{r['name']:<28} {r['f1']:>8.4f} {r['delta_f1']:>+8.4f}")
    print(f"\n报告: {fusion_dir / 'fusion_report.md'}")


if __name__ == "__main__":
    main()
