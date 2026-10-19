"""Pure hand-written rule baseline for CMeEE.

This baseline deliberately does not read the training set or build a
dictionary. It applies only manually written regular-expression rules, then
uses the shared strict span evaluator.

Run from the repository root:

    python baseline/pure_rule_baseline.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metrics import evaluate_records


@dataclass(frozen=True)
class Rule:
    name: str
    label: str
    pattern: str
    priority: int


# These rules are deliberately human-authored. They are not learned from
# train/dev and should be described as a conservative heuristic baseline.
RULES: tuple[Rule, ...] = (
    Rule(
        "microorganism_suffix",
        "mic",
        r"[一-龥A-Za-z0-9-]{2,16}(?:病毒|杆菌|球菌|支原体|衣原体|真菌|螺旋体|寄生虫)",
        90,
    ),
    Rule(
        "drug_form_suffix",
        "dru",
        r"[一-龥A-Za-z0-9-]{2,16}(?:片|胶囊|注射液|口服液|颗粒|溶液|散剂|丸剂|贴剂|滴眼液|喷雾剂)",
        80,
    ),
    Rule(
        "procedure_suffix",
        "pro",
        r"[一-龥A-Za-z0-9-]{2,16}(?:手术|切除|消融|植入|穿刺|治疗|放疗|化疗|移植|输注|置换)",
        70,
    ),
    Rule(
        "lab_item_suffix",
        "ite",
        r"[一-龥A-Za-z0-9-]{2,16}(?:检查|检测|试验|测定|化验|检验|抗体|蛋白|因子|指标)",
        60,
    ),
    Rule(
        "disease_suffix",
        "dis",
        r"[一-龥A-Za-z0-9-]{2,18}(?:炎|癌|病|感染|综合征|肿瘤|卒中|贫血|中毒|衰竭|损伤)",
        50,
    ),
    Rule(
        "symptom_word",
        "sym",
        r"[一-龥A-Za-z0-9-]{1,10}(?:疼痛|发热|咳嗽|呕吐|恶心|头晕|乏力|水肿|出血|胸闷|气促|瘙痒|麻木|抽搐|腹泻|便秘)",
        45,
    ),
    Rule(
        "body_part_suffix",
        "bod",
        r"[一-龥A-Za-z0-9-]{1,8}(?:肌|骨|腺|脏|器官|血管|神经|黏膜|关节)",
        40,
    ),
    Rule(
        "equipment_suffix",
        "equ",
        r"[一-龥A-Za-z0-9-]{2,12}(?:仪|器|机|镜|泵|导管|支架)",
        30,
    ),
    Rule(
        "department_suffix",
        "dep",
        r"[一-龥A-Za-z0-9-]{2,12}(?:科室|门诊|病房|中心|科)",
        20,
    ),
)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def collect_candidates(text: str) -> list[dict[str, Any]]:
    """Apply every rule and return candidate spans with rule metadata."""
    candidates: list[dict[str, Any]] = []
    for rule in RULES:
        compiled = re.compile(rule.pattern)
        for match in compiled.finditer(text):
            start, end_exclusive = match.span()
            entity_text = match.group(0)
            if not entity_text:
                continue
            candidates.append(
                {
                    "start_idx": start,
                    "end_idx": end_exclusive - 1,
                    "type": rule.label,
                    "entity": entity_text,
                    "rule": rule.name,
                    "priority": rule.priority,
                }
            )
    return candidates


def resolve_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep deterministic, highest-priority non-overlapping rule matches."""
    # Prefer higher-priority rules, then longer spans, then earlier positions.
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item["priority"],
            -(item["end_idx"] - item["start_idx"] + 1),
            item["start_idx"],
            item["end_idx"],
            item["type"],
        ),
    )
    selected: list[dict[str, Any]] = []
    for candidate in ordered:
        if any(
            candidate["start_idx"] <= chosen["end_idx"]
            and chosen["start_idx"] <= candidate["end_idx"]
            for chosen in selected
        ):
            continue
        selected.append({key: candidate[key] for key in ("start_idx", "end_idx", "type", "entity")})
    return sorted(selected, key=lambda item: (item["start_idx"], item["end_idx"], item["type"]))


def predict_record(record: dict[str, Any]) -> dict[str, Any]:
    text = record["text"]
    return {"text": text, "entities": resolve_candidates(collect_candidates(text))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = root / "data_raw" / "CMeEE_dev.json"
    dev = load_json(dev_path)

    started = time.perf_counter()
    predictions = [predict_record(record) for record in dev]
    result = evaluate_records(dev, predictions)
    elapsed = time.perf_counter() - started
    compact_metrics = {key: value for key, value in result.items() if key != "errors"}
    compact_metrics.update(
        {
            "method": "pure_handwritten_rules",
            "eval_file": str(dev_path),
            "rule_count": len(RULES),
            "rule_names": [rule.name for rule in RULES],
            "runtime_seconds": elapsed,
        }
    )

    prediction_path = output_dir / "pure_rules_predictions.json"
    metrics_path = output_dir / "pure_rules_metrics.json"
    errors_path = output_dir / "pure_rules_errors.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(compact_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_path.write_text(json.dumps(result["errors"], ensure_ascii=False, indent=2), encoding="utf-8")

    print("Method: pure_handwritten_rules")
    print(f"Rules: {len(RULES)}")
    print(f"Entity precision: {result['precision']:.4f}")
    print(f"Entity recall:    {result['recall']:.4f}")
    print(f"Entity F1:        {result['f1']:.4f}")
    print(f"Predictions: {prediction_path}")
    print(f"Metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
