# scripts 目录说明

本目录保存数据检查、复核整理和清洗脚本，不负责训练基线和 LLM。

## 常用入口

| 脚本 | 作用 |
|---|---|
| `inspect_data.py` | 查看原始数据结构 |
| `validate_entity_offsets.py` | 全量实体索引和标签检查 |
| `analyze_data_quality.py` | 生成数据质量报告 |
| `check_duplicate_annotations.py` | 查找重复文本及标注差异 |
| `summarize_duplicate_review.py` | 生成复核表 |
| `clean_medical_ner_data.py` | 上一版清洗入口；加 `--final` 进入最终流程 |
| `finalize_medical_ner_data.py` | 最终排除与验证实现 |
| `audit_data.cjs` | Node.js 独立审计 |

## 运行注意

- “只读检查”表示不修改原始 JSON，但可能重写自动生成报告。
- 人工复核工作簿不能被自动脚本覆盖。
- 普通模型开发只需读取最终清洗结果，不必重跑历史清洗。
- 不带 `--final` 运行清洗入口可能重写上一版输出。
- 最终流程依赖旧版数据、计划、日志、映射和验证报告，不能只保留最终 JSON。

成员2的模型训练、预测和统一指标代码在 `baseline/`，不在此目录。
