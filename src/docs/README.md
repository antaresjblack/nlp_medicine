# docs 文档索引

本目录保存数据来源、质量检查、清洗证据和最终报告文档。

## 数据和清洗

| 文件 | 用途 |
|---|---|
| `data_source.md` | 来源、字段、索引、标签分布和许可说明 |
| `data_quality_log.md` | 数据检查与复核过程 |
| `data_quality_report.txt` | 原始 train/dev 质量报告 |
| `offset_validation_report.txt` | 全量实体索引验证 |
| `duplicate_annotation_review_checked.csv.xlsx` | 人工复核工作簿 |
| `final_cleaning_summary.md` | 最终清洗决定与统计 |
| `final_validation_report.json` | 最终数据机器可读验证结果 |
| `record_index_map_final.csv` | 最终、原始和上一版记录行号映射 |
| `removed_unresolved_groups.csv` | 被整体排除的争议记录 |

## 模型与评估

模型和评估的操作说明分别以 `src/baseline/README.md` 和 `src/eval/README.md` 为准。

| 文件 | 用途 |
|---|---|
| `reproduction_scope.md` | 复现范围、非盲留出集、主结果、错误分类和报告建议措辞 |

## 最终文档待补

成员4在成员3完成后还需要补充：

- 全组最终结果表；
- 统一错误分析；
- 八节项目报告；
- 贡献声明；
- AI 使用声明；
- 演示录屏脚本；
- 提交检查清单。

文档中的成绩必须引用真实指标文件，不能手工编造或把不同 split 混在一起。
