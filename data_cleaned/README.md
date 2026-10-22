# data_cleaned 目录说明

本目录保存成员1完成的清洗结果及其重建证据。模型训练默认只读取最终版。

## 首选训练入口

```text
CMeEE_train_cleaned_final.json
```

当前规模：

```text
14,984 条文本
61,743 个实体
SHA-256 = e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc
```

成员2和成员3的模型都应使用此文件，不要混用旧版数据。

## 历史文件

- `CMeEE_train_cleaned.json`：上一版清洗集，仅用于历史追溯和最终重建；
- `cleaning_plan.json`：清洗规则及其绑定输入；
- `cleaning_log.csv`：实体修改前后值和原因；
- `record_index_map.csv`：上一版行号与原始行号映射；
- `remaining_review_items.csv`：最终排除争议记录的来源；
- `validation_report.json`：上一版验证统计；
- `cleaning_summary.md`：上一版清洗说明。

这些文件是数据证据链的一部分，不能因为已有最终 JSON 就删除。

## 当前最终决策

第 4、5、9、11、13、18、30、41 组记录整体排除，共涉及 16 条原始记录。
最终训练集回读、字段、标签、闭区间索引和原文切片验证均通过。

最新的人类可读说明见 `docs/final_cleaning_summary.md`，机器可读验证见
`docs/final_validation_report.json`。普通模型开发不需要重新执行清洗。
