# CMeEE 最终训练集清洗总结

## 本次决定与规则
按 `src/data/data_cleaned/remaining_review_items.csv` 中全部剩余组的原始零基记录索引排除整组记录，不再裁定实体。
实际组号：[4, 5, 9, 11, 13, 18, 30, 41]。与用户提供的8组编号一致。
涉及 16 条原始记录：10 条已排除，新增排除 6 条；最终累计排除 16 条。
从固定的上一版清洗集及原始索引映射筛选，不在删除后的行号上再次删除。保留记录的全部字段和标注与上一版完全一致。
原始训练集、dev/test、上一版清洗文件和人工记录均保留。test 空列表不是负样本金标；未转换 BIO、未训练模型。

## 数量统计
| 数据版本 | 样本数 | 实体数 |
| --- | ---: | ---: |
| 原始训练集 | 15000 | 61796 |
| 上一版清洗集 | 14990 | 61771 |
| 最终训练集 | 14984 | 61743 |

| 类型 | 原始 | 上一版 | 最终 | 本次变化 |
| --- | ---: | ---: | ---: | ---: |
| dis | 15843 | 15830 | 15819 | -11 |
| sym | 12269 | 12264 | 12264 | +0 |
| dru | 3930 | 3930 | 3930 | +0 |
| equ | 888 | 888 | 888 | +0 |
| pro | 6332 | 6323 | 6323 | +0 |
| bod | 17697 | 17699 | 17682 | -17 |
| ite | 2581 | 2581 | 2581 | +0 |
| mic | 1908 | 1907 | 1907 | +0 |
| dep | 348 | 349 | 349 | +0 |

## 验证与追溯
重新检查全部最终实体，不仅引用旧 validation_report；历史清洗日志逐项重放，并对照既定计划验证标注操作。
最终 JSON 回读、字段结构、九类标签、闭区间索引及文本切片通过；索引映射无重复、无越界且逐条匹配文本。
争议索引及此前排除索引均不在最终映射中；无新增实体标注变更。被排除记录明细见 `removed_unresolved_groups.csv`。
本入口每次独立构建两遍并逐字节比较；写盘后再次回读验证。可连续运行下方命令两次比较输出哈希。
最终 JSON SHA-256：`e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc`
原始训练集 SHA-256：`984f0bea3932c8c5afd96d9cccac4895b87ec77f57bd3e825d832a17943cd5d3`（运行前后不变）。

## 文件与复现
`src/data/data_cleaned/CMeEE_train_cleaned_final.json`：最终训练数据。
`src/docs/record_index_map_final.csv`：最终、原始和上一版行号对应，均为零基。
`src/docs/removed_unresolved_groups.csv`：完整原始实体、上一版实体、实体数、组号、排除状态及原因。
`src/docs/final_validation_report.json`：重新计算的统计、校验与输入哈希。
仓库根目录执行：
```bat
python -X utf8 -B src\scripts\clean_medical_ner_data.py --final
```
可用 `--input`、`--previous-dir`、`--docs-dir` 指定路径；`--final` 不重跑或覆盖上一版清洗。
本次创建前检查过所有最终输出均不存在；后续仅允许重写内容完全相同的结果，若现有结果不同则停止。

## 限制
争议通过整组排除解决训练准入，不代表医学语义已裁定或所有实体均错误；未开展新的标注审定。
第9、11、41组此前部分修订随整条记录一起排除；其余已执行修订保留。
未对标注一致的重复文本去重；数据许可及版本信息沿用现有来源文档。
## 受保护输入 SHA-256
- `src\data\data_cleaned\CMeEE_train_cleaned.json`：`bf3e16e78a7f1f42f4f1ae6722cd0794b5e1acb9b822f3321937bae71411af22`
- `src\data\data_cleaned\cleaning_log.csv`：`e62e617edd2ffd397b72b765540dc94d2b387f5d4920a8cb92b529c9f7a21d00`
- `src\data\data_cleaned\cleaning_plan.json`：`2f3196e41a013cce085985a5143d1422ece6e928c517f55c3f710d88166ba5c0`
- `src\data\data_cleaned\record_index_map.csv`：`a8ee135b47cea4a59c6b8ce207dad09dbdbc9d449ed46b47d45f89807ffa1c6a`
- `src\data\data_cleaned\remaining_review_items.csv`：`11803c297f00b96ee3a2aba88ba66be44c9a9a66e45b7d935cb2143095422ce5`
- `src\data\data_cleaned\validation_report.json`：`c2d6a1e7d709491fb11da31d95876adfe762b23ca3267ed9651449a07da38438`
- `src\data\data_raw\CMeEE_dev.json`：`bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c`
- `src\data\data_raw\CMeEE_test.json`：`dca7d9b7cf820947bba68e14557908fdc63e9286add1b4bc34f78f245e0c7ecf`
- `src\data\data_raw\CMeEE_train.json`：`984f0bea3932c8c5afd96d9cccac4895b87ec77f57bd3e825d832a17943cd5d3`
