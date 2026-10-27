# 医疗 NER 清洗副本说明

输入：`src\data\data_raw\CMeEE_train.json`
人工记录：`src\docs\duplicate_annotation_review_checked.csv.xlsx`
可执行计划：`src\data\data_cleaned\cleaning_plan.json`（绑定输入与人工记录哈希；不自动解释任意自然语言备注）

- 样本数：15000 → 14990。
- 实体数：61796 → 61771。
- 修改记录：19；人工复核操作新增实体 18 条、移除旧实体 10 条。替换按移除旧条目和新增新条目分别计数，不重复声称为独立纠错数量。
- 暂时排除 10 条记录、5 组；只排除训练副本，不能解释为所有实体均错误。
- 额外移除异常实体 2 条，所在文本保留。
- 一致重复组保留 25 组，未去重；完全统一 10 个原不一致组。
- 仍需人工介入 8 组：4、5、9、11、13、18、30、41。
- 第 9、11、41 组仅执行明确操作，未提及的差异保持各自原标注；未强行统一。清洗版并非所有标注都已裁定。
- 第 25 组建议中“保留一份一致标注”未视为明确的文本去重授权，保留全部记录。
- dev/test 不改动、不生成清洗版；test 空 entities 不作为真实负标签或用于计算 P/R/F1。
- JSON 回读、字段、九类标签、闭区间索引、实体切片、修改日志覆盖和源文件 SHA-256 检查均通过。

| 标签 | 清洗前 | 清洗后 | 变化 |
| --- | ---: | ---: | ---: |
| dis | 15843 | 15830 | -13 |
| sym | 12269 | 12264 | -5 |
| dru | 3930 | 3930 | +0 |
| equ | 888 | 888 | +0 |
| pro | 6332 | 6323 | -9 |
| bod | 17697 | 17699 | +2 |
| ite | 2581 | 2581 | +0 |
| mic | 1908 | 1907 | -1 |
| dep | 348 | 349 | +1 |

## 待人工介入事项
- 第 4 组：医疗操作类别已有倾向，但边界未最终裁定；排除不代表全部实体错误。
- 第 5 组：心功能中的“心”是否单列 bod 尚未裁定。
- 第 9 组：备注未裁定记录5685独有的“会阴部”(51,53,bod)是否应补入记录5697；保持各自原标注，等待明确决定。
当前保留标注：[{"record_index": 5685, "entities": [{"start_idx": 2, "end_idx": 3, "type": "bod", "entity": "睾丸"}, {"start_idx": 9, "end_idx": 12, "type": "bod", "entity": "睾丸引带"}, {"start_idx": 19, "end_idx": 20, "type": "bod", "entity": "睾丸"}, {"start_idx": 22, "end_idx": 23, "type": "bod", "entity": "附睾"}, {"start_idx": 32, "end_idx": 33, "type": "bod", "entity": "阴囊"}, {"start_idx": 46, "end_idx": 49, "type": "bod", "entity": "耻骨结节"}, {"start_idx": 51, "end_idx": 53, "type": "bod", "entity": "会阴部"}, {"start_idx": 55, "end_idx": 58, "type": "bod", "entity": "股内侧部"}]}, {"record_index": 5697, "entities": [{"start_idx": 2, "end_idx": 3, "type": "bod", "entity": "睾丸"}, {"start_idx": 19, "end_idx": 20, "type": "bod", "entity": "睾丸"}, {"start_idx": 22, "end_idx": 23, "type": "bod", "entity": "附睾"}, {"start_idx": 32, "end_idx": 33, "type": "bod", "entity": "阴囊"}, {"start_idx": 46, "end_idx": 49, "type": "bod", "entity": "耻骨结节"}, {"start_idx": 9, "end_idx": 12, "type": "bod", "entity": "睾丸引带"}, {"start_idx": 55, "end_idx": 58, "type": "bod", "entity": "股内侧部"}]}]
- 第 11 组：备注未裁定记录5703独有的“隐睾”(17,18,dis)是否补入记录5691；不能仅凭保留“睾丸”的结论推断。
当前保留标注：[{"record_index": 5691, "entities": [{"start_idx": 4, "end_idx": 5, "type": "bod", "entity": "睾丸"}]}, {"record_index": 5703, "entities": [{"start_idx": 4, "end_idx": 5, "type": "bod", "entity": "睾丸"}, {"start_idx": 17, "end_idx": 18, "type": "dis", "entity": "隐睾"}]}]
- 第 13 组：膀胱颈部 bod 已有结论，但医疗操作 pro 边界仍待确认；排除不代表全部实体错误。
- 第 18 组：尿路不完全性梗阻 dis 已有结论，但膀胱输尿管 bod 是否保留未裁定。
- 第 30 组：APF 的实体类别未裁定。
- 第 41 组：备注只裁定整段sym移除及首次肾瘢痕保留，未裁定记录14506独有的“肾瘢痕”(13,15,dis)与“高血压”(21,23,dis)是否补入14495；保持原标注，不自动补标。
当前保留标注：[{"record_index": 14495, "entities": [{"start_idx": 0, "end_idx": 2, "type": "dis", "entity": "高血压"}, {"start_idx": 7, "end_idx": 9, "type": "dis", "entity": "肾瘢痕"}, {"start_idx": 35, "end_idx": 37, "type": "dis", "entity": "肾瘢痕"}, {"start_idx": 53, "end_idx": 55, "type": "dis", "entity": "高血压"}]}, {"record_index": 14506, "entities": [{"start_idx": 0, "end_idx": 2, "type": "dis", "entity": "高血压"}, {"start_idx": 13, "end_idx": 15, "type": "dis", "entity": "肾瘢痕"}, {"start_idx": 21, "end_idx": 23, "type": "dis", "entity": "高血压"}, {"start_idx": 35, "end_idx": 37, "type": "dis", "entity": "肾瘢痕"}, {"start_idx": 53, "end_idx": 55, "type": "dis", "entity": "高血压"}, {"start_idx": 7, "end_idx": 9, "type": "dis", "entity": "肾瘢痕"}]}]

## 输入完整性
以下原始 JSON 和人工记录运行前后 SHA-256 完全一致：
- `src\data\data_cleaned\cleaning_plan.json`：`2f3196e41a013cce085985a5143d1422ece6e928c517f55c3f710d88166ba5c0`
- `src\data\data_raw\CMeEE_dev.json`：`bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c`
- `src\data\data_raw\CMeEE_test.json`：`dca7d9b7cf820947bba68e14557908fdc63e9286add1b4bc34f78f245e0c7ecf`
- `src\data\data_raw\CMeEE_train.json`：`984f0bea3932c8c5afd96d9cccac4895b87ec77f57bd3e825d832a17943cd5d3`
- `src\docs\duplicate_annotation_review_checked.csv.xlsx`：`6903ec4e6bfc76ae3dc61d019a78c0238f01cd2e159b831fdb2392dd4c597478`

原始下标均为零基。清洗版因排除记录而改变行序号，`record_index_map.csv` 保存清洗行号到原始行号的对应关系。
重叠与嵌套未因其存在而移除；原始 JSON、人工工作簿、已有报告均不修改。未转换 BIO。
复现：在仓库根目录运行 `python -X utf8 -B src\scripts\clean_medical_ner_data.py`。
可用 `--input`、`--review`、`--plan`、`--output-dir` 指定路径；不同数据/新备注必须先更新并核实计划，哈希不匹配会停止。
重跑会更新输出文件；不要在自动生成日志中直接写入唯一的人工复核记录。
