# data_raw 目录说明

本目录保存下载得到的原始数据，正常情况下不应修改。

| 文件 | 规模 | 标签 | 用途 |
|---|---:|---|---|
| `CMeEE_train.json` | 15,000 条 | 有 | 原始训练数据与清洗溯源 |
| `CMeEE_dev.json` | 5,000 条 | 有 | 完整开发集和内部 split 的来源 |
| `CMeEE_test.json` | 3,000 条 | **没有** | 官方盲测文本，只用于生成预测 |

## 数据来源

本地三个文件来自 Hugging Face
[`Aunderline/CMeEE`](https://huggingface.co/datasets/Aunderline/CMeEE)，
原始任务属于 CBLUE 的 CMeEE 医疗命名实体识别评测。

当前文件 SHA-256：

```text
CMeEE_train.json  984f0bea3932c8c5afd96d9cccac4895b87ec77f57bd3e825d832a17943cd5d3
CMeEE_dev.json    bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c
CMeEE_test.json   dca7d9b7cf820947bba68e14557908fdc63e9286add1b4bc34f78f245e0c7ecf
```

2026-10-22 核对时，本地文件与 `Aunderline/CMeEE` revision
`bea81a21dc077bb21f6df5a0385410f4768b97bd` 的同名文件一致。实际历史下载
日期仍未知，不能由当前 revision 反推。

## 重要限制

- `CMeEE_test.json` 的 `entities: []` 是待填写的预测模板，不是负样本。
- 官方没有公开完整 3,000 条测试答案，不能本地计算官方 F1。
- 公开的 `example_gold.json` 只有 3 条格式示例，不能用于批量评分。
- 字符索引从 0 开始，`end_idx` 包含在实体中。
- 不要把官方盲测集和团队内部的 `src/data/data_eval/eval_test.jsonl` 混淆。

数据清洗和最终训练入口见 `src/data/data_cleaned/README.md`。
