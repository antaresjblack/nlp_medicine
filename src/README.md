# src 目录说明

`src/` 用于成员3新增 LLM/BERT 模型代码。当前基础模型仍位于 `baseline/`。

建议结构：

```text
src/
├── llm/
│   ├── run_tune.py
│   ├── run_test.py
│   ├── parser.py
│   └── prompts/
└── bert/
    ├── train.py
    └── predict.py
```

## 接入要求

- 训练只使用 `data_cleaned/CMeEE_train_cleaned_final.json`；
- 模型、提示词和参数只在 `data_eval/eval_tune.jsonl` 上比较；
- few-shot 示例只能从训练集选择；
- 预测保留 `id`，实体格式与成员2一致；
- 生成 span 后用 `baseline/span_utils.validate_record` 校验；
- 统一使用 `baseline/metrics.py` 评分；
- 不得覆盖 `baseline/` 或已有 `outputs/` 历史文件；
- API 密钥只能从环境变量读取。

详细的输出合同见 `eval/README.md` 的“成员3接入合同”一节。
