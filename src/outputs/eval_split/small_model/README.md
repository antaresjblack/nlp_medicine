# 小模型冻结 split 结果

本目录由 `src/eval/run_small_model_split.py` 生成，不覆盖成员2完整 dev 的历史结果。

## 当前冻结配置

```text
模型：字符级 LinearSVC + 词典跨度特征
C：0.1
class_weight：none
seed：42
dict_min_count：1
dict_min_length：2
```

当前结果：

| split | Entity P | Entity R | Entity F1 |
|---|---:|---:|---:|
| eval_tune | 0.4276 | 0.4275 | 0.4276 |
| eval_test | 0.4301 | 0.4278 | 0.4290 |

历史完整 dev 的最佳结果仍为 `0.4301`，两套结果必须分表报告。

## tune 目录

每轮 tune 使用独立时间戳目录：

```text
tune/runs/<run-id>/
├── candidates/
│   ├── <candidate>_predictions.jsonl
│   ├── <candidate>_metrics.json
│   └── <candidate>_errors.json
├── candidate_summary.json
├── candidate_summary.md
├── selected_config.json
└── tune_run.log
```

8 个候选各产生 3 个文件，因此候选文件共 24 个。

其中：

- predictions 是该候选在 2,500 条 tune 上的完整预测；
- metrics 是统一指标；
- errors 是逐条错误；
- candidate summary 是候选排名；
- selected_config 是冻结的最佳配置。

## test 目录

```text
test/
├── test_predictions.jsonl
├── test_metrics.json
├── test_errors.json
├── test_run.log
└── test_manifest.json
```

test 已经完成一次冻结配置评估。脚本会拒绝再次覆盖现有 test 文件；如需重新计算
指标，应使用保存的预测调用 `src/eval/evaluate_predictions.py`，不要重新训练和预测。

## 公平性限制

成员2早期使用完整 5,000 条 dev 比较过模型参数，而当前 test 是该 dev 的子集。
因此当前结果应表述为“冻结配置后的补充重新确认”，不是完全盲测成绩。
