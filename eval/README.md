# eval：成员4的评估整合层

`baseline/README.md` 是成员2确定的基线与评测标准，`baseline/` 和
`outputs/` 中的结果是权威产物。本目录只做评估整合，不改写成员2的代码和结果。

## 数据布局

沿用成员2的根目录布局：

| 文件 | 用途 |
|---|---|
| `data_cleaned/CMeEE_train_cleaned_final.json` | 训练、建词典 |
| `data_raw/CMeEE_dev.json` | 完整开发集，保留成员2的原始实验口径 |
| `data_eval/eval_tune.jsonl` | 2,500 条，调参用 |
| `data_eval/eval_test.jsonl` | 2,500 条，定稿时只用一次 |

`data_eval/` 由 `data_raw/CMeEE_dev.json` 按 seed 42 切分而来。它不替换
成员2的完整 dev 结果，只用于需要留出集的补充比较。

## 命令

重建冻结评估集（已存在时会拒绝覆盖）：

```powershell
python eval/make_eval_split.py
```

在 `eval_tune` 上跑成员2的词典基线：

```powershell
python eval/run_dictionary_split.py --method longest --min-length 2 --split tune
```

输出文件：

- `outputs/eval_split/metrics/tune_dictionary_..._metrics.json`
- `outputs/eval_split/predictions/tune_dictionary_..._predictions.jsonl`
- `outputs/eval_split/errors/tune_dictionary_..._errors.json`

对其他模型保存的预测统一评分：

```powershell
python eval/evaluate_predictions.py `
  --gold data_eval/eval_tune.jsonl `
  --predictions outputs/eval_split/predictions/your_model_tune_predictions.jsonl `
  --output outputs/eval_split/metrics/your_model_tune_metrics.json
```

最终定稿时才运行 `--split test`，并且只跑一次。预测文件与金标记录顺序一致，
或两侧都带唯一 `id`；评分沿用 `baseline/metrics.py` 的严格实体级 span 匹配。

## 口径说明

- 成员2的完整 dev 表继续作为基线主表。
- `eval_tune` 只用于选择参数；`eval_test` 只用于最终留出评估。
- 不要把两套口径的分数混在同一张“最终成绩”表里。
- 本地 `data_raw/CMeEE_test.json` 没有金标，不能计算 P/R/F1。
