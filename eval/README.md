# eval：成员4的评估整合层

`baseline/README.md` 是成员2确定的基线与评测标准，`baseline/` 和
`outputs/` 中的结果是权威产物。本目录只做评估整合，不改写成员2的代码和结果。

## 团队评估冻结 v1.1

- 冻结日期：2026-10-21
- 代码基线：`081adf9`
- 指标与输出格式：**不在此重复定义**，以 [baseline/README.md](../baseline/README.md) 第五节为准。
- 本文件只补充团队的数据切分、调参/定稿规则和版本记录。

成员2原始协议以完整 dev 作为开发集；团队保留该历史结果，同时新增
`eval_tune` / `eval_test` 作为正式留出流程。两套口径必须分表报告。

v1.1 的主要变化：

- 成员2新增了纯规则基线、词典跨度特征小模型、网格搜索、特征消融和嵌套损失分析；
- 完整 dev 上的最佳基线更新为“字符级 LinearSVC + 词典跨度特征”，实体级 F1 **0.4301**；
- `metrics.py` 新增 `char_multi_label_*` 多标签字符指标，保留全部嵌套实体；
- 复现环境固定到 `requirements.txt`。

### 数据版本（SHA-256）

| 文件 | 条数 / 实体数 | SHA-256 |
|---|---:|---|
| `data_cleaned/CMeEE_train_cleaned_final.json` | 14,984 / 61,743 | `e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc` |
| `data_raw/CMeEE_dev.json` | 5,000 / 20,300 | `bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c` |
| `data_eval/eval_tune.jsonl` | 2,500 / 9,840 | `ff1e9fdc685a4b405a55cfece540eceb7cba7bfdb70492693c51c6e42d0f584b` |
| `data_eval/eval_test.jsonl` | 2,500 / 10,460 | `ca65235e83f5ba055af7498b25ea55c376747a64ab6b1d4e5bffc0dfb8e62778` |

### 使用规则

1. 训练、建词典、训练小模型只使用训练集。
2. `eval_tune` 可以反复调参、选提示词、选模型和比较超参数。
3. `eval_test` 只在配置冻结后运行一次；跑完后不得再根据其结果修改提示词或参数。
4. 成员2的完整 dev 结果作为开发集历史结果单列：当前最佳是字符级
   LinearSVC + 词典跨度特征，实体级 F1 `0.4301`；词典基线最佳为 `0.4044`。
   团队最终留出结果使用 `eval_test`，两套数字不能混在同一列。
5. 所有方法使用同一份训练集、`eval_tune`、`eval_test` 和 `baseline/metrics.py`；
   预测文件按 `id` 对齐，或保持与金标相同的记录顺序。
6. 本地 `data_raw/CMeEE_test.json` 没有金标，不参与 P/R/F1 计算。

### 当前结果状态（2026-10-21）

| 方法 | split | 状态 | 实体级 F1 |
|---|---|---:|---:|
| 纯人工规则 | 完整 dev | 已完成 | 0.0465 |
| 最长匹配词典 + 过滤单字实体 | 完整 dev | 已完成 | 0.4044 |
| 最长匹配词典 + 过滤单字实体 | eval_tune | 已完成 | 0.4010 |
| 字符级 LinearSVC + 词典跨度特征 | 完整 dev | 已完成 | **0.4301** |
| 字符级 LinearSVC + 词典跨度特征 | eval_tune / eval_test | 待补 wrapper | — |
| LLM / BERT | eval_tune / eval_test | 待成员3实现 | — |

`0.4301` 是完整 dev 结果，不能直接和词典的 tune `0.4010` 放在同一列；
在成员4补好小模型的 split wrapper 后，需要重新在 tune/test 上跑。

### 变更控制

- 数据文件、切分方式或指标定义发生变化时，协议版本继续递增并记录原因；
  所有方法必须使用新协议重跑，不能只重跑一个模型。
- 仅修复代码 bug 且不改变指标定义时，记录 commit、影响范围并重新生成受影响结果。

### 版本记录

| 版本 | 日期 | 变化 |
|---|---|---|
| v1.0 | 2026-10-19 | 首次冻结 seed 42 的 tune/test 切分和数据哈希 |
| v1.1 | 2026-10-21 | 同步成员2新基线 `0.4301`、`char_multi_label_*` 指标和复现环境 |

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

### 环境与复现

成员2的模型环境记录在仓库根目录 `requirements.txt`：

```text
Python 3.12.7
numpy 2.3.4
scikit-learn 1.9.0
```

其他评估脚本只依赖 Python 标准库。若开发机版本不同，必须在报告中记录实际
Python、numpy、scikit-learn 版本；`0.4301` 可能因此出现小幅波动。

## 命令

先安装基线依赖并复现成员2的最终基线：

```powershell
pip install -r requirements.txt
python baseline/run_final_baselines.py --config configs/final_baseline.json
```

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
- 主指标是实体级严格 Micro-F1；补充指标必须区分为：
  - `token_*`：“基于最长非重叠实体投影的字符级补充指标”；
  - `char_multi_label_*`：“基于保留全部实体（含嵌套）的逐标签多标签字符级补充指标”。
