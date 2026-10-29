# eval：成员4的评估整合层

`src/baseline/README.md` 是成员2确定的基线与评测标准，`src/baseline/` 和
`src/outputs/` 中的结果是权威产物。本目录只做评估整合，不改写成员2的代码和结果。

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
- 项目完整复现环境固定到根目录 `requirements.txt`，同时覆盖传统基线和 BERT。

### 数据版本（SHA-256）

| 文件 | 条数 / 实体数 | SHA-256 |
|---|---:|---|
| `src/data/data_cleaned/CMeEE_train_cleaned_final.json` | 14,984 / 61,743 | `e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc` |
| `src/data/data_raw/CMeEE_dev.json` | 5,000 / 20,300 | `bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c` |
| `src/data/data_eval/eval_tune.jsonl` | 2,500 / 9,840 | `ff1e9fdc685a4b405a55cfece540eceb7cba7bfdb70492693c51c6e42d0f584b` |
| `src/data/data_eval/eval_test.jsonl` | 2,500 / 10,460 | `ca65235e83f5ba055af7498b25ea55c376747a64ab6b1d4e5bffc0dfb8e62778` |

### 使用规则

1. 训练、建词典、训练小模型只使用训练集。
2. `eval_tune` 可以反复调参、选提示词、选模型和比较超参数。
3. 每个最终方法只能在配置冻结后运行一次 `eval_test`；跑完后不得再根据其结果
   修改提示词或参数。不同模型可以各自提交一次冻结后的 test 预测，但同一配置
   不得反复尝试。
4. 成员2的完整 dev 结果作为开发集历史结果单列：当前最佳是字符级
   LinearSVC + 词典跨度特征，实体级 F1 `0.4301`；词典基线最佳为 `0.4044`。
   团队最终留出结果使用 `eval_test`，两套数字不能混在同一列。
5. 所有方法使用同一份训练集、`eval_tune`、`eval_test` 和 `src/baseline/metrics.py`；
   预测文件按 `id` 对齐，或保持与金标相同的记录顺序。
6. 本地 `src/data/data_raw/CMeEE_test.json` 没有金标，不参与 P/R/F1 计算。

### 当前结果状态（2026-10-28）

| 方法 | split | 状态 | 实体级 F1 |
|---|---|---:|---:|
| 纯人工规则 | 完整 dev | 已完成 | 0.0465 |
| 纯人工规则 | eval_tune | 已完成 | **0.0436** |
| 纯人工规则 | eval_test | 已完成 | **0.0492** |
| 最长匹配词典 + 过滤单字实体 | 完整 dev | 已完成 | 0.4044 |
| 最长匹配词典 + 过滤单字实体 | eval_tune | 已完成 | 0.4010 |
| 最长匹配词典 + 过滤单字实体 | eval_test | 已完成 | **0.4077** |
| 字符级 LinearSVC + 词典跨度特征 | 完整 dev | 已完成 | **0.4301** |
| 字符级 LinearSVC + 词典跨度特征（`C=0.1`, balanced） | eval_tune | 已完成 | 0.4269 |
| 字符级 LinearSVC + 词典跨度特征（`C=0.1`, none） | eval_tune | 已完成 | **0.4276** |
| 字符级 LinearSVC + 词典跨度特征（`C=0.1`, none） | eval_test | 已完成，仅运行一次 | **0.4290** |
| BERT + GlobalPointer | eval_tune / eval_test | 已完成四组基础配置 | test 最佳 **0.6670** |
| MacBERT + GlobalPointer + 词典特征 | eval_tune / eval_test | 最新主方法，完整评估 | tune **0.6634**；test **0.6700** |
| DeepSeek 提示抽取 | eval_tune 前 100 条 | 探索性试跑 | 子集 0.4782，不与完整 test 混排 |

`0.4301` 是完整 dev 结果，不能直接和词典的 tune `0.4010` 放在同一列；
小模型已在 `eval_tune` 上重新比较候选，并以 `span_c0.1_none` 冻结后完成一次
`eval_test` 评估。选中配置和 tune 排名见
`src/outputs/eval_split/small_model/tune/runs/20261022T141507+0800/`。
词典 test 已在冻结配置 `longest/min_count=1/min_length=2` 下完成一次评估，
结果保存在 `src/outputs/eval_split/metrics/test_dictionary_longest_min_count_1_min_length_2_metrics.json`。

> 注意：成员2早期的完整 dev 调参已经覆盖了当前 `eval_test` 的 2,500 条样本。
> 因此该集合只能称为“冻结留出集（非盲）”，结果只能表述为“配置冻结后的补充
> 重新确认”，不能写成团队此前从未接触过 `eval_test` 的完全盲测成绩；报告中须
> 保留这一限制说明。

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
| `src/data/data_cleaned/CMeEE_train_cleaned_final.json` | 训练、建词典 |
| `src/data/data_raw/CMeEE_dev.json` | 完整开发集，保留成员2的原始实验口径 |
| `src/data/data_eval/eval_tune.jsonl` | 2,500 条，调参用 |
| `src/data/data_eval/eval_test.jsonl` | 2,500 条，定稿时只用一次 |

`src/data/data_eval/` 由 `src/data/data_raw/CMeEE_dev.json` 按 seed 42 切分而来。它不替换
成员2的完整 dev 结果，只用于需要留出集的补充比较。

### 环境与复现

完整复现环境记录在仓库根目录 `requirements.txt`：

```text
Python 3.10.20
torch 2.9.0+cu128
transformers 4.57.6
numpy 1.26.4
scipy 1.13.1
scikit-learn 1.7.2
```

传统基线也可以使用仅含 NumPy/scikit-learn 的 `requirements.txt`，但默认
`src/reproduce.py` 会真实运行 BERT，因此必须使用统一环境。若开发机版本不同，
必须在报告中记录实际 Python、PyTorch、Transformers、numpy、scipy 和
scikit-learn 版本；`0.4301` 可能因此出现小幅波动。

## 命令

以下命令均在项目根目录 `nlp_medicine/` 执行。

先安装基线依赖并复现成员2的最终基线：

```powershell
python -m pip install -r requirements.txt
python src/baseline/run_final_baselines.py --config src/configs/final_baseline.json
```

重建冻结评估集（已存在时会拒绝覆盖）：

```powershell
python src/eval/make_eval_split.py
```

在 `eval_tune` 上跑成员2的词典基线：

```powershell
python src/eval/run_dictionary_split.py --method longest --min-length 2 --split tune
```

输出文件：

- `src/outputs/eval_split/metrics/tune_dictionary_..._metrics.json`
- `src/outputs/eval_split/predictions/tune_dictionary_..._predictions.jsonl`
- `src/outputs/eval_split/errors/tune_dictionary_..._errors.json`

纯规则基线通过单独的 split 包装入口运行；规则本身保持冻结：

```powershell
python src/eval/run_pure_rule_split.py --split tune
python src/eval/run_pure_rule_split.py --split test
```

输出文件：

- `src/outputs/eval_split/metrics/{tune,test}_pure_rules_metrics.json`
- `src/outputs/eval_split/predictions/{tune,test}_pure_rules_predictions.jsonl`
- `src/outputs/eval_split/errors/{tune,test}_pure_rules_errors.json`

成员2的小模型使用独立追加层，不覆盖原来的完整 dev 输出：

```powershell
# 只检查候选和输入，不训练
python src/eval/run_small_model_split.py --stage tune --dry-run

# 在 eval_tune 上比较全部候选，生成 candidate_summary 和 selected_config
python src/eval/run_small_model_split.py --stage tune

# 配置冻结后只运行一次；输出已存在时脚本会拒绝重跑
python src/eval/run_small_model_split.py `
  --stage test `
  --selected-config src/outputs/eval_split/small_model/tune/runs/<run-id>/selected_config.json
```

候选列表在 `src/configs/small_model_split_candidates.json`。tune 结果保存在
`src/outputs/eval_split/small_model/tune/runs/<run-id>/`；test 结果保存在
`src/outputs/eval_split/small_model/test/`。每条预测都保留原始 `id`，可直接用于统一评分。

### 成员3接入合同

成员3开始工作前必须满足以下约定：

1. 模型、提示词、few-shot 示例和超参数只能在 `eval_tune` 上选择；
2. few-shot 示例只能从最终训练集选择，不能从 `eval_tune` 或 `eval_test` 选择；
3. 输出必须是保留 `id` 的 JSONL，实体字段为
   `{start_idx, end_idx, type, entity}`，索引采用包含式终点；
4. 配置一旦用于 `eval_test`，不得根据 test 成绩修改提示词、模型或参数；
5. 预测必须先通过原文切片校验，再交给 `src/baseline/metrics.py` 评分；
6. 同时记录模型名称、调用时间、成本、重试次数、解析失败数和实际输出文件；
7. 成员3提供预测，成员4统一评分和汇总，成员3不自行定义另一套 F1。

建议成员3交付以下文件：

```text
src/outputs/eval_split/llm/tune_predictions.jsonl
src/outputs/eval_split/llm/tune_metrics.json
src/outputs/eval_split/llm/test_predictions.jsonl
src/outputs/eval_split/llm/test_metrics.json
src/outputs/eval_split/llm/run_config.json
src/outputs/eval_split/llm/cost_latency.json
```

其中 `run_config.json` 不得包含 API 密钥。

对其他模型保存的预测统一评分：

```powershell
python src/eval/evaluate_predictions.py `
  --gold src/data/data_eval/eval_tune.jsonl `
  --predictions src/outputs/eval_split/predictions/your_model_tune_predictions.jsonl `
  --output src/outputs/eval_split/metrics/your_model_tune_metrics.json
```

最终定稿时才运行 `--split test`，并且只跑一次。预测文件与金标记录顺序一致，
或两侧都带唯一 `id`；评分沿用 `src/baseline/metrics.py` 的严格实体级 span 匹配。

## 自动错误分析

默认 `python src/reproduce.py` 会在 LinearSVC 和 BERT 的输出目录中分别生成：

```text
error_analysis.json
error_analysis.md
```

错误按互斥类别统计：

- 完全漏检：金标实体没有同标签预测跨度与之重叠；
- 完全误报：预测实体没有对应金标；
- 边界错误：类型正确，但起点或终点不同；
- 类型错误：起点和终点相同，但实体类型不同。

每类最多保留 3 个真实样例，并记录类型混淆、分类别漏检和误报数量。单独运行：

```powershell
python src/eval/analyze_errors.py `
  --errors <errors.json> `
  --gold src/data/data_eval/eval_test.jsonl `
  --output-json error_analysis.json `
  --output-md error_analysis.md `
  --method "模型名称"
```

## 口径说明

- 成员2的完整 dev 表继续作为基线主表。
- `eval_tune` 只用于选择参数；`eval_test` 只用于最终留出评估。
- 不要把两套口径的分数混在同一张“最终成绩”表里。
- 本地 `src/data/data_raw/CMeEE_test.json` 没有金标，不能计算 P/R/F1。
- 主指标是实体级严格 Micro-F1；补充指标必须区分为：
  - `token_*`：“基于最长非重叠实体投影的字符级补充指标”；
  - `char_multi_label_*`：“基于保留全部实体（含嵌套）的逐标签多标签字符级补充指标”。
