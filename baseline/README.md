# 成员 2：基线代码与实验记录

本目录保存项目 B 中成员 2 负责的非大模型基线代码。当前实现覆盖：数据检查、实体跨度处理、词典基线、字符级小模型、统一评测、实验汇总和错误分析。

## 一、刚刚完成了什么

我们以 CMeEE 医疗命名实体识别数据集为对象，完成了下面的工作：

1. 检查训练集、开发集和测试集的结构、实体位置、实体文本和标签。
2. 确认训练集和开发集中的实体标注均能和原文字符位置对应。
3. 发现并记录 CMeEE 中存在重叠/嵌套实体的事实。
4. 编写实体跨度工具，统一使用 `start_idx`、`end_idx`、`type`、`entity` 格式。
5. 实现精确词典基线。
6. 实现最长匹配词典基线。
7. 实现最低出现次数和最短实体长度的参数实验。
8. 实现字符级 LinearSVC 小模型。
9. 实现统一的实体级和字符级评测。
10. 自动保存预测结果、指标、错误样本和实验汇总表。
11. 自动分析最佳词典基线中的漏检、误报、边界错误和类型错误。

成员 1 的原始数据和清洗后数据没有被修改。

## 二、数据使用约定

| 用途 | 文件 | 说明 |
|---|---|---|
| 训练、建立词典 | `data_cleaned/CMeEE_train_cleaned_final.json` | 14,984 条文本、61,743 个实体 |
| 有金标评估 | `data_raw/CMeEE_dev.json` | 5,000 条文本、20,300 个实体 |
| 无答案预测展示 | `data_raw/CMeEE_test.json` | 3,000 条文本，`entities` 全为空 |

测试集的空 `entities` 不表示文本没有实体，因此不能用它计算 Precision、Recall 或 F1。

实体位置采用 Python 字符索引，`end_idx` 是包含式：

```python
text[start_idx:end_idx + 1] == entity
```

当前标签共有九类：

```text
dis, sym, dru, equ, pro, bod, ite, mic, dep
```

## 三、代码文件说明

| 文件 | 作用 | 主要输入 | 主要输出 |
|---|---|---|---|
| `data_check.py` | 检查数据结构、实体索引、实体文本、标签和重叠实体 | train/dev/test JSON | 终端报告或 `data_check_report.txt` |
| `span_utils.py` | 统一实体格式、实体排序、文本匹配和标签冲突处理 | 实体字典、文本 | 标准化实体或匹配位置 |
| `metrics.py` | 统一计算实体级和 token 级指标 | 金标记录、预测记录 | Precision、Recall、F1、错误记录 |
| `dictionary_baseline.py` | 从训练集建词典并运行词典预测 | train、dev | 预测文件和指标文件 |
| `small_model.py` | 训练字符级 LinearSVC 小模型 | train、dev | 小模型预测和指标 |
| `evaluate_saved.py` | 对已有预测文件重新使用统一评测 | 金标文件、预测文件 | 更新后的指标文件 |
| `error_analysis.py` | 自动区分漏检、误报、边界错误和类型错误 | 金标文件、预测文件 | 错误分析 JSON |
| `summarize_experiments.py` | 汇总所有 `*_metrics.json` | 指标文件 | `experiment_summary.md` |
| `README.md` | 记录代码用途、运行命令和实验结果 | — | 本说明文档 |

## 四、输出文件说明

| 文件类型 | 内容 |
|---|---|
| `*_predictions.json` | 每条 dev 文本的预测实体，保留原始跨度格式 |
| `*_metrics.json` | 实体级、token 级和按类别统计的指标 |
| `*_errors.json` | 每条预测中的漏检和误报详情 |
| `dictionary_best_error_analysis.json` | 最佳词典基线的错误类别和代表样例 |
| `experiment_summary.md` | 所有已运行实验的比较表 |
| `data_check_report.txt` | 第一步数据检查报告 |

## 五、运行命令

所有命令都在仓库根目录执行。请把 `<REPO_ROOT>` 替换成你电脑上克隆本项目的实际目录；README 不依赖任何特定电脑路径。

```powershell
cd "<REPO_ROOT>"
```

### 1. 检查数据

```powershell
python baseline/data_check.py --report outputs/data_check_report.txt
```

### 2. 运行词典基线

```powershell
python baseline/dictionary_baseline.py --method exact
python baseline/dictionary_baseline.py --method longest
python baseline/dictionary_baseline.py --method longest --min-count 2
python baseline/dictionary_baseline.py --method longest --min-length 2
python baseline/dictionary_baseline.py --method longest --min-length 3
```

其中 `exact` 保留所有匹配；`longest` 优先保留最长实体；`--min-count` 过滤低频短语；`--min-length` 过滤过短实体。

### 3. 运行字符级小模型

```powershell
python baseline/small_model.py
```

该模型使用当前字符、前后字符、数字/英文特征和词典字符特征训练 LinearSVC。

### 4. 重新评测已有预测

```powershell
python baseline/evaluate_saved.py `
  --gold data_raw/CMeEE_dev.json `
  --predictions outputs/dictionary_longest_min_count_1_min_length_2_predictions.json `
  --output outputs/dictionary_best_metrics_recomputed.json
```

### 5. 生成错误分析和实验汇总

```powershell
python baseline/error_analysis.py `
  --gold data_raw/CMeEE_dev.json `
  --predictions outputs/dictionary_longest_min_count_1_min_length_2_predictions.json `
  --output outputs/dictionary_best_error_analysis.json

python baseline/summarize_experiments.py
```

## 六、已经冻结的评测协议

本项目采用 CMeEE/CBLUE 的任务口径：医疗命名实体识别的官方主要指标是 **实体级严格 Micro-F1**。CBLUE 的 CMeEE 说明和示例见：

- [CBLUE 官方仓库](https://github.com/CBLUEbenchmark/CBLUE)
- [CMeEE 数据说明与提交格式](https://huggingface.co/datasets/Aunderline/CMeEE)

### 1. 数据划分

| 数据 | 用途 | 是否计算指标 |
|---|---|---|
| `data_cleaned/CMeEE_train_cleaned_final.json` | 训练模型、建立词典、选择训练参数 | 不作为最终报告指标 |
| `data_raw/CMeEE_dev.json` | 本地开发、参数选择和方法比较 | 可以计算指标 |
| `data_raw/CMeEE_test.json` | 生成最终预测文件或展示结果 | 当前没有公开金标，不计算 P/R/F1 |

本地 test 的 `entities` 全为空，不能把空列表当作“没有实体”。因此当前仓库中的数值结果都应称为 **dev 集开发结果**，不能写成官方 test 集成绩。若之后取得带金标的测试集，应在所有方法和参数冻结后，只运行一次最终测试评估。

为了公平比较，所有方法必须使用同一版训练集、同一份 dev 集、同一份金标和同一套评测代码。不能让某个方法使用其他方法没有看到的 dev 标注来建词典或调参数。

### 2. 统一输出格式

内部预测文件与 CMeEE 原始格式保持一致：顶层是列表，每条记录保留原文 `text` 和预测实体列表 `entities`。每个实体统一使用下面四个字段：

```json
{
  "start_idx": 10,
  "end_idx": 11,
  "type": "dis",
  "entity": "肺炎"
}
```

规则如下：

- `start_idx` 和 `end_idx` 是从 0 开始的字符位置；
- `end_idx` 是包含式，实体文本必须满足 `text[start_idx:end_idx + 1] == entity`；
- `type` 必须是 CMeEE 的九类之一；
- `entity` 必须是原文对应切片；
- 预测记录必须与输入记录保持相同顺序；
- 评测时使用 `(record_index, start_idx, end_idx, type)` 作为实体键，不能只按实体文字匹配；
- 对外提交时沿用 CMeEE 要求的 `text`、`entities` 结构；实体字段至少包含 `start_idx`、`end_idx`、`type`，本项目内部保留 `entity` 便于审计。

这样，词典、LinearSVC、BERT 和 LLM 的结果都可以先转换到同一个 span 格式，再调用同一个评测脚本。

### 3. 主指标：实体级严格 Micro-F1

预测实体只有在下面三项全部相同才算正确：

```text
start_idx 相同
end_idx 相同
type 相同
```

然后在整个评估集上汇总：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
```

主结果报告总体 Micro Precision、Micro Recall、Micro F1，并同时报告九类实体各自的 P/R/F1。

### 4. token/字符级指标：补充指标

CMeEE 存在嵌套实体，例如同一段文字可以同时标注为 `bod` 和 `sym`。单层 BIO 或单层字符标签无法在一个位置同时保存多层实体，因此 token 指标不能替代官方实体级指标。

当前代码采用明确的可复现投影：同一文本中优先保留最长的非重叠实体，再逐字符计算 Accuracy、实体字符 Precision、Recall、F1。报告中必须把它写成：

> 基于最长非重叠实体投影的字符级补充指标。

不能把这个投影结果称为完整的嵌套实体 token 金标准。实体级主指标仍然直接使用原始 span 标注，保留所有重叠实体。

## 七、当前实验结果

结果来自 `data_raw/CMeEE_dev.json`：

| 方法 | Entity Precision | Entity Recall | Entity F1 | Token F1 | Token Accuracy |
|---|---:|---:|---:|---:|---:|
| 精确词典 | 0.1520 | 0.5415 | 0.2374 | 0.4977 | 0.7281 |
| 最长匹配词典 | 0.2875 | 0.4916 | 0.3628 | 0.4977 | 0.7281 |
| 最长匹配，实体长度≥2 | **0.3496** | **0.4796** | **0.4044** | **0.5060** | **0.7389** |
| 最长匹配，实体长度≥3 | 0.4610 | 0.3126 | 0.3725 | 0.4581 | 0.7392 |
| 最长匹配，出现次数≥2 | 0.3067 | 0.4433 | 0.3625 | 0.4703 | 0.7355 |
| 字符级 LinearSVC | 0.1161 | 0.3513 | 0.1745 | 0.4838 | 0.6976 |

当前最佳基线是“最长匹配词典 + 过滤单字实体”，实体级 F1 为 `0.4044`。过滤单字实体减少了误报；过滤到长度≥3 后召回下降过多，F1 反而下降。频次过滤没有超过长度过滤方案。

## 八、错误分析结果

最佳词典基线的错误分析文件为 `outputs/dictionary_best_error_analysis.json`：

| 错误类型 | 数量 |
|---|---:|
| 误报实体 | 22,465 |
| 漏检实体 | 14,914 |
| 边界错误 | 4,137 |
| 类型错误 | 1,249 |

主要问题是词典短语在新文本中产生误报、未登录实体漏检、嵌套实体造成边界冲突，以及同一短语的类型冲突。

## 九、重要限制

1. CMeEE 有重叠/嵌套实体，单层 BIO 或单层 CRF 无法无损表示全部标注。
2. 词典基线只能识别训练集中出现过的实体短语。
3. 当前字符级 LinearSVC 使用最长非重叠投影训练，不能完整学习嵌套实体。
4. 当前评估使用有金标的 dev 集；本地 test 没有公开答案，不能计算有效 F1。
5. 当前 dev 结果用于开发和方法比较，不能替代隐藏 test 的官方成绩。
6. 成员 3 的 BERT/LLM 结果需要转换为相同的 span 输出格式后，再使用 `metrics.py` 评测。

## 十、成员 2 的贡献证据

本目录中的数据检查脚本、实体跨度工具、词典基线、字符级小模型、统一评测模块、错误分析脚本、预测文件、指标文件和实验汇总表，构成成员 2 的代码与实验产出证据。
