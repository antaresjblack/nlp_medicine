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
8. 实现只依赖人工正则规则的纯规则基线。
9. 实现字符级 LinearSVC 小模型。
10. 实现统一的实体级和字符级评测。
11. 自动保存预测结果、指标、错误样本和实验汇总表。
12. 自动分析最佳词典基线和纯规则基线中的漏检、误报、边界错误和类型错误。
13. 为词典、纯规则和小模型统一增加 `--eval-file`，可在无金标的测试集上直接生成最终预测（自动进入 predict-only 模式，不计算指标）。
14. 一键复现脚本纳入错误分析步骤，config 驱动完整流程：规则 → 词典 → 小模型 → 错误分析 → 实验汇总。
15. 小模型支持 `--seed`、`--C`、`--class-weight` 参数；词典字符特征改为预计算字符集合，特征提取从小时级降到秒级。
16. 新增 `requirements.txt`，记录已验证的 numpy/scikit-learn 版本。
17. 补充 token 级指标：新增逐类 token 指标（`token_per_label`）和保留嵌套实体的多标签字符级指标（`char_multi_label_*`），后者不依赖最长投影，能反映嵌套层的漏检；`evaluate_saved.py` 新增 `--batch-dir` 批量刷新历史指标文件。
18. 小模型超参数在 dev 上网格搜索（C × class_weight，共 8 组，1 组因训练耗时约 100 分钟且趋势无竞争力而跳过）：去掉 balanced 权重后实体虚报减半（61,408 → 31,651），实体级 F1 从 0.1745 提升到 0.2281。网格结果存于 `outputs/grid/`，最终配置 C=1.0、class_weight=none、seed=42。

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
| `dictionary_baseline.py` | 从训练集建词典并运行词典预测 | train、dev 或 test（`--eval-file`） | 预测文件和指标文件；无金标时仅预测文件 |
| `pure_rule_baseline.py` | 只使用人工正则规则运行纯规则预测，不读取训练词典 | dev 或 test（`--eval-file`） | 规则预测文件和指标文件；无金标时仅预测文件 |
| `small_model.py` | 训练字符级 LinearSVC 小模型，支持 `--seed`、`--C`、`--class-weight` | train、dev 或 test（`--eval-file`） | 小模型预测和指标；无金标时仅预测文件 |
| `small_model_grid_search.py` | dev 上网格搜索小模型超参数，结果存 `outputs/grid/` | train、dev | 每组组合的指标文件和网格汇总 |
| `evaluate_saved.py` | 对已有预测文件重新使用统一评测；`--batch-dir` 可批量刷新目录内全部历史指标文件 | 金标文件、预测文件 | 更新后的指标文件 |
| `error_analysis.py` | 自动区分漏检、误报、边界错误和类型错误 | 金标文件、预测文件 | 错误分析 JSON |
| `summarize_experiments.py` | 汇总所有 `*_metrics.json` | 指标文件 | `experiment_summary.md` |
| `run_final_baselines.py` | 按 `configs/final_baseline.json` 一键重跑规则、词典、小模型、错误分析和汇总 | 配置文件 | 全部最终输出 |
| `requirements.txt` | 模型环境依赖（numpy、scikit-learn），其余脚本只用标准库 | — | — |
| `README.md` | 记录代码用途、运行命令和实验结果 | — | 本说明文档 |

## 四、输出文件说明

| 文件类型 | 内容 |
|---|---|
| `*_predictions.json` | 每条 dev 文本的预测实体，保留原始跨度格式 |
| `*_test_predictions.json` | 对无金标测试集的最终预测，predict-only 模式产出，**没有对应指标文件** |
| `*_metrics.json` | 实体级、投影/多标签字符级（`token_*`/`char_multi_label_*`）和按类别统计的指标 |
| `*_errors.json` | 每条预测中的漏检和误报详情 |
| `dictionary_best_error_analysis.json` | 最佳词典基线的错误类别和代表样例 |
| `grid/` | 小模型超参数网格搜索结果（每组指标 + 汇总 md/json），属调参实验，不进入主汇总表 |
| `experiment_summary.md` | 所有已运行实验的比较表 |
| `data_check_report.txt` | 第一步数据检查报告 |

## 五、运行命令

所有命令都在仓库根目录执行。请把 `<REPO_ROOT>` 替换成你电脑上克隆本项目的实际目录；README 不依赖任何特定电脑路径。

```powershell
cd "<REPO_ROOT>"
```

### 0. 安装依赖

只有 `small_model.py` 需要第三方库；验证环境为 Python 3.12.7、numpy 2.3.4、scikit-learn 1.9.0：

```powershell
pip install -r requirements.txt
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
python baseline/dictionary_baseline.py --method longest --min-count 3
python baseline/dictionary_baseline.py --method longest --min-count 5
python baseline/dictionary_baseline.py --method longest --min-length 2
python baseline/dictionary_baseline.py --method longest --min-length 3
```

其中 `exact` 保留所有匹配；`longest` 优先保留最长实体；`--min-count` 过滤低频短语；`--min-length` 过滤过短实体。

对无金标的测试集生成最终预测（自动进入 predict-only 模式，只写预测文件、不计算指标）：

```powershell
python baseline/dictionary_baseline.py --method longest --min-length 2 --eval-file data_raw/CMeEE_test.json
```

### 3. 运行纯规则基线

```powershell
python baseline/pure_rule_baseline.py
python baseline/pure_rule_baseline.py --eval-file data_raw/CMeEE_test.json
```

纯规则基线只使用 `pure_rule_baseline.py` 中人工编写的正则规则，不读取训练集实体，也不建立词典。当前规则覆盖药物后缀、疾病后缀、症状词、检验项目、医疗程序、身体部位、设备、微生物和科室等模式。第二条命令对测试集生成最终预测。

### 4. 运行字符级小模型

```powershell
python baseline/small_model.py
python baseline/small_model.py --eval-file data_raw/CMeEE_test.json
```

该模型使用当前字符、前后字符、数字/英文特征和词典字符特征训练 LinearSVC。最终配置的超参数为 `--seed 42 --C 1.0 --class-weight none`（与默认值一致，来自第 4.1 节的 dev 网格调参）；第二条命令对测试集生成最终预测。

### 4.1 小模型超参数网格搜索（调参实验）

```powershell
python baseline/small_model_grid_search.py
```

在 dev 上遍历 `C ∈ {0.01, 0.1, 1.0, 10.0}` × `class_weight ∈ {balanced, none}`，结果写入 `outputs/grid/`（`small_model_grid_summary.md/json`），不进入主实验汇总表。已运行 7/8 组：`C=10.0/none` 因训练耗时约 100 分钟且同 C 下 balanced 已确认更差（F1 0.1739）、趋势无竞争力而跳过，在汇总表中标记。最终按 dev 上实体级 F1 最高选择 `C=1.0、class_weight=none`。此过程属于开发集调参，报告中已如实说明。

### 5. 重新评测已有预测

```powershell
python baseline/evaluate_saved.py `
  --gold data_raw/CMeEE_dev.json `
  --predictions outputs/dictionary_longest_min_count_1_min_length_2_predictions.json `
  --output outputs/dictionary_best_metrics_recomputed.json
```

### 6. 生成错误分析和实验汇总

错误分析已纳入第 7 节的一键复现脚本，单独运行命令如下（手工调参时使用）：

```powershell
python baseline/error_analysis.py `
  --gold data_raw/CMeEE_dev.json `
  --predictions outputs/dictionary_longest_min_count_1_min_length_2_predictions.json `
  --output outputs/dictionary_best_error_analysis.json

python baseline/summarize_experiments.py
```

### 7. 一键复现最终配置

按 `configs/final_baseline.json` 依次执行：纯规则 → 最佳词典 → 小模型 → 错误分析 → 实验汇总：

```powershell
python baseline/run_final_baselines.py --config configs/final_baseline.json
```

开发期间可用 `--skip-small-model` 跳过最慢的小模型步骤。测试集预测不属于该流程：按冻结协议，应在所有参数冻结后单独执行第 2、3、4 节的 `--eval-file data_raw/CMeEE_test.json` 命令，且只执行一次。

## 六、已经冻结的评测协议

本项目采用 CMeEE/CBLUE 的任务口径：医疗命名实体识别的官方主要指标是 **实体级严格 Micro-F1**。CBLUE 的 CMeEE 说明和示例见：

- [CBLUE 官方仓库](https://github.com/CBLUEbenchmark/CBLUE)
- [CMeEE 数据说明与提交格式](https://huggingface.co/datasets/Aunderline/CMeEE)

### 1. 数据划分

| 数据 | 用途 | 是否计算指标 |
|---|---|---|
| `data_cleaned/CMeEE_train_cleaned_final.json` | 训练模型、建立词典、选择训练参数 | 不作为最终报告指标 |
| `data_raw/CMeEE_dev.json` | 本地开发、参数选择和方法比较 | 可以计算指标 |
| `data_raw/CMeEE_test.json` | 生成最终预测文件或展示结果 | 当前没有公开金标，不计算 P/R/F1；用 `--eval-file` 指向它时自动 predict-only |

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

当前代码采用明确的可复现投影：同一文本中优先保留最长的非重叠实体，再逐字符计算 Accuracy、实体字符 Precision、Recall、F1。指标文件中还按九个类别分别报告 token 级 P/R/F1（`token_per_label`）。报告中必须把它写成：

> 基于最长非重叠实体投影的字符级补充指标。

不能把这个投影结果称为完整的嵌套实体 token 金标准。实体级主指标仍然直接使用原始 span 标注，保留所有重叠实体。

### 5. 多标签字符级指标：嵌套感知补充指标

最长投影会丢弃嵌套实体的内层标签，为了不让这个事实被隐藏，指标文件同时保留一组**不投影**的字符级补充指标（`char_multi_label_*`）：每个字符位置保留该位置上的全部实体标签集合——金标直接使用原始嵌套标注，预测同样处理，然后逐标签计算字符级 Precision、Recall、F1（多标签微平均），并额外报告逐位置标签集合完全一致的比例作为多标签字符准确率。例如金标同时标注 `bod`（腹部）和 `sym`（腹部疼痛）、预测只给出 `sym` 时，投影口径显示满分，而多标签口径会把 `bod` 记为漏检。报告中必须把它写成：

> 基于保留全部实体（含嵌套）的逐标签多标签字符级补充指标。

两组字符级补充指标并列：`token_*` 是投影口径，`char_multi_label_*` 是非投影口径；两者都不能替代官方实体级主指标。指标代码变更后，可用 `evaluate_saved.py --batch-dir outputs` 批量重算全部历史指标文件，无需重跑模型。

## 七、当前实验结果

结果来自 `data_raw/CMeEE_dev.json`：

| 方法 | Entity Precision | Entity Recall | Entity F1 | Token F1 | Token Accuracy | 多标签字符 F1 |
|---|---:|---:|---:|---:|---:|---:|
| 精确词典 | 0.1520 | 0.5415 | 0.2374 | 0.4977 | 0.7281 | 0.4815 |
| 最长匹配词典 | 0.2875 | 0.4916 | 0.3628 | 0.4977 | 0.7281 | 0.5116 |
| 最长匹配，实体长度≥2 ★ | **0.3496** | **0.4796** | **0.4044** | **0.5060** | **0.7389** | **0.5188** |
| 最长匹配，实体长度≥3 | 0.4610 | 0.3126 | 0.3725 | 0.4581 | 0.7392 | 0.4647 |
| 最长匹配，出现次数≥2 | 0.3067 | 0.4433 | 0.3625 | 0.4703 | 0.7355 | 0.4873 |
| 最长匹配，出现次数≥3 | 0.3162 | 0.4135 | 0.3584 | 0.4489 | 0.7345 | 0.4664 |
| 最长匹配，出现次数≥5 | 0.3217 | 0.3678 | 0.3432 | 0.4033 | 0.7234 | 0.4218 |
| 纯人工正则规则 ★ | 0.0908 | 0.0312 | 0.0465 | 0.2190 | 0.5972 | 0.2196 |
| 字符级 LinearSVC ★ | 0.1872 | 0.2918 | **0.2281** | 0.5111 | 0.7661 | 0.5178 |

★ 标注的是 `configs/final_baseline.json` 一键复现的最终配置（词典取长度≥2；小模型经 dev 网格调参取 C=1.0、class_weight=none、seed=42）。表中其余行是一次性调参实验，仅用于支撑方法选择，不属于最终复现范围。

当前最佳基线是”最长匹配词典 + 过滤单字实体”，实体级 F1 为 `0.4044`。过滤单字实体减少了误报；过滤到长度≥3 后召回下降过多，F1 反而下降。频次过滤（≥2、≥3、≥5）整体都不如长度过滤方案。

字符级 LinearSVC 经网格调参后实体级 F1 从 0.1745 提升到 0.2281：class_weight 从 balanced 改为 none 后实体虚报减半（61,408 → 31,651），Precision 翻倍（0.1161 → 0.1872），代价是 Recall 从 0.3513 降到 0.2918。小模型超参数在 dev 上选出（`outputs/grid/`），属于开发集调参；它仍然低于词典基线，与”简单统计模型不如词典”的结论一致。

## 八、错误分析结果

最佳词典基线的错误分析文件为 `outputs/dictionary_best_error_analysis.json`；纯规则基线的错误分析文件为 `outputs/pure_rules_error_analysis.json`。

最佳词典基线的错误分布：

| 错误类型 | 数量 |
|---|---:|
| 误报实体 | 22,465 |
| 漏检实体 | 14,914 |
| 边界错误 | 4,137 |
| 类型错误 | 1,249 |

主要问题是词典短语在新文本中产生误报、未登录实体漏检、嵌套实体造成边界冲突，以及同一短语的类型冲突。

纯规则基线的错误分布为误报 3,887、漏检 17,203、边界错误 2,900、类型错误 197，以漏检为主——规则只覆盖有限的词尾模式，覆盖不到的实体全部漏检。

## 九、重要限制

1. CMeEE 有重叠/嵌套实体，单层 BIO 或单层 CRF 无法无损表示全部标注。
2. 词典基线只能识别训练集中出现过的实体短语。
3. 当前字符级 LinearSVC 使用最长非重叠投影训练，不能完整学习嵌套实体。
4. 当前评估使用有金标的 dev 集；本地 test 没有公开答案，不能计算有效 F1。
5. 当前 dev 结果用于开发和方法比较，不能替代隐藏 test 的官方成绩。
6. 成员 3 的 BERT/LLM 结果需要转换为相同的 span 输出格式后，再使用 `metrics.py` 评测。

## 十、成员 2 的贡献证据

本目录中的数据检查脚本、实体跨度工具、词典基线、字符级小模型、统一评测模块、错误分析脚本、预测文件、指标文件和实验汇总表，构成成员 2 的代码与实验产出证据。
