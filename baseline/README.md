# 成员 2：非大模型基线（交接文档）

> 本目录是成员 2 在 CMeEE 医疗命名实体识别任务上完成的**全部非大模型基线**（词典、纯规则、字符级小模型）及其评测与实验设施。本文档面向接手者：5 分钟了解"有什么、怎么跑、注意什么"。**每一步的原理、被覆盖的中间结果与不足清单，见《成员2_基线工作过程.md》。**

## 1. 最终成绩（dev 集，实体级严格 micro-F1）

| 方法 | Entity F1 | 一句话说明 |
|---|---:|---|
| 字符级 LinearSVC + 词典跨度特征 | **0.4301** | 最佳方法 |
| 最长匹配词典 + 过滤单字实体 | 0.4044 | 原最佳，词典方法参照 |
| 纯人工正则规则 | 0.0465 | 不读训练集的下界参照 |

小模型演进：0.1745 →（第一轮调参）0.2281 →（词典跨度特征）0.4201 →（第二轮调参）**0.4301**。

> ⚠️ 0.4301 是真实成绩。嵌套分析报告中的 0.4320 是"假设未恢复内层全部预测正确"的**理论上限**，不是成绩，引用时必须带假设限定语。

## 2. 数据文件约定

| 用途 | 文件 |
|---|---|
| 训练、建词典、建特征 | `data_cleaned/CMeEE_train_cleaned_final.json`（14,984 条 / 61,743 实体） |
| 有金标评估与调参 | `data_raw/CMeEE_dev.json`（5,000 条 / 20,300 实体） |
| 最终预测展示 | `data_raw/CMeEE_test.json`（3,000 条，`entities` 全空——**不代表没有实体，不能算 P/R/F1**） |

实体索引：`start_idx` 从 0 起、`end_idx` 包含式，恒有 `text[start_idx:end_idx+1] == entity`。九类标签：dis, sym, dru, equ, pro, bod, ite, mic, dep。

## 3. 文件与用途（按需查表）

| 文件 | 干什么 | 什么时候跑 |
|---|---|---|
| `run_final_baselines.py` | **一键复现最终配置**：规则 → 词典 → 小模型 → 3 个错误分析 → 汇总表 | 复现时跑一次；开发时加 `--skip-small-model` |
| `small_model.py` | 字符级 LinearSVC 小模型，默认参数即最终配置 | 需要小模型结果时 |
| `dictionary_baseline.py` | 词典基线（exact / longest / min-count / min-length） | 调参实验时 |
| `pure_rule_baseline.py` | 纯规则基线（9 条人工正则，不读训练集） | 调参实验时 |
| `dict_span_features.py` | `DictSpanAnnotator`：6 项词典跨度特征（被 small_model 复用，成员 3 也可用作 BERT 附加特征） | 不单独跑；自带 demo 可验证 |
| `small_model_grid_search.py` | 小模型超参数网格搜索（两轮：无跨度特征 8 组 / 有跨度特征 4 组），结果存 `outputs/grid/` | 重做调参时 |
| `ablation_dict_features.py` | 逐组移除跨度特征的消融实验，报告存 `outputs/ablation/` | 重做消融时 |
| `nested_loss_analysis.py` | 单层 BIO 训练投影的信息损失分析（几秒跑完） | 报告引用该结论时跑一次 |
| `metrics.py` / `span_utils.py` | 统一评测与 span 工具（被所有脚本 import） | 不单独跑 |
| `evaluate_saved.py` | 重评已有预测；`--batch-dir` 批量刷新历史指标（保留元数据） | 改了评测代码之后 |
| `error_analysis.py` / `summarize_experiments.py` | 错误分类（漏检/误报/边界/类型）与实验汇总表 | runner 自动跑 |
| `data_check.py` | 数据只读检查（结构/索引/重叠实体/嵌套多标签位置统计） | 接手数据时 |
| `requirements.txt` | 环境依赖（其余脚本仅标准库） | 新机器装一次 |

## 4. 快速运行

所有命令在**仓库根目录**执行。

```powershell
# 4.1 环境
pip install -r requirements.txt          # 验证环境：Python 3.12.7 / numpy 2.3.4 / scikit-learn 1.9.0

# 4.2 一键复现最终配置（推荐）
python baseline/run_final_baselines.py --config configs/final_baseline.json

# 4.3 单个方法
python baseline/small_model.py                                   # 小模型，默认=最终配置
python baseline/dictionary_baseline.py --method longest --min-length 2
python baseline/pure_rule_baseline.py

# 4.4 测试集预测（协议：参数冻结后只执行一次）
python baseline/small_model.py --eval-file data_raw/CMeEE_test.json
python baseline/dictionary_baseline.py --method longest --min-length 2 --eval-file data_raw/CMeEE_test.json
python baseline/pure_rule_baseline.py --eval-file data_raw/CMeEE_test.json
# 无金标文件自动进入 predict-only：只写 *_test_predictions.json，不计算指标

# 4.5 评测代码改动后刷新全部历史指标（无需重跑模型）
python baseline/evaluate_saved.py --gold data_raw/CMeEE_dev.json --batch-dir outputs
```

## 5. 冻结的评测协议（后续编码必须遵守）

1. **主指标 = 实体级严格 micro-F1**（起点+终点+类型全对才算对），CBLUE 官方口径；
2. **补充指标两组**，报告中必须使用固定措辞：
   - 投影口径 → "基于最长非重叠实体投影的字符级补充指标"（`token_*`）；
   - 多标签口径 → "基于保留全部实体（含嵌套）的逐标签多标签字符级补充指标"（`char_multi_label_*`）；
3. **数据规则**：词典/特征只从 train 构建；dev 只用于调参和比较（调参须在报告中如实披露）；test 无金标，冻结后只跑一次；
4. **输出格式**：实体 = `{start_idx, end_idx, type, entity}`，与输入记录同序；
5. **公平比较**：所有方法必须经同一套 `metrics.py` 评测，不得另起口径。

## 6. 后续编码注意事项（给成员 3 / 4 与接手者）

- **成员 3（BERT/LLM）接入**：输出转成第 5.4 条的 span 格式 → 用 `span_utils.validate_record` 校验文本切片 → `metrics.py` 评测。`DictSpanAnnotator` 可直接作为 BERT 的词典特征源（做法：把 6 项跨度特征拼进 BERT 表示或输入层）。
- **已知环境坑**：本机 scikit-learn 环境下 `DictVectorizer` 输出 int64 稀疏索引，LinearSVC 会拒绝——**任何新模型脚本**对向量化结果都要过 `small_model.sparse_int32` 转换（该函数已用显式 int32 构造，可直接 import）。
- **嵌套实体**：不要为适配单层 BIO 去改共享金标。单层结构的嵌套损失已量化：投影仅丢弃 6.0% 金标实体、其中 91.8% 已被词典跨度特征隐式恢复、理论增益上限 <0.2 个 F1 点（`nested_loss_analysis.py` 可复现）。
- **方法天花板**：规则/词典/字符级线性模型的实体级 F1 上限约 0.5；公开资料中 0.6+ 来自预训练模型微调（成员 3 方向）。
- **不要动**：`data_raw/`、`data_cleaned/` 中的数据文件；本协议；`outputs/grid/` 中的旧结果（是被覆盖配置的存档，用于追溯）。

## 7. 已解决的问题（简记）

- 一键复现 runner 曾因参数不匹配在运行中崩溃；小模型特征提取曾是小时级（修复后 0.9 秒）；
- 投影 token 指标曾系统性低估 Precision（把命中嵌套内层的正确预测误记为误报）→ 补充多标签字符指标；
- 小模型曾虚报 3 倍实体（F1 0.1745）→ 两轮调参 + 词典跨度特征 → **0.4301**，超过词典基线；
- 复现链路缺口（test 预测能力、一键复现、环境冻结、指标批量刷新）→ 全部补齐。

**未做/遗留**：词典匹配性能优化（Aho-Corasick）；CRF 进阶实验（预估 +3~5 点，如需超 0.45 再做）；test 集最终预测（按协议等全部参数冻结后执行一次）。

## 8. 报告索引

| 内容 | 位置 |
|---|---|
| 主结果对比表 | `outputs/experiment_summary.md` |
| 两轮网格调参（含被覆盖旧配置存档） | `outputs/grid/` |
| 跨度特征消融报告 | `outputs/ablation/ablation_report.md` |
| 单层投影信息损失分析 | `outputs/nested_loss_analysis_report.md` |
| 三个方法的错误分析 | `outputs/*_error_analysis.json` |
| 完整过程与不足记录 | 《成员2_基线工作过程.md》（本目录） |
