# 复现范围与报告口径

最后整理：2026-10-29。

本文给出最终报告、摘要和演示中应统一使用的复现范围、核心数字与限制，避免把
“checkpoint 推理复现”写成“BERT 训练复现”，或把非盲留出集写成盲测。

## 默认一键复现实际做了什么

`python src/reproduce.py` 会：

1. 检查 Python、PyTorch、Transformers 和 scikit-learn 环境；
2. 下载或读取冻结数据包，校验压缩包和四份数据文件 SHA-256；
3. 校验 LinearSVC 冻结配置及 BERT 已提交证据；
4. 使用 seed 42 从训练集重新训练字符级 LinearSVC + 词典跨度特征；
5. 在完整 2,500 条 `eval_test` 上评估 LinearSVC；
6. 加载固定 MacBERT 词典融合 checkpoint 并执行完整 `eval_test` 推理；
7. 分别生成两个模型的 metrics、predictions、errors 和自动错误分析；
8. 对比实际 F1 与冻结期望值，两项均通过才返回 PASS。

**默认不重新训练 BERT。** 五个历史 BERT 配置的分数来自已提交证据，仅用于对照。
如果需要重新训练，使用 README 中“从零重训历史 BERT 配置”的可选命令。

## 数据切分限制

- dev：5,000 条，20,300 个实体；
- `eval_tune`：2,500 条，9,840 个实体；
- `eval_test`：2,500 条，10,460 个实体；
- 前两者来自 dev 的 seed 42 切分。

成员 2 早期的完整 dev 调参已经覆盖当前 `eval_test` 的样本。因此 `eval_test` 是
**冻结留出集（非盲）**，不是从未观察过的盲测集，不能与公开盲测榜单直接比较。

## 主结果和比较边界

| 方法 | eval_test 实体级 P | R | F1 |
|---|---:|---:|---:|
| 字符 LinearSVC + 词典特征 | 0.4301 | 0.4278 | 0.4290 |
| MacBERT + GlobalPointer + 词典特征 | 0.6770 | 0.6632 | **0.6700** |

BERT 相对 LinearSVC 的绝对 F1 差约为 `+0.2411`，相对提升约 `+56.2%`。该差距同时
包含预训练、span 建模和词典特征三个因素，不能全部归因于“BERT 更强”。

词典融合模型相对无词典的 5 轮 warmup 模型：

```text
0.670047 - 0.667023 = +0.003025
相对提升约 +0.45%
```

该比较只有一个 seed。按 eval_test 的记录做 10,000 次 paired bootstrap：

```text
95% CI = [-0.0006491813, +0.0066428184]
p      = 0.1122
```

区间包含 0，未达到 0.05 水平的统计显著。因此应表述为“点估计为正但未达显著”，
不能写成“显著提升”。证据文件为 `src/outputs/bert_gp/dict_gain_bootstrap.json`。

复现命令：

```powershell
python src/eval/bootstrap_dict_gain.py `
  --gold src/data/data_eval/eval_test.jsonl `
  --dict-predictions <dict_predictions.jsonl> `
  --warmup-predictions <warmup_predictions.jsonl> `
  --output src/outputs/bert_gp/dict_gain_bootstrap.json `
  --bootstrap-samples 10000 --seed 42
```

## 错误分析

一键复现会自动生成：

```text
small_model/test/error_analysis.json
small_model/test/error_analysis.md
bert_gp/error_analysis.json
bert_gp/error_analysis.md
```

四类错误为：

- 完全漏检：无同标签预测跨度与金标重叠；
- 完全误报：预测实体没有对应金标；
- 边界错误：类型正确，但边界不同；
- 类型错误：起止跨度一致，但类型不同。

本次参考运行中的错误事件数量。边界和类型错误按一条“金标—预测”配对计 1 个事件：

| 模型 | 漏检 | 误报 | 边界 | 类型 | 错误事件合计 | 金标错误实体 | 预测错误实体 | 错误实体合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 字符 LinearSVC | 3,375 | 3,318 | 2,064 | 546 | 9,303 | 5,985 | 5,928 | 11,913 |
| MacBERT + 词典特征 | 2,339 | 2,125 | 683 | 501 | 5,648 | 3,523 | 3,309 | 6,832 |

配对恒等式：

```text
漏检 + 误报 + 2 × (边界 + 类型) = 金标错误实体 + 预测错误实体
```

LinearSVC 的错误计数是参考机器结果；BLAS、sklearn 或浮点实现不同时可能波动数个
实体。BERT 固定 checkpoint 推理应保持逐位一致。

具体样例、类型混淆、分类别漏检/误报/边界统计见自动生成的 Markdown/JSON。

## 覆盖与运行时间

- eval_test 有 `8 / 10,460` 个金标实体无法精确映射到 tokenizer token，占 0.076%；
- 本机 GPU 完整流程约 82—165 秒；
- 评审 CPU 环境完整流程约 274.6 秒；
- BERT 在 CPU 上较慢，但固定 checkpoint 推理结果可逐位一致。

## 离线执行

checkpoint 推理不需要预训练基座权重。若 Hugging Face 无法访问：

1. 准备只含 `config.json` 和 tokenizer 文件的普通目录；
2. 设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；
3. 将数据和 checkpoint 通过 `--data-package`、`--bert-checkpoint` 传入；
4. 用 `--model-dir` 指向本地 config/tokenizer 目录。
