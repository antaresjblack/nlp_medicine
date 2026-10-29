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

该比较只有一个 seed，未做 paired bootstrap 或置信区间，不能表述为“显著提升”。
更稳妥的写法是“方向一致的小幅增益，显著性未检验”。

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

本次参考运行中的错误实体数量：

| 模型 | 漏检 | 误报 | 边界 | 类型 |
|---|---:|---:|---:|---:|
| 字符 LinearSVC | 3,375 | 3,318 | 2,064 | 546 |
| MacBERT + 词典特征 | 2,339 | 2,125 | 683 | 501 |

具体样例和类型混淆统计见自动生成的 Markdown/JSON。不同 sklearn 版本可能使
LinearSVC 的少量数字变化，BERT checkpoint 推理应保持完全一致。

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

## 报告建议措辞

> 本次复现重新训练了字符级 LinearSVC，并加载已发布的 MacBERT 词典融合 checkpoint
> 在冻结 eval_test 上推理。BERT 的训练过程未在本次复现中重跑；五个历史变体的分数
> 来自仓库已提交证据，仅用于对照，不作为本次运行结果。

> eval_test 是由 dev 按 seed 42 切分的冻结留出集。由于早期完整 dev 调参已覆盖该子集，
> 本文不将其称为盲测，只将其用于配置冻结后的补充确认。

> 相对无词典 5 轮模型，词典融合模型 F1 提升 +0.0030。该差异在单 seed 下不足以
> 单独证明统计显著性，因此本文将其表述为方向一致的小幅增益。
