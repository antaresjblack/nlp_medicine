# 复现范围与报告口径

最后整理：2026-10-29。

本文给出最终报告、摘要和演示中应统一使用的复现范围、核心数字与限制，避免把
“checkpoint 推理复现”写成“BERT 训练复现”，或把非盲留出集写成盲测。

## 默认一键复现实际做了什么

`python src/reproduce.py` 会：

1. 检查 Python、PyTorch、Transformers 和 scikit-learn 环境；
2. 下载或读取冻结数据包，校验压缩包和四份数据文件 SHA-256；
3. 校验 LinearSVC 冻结配置及 BERT 已提交证据；
4. 在完整 2,500 条 `eval_test` 上执行 9 条纯人工规则；
5. 从训练集构建最长匹配词典并在同一测试集上评估；
6. 使用 seed 42 从训练集重新训练字符级 LinearSVC + 词典跨度特征；
7. 加载固定 MacBERT 词典融合 checkpoint 并执行完整 `eval_test` 推理；
8. 为四个方法生成 metrics、predictions 和 errors，并为 LinearSVC/BERT 生成自动错误分析；
9. 对比四个方法的实际 F1 与冻结期望值，全部通过且 BERT 推理成功才返回 PASS。

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
| 纯人工规则（9 条正则） | 0.0965 | 0.0330 | 0.0492 |
| 最长匹配词典（过滤单字） | 0.3540 | 0.4806 | **0.4077** |
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
# 1) 先用无词典 5 轮 checkpoint 生成 warmup 预测
python src/bert_gp/train.py `
  --run-name warmup_for_bootstrap `
  --checkpoint <macbert_warmup_5ep_v1.pt> `
  --model-dir <config和tokenizer目录> `
  --final --output-dir <warmup输出目录>

# 2) <warmup输出目录>/predictions.jsonl 是 bootstrap 所需 warmup 预测。
#    dict 预测来自默认复现输出：<复现输出目录>/bert_gp/predictions.jsonl。
python src/eval/bootstrap_dict_gain.py `
  --gold src/data/data_eval/eval_test.jsonl `
  --dict-predictions <复现输出目录>/bert_gp/predictions.jsonl `
  --warmup-predictions <warmup输出目录>/predictions.jsonl `
  --output src/outputs/bert_gp/dict_gain_bootstrap.json `
  --bootstrap-samples 10000 --seed 42
```

## 错误分析

一键复现会自动生成：

```text
pure_rules/errors/test_pure_rules_errors.json
dictionary/errors/test_dictionary_longest_min_count_1_min_length_2_errors.json
small_model/test/error_analysis.json
small_model/test/error_analysis.md
bert_gp/error_analysis.json
bert_gp/error_analysis.md
```

人工规则和词典基线生成原始 errors；LinearSVC 和 MacBERT 还会生成四类错误的
自动分类报告。

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
- 本次四方法流程在本机离线缓存环境实测约 101 秒；
- 评审 CPU 环境的总耗时会更长，具体以本次 `reproduction_report.txt` 为准；
- CPU 是必需硬件；GPU/CUDA 是可选的加速条件，代码检测不到 CUDA 时会回退到 CPU；
- BERT 在 CPU 上较慢，但固定 checkpoint 推理结果可逐位一致。

## 离线执行

checkpoint 推理不需要预训练基座权重。若 Hugging Face 无法访问：

1. 设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；
2. 将冻结数据包和 checkpoint 通过 `--data-package`、`--bert-checkpoint` 传入；
3. 默认自动使用源码包内的 `src/bert_gp/pretrained_config/`；
4. 如需使用其他 tokenizer/config，再用 `--model-dir` 覆盖。

数据包和 BERT checkpoint 默认先尝试 GitHub Release 下载地址；若失败，脚本会自动切换到
对应 Asset API，再执行同一个 SHA-256 校验。源码包已经包含 config/tokenizer，
因此默认 BERT checkpoint 推理不需要 Hugging Face 在线缓存。损坏缓存会被丢弃并
重新下载；每次下载先写入唯一系统临时文件并支持断点续传，校验后在目标盘内暂存并
原子替换，失败后自动清理。因此项目路径含中文时，不需要由 curl 直接写项目目录。若本机仍限制 curl
或系统临时目录，可将项目放到纯 ASCII 路径，或通过参数使用本地数据和权重。
