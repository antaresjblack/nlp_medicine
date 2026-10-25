# BERT + GlobalPointer

独立模块，不改共享数据与LLM结果。需要 medical-ner 环境：torch 2.9.0+cu128、transformers 4.57.6、numpy 1.26.4。首次运行从 Hugging Face 下载 google-bert/bert-base-chinese 或 hfl/chinese-macbert-base（需要网络）。这是跨度多标签模型，保留全部可对齐的嵌套/重叠标签，非单层BIO。

```powershell
conda activate medical-ner
cd D:\desktop_programming\nlp\nlp_medicine
# 首先验证少量数据的分词映射，只下载tokenizer，不训练
python bert_gp/train.py --run-name data_check --train-limit 64 --tune-limit 32 --check-data
# 小规模检查，不是正式成绩
python bert_gp/train.py --run-name smoke --train-limit 128 --tune-limit 64 --epochs 1
```

## 冻结的四个模型

清理后保留四个正式模型，test 分数来自 `outputs/bert_gp/test_four_models_summary.json`（阈值 0，严格实体级 F1）：

| run-name | 编码器 | 轮数 | 学习率调度 | test F1 |
|---|---|---|---|---|
| full_v1 | google-bert/bert-base-chinese | 3 | constant（无预热） | 0.638 |
| macbert_base_v1 | hfl/chinese-macbert-base | 3 | constant（无预热） | 0.657 |
| macbert_warmup_v1 | hfl/chinese-macbert-base | 3 | linear + warmup 0.1 | 0.662 |
| macbert_warmup_5ep_v1 | hfl/chinese-macbert-base | 5 | linear + warmup 0.1 | 0.667 |

其余超参统一：max_length 256、overlap 64、batch 2、梯度累积 8、lr 2e-5、seed 42、混合精度、梯度检查点、阈值 0。

## 复现流程（不上传权重）

课程提交不包含权重（best.pt 约 400MB/个），复现分两条路：

**A. 复现成绩**：直接读取已提交的 `outputs/bert_gp/<run-name>/metrics.json`（tune）与 `outputs/bert_gp/test_<run-name>/metrics.json`（test），以及汇总 `outputs/bert_gp/test_four_models_summary.json`。这些是记录在案的证据，无需权重、无需重算。

**B. 复现训练**：从零重训。脚本从 Hugging Face 下载预训练编码器，在共享训练集 `data_cleaned/CMeEE_train_cleaned_final.json` 上用 seed 42 训练，重新生成 best.pt。四条命令（`full_v1`/`macbert_base_v1` 的 manifest 里 schedule 记录为 null，是当时代码尚无 `--schedule` 参数；当前默认 `--schedule constant` 即等价的无预热行为）：

```powershell
# 1) 原 BERT 3轮
python bert_gp/train.py --run-name full_v1 --model google-bert/bert-base-chinese --epochs 3
# 2) MacBERT 3轮
python bert_gp/train.py --run-name macbert_base_v1 --model hfl/chinese-macbert-base --epochs 3
# 3) MacBERT 3轮 + 预热衰减
python bert_gp/train.py --run-name macbert_warmup_v1 --model hfl/chinese-macbert-base --epochs 3 --schedule linear --warmup-ratio 0.1
# 4) MacBERT 5轮 + 预热衰减
python bert_gp/train.py --run-name macbert_warmup_5ep_v1 --model hfl/chinese-macbert-base --epochs 5 --schedule linear --warmup-ratio 0.1
```

同名目录拒绝覆盖，重跑前改名或删除旧目录。最终 test 推理（冻结后，不据结果反向调参）：

```powershell
python bert_gp/train.py --run-name test_full_v1 --checkpoint outputs/bert_gp/full_v1/best.pt --final
```

相同种子不保证跨硬件、CUDA 版本、依赖版本完全一致，重训可能与提交的 metrics.json 有小数点级差异。以提交的 `metrics.json` / `test_four_models_summary.json` 为正式成绩，重训仅作可复现性验证。

## 实现与输出

默认长度256、batch=2、梯度累积8、混合精度、梯度检查点、学习率2e-5、阈值0。显存不足先降batch为1。不得承诺某个成绩。CPU可运行但非常慢。

fast tokenizer offset_mapping 将原文含尾字符跨度转换为token边界；预测映射回原文含尾索引。滑动窗口重叠64 tokens，合并按原记录/跨度/类别去重。无效位置与反向跨度屏蔽。无法精确对齐、超过窗口的实体计入manifest覆盖损失，不删除共享金标；评估对原始完整金标进行。检查coverage，不能隐藏未覆盖实体。

outputs/bert_gp/<run-name>/保存manifest（数据哈希、版本、覆盖统计）、training.jsonl、best.pt、predictions.jsonl、metrics.json、errors.json。复用baseline/metrics.py，记录ID，不另定义评分。模型权重不放课程提交zip，不提交数据或缓存。训练集选取、超参数与实验结果分析由小组独立决定，声明AI代码辅助。

从检查点推理时保存训练时模型名、长度、重叠与阈值，不允许悄悄改动：

```powershell
# tune推理
python bert_gp/train.py --run-name full_v1_tune --checkpoint outputs/bert_gp/full_v1/best.pt
# 最终test，参数冻结后才使用，不根据结果反向调参
python bert_gp/train.py --run-name full_v1_test --checkpoint outputs/bert_gp/full_v1/best.pt --final
```

已有运行目录拒绝覆盖。脚本不支持训练中断恢复，best.pt仅作推理/最优检查点；中断后重新训练用新运行名。模型下载来源需固定缓存版本并保留manifest；torch检查点仅加载自己生成的可信文件。当前实现使用标准旋转位置编码GlobalPointer和多标签交叉熵，无CRF、无LoRA。
