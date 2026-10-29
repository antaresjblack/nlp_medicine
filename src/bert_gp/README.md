# BERT + GlobalPointer

独立模块，不改共享数据与LLM结果。项目完整环境由仓库根目录
`requirements.txt` 定义；已验证 `medical-ner-unified`：Python 3.10.20、
torch 2.9.0+cu128、transformers 4.57.6、numpy 1.26.4、scipy 1.13.1、
scikit-learn 1.7.2。默认复现从源码内 `pretrained_config/` 读取 tokenizer/config，
不需要连接 Hugging Face。
这是跨度多标签模型，保留全部可对齐的嵌套/重叠标签，非单层BIO。

使用完整微调 checkpoint 做推理时，模型结构从 `AutoConfig` 创建，随后加载 checkpoint
中的编码器参数，因此不会下载或使用基础模型权重。默认复现自动读取
`bert_gp/pretrained_config/` 中的 config/tokenizer；也可以通过 `--model-dir <目录>` 覆盖。

最新主方法是“MacBERT + GlobalPointer + 门控词典特征”，正式 checkpoint 可从
GitHub Release 获取：

```text
https://github.com/antaresjblack/nlp_medicine/releases/download/macbert-dict-5ep-v1/macbert_dict_5ep_v1.pt
```

SHA-256：

```text
595EB45CBFD920F93E175668A7FE50A11F8B949222E882BB2F7FD4C91AE9B1D8
```

```powershell
Set-Location '<仓库根目录>'
python -m pip install -r requirements.txt
# 默认真实复现人工规则、词典、LinearSVC 和最新 BERT
python src/reproduce.py
# 首先验证少量数据的分词映射，只下载tokenizer，不训练
python src/bert_gp/train.py --run-name data_check --train-limit 64 --tune-limit 32 --check-data
# 小规模检查，不是正式成绩
python src/bert_gp/train.py --run-name smoke --train-limit 128 --tune-limit 64 --epochs 1
```

完全离线推理时，默认直接使用源码内 `pretrained_config/`；把 data 和 checkpoint
通过参数传入后执行：

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python src/bert_gp/train_dict.py `
  --run-name offline_check `
  --checkpoint D:\offline\macbert_dict_5ep_v1.pt `
  --model-dir src\bert_gp\pretrained_config `
  --final --output-dir D:\offline\bert
```

checkpoint 模式不会再请求基座权重；本地目录只包含约 0.4 MB 的 config/tokenizer 文件。
从零训练时不传 `--model-dir`，仍从预训练模型正常加载全部权重。

## 基础对比与最新主方法

四个基础配置的 test 分数来自 `src/outputs/bert_gp/test_four_models_summary.json`；
最新主方法的证据来自 `src/outputs/bert_gp/test_macbert_dict_5ep_v1/`。所有分数均为
阈值 0、严格实体级 F1：

| run-name | 编码器 | 轮数 | 学习率调度 | test F1 |
|---|---|---|---|---|
| full_v1 | google-bert/bert-base-chinese | 3 | constant（无预热） | 0.638 |
| macbert_base_v1 | hfl/chinese-macbert-base | 3 | constant（无预热） | 0.657 |
| macbert_warmup_v1 | hfl/chinese-macbert-base | 3 | linear + warmup 0.1 | 0.662 |
| macbert_warmup_5ep_v1 | hfl/chinese-macbert-base | 5 | linear + warmup 0.1 | 0.667 |
| **macbert_dict_5ep_v1** | hfl/chinese-macbert-base + 词典特征 | 5 | linear + warmup 0.1 | **0.670** |

其余超参统一：max_length 256、overlap 64、batch 2、梯度累积 8、lr 2e-5、seed 42、混合精度、梯度检查点、阈值 0。

## 复现流程（权重通过 Release 提供）

课程提交包不包含权重，最新 `best.pt` 约 414 MB，通过 Release 提供。复现分两条路：

**A. 默认复现成绩**：在仓库根目录运行 `python src/reproduce.py`。脚本会执行纯人工规则、
最长匹配词典，重新训练字符级 LinearSVC，并下载或复用最新 BERT checkpoint，真实执行
`eval_test` 推理；任一冻结结果检查失败时整体不会返回 PASS。

**B. 复现训练**：从开源预训练编码器重新微调，不是随机初始化预训练 BERT。脚本从 Hugging Face 下载预训练编码器，在共享训练集 `src/data/data_cleaned/CMeEE_train_cleaned_final.json` 上用 seed 42 训练，重新生成 best.pt。五条命令（`full_v1`/`macbert_base_v1` 的 manifest 里 schedule 记录为 null，是当时代码尚无 `--schedule` 参数；当前默认 `--schedule constant` 即等价的无预热行为）：

```powershell
# 1) 原 BERT 3轮
python src/bert_gp/train.py --run-name full_v1 --model google-bert/bert-base-chinese --epochs 3
# 2) MacBERT 3轮
python src/bert_gp/train.py --run-name macbert_base_v1 --model hfl/chinese-macbert-base --epochs 3
# 3) MacBERT 3轮 + 预热衰减
python src/bert_gp/train.py --run-name macbert_warmup_v1 --model hfl/chinese-macbert-base --epochs 3 --schedule linear --warmup-ratio 0.1
# 4) MacBERT 5轮 + 预热衰减
python src/bert_gp/train.py --run-name macbert_warmup_5ep_v1 --model hfl/chinese-macbert-base --epochs 5 --schedule linear --warmup-ratio 0.1
# 5) 最新主方法：MacBERT + 门控词典特征
python src/bert_gp/train_dict.py --run-name macbert_dict_5ep_v1 --epochs 5
```

同名目录拒绝覆盖，重跑前改名或删除旧目录。最终 test 推理（冻结后，不据结果反向调参）：

```powershell
python src/bert_gp/train.py --run-name test_full_v1 --checkpoint src/outputs/bert_gp/full_v1/best.pt --final
# 最新主方法
python src/bert_gp/train_dict.py --run-name test_macbert_dict_5ep_v1 `
  --checkpoint src/outputs/bert_gp/macbert_dict_5ep_v1/best.pt --final
```

相同种子不保证跨硬件、CUDA 版本、依赖版本完全一致，重训可能与提交的 metrics.json 有小数点级差异。以提交的 `metrics.json` / `test_four_models_summary.json` 为正式成绩，重训仅作可复现性验证。

## 实现与输出

默认长度256、batch=2、梯度累积8、混合精度、梯度检查点、学习率2e-5、阈值0。显存不足先降batch为1。不得承诺某个成绩。CPU可运行但非常慢。

fast tokenizer offset_mapping 将原文含尾字符跨度转换为token边界；预测映射回原文含尾索引。滑动窗口重叠64 tokens，合并按原记录/跨度/类别去重。无效位置与反向跨度屏蔽。无法精确对齐、超过窗口的实体计入manifest覆盖损失，不删除共享金标；评估对原始完整金标进行。检查coverage，不能隐藏未覆盖实体。

`src/outputs/bert_gp/<run-name>/` 保存 manifest（数据哈希、版本、覆盖统计）、training.jsonl、best.pt、predictions.jsonl、metrics.json、errors.json。复用 `src/baseline/metrics.py`，记录 ID，不另定义评分。模型权重不放课程提交 zip，不提交数据或缓存。训练集选取、超参数与实验结果分析由小组独立决定，声明 AI 代码辅助。

从检查点推理时保存训练时模型名、长度、重叠与阈值，不允许悄悄改动。
词典融合模型也支持 `--output-dir`，便于复现入口把结果写入独立时间戳目录：

```powershell
# tune推理
python src/bert_gp/train.py --run-name full_v1_tune --checkpoint src/outputs/bert_gp/full_v1/best.pt
# 最终test，参数冻结后才使用，不根据结果反向调参
python src/bert_gp/train.py --run-name full_v1_test --checkpoint src/outputs/bert_gp/full_v1/best.pt --final
```

```powershell
python src/bert_gp/train_dict.py --run-name test_macbert_dict_5ep_v1 `
  --checkpoint src/outputs/bert_gp/macbert_dict_5ep_v1/best.pt --final `
  --output-dir src/reproduction_outputs/manual-bert
```

已有运行目录拒绝覆盖。脚本不支持训练中断恢复，best.pt仅作推理/最优检查点；
中断后重新训练用新运行名。模型下载来源需固定缓存版本并保留manifest；
torch检查点仅加载自己生成的可信文件。当前实现使用标准旋转位置编码GlobalPointer、
门控词典特征和多标签交叉熵，无CRF、无LoRA。
