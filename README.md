# 项目说明及复现指南

## 环境要求

| 组件         | 版本        |
| ------------ | ----------- |
| Python       | 3.10.20     |
| PyTorch      | 2.9.0+cu128 |
| Transformers | 4.57.6      |
| NumPy        | 1.26.4      |
| SciPy        | 1.13.1      |
| scikit-learn | 1.7.2       |
| CPU          | 可选        |
| CUDA         | 可用        |

## 依赖

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 是项目唯一完整环境，同时安装传统基线和 BERT 所需依赖。

## 数据集来源

数据来自 CBLUE CMeEE 医疗命名实体识别任务。课程提交包不包含数据；获取方式和
许可边界见 `src/docs/data_source.md`。冻结数据包发布在：

[cmeee-frozen-data-v1](https://github.com/antaresjblack/nlp_medicine/releases/tag/cmeee-frozen-data-v1)

压缩包 SHA-256：

```text
B9E48B664AB0135892E688DB2FFB25DF6D6A1A726FABDD6B5FD0A1611A41CFEF
```

默认复现会检查所需数据；如果源码包不含 `src/data/`，会自动下载约 4.45 MB 的
冻结数据包到 `src/.data_packages/`，校验压缩包后解压，并再次校验每个数据文件。
所需数据已齐全时不会重复下载；存在缺失文件时会解压固定数据包补齐。最终所需的目录布局为：

```text
src/data/data_raw/CMeEE_dev.json
src/data/data_raw/CMeEE_test.json
src/data/data_cleaned/CMeEE_train_cleaned_final.json
src/data/data_eval/eval_tune.jsonl
src/data/data_eval/eval_test.jsonl
```

## 复现命令

### 校验复现

只校验环境、数据哈希和冻结结果：

```powershell
python src/reproduce.py --check-only
```

### 完整复现

复现重新训练冻结的字符级 LinearSVC，并加载最新 MacBERT 词典融合
checkpoint，在 `eval_test` 上分别计算主结果：

```powershell
python src/reproduce.py
```

默认结果保存在项目源码目录下：

```text
<仓库根目录>/src/reproduction_outputs/<时间戳>/
```

默认会优先使用本地最新 checkpoint；若不存在，则自动从 GitHub Release 下载约
414 MB 的权重并校验 SHA-256。数据包和权重都可以显式指定本地路径或 HTTPS 地址：

```powershell
python src/reproduce.py --bert-checkpoint https://api.github.com/repos/antaresjblack/nlp_medicine/releases/assets/684231335
python src/reproduce.py --data-package <本地zip或HTTPS链接>
```

权重 SHA-256：

```text
595EB45CBFD920F93E175668A7FE50A11F8B949222E882BB2F7FD4C91AE9B1D8
```

## 随机种子

项目统一使用随机种子 `42`：

- `src/data/data_raw/CMeEE_dev.json` 的 5,000 条记录通过 `random.Random(42)` 打乱，
  切分为各 2,500 条的 `eval_tune` 和 `eval_test`。
- 字符级 LinearSVC 重新训练时使用冻结配置中的 `seed=42`。
- 最新 BERT checkpoint 的训练记录为 `seed=42`，同时设置 Python、NumPy、
  PyTorch 和 CUDA 的随机种子，并控制训练时窗口打乱顺序。
- 默认复现只加载最新 BERT checkpoint 做推理，不重新训练 BERT；推理阶段关闭
  dropout，因此 seed 42 主要用于追溯原训练实验。

固定随机种子用于保证同一份数据、配置和环境下得到可比较的结果，但不保证不同
GPU、CUDA、PyTorch 版本之间每一位数字完全相同。因此复现默认允许 F1 存在
`0.01` 以内的数值差异。

## 预期输出

以下为本机已验证的代表性输出。时间、绝对路径和小数末位可能略有变化；通过条件
是 LinearSVC 和 BERT 均执行成功，且实际 F1 与冻结结果的差值不超过容差。

### 默认完整复现

```text
[PASS] Environment, required data, hashes and frozen small-model configuration.
[PASS] Committed macbert_dict_5ep_v1 eval_test evidence: actual=0.670047, expected=0.670047, |delta|=0.000000, tolerance=0.000000

$ <当前环境的Python> eval/run_small_model_split.py --stage test --selected-config ... --output-dir ...

stage=test root=<仓库根目录>\src
selected_config=<仓库根目录>\src\outputs\eval_split\small_model\tune\runs\<run-id>\selected_config.json
train=<仓库根目录>\src\data\data_cleaned\CMeEE_train_cleaned_final.json
test=<仓库根目录>\src\data\data_eval\eval_test.jsonl
test: P=0.4302 R=0.4278 F1=0.4290 time=...
manifest=<输出目录>\small_model\test\test_manifest.json
Test metrics: <输出目录>\small_model\test\test_metrics.json
[PASS] Retrained LinearSVC eval_test: actual=0.428989, expected=0.428969, |delta|=0.000021, tolerance=0.010000
[PASS] BERT checkpoint SHA-256 verified: 595eb45cbfd920f93e175668a7fe50a11f8b949222e882bb2f7fd4c91ae9b1d8

$ <当前环境的Python> bert_gp/train_dict.py --run-name ... --checkpoint ... --final --output-dir ... --quiet

Completed: <输出目录>\bert_gp F1: 0.6700473292765381
[PASS] macbert_dict_5ep_v1 eval_test: actual=0.670047, expected=0.670047, |delta|=0.000000, tolerance=0.010000

==============================================================================
CMeEE 医疗命名实体识别：实验复现报告
==============================================================================
状态             : PASS
代码提交         : <git commit>
工作区           : dirty=<true/false>
运行时间         : <开始时间>  ->  <结束时间>
耗时             : <秒数> 秒
Python           : 3.10.20
PyTorch          : 2.9.0+cu128
Transformers     : 4.57.6
NumPy/SciPy/sklearn: 1.26.4 / 1.13.1 / 1.7.2
计算设备         : NVIDIA GeForce RTX 4060 Laptop GPU

[1] 数据范围与完整性
  train     14984 条 /  61743 实体
    sha256=e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc
  dev        5000 条 /  20300 实体
    sha256=bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c
  eval_tune  2500 条 /   9840 实体
    sha256=ff1e9fdc685a4b405a55cfece540eceb7cba7bfdb70492693c51c6e42d0f584b
  eval_test  2500 条 /  10460 实体
    sha256=ca65235e83f5ba055af7498b25ea55c376747a64ab6b1d4e5bffc0dfb8e62778
  评估口径         严格实体级 Micro-F1：起点、终点和标签全部匹配才算正确
  未使用           官方无标签 CMeEE_test；F1 仅来自冻结 eval_test

[2] 冻结配置与模型范围
  执行方式         LinearSVC 重新训练；BERT 使用固定 checkpoint 推理
  LinearSVC       C=0.1, class_weight=none, dict_span_features=True, seed=42
  BERT             hfl/chinese-macbert-base + GlobalPointer + 词典特征
  BERT checkpoint  outputs/bert_gp/macbert_dict_5ep_v1/best.pt
  checkpoint SHA
    595eb45cbfd920f93e175668a7fe50a11f8b949222e882bb2f7fd4c91ae9b1d8
  BERT 参数        threshold=0, max_length=256, overlap=64

[3] 冻结 eval_test 主结果
  char LinearSVC + dict features
    P=0.430164 R=0.427820 F1=0.428989
    Token-F1=0.516310 CharML-F1=0.527337
  MacBERT + GP + dict features
    P=0.677045 R=0.663193 F1=0.670047
    Token-F1=0.710578 CharML-F1=0.716031
  BERT - LinearSVC 绝对 F1 提升：+0.241058
  BERT 相对 LinearSVC F1 提升：+56.19%

[4] 九类实体严格 F1
  标签      LinearSVC      BERT       Delta
  dis        0.501160   0.764070  +0.262910
  ...
  dep        0.325581   0.549020  +0.223438

[5] 本次输出
  输出目录         src/reproduction_outputs/<时间戳>
  控制台报告       src/reproduction_outputs/<时间戳>/reproduction_report.txt
  LinearSVC        src/reproduction_outputs/<时间戳>/small_model/test/test_metrics.json
  BERT             src/reproduction_outputs/<时间戳>/bert_gp/metrics.json
  总摘要           src/reproduction_outputs/<时间戳>/reproduction_summary.json
==============================================================================

Reproduction status: PASS
Summary: <仓库根目录>\src\reproduction_outputs\<时间戳>\reproduction_summary.json
```

`[4]` 中实际会完整打印九个类别；上面的 `...` 只是为了缩短 README。

### 首次下载 BERT 权重

如果本地没有 `src/outputs/bert_gp/macbert_dict_5ep_v1/best.pt`，也没有下载缓存，
BERT 命令前会额外出现：

```text
[INFO] Downloading checkpoint: https://api.github.com/repos/antaresjblack/nlp_medicine/releases/assets/684231335
...curl 下载进度...
[PASS] Downloaded and verified checkpoint: .checkpoints/macbert_dict_5ep_v1.pt
```

已有下载缓存时显示：

```text
[INFO] Using cached checkpoint: .checkpoints/macbert_dict_5ep_v1.pt
```

### 首次下载冻结数据

源码包不含数据时，环境检查后会出现：

```text
[INFO] Missing frozen data files:
       - data/data_cleaned/CMeEE_train_cleaned_final.json
       - data/data_raw/CMeEE_dev.json
       - data/data_eval/eval_tune.jsonl
       - data/data_eval/eval_test.jsonl
[INFO] Downloading frozen data package: https://github.com/.../cmeee_frozen_data_v1.zip
[PASS] Downloaded and verified data package: .data_packages/cmeee_frozen_data_v1.zip
[PASS] Extracted frozen data package to: data
```

已有完整数据时只打印：

```text
[PASS] Frozen data files are already present; download was not needed.
```

### 只检查模式

执行 `python src/reproduce.py --check-only` 时不会训练或运行模型，预期输出：

```text
[PASS] Environment, required data, hashes and frozen small-model configuration.
[PASS] Committed macbert_dict_5ep_v1 eval_test evidence: actual=0.670047, expected=0.670047, |delta|=0.000000, tolerance=0.000000
[INFO] Check-only mode completed; no model was trained or executed.

Reproduction status: CHECK-ONLY-PASS
Summary: <仓库根目录>\src\reproduction_outputs\<时间戳>\reproduction_summary.json
```

默认流程结束后，控制台实验摘要同时保存为
`src/reproduction_outputs/<时间戳>/reproduction_report.txt`。完整指标、预测和错误
明细分别保存在同目录的 `small_model/` 与 `bert_gp/` 下。

## 可选复现内容

### 选择历史 BERT 权重复现

默认 `python src/reproduce.py` 使用最新词典融合模型。需要比较历史 BERT 配置时，
使用 `--bert-variant` 选择对应冻结权重；每次仍会重新训练 LinearSVC，并在同一份
`eval_test` 上评估所选 BERT：

```powershell
python src/reproduce.py --bert-variant full_v1
python src/reproduce.py --bert-variant macbert_base_v1
python src/reproduce.py --bert-variant macbert_warmup_v1
python src/reproduce.py --bert-variant macbert_warmup_5ep_v1
```

| 变体                    | 模型与训练配置                           | 冻结 eval_test F1 |
| ----------------------- | ---------------------------------------- | ----------------: |
| `dict_5ep`              | MacBERT + GlobalPointer + 词典特征，5 轮 |      **0.670047** |
| `full_v1`               | BERT-base-chinese，3 轮                  |          0.637723 |
| `macbert_base_v1`       | MacBERT，3 轮                            |          0.657026 |
| `macbert_warmup_v1`     | MacBERT，3 轮 + warmup                   |          0.662192 |
| `macbert_warmup_5ep_v1` | MacBERT，5 轮 + warmup                   |          0.667023 |

四份历史 BERT 权重已发布在
[bert-legacy-checkpoints-v1](https://github.com/antaresjblack/nlp_medicine/releases/tag/bert-legacy-checkpoints-v1)；
最新词典融合权重和旧 5 轮权重另有独立 Release。脚本会严格校验所选变体的 SHA-256。
也可以使用 `--bert-checkpoint <本地路径或HTTPS链接>` 覆盖下载源，但权重必须与
`--bert-variant` 指定的版本一致。

### 从零重训历史 BERT 配置

以下命令不使用已发布 checkpoint，而是从 Hugging Face 预训练编码器重新训练。
这些训练耗时明显长于默认复现，且跨硬件可能有少量指标波动：

```powershell
python src/bert_gp/train.py --run-name optional_full_v1 --model google-bert/bert-base-chinese --epochs 3 --seed 42
python src/bert_gp/train.py --run-name optional_macbert_base_v1 --model hfl/chinese-macbert-base --epochs 3 --schedule constant --seed 42
python src/bert_gp/train.py --run-name optional_macbert_warmup_v1 --model hfl/chinese-macbert-base --epochs 3 --schedule linear --warmup-ratio 0.1 --seed 42
python src/bert_gp/train.py --run-name optional_macbert_warmup_5ep_v1 --model hfl/chinese-macbert-base --epochs 5 --schedule linear --warmup-ratio 0.1 --seed 42
python src/bert_gp/train_dict.py --run-name optional_macbert_dict_5ep_v1 --model hfl/chinese-macbert-base --epochs 5 --warmup-ratio 0.1 --seed 42
```

输出目录为 `src/outputs/bert_gp/<新的run-name>/`。已有目录不会被覆盖，因此上述命令
使用新的 `optional_` 前缀。旧 checkpoint 只用于冻结推理，旧训练脚本不支持从中断处
继续训练。

### DeepSeek 100 条探索性复现

该路线用于展示 LLM 提示抽取能力，不属于默认主复现，也不能与完整 2,500 条
`eval_test` 上的 BERT 成绩直接排名。首次调用 API 前先做不产生费用的输入检查：

```powershell
python src/llm/run.py --dry-run --split tune --limit 100 --run-name optional_deepseek_dryrun
```

确认允许将文本发送到外部 API 后，配置当前终端的环境变量并执行：

```powershell
. .\src\llm\set_key.ps1
python src/llm/run.py --split tune --limit 100 --run-name optional_deepseek_tune100
```

本次历史试跑使用 `deepseek-v4-pro`、温度 0，范围为 `eval_tune` 的
`tune-0001` 至 `tune-0100`。统一实体级指标为：

```text
gold entities      = 383
predicted entities = 441
true positive      = 197

Precision = 0.4467
Recall    = 0.5144
F1        = 0.4782
```

该分数来自“DeepSeek 原始 JSON → 合法标签过滤与位置归一化 → 统一实体级评分”的完整
流水线，不是原始模型回答的独立分数。结果文件包括 `manifest.json`、`metrics.json`、
`errors.json`、`predictions.jsonl` 和 `usage.json`；原始输入和 API 审计材料按项目
规则留在本地，不进入课程代码包。

复现时需要注意：

- API 调用会产生费用，`usage.json` 只能自动统计 token 和耗时，实际费用必须按调用
  当日价格和账单填写，不能把未知费用写成 0。
- 历史记录为 100 次请求、145,171 tokens，总等待时间约 199.31 秒，平均约
  1.99 秒/条。
- 这只是 100 条 tune 子集，没有运行完整 2,500 条 tune，也没有运行 `eval_test`，
  不能当作最终大模型成绩。
- 当前 `normalize()` 对“多次出现但位置不明确”的实体采用最近位置修复，而现有测试
  和 LLM README 更倾向于排除。报告必须如实说明这次试跑包含位置后处理；在规则统一
  前不得把该分数与正式 BERT 主结果直接比较。

### 其他选项

- 只验证所选变体的环境、数据哈希和历史证据：
  `python src/reproduce.py --bert-variant <变体名> --check-only`。
- 从指定位置准备冻结数据：
  `python src/reproduce.py --data-package <本地zip或HTTPS链接>`。

`final_submit` 分支不提交 `src/data/` 下的实际数据、模型权重、下载缓存或重复的
历史复跑目录。默认复现会从固定 Release 获取数据和模型权重，并执行 SHA-256 校验；
课程 ZIP 也不要重新加入这些文件。
