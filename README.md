# MedicalNER_Project 文件使用与组员交接说明

更新时间：2026-10-27。适用于当前项目根目录 `nlp_medicine/`。

> 当前整合标准：成员2的 `src/baseline/` 代码和 `src/outputs/` 结果是基线权威版本；
> 数据统一位于 `src/data/data_raw/`、`src/data/data_cleaned/`。成员4新增的冻结评估集在
> `src/data/data_eval/`，统一评分入口在 `src/eval/`。详细说明见 `src/baseline/README.md` 和
> `src/eval/README.md`。

目录已统一为 `src/` 下组织。**新运行的路径以 `src/` 为运行时根目录**：数据在
`src/data/`、配置在 `src/configs/`、结果在 `src/outputs/`。历史冻结配置和指标中
仍可能记录整理前的 `data_eval/...`、`outputs/...` 等相对路径；这些是带哈希的
实验证据，不能直接改写。`src/reproduce.py` 和评估脚本会兼容映射这些旧路径。

## 复现入口（先看这里）

环境要求：Python 3.12+，并先安装基线依赖。

```powershell
python -m pip install -r requirements.txt
```

### 已验证的统一环境

当前机器已经创建并验证 `medical-ner-unified`，它同时可以运行传统基线和 BERT：

```text
Python 3.10.20
torch 2.9.0+cu128
transformers 4.57.6
numpy 1.26.4
scipy 1.13.1
scikit-learn 1.7.2
```

对应依赖文件是 `requirements-unified.txt`。在新机器上可以先创建环境：

```powershell
conda create -n medical-ner-unified python=3.10.20 -y
conda activate medical-ner-unified
python -m pip install -r requirements-unified.txt
```

如果不方便激活 Conda 环境，可以直接使用解释器：

```powershell
& 'D:\anaconda3\envs\medical-ner-unified\python.exe' -m pip check
```

当前统一环境已经实测通过：

```text
LinearSVC eval_test F1: 0.428989
MacBERT eval_test F1: 0.667023
pip check: No broken requirements found
CUDA: RTX 4060 available
```

`reproduce.py` 要求以下数据文件已经位于仓库对应目录：

```text
src/data/data_cleaned/CMeEE_train_cleaned_final.json
src/data/data_raw/CMeEE_dev.json
src/data/data_eval/eval_tune.jsonl
src/data/data_eval/eval_test.jsonl
```

当前 GitHub clone 已包含这些文件。课程最终 zip 按任务书不打包数据，因此提交前的
README 还应给出一个准备版数据包的可访问链接；来源和许可依据见
`src/docs/data_source.md`。不要把下载链接写成当前机器的绝对路径。

安装依赖后，在仓库根目录执行**一条命令**：

```powershell
python src/reproduce.py
```

该命令会自动完成以下检查：

1. 校验训练集、dev、`eval_tune`、`eval_test` 的固定 SHA-256；
2. 校验小模型冻结配置、候选配置和 tune 摘要未被修改；
3. 按冻结配置和 seed 42 重新训练字符级 LinearSVC；
4. 在 `eval_test` 上重新评估，预期实体级 F1 约为 **0.4290**；
5. 核对已提交的 MacBERT `eval_test` 证据，预期 F1 约为 **0.6670**。

默认只重新训练约需 1 分钟的轻量基线，BERT 部分核对提交的指标和训练来源记录。
输出保存在 `src/reproduction_outputs/<时间戳>/`，最终状态打印为 `Reproduction status: PASS`。

预期关键输出：

```text
[PASS] Environment, required data, hashes and frozen small-model configuration.
[PASS] Committed BERT eval_test evidence: actual=0.667023, expected=0.667023
[PASS] Retrained LinearSVC eval_test: actual=0.428969, expected=0.428969
[SKIP] BERT inference was not run because no checkpoint was supplied.
Reproduction status: PASS
```

`eval_test` 的复跑只用于核验已冻结结果，不得根据新结果修改参数或提示词。

如果已经取得可信的 MacBERT 权重，可以在安装 PyTorch/Transformers 的环境中执行：

```powershell
python src/reproduce.py --bert-checkpoint <best.pt 的绝对路径>
```

该命令会额外运行 BERT 最终推理，并比较新的 `eval_test` F1。项目没有把约 400 MB
的权重放进仓库，因此没有权重时只能核对历史证据，不能声称 BERT 已经在当前机器上运行。
BERT 权重应放在外部网盘，并在 README 中提供链接和 SHA-256；完整环境与训练参数见
`src/bert_gp/README.md`。

只检查数据和已提交证据、不训练模型：

```powershell
python src/reproduce.py --check-only
```

随机种子固定为 `42`；完整实验说明和分模块命令见对应目录的 README。课程提交包不包含
数据集和模型权重，数据获取来源见 `src/docs/data_source.md`。

## 当前状态（2026-10-27）

| 工作流 | 状态 | 当前结果 / 产物 |
|---|---|---|
| 数据与清洗（成员1） | 已完成 | 最终训练集 14,984 条 / 61,743 实体；dev 5,000 条 / 20,300 实体 |
| 非大模型基线（成员2） | 已完成 | 纯规则、词典、字符级小模型；最佳完整 dev 实体级 F1 **0.4301** |
| 评估整合（成员4） | 基础层与成员2小模型接入已完成 | 词典 tune F1 **0.4010**；小模型 tune F1 **0.4276**、test F1 **0.4290** |
| LLM / BERT（成员3） | BERT 主实验完成；LLM 部分完成 | MacBERT `eval_test` F1 **0.6670**；DeepSeek 仅完成 100 条 tune 子集，F1 **0.4782** |
| 报告 / 录屏 / 提交包 | 未完成 | 八节报告、摘要、贡献声明、AI 声明、zip 待补 |

当前最佳基线是“字符级 LinearSVC + 词典跨度特征”，完整 dev 实体级 F1 为
**0.4301**，冻结 `eval_test` F1 为 **0.4290**；在相同 `eval_test` 上，
MacBERT + GlobalPointer 的 F1 为 **0.6670**。完整 dev、冻结 test 和 DeepSeek
100 条子集必须分表报告，不能混排。

### 基线快速复现

```powershell
pip install -r requirements.txt
python src/baseline/run_final_baselines.py --config src/configs/final_baseline.json
```

详细结果见 `src/baseline/README.md`，留出评估规则见 `src/eval/README.md`。
`requirements.txt` 记录的是 Python 3.12.7 / numpy 2.3.4 / scikit-learn 1.9.0；
如果本机版本不同，`0.4301` 可能有小幅波动，报告中应注明实际环境。

### 成员3开始前的入口

成员3开始实现 LLM/BERT 前，先阅读以下文档：

- [src/baseline/README.md](src/baseline/README.md)：输入输出格式、嵌套实体限制和成员2基线；
- [src/eval/README.md](src/eval/README.md)：tune/test 使用规则、当前结果和接入合同；
- [src/data/data_eval/README.md](src/data/data_eval/README.md)：两份冻结评估集的含义；
- [src/configs/README.md](src/configs/README.md)：历史配置、split 候选和冻结配置的区别；
- [src/docs/README.md](src/docs/README.md)：数据证据、过程文档和最终报告索引；
- [src/outputs/README.md](src/outputs/README.md)：历史结果、tune 结果和 test 结果的目录边界；
- 各模块根目录下的 `README.md`：该模块的输入、输出和运行方式。

成员3只能使用最终训练集训练或选择 few-shot 示例，只能在 `eval_tune` 上比较
模型、提示词和参数。配置冻结后，每个最终方法在 `eval_test` 上只运行一次。
API 密钥只能通过环境变量提供，不能写入代码、日志或预测文件。

## 小组直接使用说明

本小组仓库包含原始数据、清洗结果、基线代码及复核材料，供组员直接使用。下载或克隆整个仓库后，**训练读取 `src/data/data_cleaned/CMeEE_train_cleaned_final.json`，无需重复清洗**；完整开发集读取 `src/data/data_raw/CMeEE_dev.json`，需要留出评估时读取 `src/data/data_eval/eval_tune.jsonl` 和 `src/data/data_eval/eval_test.jsonl`。目录中的旧版数据、日志和脚本保留用于追溯，不是组员开始建模前必须执行的步骤。

在自己的电脑上把代码中的项目根目录设置为实际下载位置；本说明里的 `C:\Users\home\...` 路径只是数据负责人本机路径。共享范围和来源许可记录见 `src/docs/data_source.md`。GitHub 小组共享与课程最终提交是两件事，课程压缩包仍按任务书排除数据集。

## 1. 新组员先看这里

本项目是自然语言处理课程 B 方向的医疗命名实体识别（NER）项目。当前已完成成员 1 的数据检查、人工复核记录整理及保守清洗，成员 2 的纯规则、词典、字符级小模型基线和统一评测，以及成员 3 的 MacBERT + GlobalPointer 主实验。冻结 `eval_test` 上，小模型 F1 为 **0.4290**，MacBERT F1 为 **0.6670**。DeepSeek 提示抽取只完成 100 条 tune 子集，不能当作最终全量成绩。基线代码、输出格式和评测协议以 `src/baseline/README.md` 为准，成员 4 的评估整合说明见 `src/eval/README.md`。

**后续训练使用 `src/data/data_cleaned/CMeEE_train_cleaned_final.json`。不要误用文件名相似的上一版 `CMeEE_train_cleaned.json`。**

| 用途 | 应使用的文件 | 当前数量与限制 |
| --- | --- | --- |
| 模型训练、训练词典构建 | `src/data/data_cleaned/CMeEE_train_cleaned_final.json` | 14,984 条文本，61,743 个实体 |
| 有标签的开发验证 | `src/data/data_raw/CMeEE_dev.json` | 5,000 条文本，20,300 个实体；具体调参／最终评估使用方式须组内冻结 |
| 调参与留出评估（成员4补充） | `src/data/data_eval/eval_tune.jsonl`、`src/data/data_eval/eval_test.jsonl` | 各 2,500 条；由 dev 按 seed 42 切分，`eval_test` 定稿时只用一次 |
| 无公开答案文本上的预测展示 | `src/data/data_raw/CMeEE_test.json` | 3,000 条；实体列表全部为空，不能据此计算有效实体级 P/R/F1 |
| 了解清洗与验证结果 | `src/docs/final_cleaning_summary.md`、`src/docs/final_validation_report.json` | 最新结果；包含类别统计、检查项与哈希 |
| 追溯最终样本原始位置 | `src/docs/record_index_map_final.csv` | 最终行号、原始行号、上一版行号对应表，均从 0 开始 |
| 查看数据来源与使用限制 | `src/docs/data_source.md` | 来源为使用者确认的 Hugging Face `Aunderline/CMeEE`；下载时间、下载 commit 和数据内容独立授权条款仍待核实 |

正式交接时先阅读本说明，再阅读最终清洗总结及数据来源文档。早期文档中的“尚未清洗”“仍需人工介入”等描述属于当时阶段，不应覆盖最终训练版的决定。

## 2. 目录如何分工

```text
MedicalNER_Project/
├── README.md                        本说明
├── requirements.txt                 基线依赖
├── src/
│   ├── reproduce.py                 一条命令复现入口
│   ├── baseline/                    成员2：基线代码与权威结果
│   ├── bert_gp/                     成员3：MacBERT + GlobalPointer
│   ├── llm/                         成员3：DeepSeek 抽取
│   ├── configs/                    实验配置
│   ├── data/
│   │   ├── data_raw/               原始数据
│   │   ├── data_cleaned/           清洗后训练数据
│   │   └── data_eval/              eval_tune / eval_test
│   ├── docs/                       数据说明与质量证据
│   ├── eval/                       成员4：统一评估
│   ├── outputs/                    指标、预测和实验汇总
│   └── scripts/                    检查与清洗脚本
└── ignore/                         不提交的本地整理材料
```

成员 2、3 如确需生成分词、分块、BIO 或其他模型输入，应将其作为独立派生步骤放在新目录或 `src/outputs/` 下并说明规则；不要改写原始数据、最终跨度格式数据或成员2已提交的权威结果。重叠、嵌套实体不能被静默丢弃。

新增的 LLM/BERT 代码放在 `src/llm/` 或 `src/bert_gp/`，预测和运行证据放在新的
`src/outputs/` 子目录。不要覆盖 `src/baseline/`、`src/outputs/*_predictions.json`、
`src/outputs/grid/` 或已经冻结的 small-model split 结果。

## 3. 每个数据文件的用途

### src/data/data_raw：原始证据与开发集

| 文件 | 做什么／有什么用 | 后续如何使用 |
| --- | --- | --- |
| `src/data/data_raw/CMeEE_train.json` | 原始训练集，15,000 条、61,796 个实体条目，包含两条非法实体及重复标注争议 | 用于溯源、重建和核对，不作为当前推荐训练入口 |
| `src/data/data_raw/CMeEE_dev.json` | 原始有标签开发集，5,000 条、20,300 个实体 | 用于组内确定的开发验证流程，保留原文件不变 |
| `src/data/data_raw/CMeEE_test.json` | 原始测试文本，3,000 条，`entities` 均为空 | 用于预测展示或约定的提交格式；空列表不能当作真实负标签 |

### src/data/data_cleaned：最终训练入口与重建依据

| 文件 | 做什么／有什么用 | 是否为后续主要入口 |
| --- | --- | --- |
| `src/data/data_cleaned/CMeEE_train_cleaned_final.json` | 最终训练集，排除全部 8 组未解决争议的 16 条记录；61,743 个实体 | **是，成员 2、3 共用** |
| `src/data/data_cleaned/CMeEE_train_cleaned.json` | 上一版清洗集，14,990 条、61,771 个实体；部分争议记录当时仍保留 | 仅历史版本和最终重建输入，不应混作最终训练数据 |
| `src/data/data_cleaned/cleaning_plan.json` | 绑定原始训练集和人工工作簿 SHA-256，列明逐组允许的实体新增／移除及异常实体处理 | 上一版清洗的执行依据；最终流程用于核对历史改动 |
| `src/data/data_cleaned/cleaning_log.csv` | 每次历史实体修改、原先争议排除、异常实体移除的前后值与原因 | 最终流程逐条重放的证据，必须保留 |
| `src/data/data_cleaned/record_index_map.csv` | 上一版清洗行号与原始训练行号的映射 | 最终流程定位记录的依据，必须保留 |
| `src/data/data_cleaned/remaining_review_items.csv` | 上一版遗留 8 组的组号、全部原始索引、文本和未裁定原因 | **最终排除范围的直接依据**；历史 pending 不代表最终版仍保留这些记录 |
| `src/data/data_cleaned/validation_report.json` | 上一版样本／实体统计、哈希及校验记录 | 最终流程交叉核对输入，同时重新验证实际数据；不能单独代替最终校验 |
| `src/data/data_cleaned/cleaning_summary.md` | 上一版清洗决定、数量变化和复现命令 | 历史说明；判断当前使用状态应看最终总结 |

**不要只留最终 JSON 就删除上一版文件。** 当前 `--final` 流程依赖上述上一版数据、计划、日志、映射、待复核表和校验报告，缺失会导致无法复现。

## 4. 每个 docs 文件的用途

| 文件 | 做什么／有什么用 | 后续用途与注意事项 |
| --- | --- | --- |
| `src/docs/data_source.md` | 来源链接、字段结构、索引规则、标签分布及许可证据 | 报告的数据与伦理章节依据；部分段落形成于清洗前，最终清洗状态看最终总结 |
| `src/docs/data_quality_log.md` | 阶段性检查与复核进展记录 | 成员 1 的过程证据，不是自动清洗命令或最新训练配置 |
| `src/docs/data_quality_report.txt` | 原始 train/dev 类别、索引、重叠、重复文本及异常示例；test 仅检查结构 | 数据质量说明和原始数据统计，不能直接当作最终训练版类别分布 |
| `src/docs/entity_label_distribution.csv` | 原始 train/dev 的九类数量及占比 | 用于原始数据分布图；最终版分布从 `final_validation_report.json` 取 |
| `src/docs/offset_validation_report.txt` | 全量索引验证：82,094 个有效实体支持包含式结束索引，并定位两条异常 | 边界与索引规则证据 |
| `src/docs/offset_samples.txt` | 最初 3 条训练样本的两种索引切片对照 | 初学者理解切片的辅助材料，不足以单独支持全量结论 |
| `src/docs/data_audit.json` | 早期 Node.js 独立审计的规模、类别、哈希、重叠和跨集重复统计 | 历史交叉核验；早期“嵌套”计数包含同跨度，比较时注意与后期互斥分类的口径差异 |
| `src/docs/成员1_数据检查与交接.md` | 成员 1 初期职责、检查结果和待办清单 | 历史交接记录，其中环境及待办状态可能已过时 |
| `src/docs/duplicate_annotation_review.txt` | 原始 train 的 43 组重复文本，排序四元组比较，25 组一致、18 组不一致 | 人工核查的原始自动证据；不一致组附实体明细 |
| `src/docs/duplicate_annotation_review.csv` | 将上述文本报告整理为逐组复核表；分类为初步结构判断 | 初始自动生成表，不是更新后的人工裁定，不应拿它替代 checked 工作簿 |
| `src/docs/duplicate_annotation_summary.log` | CSV 整理的组数、初步分类、解析结果及源报告哈希 | 证明 43 组完整解析，不能当作人工审定证明 |
| `src/docs/duplicate_annotation_review_checked.csv.xlsx` | 人工填写的状态、备注与清洗建议 | **实际是 Excel 文件**，不是 CSV；上一版清洗计划绑定其哈希，勿改后直接重跑旧计划 |
| `src/docs/final_cleaning_summary.md` | 当前最终排除策略、三版规模、类别变化、复现方式与限制 | **最新交接说明**，报告引用最终清洗结果时使用 |
| `src/docs/final_validation_report.json` | 重新计算的最终统计、全部排除索引、输入哈希和最终哈希 | **最新机器可读证据**，评估整合负责人可读取并核对数据版本 |
| `src/docs/record_index_map_final.csv` | `final_record_index`、`original_record_index`、`previous_cleaned_record_index` | 错误分析时从模型样本追溯原文；不要把行号当字符位置 |
| `src/docs/removed_unresolved_groups.csv` | 16 条争议记录的组号、原始索引、文本、实体数、原始／上一版实体和排除原因 | 排除证据；区分原先已排除 10 条与本次新增 6 条，避免重复计数 |

第 4、5、9、11、13、18、30、41 组已按负责人决定整体退出最终训练集，不再要求为了本轮训练逐项裁定。排除不表示这些记录的全部标注均错误。

## 5. 每个脚本做什么、会写什么

Python 脚本使用标准库，Python 3.8 以上均可运行。`audit_data.cjs` 使用 Node.js，与 Python 主流程独立。

| 脚本 | 功能 | 写出的文件／运行注意事项 |
| --- | --- | --- |
| `src/scripts/inspect_data.py` | 查看三个原始 JSON 的顶层结构、数量和首条样本 | 仅终端输出；最后等待回车，初期预览工具 |
| `src/scripts/check_entity_offsets.py` | 检查训练集前 3 条样本的包含式／排除式切片 | 重写 `src/docs/offset_samples.txt`；初期示例工具 |
| `src/scripts/validate_entity_offsets.py` | 全量索引、实体文本和标签计数检查；同时提供路径常量与验证函数 | 重写 `src/docs/offset_validation_report.txt`；**多个脚本会导入它，不能因曾运行过就删除** |
| `src/scripts/analyze_data_quality.py` | 第二阶段质量检查，复用上述验证函数 | 重写质量报告及原始类别 CSV；不修改 JSON |
| `src/scripts/audit_data.cjs` | Node.js 独立只读审计 | 重写 `src/docs/data_audit.json`；不是当前清洗入口 |
| `src/scripts/check_duplicate_annotations.py` | 原始 train 按完全相同文本分组并比较排序四元组 | 重写 `src/docs/duplicate_annotation_review.txt` |
| `src/scripts/summarize_duplicate_review.py` | 从文本报告生成复核 CSV，给不一致组作初步差异分类 | 重写初始 CSV 和 summary.log；状态全部重置为 `pending`，不能覆盖唯一人工记录 |
| `src/scripts/clean_medical_ner_data.py` | 上一版清洗入口；添加 `--final` 时转入最终排除流程；也提供共用函数 | **不带 `--final` 会重写上一版输出**；只需使用最终数据的组员无需重跑它 |
| `src/scripts/finalize_medical_ner_data.py` | 最终流程实现：核对历史改动、按原始索引排除剩余组、验证映射与数据完整性 | 由 `clean_medical_ner_data.py --final` 调用；只产生最终数据和最终 docs 输出，不改上一版 |
| `src/scripts/__pycache__/validate_entity_offsets.cpython-313.pyc`（如存在） | Python 导入产生的字节码缓存 | 无需人工使用，可删除，之后可能自动重建；其他 `.pyc` 同理 |

“只读检查”表示不改输入数据，**不表示不会覆盖自动生成的检查报告**。人工复核内容应单独保存在 checked 工作簿或新增人工记录中。

## 6. 推荐操作：读取与验证最终数据

### 6.1 只做模型开发，不必重跑全部历史步骤

在项目根目录运行自己新增的模型脚本，读取最终训练集和原始 dev。以下是数据读取示意，可直接用于你们自己的加载函数，不会修改文件：

```python
import json
from pathlib import Path

# 从项目根目录运行时成立；正式模型脚本也可按自身位置定位项目根目录。
root = Path.cwd()
train = json.loads((root / "src/data/data_cleaned/CMeEE_train_cleaned_final.json").read_text(encoding="utf-8"))
dev = json.loads((root / "src/data/data_raw/CMeEE_dev.json").read_text(encoding="utf-8-sig"))

for record in train:
    text = record["text"]
    for e in record["entities"]:
        assert text[e["start_idx"]:e["end_idx"] + 1] == e["entity"]
```

顶层为样本列表；每条样本含 `text` 和 `entities`；每个实体含 `start_idx`、`end_idx`、`type`、`entity`。索引从 0 开始，终点包含在实体中。九类为 `dis/sym/dru/equ/pro/bod/ite/mic/dep`；类别含义和歧义边界仍应参考上游规范。官方 README 已确认 `dis` 为疾病、`pro` 为医疗程序，不能只凭词面自动修复原始争议。

### 6.2 重建并验证最终训练集

CMD 中运行：

```bat
cd /d <仓库根目录>
python -X utf8 -B src\scripts\clean_medical_ner_data.py --final
```

该命令核验原始数据、历史日志、计划、上一版数据和映射，从上一版筛选而不是从最终版反复删除。每次独立构建两遍，写盘后再检查；现有最终文件若与重建结果不同会停止，防止覆盖人工改动。此前已实际连续运行两次，五个最终输出文件的哈希一致。

预期终端结果：8 组累计排除 16 条（此前 10、本次新增 6），最终 14,984 条、61,743 个实体，验证通过。

最终训练集 SHA-256：

```text
e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc
```

可用 PowerShell 只读核对文件哈希：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'src\data\data_cleaned\CMeEE_train_cleaned_final.json'
```

### 6.3 换电脑时注意

数据路径大部分由脚本位置推导，不要求队友也有同名桌面路径。队友可使用自己的 Python 解释器；上面的绝对解释器路径仅针对本机。当前清洗和质量脚本不需要安装第三方依赖，模型环境另行记录和配置。

**旧 `validation_report.json` 和部分其他证据含本机绝对路径及文件哈希。** 最终脚本会校验其来源一致性，迁移时应保留证据，不要简单替换其中路径后声称哈希未变。可以直接读取最终训练集开发模型；如要在另一台电脑从头重建全部历史链，需要数据负责人处理路径迁移和重新生成依赖证据，当前未验证跨电脑完整重建。

## 7. 组员分别从哪里继续

### 成员 1：数据与标注负责人

当前主要交付是最终训练集、原始 dev/test、最终索引映射、排除清单、数据来源和质量证据。后续负责解释版本、补充实际下载日期／commit 与数据授权证据，并与模型和评估成员确认输入输出口径。以后如新增标注裁定，应新建版本与日志，不能直接编辑当前最终 JSON。

### 成员 2：非大模型基线负责人

已完成纯规则基线、精确/最长匹配词典、词典参数扫描、字符级 LinearSVC、词典跨度特征、两轮网格搜索、特征消融、嵌套投影损失分析、统一评测、错误分析和实验汇总。最佳方法为“字符级 LinearSVC + 词典跨度特征”，完整 dev 实体级 F1 **0.4301**。代码在 `src/baseline/`，说明、运行命令和结果在 `src/baseline/README.md`，该目录是基线权威版本；一键入口为 `src/baseline/run_final_baselines.py --config src/configs/final_baseline.json`。

后续如扩展基线，继续保留原始跨度格式输出；如模型需要单层 BIO，应单独编写转换，明确重叠／嵌套实体的兼容限制并统计转换损失。嵌套损失已经量化，不要为适配单层 BIO 直接改共享金标，也不要覆盖已提交的权威结果。

### 成员 3：LLM 与特色模块负责人

BERT + GlobalPointer 主实验已经完成，四组配置均有 tune/test 指标和训练记录；
代码与说明见 `src/bert_gp/README.md`。DeepSeek 提示抽取代码已经实现，但当前只有
`eval_tune` 前 100 条结果，API 成本仍待按账单填写，不能当作全量最终成绩。

生成的实体必须能够回到原文位置，并采用组内一致的索引与标签口径。API 密钥通过环境变量提供，不写入代码或预测结果。对外调用前遵循课程隐私要求及数据许可。

### 成员 4：评估与工程整合负责人

以 `src/docs/final_validation_report.json` 冻结训练数据版本，并汇总模型共用的样本、金标、预测格式和评测脚本。实体级严格匹配、`token_*` 最长非重叠投影和 `char_multi_label_*` 多标签字符指标以 `src/baseline/README.md` 第五节为准；统一评分直接复用 `src/baseline/metrics.py`。

已完成 `src/data/data_eval/` 的 seed 42 切分和 `src/eval/` 的评估封装：`eval_tune` 用于调参，`eval_test` 用于最终留出评估，完整 dev 结果继续作为成员2的历史基线。词典基线已在 tune 上复算为 `0.4010`；小模型已在 tune 上重新比较 8 个候选，选中 `C=0.1`、`class_weight=none`、词典跨度特征开启，tune F1 为 `0.4276`，冻结配置后 `eval_test` 单次 F1 为 `0.4290`。运行命令和口径说明见 `src/eval/README.md`；不能用空标注 test 替代金标。错误分析通过最终索引映射定位原始文本。报告中的指标、成本和曲线须来自真实运行。

后续重点是接入成员 3 的 LLM 预测、统一预测文件格式、补齐最终对比表和复现命令；模型专用配置、日志和权重另建目录保存。

## 8. 文件保留、修改和提交约定

- 原始 JSON、人工 checked 工作簿、清洗计划与历史日志是证据，保留原样。
- 当前最终流程依赖的旧版文件不能直接删；需要精简时应先改变复现结构并验证。
- 缓存 `__pycache__` 可删除；初期预览文件不是模型依赖，但它们仍能帮助解释检查过程。
- 后续预测、模型权重和日志另存，避免与共享金标和人工复核文件同名覆盖。
- 同一组模型比较应使用同一版本训练集与同一评估协议，不能把上一版或原始 train 与最终 train 混用却不作说明。
- 按课程任务书，提交压缩包不打入数据集、模型权重或虚拟环境；提供数据来源及复现方式，并核实授权。可执行源码、文档和允许提交的证据按要求组织。
- 当前已有成员2的完整非大模型基线及其 tune/test 接入、成员4的评估封装和 `src/reproduce.py` 入口、成员3的 BERT 主实验；仍待补齐 BERT 权重/精确环境、DeepSeek 费用与全量边界说明、最终报告、录屏、贡献声明和提交包。

本文只新增使用说明，不修改数据、人工记录、清洗代码或既有报告。
