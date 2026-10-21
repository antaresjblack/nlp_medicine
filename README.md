# MedicalNER_Project 文件使用与组员交接说明

更新时间：2026-10-19。适用于当前项目根目录 `nlp_medicine/`。

> 当前整合标准：成员2的 `baseline/` 代码和 `outputs/` 结果是基线权威版本；
> 数据布局沿用根目录的 `data_raw/`、`data_cleaned/`。成员4新增的冻结评估集在
> `data_eval/`，统一评分入口在 `eval/`。详细说明见 `baseline/README.md` 和
> `eval/README.md`。

## 小组直接使用说明

本小组仓库包含原始数据、清洗结果、基线代码及复核材料，供组员直接使用。下载或克隆整个仓库后，**训练读取 `data_cleaned/CMeEE_train_cleaned_final.json`，无需重复清洗**；完整开发集读取 `data_raw/CMeEE_dev.json`，需要留出评估时读取 `data_eval/eval_tune.jsonl` 和 `data_eval/eval_test.jsonl`。目录中的旧版数据、日志和脚本保留用于追溯，不是组员开始建模前必须执行的步骤。

在自己的电脑上把代码中的项目根目录设置为实际下载位置；本说明里的 `C:\Users\home\...` 路径只是数据负责人本机路径。共享范围和来源许可记录见 `docs/data_source.md`。GitHub 小组共享与课程最终提交是两件事，课程压缩包仍按任务书排除数据集。

## 1. 新组员先看这里

本项目是自然语言处理课程 B 方向的医疗命名实体识别（NER）项目。当前已完成成员 1 的数据检查、人工复核记录整理及保守清洗，以及成员 2 的词典／字符级小模型基线和统一评测；大模型方案仍待成员 3 实现。基线代码、输出格式和评测协议以 `baseline/README.md` 为准，成员 4 的评估整合说明见 `eval/README.md`。

**后续训练使用 `data_cleaned/CMeEE_train_cleaned_final.json`。不要误用文件名相似的上一版 `CMeEE_train_cleaned.json`。**

| 用途 | 应使用的文件 | 当前数量与限制 |
| --- | --- | --- |
| 模型训练、训练词典构建 | `data_cleaned/CMeEE_train_cleaned_final.json` | 14,984 条文本，61,743 个实体 |
| 有标签的开发验证 | `data_raw/CMeEE_dev.json` | 5,000 条文本，20,300 个实体；具体调参／最终评估使用方式须组内冻结 |
| 调参与留出评估（成员4补充） | `data_eval/eval_tune.jsonl`、`data_eval/eval_test.jsonl` | 各 2,500 条；由 dev 按 seed 42 切分，`eval_test` 定稿时只用一次 |
| 无公开答案文本上的预测展示 | `data_raw/CMeEE_test.json` | 3,000 条；实体列表全部为空，不能据此计算有效实体级 P/R/F1 |
| 了解清洗与验证结果 | `docs/final_cleaning_summary.md`、`docs/final_validation_report.json` | 最新结果；包含类别统计、检查项与哈希 |
| 追溯最终样本原始位置 | `docs/record_index_map_final.csv` | 最终行号、原始行号、上一版行号对应表，均从 0 开始 |
| 查看数据来源与使用限制 | `docs/data_source.md` | 来源为使用者确认的 Hugging Face `Aunderline/CMeEE`；下载时间、下载 commit 和数据内容独立授权条款仍待核实 |

正式交接时先阅读本说明，再阅读最终清洗总结及数据来源文档。早期文档中的“尚未清洗”“仍需人工介入”等描述属于当时阶段，不应覆盖最终训练版的决定。

## 2. 目录如何分工

```text
MedicalNER_Project/
├── README.md              本说明
├── baseline/              成员2：基线代码、评测模块与权威实验结果
├── data_raw/              原始文件，保留不改
├── data_cleaned/          上一版与最终版清洗数据、日志和依赖
├── data_eval/             成员4：冻结的调参集与留出评估集
├── docs/                  数据说明、质量报告、复核证据和最终映射
├── eval/                  成员4：评估整合与统一评分入口
├── outputs/               预测、指标、错误分析和实验汇总
└── scripts/               检查、复核整理与清洗脚本
```

成员 2、3 如确需生成分词、分块、BIO 或其他模型输入，应将其作为独立派生步骤放在新目录或 `outputs/` 下并说明规则；不要改写原始数据、最终跨度格式数据或成员2已提交的权威结果。重叠、嵌套实体不能被静默丢弃。

## 3. 每个数据文件的用途

### data_raw：原始证据与开发集

| 文件 | 做什么／有什么用 | 后续如何使用 |
| --- | --- | --- |
| `data_raw/CMeEE_train.json` | 原始训练集，15,000 条、61,796 个实体条目，包含两条非法实体及重复标注争议 | 用于溯源、重建和核对，不作为当前推荐训练入口 |
| `data_raw/CMeEE_dev.json` | 原始有标签开发集，5,000 条、20,300 个实体 | 用于组内确定的开发验证流程，保留原文件不变 |
| `data_raw/CMeEE_test.json` | 原始测试文本，3,000 条，`entities` 均为空 | 用于预测展示或约定的提交格式；空列表不能当作真实负标签 |

### data_cleaned：最终训练入口与重建依据

| 文件 | 做什么／有什么用 | 是否为后续主要入口 |
| --- | --- | --- |
| `data_cleaned/CMeEE_train_cleaned_final.json` | 最终训练集，排除全部 8 组未解决争议的 16 条记录；61,743 个实体 | **是，成员 2、3 共用** |
| `data_cleaned/CMeEE_train_cleaned.json` | 上一版清洗集，14,990 条、61,771 个实体；部分争议记录当时仍保留 | 仅历史版本和最终重建输入，不应混作最终训练数据 |
| `data_cleaned/cleaning_plan.json` | 绑定原始训练集和人工工作簿 SHA-256，列明逐组允许的实体新增／移除及异常实体处理 | 上一版清洗的执行依据；最终流程用于核对历史改动 |
| `data_cleaned/cleaning_log.csv` | 每次历史实体修改、原先争议排除、异常实体移除的前后值与原因 | 最终流程逐条重放的证据，必须保留 |
| `data_cleaned/record_index_map.csv` | 上一版清洗行号与原始训练行号的映射 | 最终流程定位记录的依据，必须保留 |
| `data_cleaned/remaining_review_items.csv` | 上一版遗留 8 组的组号、全部原始索引、文本和未裁定原因 | **最终排除范围的直接依据**；历史 pending 不代表最终版仍保留这些记录 |
| `data_cleaned/validation_report.json` | 上一版样本／实体统计、哈希及校验记录 | 最终流程交叉核对输入，同时重新验证实际数据；不能单独代替最终校验 |
| `data_cleaned/cleaning_summary.md` | 上一版清洗决定、数量变化和复现命令 | 历史说明；判断当前使用状态应看最终总结 |

**不要只留最终 JSON 就删除上一版文件。** 当前 `--final` 流程依赖上述上一版数据、计划、日志、映射、待复核表和校验报告，缺失会导致无法复现。

## 4. 每个 docs 文件的用途

| 文件 | 做什么／有什么用 | 后续用途与注意事项 |
| --- | --- | --- |
| `docs/data_source.md` | 来源链接、字段结构、索引规则、标签分布及许可证据 | 报告的数据与伦理章节依据；部分段落形成于清洗前，最终清洗状态看最终总结 |
| `docs/data_quality_log.md` | 阶段性检查与复核进展记录 | 成员 1 的过程证据，不是自动清洗命令或最新训练配置 |
| `docs/data_quality_report.txt` | 原始 train/dev 类别、索引、重叠、重复文本及异常示例；test 仅检查结构 | 数据质量说明和原始数据统计，不能直接当作最终训练版类别分布 |
| `docs/entity_label_distribution.csv` | 原始 train/dev 的九类数量及占比 | 用于原始数据分布图；最终版分布从 `final_validation_report.json` 取 |
| `docs/offset_validation_report.txt` | 全量索引验证：82,094 个有效实体支持包含式结束索引，并定位两条异常 | 边界与索引规则证据 |
| `docs/offset_samples.txt` | 最初 3 条训练样本的两种索引切片对照 | 初学者理解切片的辅助材料，不足以单独支持全量结论 |
| `docs/data_audit.json` | 早期 Node.js 独立审计的规模、类别、哈希、重叠和跨集重复统计 | 历史交叉核验；早期“嵌套”计数包含同跨度，比较时注意与后期互斥分类的口径差异 |
| `docs/成员1_数据检查与交接.md` | 成员 1 初期职责、检查结果和待办清单 | 历史交接记录，其中环境及待办状态可能已过时 |
| `docs/duplicate_annotation_review.txt` | 原始 train 的 43 组重复文本，排序四元组比较，25 组一致、18 组不一致 | 人工核查的原始自动证据；不一致组附实体明细 |
| `docs/duplicate_annotation_review.csv` | 将上述文本报告整理为逐组复核表；分类为初步结构判断 | 初始自动生成表，不是更新后的人工裁定，不应拿它替代 checked 工作簿 |
| `docs/duplicate_annotation_summary.log` | CSV 整理的组数、初步分类、解析结果及源报告哈希 | 证明 43 组完整解析，不能当作人工审定证明 |
| `docs/duplicate_annotation_review_checked.csv.xlsx` | 人工填写的状态、备注与清洗建议 | **实际是 Excel 文件**，不是 CSV；上一版清洗计划绑定其哈希，勿改后直接重跑旧计划 |
| `docs/final_cleaning_summary.md` | 当前最终排除策略、三版规模、类别变化、复现方式与限制 | **最新交接说明**，报告引用最终清洗结果时使用 |
| `docs/final_validation_report.json` | 重新计算的最终统计、全部排除索引、输入哈希和最终哈希 | **最新机器可读证据**，评估整合负责人可读取并核对数据版本 |
| `docs/record_index_map_final.csv` | `final_record_index`、`original_record_index`、`previous_cleaned_record_index` | 错误分析时从模型样本追溯原文；不要把行号当字符位置 |
| `docs/removed_unresolved_groups.csv` | 16 条争议记录的组号、原始索引、文本、实体数、原始／上一版实体和排除原因 | 排除证据；区分原先已排除 10 条与本次新增 6 条，避免重复计数 |

第 4、5、9、11、13、18、30、41 组已按负责人决定整体退出最终训练集，不再要求为了本轮训练逐项裁定。排除不表示这些记录的全部标注均错误。

## 5. 每个脚本做什么、会写什么

Python 脚本使用标准库，Python 3.8 以上均可运行。`audit_data.cjs` 使用 Node.js，与 Python 主流程独立。

| 脚本 | 功能 | 写出的文件／运行注意事项 |
| --- | --- | --- |
| `scripts/inspect_data.py` | 查看三个原始 JSON 的顶层结构、数量和首条样本 | 仅终端输出；最后等待回车，初期预览工具 |
| `scripts/check_entity_offsets.py` | 检查训练集前 3 条样本的包含式／排除式切片 | 重写 `docs/offset_samples.txt`；初期示例工具 |
| `scripts/validate_entity_offsets.py` | 全量索引、实体文本和标签计数检查；同时提供路径常量与验证函数 | 重写 `docs/offset_validation_report.txt`；**多个脚本会导入它，不能因曾运行过就删除** |
| `scripts/analyze_data_quality.py` | 第二阶段质量检查，复用上述验证函数 | 重写质量报告及原始类别 CSV；不修改 JSON |
| `scripts/audit_data.cjs` | Node.js 独立只读审计 | 重写 `docs/data_audit.json`；不是当前清洗入口 |
| `scripts/check_duplicate_annotations.py` | 原始 train 按完全相同文本分组并比较排序四元组 | 重写 `docs/duplicate_annotation_review.txt` |
| `scripts/summarize_duplicate_review.py` | 从文本报告生成复核 CSV，给不一致组作初步差异分类 | 重写初始 CSV 和 summary.log；状态全部重置为 `pending`，不能覆盖唯一人工记录 |
| `scripts/clean_medical_ner_data.py` | 上一版清洗入口；添加 `--final` 时转入最终排除流程；也提供共用函数 | **不带 `--final` 会重写上一版输出**；只需使用最终数据的组员无需重跑它 |
| `scripts/finalize_medical_ner_data.py` | 最终流程实现：核对历史改动、按原始索引排除剩余组、验证映射与数据完整性 | 由 `clean_medical_ner_data.py --final` 调用；只产生最终数据和最终 docs 输出，不改上一版 |
| `scripts/__pycache__/validate_entity_offsets.cpython-313.pyc`（如存在） | Python 导入产生的字节码缓存 | 无需人工使用，可删除，之后可能自动重建；其他 `.pyc` 同理 |

“只读检查”表示不改输入数据，**不表示不会覆盖自动生成的检查报告**。人工复核内容应单独保存在 checked 工作簿或新增人工记录中。

## 6. 推荐操作：读取与验证最终数据

### 6.1 只做模型开发，不必重跑全部历史步骤

在项目根目录运行自己新增的模型脚本，读取最终训练集和原始 dev。以下是数据读取示意，可直接用于你们自己的加载函数，不会修改文件：

```python
import json
from pathlib import Path

# 从项目根目录运行时成立；正式模型脚本也可按自身位置定位项目根目录。
root = Path.cwd()
train = json.loads((root / "data_cleaned/CMeEE_train_cleaned_final.json").read_text(encoding="utf-8"))
dev = json.loads((root / "data_raw/CMeEE_dev.json").read_text(encoding="utf-8-sig"))

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
python -X utf8 -B scripts\clean_medical_ner_data.py --final
```

该命令核验原始数据、历史日志、计划、上一版数据和映射，从上一版筛选而不是从最终版反复删除。每次独立构建两遍，写盘后再检查；现有最终文件若与重建结果不同会停止，防止覆盖人工改动。此前已实际连续运行两次，五个最终输出文件的哈希一致。

预期终端结果：8 组累计排除 16 条（此前 10、本次新增 6），最终 14,984 条、61,743 个实体，验证通过。

最终训练集 SHA-256：

```text
e0e0d1bba812040e8e7f3e1096234e361db7ccc048db9575943fcfd9b3eb86dc
```

可用 PowerShell 只读核对文件哈希：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'data_cleaned\CMeEE_train_cleaned_final.json'
```

### 6.3 换电脑时注意

数据路径大部分由脚本位置推导，不要求队友也有同名桌面路径。队友可使用自己的 Python 解释器；上面的绝对解释器路径仅针对本机。当前清洗和质量脚本不需要安装第三方依赖，模型环境另行记录和配置。

**旧 `validation_report.json` 和部分其他证据含本机绝对路径及文件哈希。** 最终脚本会校验其来源一致性，迁移时应保留证据，不要简单替换其中路径后声称哈希未变。可以直接读取最终训练集开发模型；如要在另一台电脑从头重建全部历史链，需要数据负责人处理路径迁移和重新生成依赖证据，当前未验证跨电脑完整重建。

## 7. 组员分别从哪里继续

### 成员 1：数据与标注负责人

当前主要交付是最终训练集、原始 dev/test、最终索引映射、排除清单、数据来源和质量证据。后续负责解释版本、补充实际下载日期／commit 与数据授权证据，并与模型和评估成员确认输入输出口径。以后如新增标注裁定，应新建版本与日志，不能直接编辑当前最终 JSON。

### 成员 2：非大模型基线负责人

已完成精确词典、最长匹配词典、参数扫描、字符级 LinearSVC、统一实体级／字符级评测、错误分析和实验汇总；代码在 `baseline/`，说明和结果在 `baseline/README.md`，该目录是基线权威版本。

后续如扩展基线，继续保留原始跨度格式输出；如模型需要单层 BIO，应单独编写转换，明确重叠／嵌套实体的兼容限制并统计转换损失。不要为适配基线直接改共享金标，也不要覆盖已提交的权威结果。

### 成员 3：LLM 与特色模块负责人

使用相同的最终训练版和组内冻结的评估文本；少样本示例从训练集选择。新增提示模板、模型／API 配置、抽取结果校验和成本延迟记录。当前项目没有模型调用或提示抽取实现，需要自行开发。

生成的实体必须能够回到原文位置，并采用组内一致的索引与标签口径。API 密钥通过环境变量提供，不写入代码或预测结果。对外调用前遵循课程隐私要求及数据许可。

### 成员 4：评估与工程整合负责人

以 `docs/final_validation_report.json` 冻结训练数据版本，并汇总模型共用的样本、金标、预测格式和评测脚本。实体级严格匹配口径（样本、起点、终点、类型）和逐 token 投影规则以 `baseline/README.md` 为准；统一评分直接复用 `baseline/metrics.py`。

已完成 `data_eval/` 的 seed 42 切分和 `eval/` 的评估封装：`eval_tune` 用于调参，`eval_test` 用于最终留出评估，完整 dev 结果继续作为成员2的基线主表。运行命令和口径说明见 `eval/README.md`；不能用空标注 test 替代金标。错误分析通过最终索引映射定位原始文本。报告中的指标、成本和曲线须来自真实运行。

后续重点是接入成员 3 的 LLM 预测、统一预测文件格式、补齐最终对比表和复现命令；模型专用配置、日志和权重另建目录保存。

## 8. 文件保留、修改和提交约定

- 原始 JSON、人工 checked 工作簿、清洗计划与历史日志是证据，保留原样。
- 当前最终流程依赖的旧版文件不能直接删；需要精简时应先改变复现结构并验证。
- 缓存 `__pycache__` 可删除；初期预览文件不是模型依赖，但它们仍能帮助解释检查过程。
- 后续预测、模型权重和日志另存，避免与共享金标和人工复核文件同名覆盖。
- 同一组模型比较应使用同一版本训练集与同一评估协议，不能把上一版或原始 train 与最终 train 混用却不作说明。
- 按课程任务书，提交压缩包不打入数据集、模型权重或虚拟环境；提供数据来源及复现方式，并核实授权。可执行源码、文档和允许提交的证据按要求组织。
- 当前已有成员2的基线代码与成员4的评估封装；成员3的大模型入口、模型依赖、随机种子和最终报告仍待补齐。

本文只新增使用说明，不修改数据、人工记录、清洗代码或既有报告。
