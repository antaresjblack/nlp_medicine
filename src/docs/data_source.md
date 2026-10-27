# CMeEE 数据来源与数据说明

编写日期：2026-10-18。本文区分本地检查结果、官方说明和待核实信息，不代表已经完成人工标注复核。

## 1. 数据集与本地文件

数据集名称为 CMeEE，任务为医疗领域命名实体识别（Medical NER）：从文本中识别实体跨度并赋予实体类别。CMeEE 是 CBLUE 中文医疗语言理解评测中的任务之一，背景与任务说明见 [CBLUE 官方仓库](https://github.com/CBLUEbenchmark/CBLUE)。本地文件位于项目的 `src/data/data_raw/`，顶层均为 JSON 列表。

| 集合 | 本地文件 | 已确认样本数 | 实体条目数 |
| --- | --- | ---: | ---: |
| train | `src/data/data_raw/CMeEE_train.json` | 15,000 | 61,796 |
| dev | `src/data/data_raw/CMeEE_dev.json` | 5,000 | 20,300 |
| test | `src/data/data_raw/CMeEE_test.json` | 3,000 | 0（本地实体列表均为空，不表示真实无实体） |

样本数和实体数来自实际运行的本地检查报告，而非仅引用预期规模。

## 2. JSON 字段与索引规则

| 层级 | 字段 | 本地类型 | 含义 |
| --- | --- | --- | --- |
| 样本 | `text` | 字符串 | 原始文本 |
| 样本 | `entities` | 列表 | 实体标注列表；测试集中的空列表不能解释为真实负标签 |
| 实体 | `start_idx` | 整数 | 实体起始字符位置，从 0 开始 |
| 实体 | `end_idx` | 整数 | 实体末尾字符位置，包含该字符 |
| 实体 | `type` | 字符串 | 实体类别代码 |
| 实体 | `entity` | 字符串 | 标注的实体文本；已发现两个空字符串异常条目 |

字符位置按照 Python 字符串的 Unicode 字符索引理解，不是 UTF-8 字节偏移。实体提取方式为：

```python
entity_text = text[start_idx:end_idx + 1]
```

训练集 61,794 个有效实体与验证集 20,300 个有效实体，合计 **82,094** 个实体全部匹配包含式索引，包含式不匹配为 0、排除式匹配为 0。两个异常实体不进入有效匹配统计。官方基线 `EEDataProcessor` 中使用 `range(start, end + 1)`，与本地验证结论一致，见 [官方数据处理代码](https://github.com/CBLUEbenchmark/CBLUE/blob/main/cblue/data/data_process.py)。测试集没有可用于验证索引的实体条目。

## 3. 实体标签与类别分布

本地 train/dev 实际出现以下九种代码，未发现九类之外的标签。表中中文名称用于阅读辅助：**最终标签含义、实体边界和歧义判定必须以已核实且与本地版本对应的 CMeEE/CBLUE 标注指南为依据**。目前未取得本地下载包的原始标签说明，代码与中文定义的逐项来源及细粒度判定规则仍待核实，不能用本表代替正式标注规范。

| 代码 | 中文含义（待对照对应版本指南核实） | train 数量 | train 占比 | dev 数量 | dev 占比 |
| --- | --- | ---: | ---: | ---: | ---: |
| `dis` | 疾病 | 15,843 | 25.637582% | 4,935 | 24.310345% |
| `sym` | 临床表现／症状 | 12,269 | 19.854036% | 4,130 | 20.344828% |
| `dru` | 药物 | 3,930 | 6.359635% | 1,440 | 7.093596% |
| `equ` | 医疗设备 | 888 | 1.436986% | 238 | 1.172414% |
| `pro` | 医疗程序 | 6,332 | 10.246618% | 2,057 | 10.133005% |
| `bod` | 身体部位 | 17,697 | 28.637776% | 5,883 | 28.980296% |
| `ite` | 医学检验项目 | 2,581 | 4.176646% | 923 | 4.546798% |
| `mic` | 微生物 | 1,908 | 3.087578% | 584 | 2.876847% |
| `dep` | 科室 | 348 | 0.563143% | 110 | 0.541872% |

来源：`src/docs/entity_label_distribution.csv`。占比分母为各集合全部实体条目数（含异常条目），不是文本数或有效实体数。显示值经过四舍五入。

## 4. 已完成的数据质量检查

### 4.1 字段、实体文本及索引

依据 `src/docs/offset_validation_report.txt` 和 `src/docs/data_quality_report.txt`：train/dev 未发现非字典样本、非法文本字段、非法实体列表字段、非字典实体或未知标签。合法索引、非空实体文本范围内的原文切片均匹配；包含式越界为 0。

训练集有两个实体同时存在空文本与反向索引，所有下标从 0 开始：

| 记录索引 | 实体索引 | 类型 | start_idx | end_idx | entity |
| --- | --- | --- | ---: | ---: | --- |
| 6051 | 1 | `pro` | 16 | 15 | `""` |
| 6421 | 3 | `mic` | 15 | 14 | `""` |

这是 **2 个异常实体、4 次异常事件**，不能误写为 4 个异常实体。两个实体仍保留在原始文件中，等待上下文核验、上游版本比对和团队处理决定。观察到异常不代表已确认其成因。

### 4.2 重叠实体

仅对合法且未越界的跨度检查；同一记录内每对实体仅计一次，包含式跨度有公共字符即计为重叠。相同跨度、严格包含、交叉重叠三类互斥。

| 指标 | train | dev |
| --- | ---: | ---: |
| 重叠实体对合计 | 3,681 | 1,225 |
| 相同跨度 | 33 | 16 |
| 包含关系（不含相同跨度） | 3,645 | 1,209 |
| 交叉重叠 | 3 | 0 |
| 含重叠实体的记录 | 1,666 | 524 |

两个非法跨度实体未参与重叠检查。重叠或嵌套是需要保留的数据特征，不能自动判为错误。单层 BIO 通常无法无损表达这类数据，模型输入兼容方式需与模型负责人确认。

### 4.3 重复文本与标注差异

按 `text` 完全相同分组，不去空格、不转换大小写、不进行归一化。train 内有 43 组重复文本，多余记录数为 43；dev 内为 0。train/dev 之间完全相同文本交集为 0，此结论不涵盖近似重复。

实体使用 `(start_idx, end_idx, type, entity)` 四元组排序后比较，忽略排列顺序但保留实体重复次数：43 组中 **25 组一致、18 组不一致**。

| 初步差异分类 | 组数 |
| --- | ---: |
| `type_conflict` | 2 |
| `possible_missing_entity` | 7 |
| `boundary_conflict` | 1 |
| `multiple_conflicts` | 8 |

分类仅反映结构差异，不能决定哪条标注正确。整理时报告解析失败为 0，复核表初始化时全部为 `pending`，不等于人工复核已完成。证据见 `src/docs/duplicate_annotation_review.txt`、`src/docs/duplicate_annotation_review.csv` 和 `src/docs/duplicate_annotation_summary.log`。原报告未给出一致组的实体明细，CSV 对应空单元格表示未提供，不表示空实体列表。

## 5. 测试集用途与评估限制

本地 test 的 3,000 条记录均包含字符串 `text` 和列表 `entities`，且 `entities` 全为空。[CBLUE 官方 README 的格式检查说明](https://github.com/CBLUEbenchmark/CBLUE/blob/main/README_ZH.md#格式检查check-format) 将原始测试文件说明为没有答案的文件，并以 CMeEE 为例。

因此，本地空列表与官方无公开答案测试集的用途相符，但仍不能据此证明本地文件与某个上游版本完全一致。**不能将本地 test 直接当作真实负样本，也不能依据这些空列表计算有效的实体级 Precision、Recall、F1。** 本项目的调参集、最终评估集划分和金标准来源仍需团队确认并记录。

## 6. 数据来源、下载与许可证据

| 项目 | 当前状态与证据 |
| --- | --- |
| 数据集及基准身份 | 已确认 CMeEE 为 CBLUE 的医疗 NER 任务，依据官方仓库任务说明；本地文件名、字段及规模与其相符 |
| 官方仓库 | [CBLUEbenchmark/CBLUE](https://github.com/CBLUEbenchmark/CBLUE) |
| 官方 README 提供的数据下载入口 | [天池 dataId=95414](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414)，链接来源为[官方 README 的数据准备部分](https://github.com/CBLUEbenchmark/CBLUE/blob/main/README_ZH.md#数据准备data-preparation)；本轮只核实了 README 的链接，未验证下载服务当前可用性 |
| 本地文件实际下载链接／渠道 | 使用者确认三个 JSON 来自 Hugging Face 数据集仓库 [Aunderline/CMeEE](https://huggingface.co/datasets/Aunderline/CMeEE)；具体文件页面及仓库说明见下方，访问日期为 2026-10-18。此为使用者提供的来源记录，尚未对本地文件与远程版本进行逐字节比对 |
| 实际下载日期 | **待核实**。文件修改时间不能证明下载日期 |
| 本地数据版本／发布批次／实际下载 commit hash | **待核实**。提供的链接指向可变的 `main` 分支；当前页面显示的提交信息不能直接证明实际下载时的版本，不能仅凭规模、文件名认定版本或与上游逐字一致 |
| 仓库页面标注的许可证 | 2026-10-18 访问 Hugging Face 仓库页面及 README，页面显示 `License: mit`，README 元数据为 `license: mit`，这是该仓库的数据集卡许可证标注 |
| 数据内容本身的授权范围／独立使用条款 | **待核实**。本次查看的来源页面未提供可单独确认原始数据内容授权范围的详细条款或授权链证据；不将仓库页面的 MIT 标注直接推断为原始数据全部内容及第三方材料均获 MIT 授权，也不据此宣称任意使用或再分发已经得到确认 |
| 标注指南及九类定义原文 | **待核实**。需提供本地下载包 README、标签说明或可追溯的官方指南 |

### 本地数据下载来源记录

以下地址由使用者补充确认为本地三个 JSON 文件的下载来源页面，本次逐一访问日期均为 **2026-10-18**：

- 训练集：[CMeEE_train.json](https://huggingface.co/datasets/Aunderline/CMeEE/blob/main/CMeEE_train.json)。
- 验证集：[CMeEE_dev.json](https://huggingface.co/datasets/Aunderline/CMeEE/blob/main/CMeEE_dev.json)。
- 测试集：[CMeEE_test.json](https://huggingface.co/datasets/Aunderline/CMeEE/blob/main/CMeEE_test.json)。
- 仓库说明：[README.md](https://huggingface.co/datasets/Aunderline/CMeEE/blob/main/README.md)。

该 README 说明测试集提交时需要为每条记录填充 `entities` 列表，并列出实体需要的字段及严格 Micro-F1 评估口径，与本地测试集不能直接用空列表充当金标准的说明一致。这里记录的是来源页面说明，未据此修改本地评估或标注。

本次外部说明访问日期为 2026-10-18，**不是实际下载日期**；实际下载日期和下载时的版本／commit hash 均继续标为待核实。未记录的历史下载信息不从当前 `main` 页面或本地文件修改时间反推。

许可证证据目前只能确认仓库页面及数据集卡标注为 MIT。数据内容的独立授权条款、原始权利人授权及具体使用／分享范围尚未从这些页面确认；这表示证据尚不充分，并非断言没有授权。后续如取得原始数据协议或明确授权说明，应据其更新记录。

## 7. 数据保留、完整性与复现

原始 JSON 未被本轮检查脚本修改；未执行自动纠错、重复记录删除、标注修复或 BIO 转换。此前检查记录包含运行前后 SHA-256 核对，本文只整理已有证据，没有重新运行会写出报告的脚本。

以下为已有报告记录的源文件 SHA-256，用于本地版本识别，不证明与上游一致：

```text
CMeEE_train.json  984f0bea3932c8c5afd96d9cccac4895b87ec77f57bd3e825d832a17943cd5d3
CMeEE_dev.json    bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c
CMeEE_test.json   dca7d9b7cf820947bba68e14557908fdc63e9286add1b4bc34f78f245e0c7ecf
```

在项目根目录可复现已有检查（这些命令会重新生成相应检查报告和复核表，人工填写后应先保留人工版本）：

```powershell
Set-Location '<仓库根目录>'
python -X utf8 src/scripts/validate_entity_offsets.py
python -X utf8 src/scripts/analyze_data_quality.py
python -X utf8 src/scripts/check_duplicate_annotations.py
python -X utf8 src/scripts/summarize_duplicate_review.py
```

现有运行环境已确认 Python 3.13.5，以上脚本仅依赖 Python 标准库。本文引用的本地依据为 `src/docs/data_quality_log.md`、`src/docs/data_quality_report.txt`、`src/docs/offset_validation_report.txt`、`src/docs/entity_label_distribution.csv` 以及重复标注复核报告与日志。后续人工决定应单独留痕，不覆盖原始数据。
