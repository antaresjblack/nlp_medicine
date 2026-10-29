# 成员3：DeepSeek 医疗实体抽取

本模块是实现辅助代码。实验设计、标签规范确认、参数选择和结果分析由小组自行完成，并如实声明 AI 辅助范围。仅用 Python 标准库，复用成员2评分代码，不改数据与既有结果。

## 开始使用

以下命令在仓库根目录运行（先确认数据允许发送外部 API）：

```powershell
. .\src\llm\set_key.ps1
python src/llm/run.py --run-name smoke10 --limit 10
```

第一条命令会要求隐藏输入你自己的 **DeepSeek** Key，直接粘贴即可。它只保留在当前终端环境中，无须修改代码，关闭终端后需重新设置。不要使用 OpenAI Key，不要在聊天、截图或仓库公开密钥。

默认模型 `deepseek-v4-pro`，API 地址、模型和调用参数在 `src/llm/config.json` 中修改。这里的提示词是起始模板，不是已经验证的最佳配置；九类名称和边界规则仍须人工对照组内标注规范。

不调用 API 的检查：

```powershell
python -m src.llm.test_run
python src/llm/run.py --dry-run --run-name input_check --limit 3
```

测试脚本必须使用模块方式从仓库根目录运行；直接执行 `python src/llm/test_run.py`
会因为仓库根目录不在 Python 导入路径而报 `ModuleNotFoundError`。

## 输出与恢复

每次运行保存在 `src/outputs/llm/<run-name>/`：
- `manifest.json`：配置、完整提示词、输入文件哈希和所选样本ID。
- `inputs.jsonl`：只含ID与原文，不含金标；请求中也不含待评估样本答案。
- `audit.jsonl`：每次请求的原始响应、usage、耗时、错误和校验记录。
- `predictions.jsonl`：统一实体格式，含尾索引；保留嵌套、按跨度和类型去重。
- `metrics.json` / `errors.json`：复用 `src/baseline/metrics.py` 的真实评分。
- `usage.json`：累计用量与请求耗时；费用留空，按实际日期价格和账单填写。

第一次失败即停止，不把失败样本伪装成无实体或跳过评分。修复原因后用原命令加 `--resume`，复用已保存成功结果。配置、提示词、数据或样本选择改变会拒绝恢复，应使用新运行名。请求超时可能仍有服务端费用；日志只统计已知用量。原始响应与文本属于数据材料，不应随课程代码包提交。

位置不匹配时，如果实体原词在全文恰好出现一次，会自动修复位置并记录。多次出现时，
仅在模型给出合法整数 `start_idx` 的情况下选择距离该提示最近的匹配；位置字段缺失、
原文不存在、未知标签等会记录并排除，不会自动展开到所有位置。评估反映包含此后处理
的完整流水线，不是原始模型输出的独立分数。

## 全量与最终评估

```powershell
python src/llm/run.py --run-name tune_final --limit 0
```

默认10条只是接通检查，不能当作全量成绩。冻结参数并经组内确认后才运行：

```powershell
python src/llm/run.py --split test --limit 0 --final --run-name test_final
```

全量运行会逐条付费调用；请先检查小样本的费用、速度与输出长度。定稿评估不得用于反复调整参数。完整dev历史成绩不能和tune/test成绩混列，且历史基线曾用完整dev调参，后拆test并非对基线完全独立的留出集，报告须披露。

官方接口依据：https://api-docs.deepseek.com/api/create-chat-completion/
JSON输出说明：https://api-docs.deepseek.com/guides/json_mode/
API密钥与账单：https://platform.deepseek.com/
