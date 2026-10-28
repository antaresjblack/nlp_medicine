# 项目说明及复现指南

> 本文件是最终提交包的复现入口；完整项目说明、成员分工和结果解释见 `README.md`。

## 环境要求

| 组件         | 版本                   |
| ------------ | ---------------------- |
| Python       | 3.10.20                |
| PyTorch      | 2.9.0+cu128            |
| Transformers | 4.57.6                 |
| NumPy        | 1.26.4                 |
| SciPy        | 1.13.1                 |
| scikit-learn | 1.7.2                  |
| GPU          | NVIDIA RTX 4060 Laptop |
| CUDA         | 可用                   |

## 依赖

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

## 数据集来源

数据来自 CBLUE CMeEE 医疗命名实体识别任务。课程提交包不包含数据；获取方式和
许可边界见 `src/docs/data_source.md`。解压或下载后应保持以下布局：

```text
src/data/data_raw/CMeEE_dev.json
src/data/data_raw/CMeEE_test.json
src/data/data_cleaned/CMeEE_train_cleaned_final.json
src/data/data_eval/eval_tune.jsonl
src/data/data_eval/eval_test.jsonl
```

## 复现命令

只校验环境、数据哈希和冻结结果：

```powershell
python src/reproduce.py --check-only
```

重新训练冻结的字符级 LinearSVC，并在 `eval_test` 上复核主结果：

```powershell
python src/reproduce.py
```

如已下载可信的 MacBERT checkpoint，可额外复现 BERT 推理：

```powershell
python src/reproduce.py --bert-checkpoint https://api.github.com/repos/antaresjblack/nlp_medicine/releases/assets/683593804
```

权重 SHA-256：

```text
868FBEB8530125552FF06EF8CA4C29F9CFAD8B25F087BE1E1D61687B41038CF7
```

## 预期输出

```text
[PASS] Environment, required data, hashes and frozen small-model configuration.
[PASS] Committed BERT eval_test evidence: actual=0.667023, expected=0.667023
[PASS] Retrained LinearSVC eval_test: actual≈0.428969
Reproduction status: PASS
```

复现结果保存在 `src/reproduction_outputs/<时间戳>/`。未提供 BERT checkpoint 时，
脚本会明确打印 `[SKIP]`，只核验已提交的 BERT 指标证据。
