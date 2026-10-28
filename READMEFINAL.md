# 项目说明及复现指南

> 本文件是最终提交包的复现入口；完整项目说明、成员分工和结果解释见 `README.md`。

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
python -m pip install -r requirements-unified.txt
```

`requirements-unified.txt` 是项目唯一完整环境，同时安装传统基线和 BERT 所需依赖。

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

默认完整复现：重新训练冻结的字符级 LinearSVC，并加载最新 MacBERT 词典融合
checkpoint，在 `eval_test` 上分别计算主结果：

```powershell
python src/reproduce.py
```

默认会优先使用本地最新 checkpoint；若不存在，则自动从 GitHub Release 下载约
414 MB 的权重并校验 SHA-256。也可以显式指定 checkpoint：

```powershell
python src/reproduce.py --bert-checkpoint https://api.github.com/repos/antaresjblack/nlp_medicine/releases/assets/684231335
```

权重 SHA-256：

```text
595EB45CBFD920F93E175668A7FE50A11F8B949222E882BB2F7FD4C91AE9B1D8
```

## 预期输出

```text
[PASS] Environment, required data, hashes and frozen small-model configuration.
[PASS] Committed BERT eval_test evidence: actual=0.670047, expected=0.670047
[PASS] Retrained LinearSVC eval_test: actual=0.428989, expected=0.428969
[PASS] MacBERT dictionary-fusion eval_test: actual=0.670047, expected=0.670047
Reproduction status: PASS
```

复现结果保存在 `src/reproduction_outputs/<时间戳>/`。默认命令要求 BERT 推理真实执行；
只有 `--check-only` 会跳过训练和推理，仅校验环境、数据哈希和冻结证据。

## 可选复现内容

- 使用自定义 checkpoint：`python src/reproduce.py --bert-checkpoint <本地路径或HTTPS链接>`。
- 降低对权重数值波动的容差：`python src/reproduce.py --tolerance 0.001`。
