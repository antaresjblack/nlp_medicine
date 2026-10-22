# outputs 目录说明

本目录保存实验预测、指标、错误分析和运行日志。它不是源代码目录，但其中的结果
是报告和复现检查的证据。

## 历史基线

历史基线结果直接位于 `outputs/`，包括：

- `*_predictions.json`：预测；
- `*_metrics.json`：指标；
- `*_errors.json`：逐条错误；
- `*_error_analysis.json`：错误分类；
- `experiment_summary.md`：完整 dev 结果汇总；
- `grid/`：历史调参；
- `ablation/`：特征消融；
- `nested_loss_analysis_report.md`：嵌套投影损失分析。

这些是成员2权威历史结果，不要覆盖。

## 冻结 split

`outputs/eval_split/` 保存成员4新增的 tune/test 评估：

```text
eval_split/
├── metrics/             词典 tune 指标
├── predictions/         词典 tune 预测
├── errors/              词典 tune 错误
└── small_model/
    ├── tune/runs/       每轮候选实验
    └── test/            冻结配置的唯一一次 test
```

小模型的具体文件说明见
`outputs/eval_split/small_model/README.md`。

## 文件原则

- 预测文件必须保留记录 `id`；
- metrics 使用同一套 `baseline/metrics.py`；
- errors 保存金标、预测、漏检和误报；
- test 结果不得根据指标再反向调参；
- 新模型使用新的子目录，不覆盖旧结果；
- 不保存 API 密钥、原始患者隐私数据或虚拟环境。
