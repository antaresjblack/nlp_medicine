# 基线开发：第一步数据检查

`data_check.py` 是成员 2 的第一个脚本。它只读取数据，不修改任何 JSON 文件，检查：

- 训练集、开发集、测试集是否存在并能读取；
- 每条实体的起止位置是否有效；
- 原文切片是否等于标注的 `entity`；
- 标签是否属于 CMeEE 的九类标签；
- 数据中是否存在重叠实体。

在仓库根目录执行：

```powershell
python baseline/data_check.py --report outputs/data_check_report.txt
```

如果检查通过，程序退出码为 `0`；如果发现问题，退出码为 `1`，但仍会打印完整报告。

当前数据的使用约定：

- 训练和建词典：`data_cleaned/CMeEE_train_cleaned_final.json`
- 有金标的评估：`data_raw/CMeEE_dev.json`
- `data_raw/CMeEE_test.json` 的实体列表为空，不能把它当成全是负例，也不能用它计算 P/R/F1。
