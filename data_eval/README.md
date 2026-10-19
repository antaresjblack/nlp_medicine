# data_eval：冻结的评估集

> 这两份文件是全组共用的考卷，生成后不再修改。
> 改动会导致所有人的分数失去可比性。

## 文件

| 文件 | 用途 | 条数 | 实体数 |
| --- | --- | ---: | ---: |
| `eval_tune.jsonl` | 调参：提示词、超参数、模型选择只能用这份 | 2500 | 9840 |
| `eval_test.jsonl` | 最终评估：报告里的分数只能来自这份 | 2500 | 10460 |

原始 `CMeEE_test.json` 的 3,000 条实体列表全为空，是官方无答案测试集，
不能用来计算 F1，本目录不使用它。

## 怎么切出来的

- 来源：`data_raw/CMeEE_dev.json`（5,000 条，有标注）
- 随机种子：`42`
- 切分单位：整条记录（不按实体切，避免同一句拆到两边）
- 比例：打乱后取前 2,500 条为 tune，其余 2,500 条为 test
- 索引口径：沿用原始含尾索引，`text[start_idx:end_idx + 1] == entity`

复现命令：

```powershell
python src/eval/make_eval_split.py
```

## 校验值

| 文件 | SHA-256 |
| --- | --- |
| `eval_tune.jsonl` | `ff1e9fdc685a4b405a55cfece540eceb7cba7bfdb70492693c51c6e42d0f584b` |
| `eval_test.jsonl` | `ca65235e83f5ba055af7498b25ea55c376747a64ab6b1d4e5bffc0dfb8e62778` |

## 类别分布

| 类型 | tune | 占比 | test | 占比 |
| --- | ---: | ---: | ---: | ---: |
| dis | 2397 | 24.36% | 2538 | 24.26% |
| sym | 2082 | 21.16% | 2048 | 19.58% |
| dru | 709 | 7.21% | 731 | 6.99% |
| equ | 138 | 1.40% | 100 | 0.96% |
| pro | 1046 | 10.63% | 1011 | 9.67% |
| bod | 2716 | 27.60% | 3167 | 30.28% |
| ite | 436 | 4.43% | 487 | 4.66% |
| mic | 253 | 2.57% | 331 | 3.16% |
| dep | 63 | 0.64% | 47 | 0.45% |

## 使用规则

1. `eval_tune` 可以反复跑、反复改参数，不会影响最终结论。
2. `eval_test` 只在定稿时跑一次，跑完就写进报告。
3. 预测文件命名：`outputs/predictions/{方法名}_tune_predictions.jsonl` 和
   `outputs/predictions/{方法名}_test_predictions.jsonl`。
   这里的 `test` 指本目录的 `eval_test.jsonl`，不是官方无答案的 `CMeEE_test.json`。
4. 本目录任何文件都不允许改；确需重新切分，必须全组同意并重跑所有方法。
