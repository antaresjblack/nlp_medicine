# configs 目录说明

本目录保存可复现实验配置。配置一旦被用于正式测试，就不能根据测试成绩修改。

## 配置文件的区别

| 文件 | 性质 | 用途 |
|---|---|---|
| `final_baseline.json` | 成员2历史基线配置 | 复现完整 dev 上的规则、词典和小模型结果，不要为 split 实验改写 |
| `small_model_split_candidates.json` | 冻结前的候选清单 | 在 `eval_tune` 上比较 8 组小模型参数 |
| `outputs/eval_split/small_model/.../selected_config.json` | 运行后生成的冻结配置 | 记录 tune 选中的唯一配置、数据哈希和来源 commit |

## 小模型候选

`small_model_split_candidates.json` 固定了：

- `C`：`0.1` 或 `1.0`；
- `class_weight`：`balanced` 或 `none`；
- 词典跨度特征：开启或关闭；
- 随机种子：`42`；
- 单层 BIO 训练投影：最长非重叠实体。

共同设置 `2 × 2 × 2 = 8` 个候选。候选集合必须在查看 tune 结果前确定。

## 使用规则

1. `final_baseline.json` 是对应历史结果的配置，不要覆盖。
2. 小模型 split 的候选配置用于 `eval_tune`，不能用于直接挑 test。
3. `selected_config.json` 由
   `eval/run_small_model_split.py --stage tune` 自动生成。
4. 冻结配置中记录训练集、tune 集、数据哈希、候选配置哈希和代码 commit。
5. test 阶段必须显式传入冻结配置；不能手工修改后再声称仍是同一实验。
6. API 密钥和本机绝对用户路径不得写入配置。

完整流程见 `eval/README.md`。
