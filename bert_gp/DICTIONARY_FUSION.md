# 训练时词典特征融合

独立入口 `bert_gp/train_dict.py`，不修改原train.py、原检查点或成员2匹配实现。
复用 `DictSpanAnnotator.match_all()`，min_count=1/min_length=1；保留重复、嵌套、多类别计数。

31维token特征：9类覆盖分布、9类起始分布、9类末尾分布，命中标记、对数频次、长度、多类别冲突。
同位置多词条逐维max聚合，类别值是线索，聚合后不保证和为1；不是模型输出概率。字符到token通过offset_mapping覆盖聚合，special/padding全零。
匹配先聚合到字符，token内max；词条跨窗口时只保留窗口内的可见特征，不伪造额外起止点。

MacBERT表示与特征共同计算门控，词典投影以残差加入表示，再输入原GlobalPointer。
投影零初始化、无命中时残差为0，原上下文表示保持可用。不硬替换类别、边界，不禁止未命中词典的新实体。

训练词典只从当前选用的训练记录构建；训练每条文本生成特征前，扣除所有同原文记录对词频的贡献，减少“当前标签直接作为自己输入”的捷径。
训练阶段另以0.3概率整组遮掉某token的词典特征；tune推理使用固定训练词典、不遮盖。不使用tune/test金标建词典。
这些策略减轻自包含偏差，不保证消除过拟合；训练/推理词典覆盖差异会记录在manifest。

```powershell
conda activate medical-ner
python bert_gp/test_dict.py
python bert_gp/train_dict.py --run-name dict_data_check --train-limit 128 --tune-limit 64 --check-data
python bert_gp/train_dict.py --run-name dict_smoke --train-limit 128 --tune-limit 64 --epochs 1
# 小规模检查通过后再运行，完整5轮；不会自动启动
python bert_gp/train_dict.py --run-name macbert_dict_5ep_v1 --epochs 5
```

默认其余配置与五轮MacBERT相同：256长度、batch2、累积8、lr2e-5、预热0.1、线性衰减、阈值0。
从预训练编码器重新训练，额外融合层也需学习；不把已有非融合检查点冒充本模型。
检查点包含训练词典的各类别计数，推理不再依赖外部重新构建词典：

```powershell
python bert_gp/train_dict.py --run-name dict_tune_inference --checkpoint outputs/bert_gp/macbert_dict_5ep_v1/best.pt
```

只提供tune入口，没有test开关。当前已有test被查看，新增方案在tune探索，不据test改模型。
旧threshold_scan.py不能加载本融合结构；需要阈值扫描时须使用匹配本模型的推理代码。
结果在outputs/bert_gp/<run-name>/，保留原始金标评分、嵌套跨度及对齐损失统计。
best.pt仅支持推理，不支持断点恢复训练。收益未知，实验结论由小组根据真实结果独立分析并声明AI辅助实现范围。
