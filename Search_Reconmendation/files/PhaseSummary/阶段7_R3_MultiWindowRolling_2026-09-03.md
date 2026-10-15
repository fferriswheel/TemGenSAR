# 阶段 7：R3 多窗口滚动时序评估

日期：2026-09-03

## 1. 阶段目标

本阶段不再只观察单次 `W1 -> W2` 更新，而是在固定超参数后执行连续季度滚动实验，回答以下问题：

1. 选择性更新 task-specific suffix 的收益能否跨多个时间窗口持续存在；
2. 多轮局部换码是否会导致 SID collision 累积恶化；
3. 收益是否来自漂移选择，而不是相同预算下的任意换码；
4. 只更新 task adapter/LoRA 的 SID 学习成本实际有多大；
5. Search 与 Rec 是否都能从同一套联合更新机制中稳定获益。

## 2. 数据与时间协议

实验使用 20K item 数据集。校准期为 `2010Q1-2011Q4`，监控期为：

```text
W1=2012Q1, W2=2012Q2, ..., W9=2014Q1, W10=2014Q2
```

采用严格的 `test-then-train` 协议：在 `W_t` 完成更新后只评估未来的 `W_{t+1}`；随后 `W_{t+1}` 的数据可作为下一轮更新数据。最终覆盖 9 个未来评估窗口 `W2-W10`。

- Search：从 W1 起持续使用 controlled drift 数据；
- Rec：使用季度数据中自然发生的行为漂移；
- SID representation：每轮使用截至当前窗口的累计快照，不使用评估窗口及更未来的信息；
- 下游训练：所有方法都使用当前窗口的 Search/Rec update 样本增量训练，再测试下一窗口；
- 每个评估窗口各含 5,000 条 Rec 和 5,000 条 Search 样本。

### 协议修正

早期 R2b 联合实验中，SID 基于 controlled Search drift 生成，但联合下游数据的 `src_test.pkl` 实际指向自然季度 Search 数据。R3 开始前，Fixed、Selective、Random 三条 W1 分支均使用 controlled quarterly 数据重新物化并训练。因此本阶段三种方法从 W2 到 W10 的比较使用相同数据、任务、训练预算和初始化规则，只有 SID 更新策略不同。

修正后的 W1 更新、W2 测试结果为：

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| Fixed | 0.0520 | 0.02967 | 0.0772 | 0.04005 |
| Selective | 0.0522 | 0.03213 | 0.0768 | 0.04028 |
| Random | 0.0518 | 0.02957 | 0.0770 | 0.04020 |

## 3. 三种滚动策略

### Fixed

始终保留 t0 的 shared prefix、Search suffix 和 Rec suffix。下游 T5 仍会用每个新窗口的数据继续训练，因此它是“下游正常更新，但 SID 不更新”的基线。

### Selective

1. 根据相邻窗口的任务级漂移指标选择 Search/Rec 候选 item；
2. shared prefix 和 shared codebook 始终冻结；
3. Search adapter 与 Rec adapter 分别用本任务新信号和 replay 样本训练；
4. 使用 sticky ratio `0.1`，仅在新码相对旧码具有足够优势时替换对应 suffix；
5. Search 与 Rec suffix 独立更新，再与不变的 shared prefix 合成完整 SID；
6. 下一窗口从当前 adapter、SID mapping 和下游 checkpoint 继续滚动。

### Random

使用与 Selective 匹配的换码预算和相同训练流程，但随机选择需要换码的 item，用于检验收益是否只是来自“更新了一批 SID”。Random 和 Selective 分别维护自己的历史 mapping，避免跨分支污染。

## 4. 下游训练控制

三种策略均采用同一 GenSAR full-task 增量训练设置：

- backbone：`T5-small`；
- train tasks：`SeqRecWithSrcHis, PerSrcWithRecHis, Item2Index, Index2Item`；
- Search/Rec SID alignment 使用 task-local item pool；
- `Item2Index` 和 `Index2Item` 的每任务样本预算均为 1,800；
- 每窗口训练 1 epoch，seed 为 2024；
- 每条分支从自己上一窗口的最佳 checkpoint warm-start；
- 候选评估为 1 个 ground truth 加 999 个 negatives，beam size 为 10。

因此，Fixed 与两种换码策略都承担相同的下游增量训练；实验隔离的是 SID 更新策略，而不是比较“完全不训练”与“继续训练”。

## 5. 九窗口结果

下表用两个任务 `nDCG@10` 的算术平均作为紧凑的跨窗口主指标：

| Eval window | Fixed | Selective | Random | Selective-Fixed | Selective-Random |
|---|---:|---:|---:|---:|---:|
| W2 | 0.034860 | 0.036204 | 0.034888 | +0.001344 | +0.001317 |
| W3 | 0.036224 | 0.036828 | 0.036240 | +0.000604 | +0.000588 |
| W4 | 0.028863 | 0.028396 | 0.029449 | -0.000467 | -0.001053 |
| W5 | 0.028972 | 0.029459 | 0.028706 | +0.000487 | +0.000754 |
| W6 | 0.026122 | 0.026548 | 0.025988 | +0.000426 | +0.000561 |
| W7 | 0.021464 | 0.021842 | 0.021757 | +0.000378 | +0.000085 |
| W8 | 0.021999 | 0.022070 | 0.022000 | +0.000071 | +0.000070 |
| W9 | 0.025383 | 0.025837 | 0.026373 | +0.000455 | -0.000536 |
| W10 | 0.025754 | 0.026153 | 0.024966 | +0.000399 | +0.001188 |

九窗口聚合结果：

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 | Mean nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Fixed | 0.039422 | **0.024352** | 0.057733 | 0.031124 | 0.027738 |
| Selective | **0.039578** | 0.024321 | **0.058733** | **0.031977** | **0.028149** |
| Random | 0.039156 | 0.024083 | 0.058533 | 0.031554 | 0.027818 |

注意：表中的粗体表示三种方法中的数值最大；Selective 的 Rec nDCG 实际略低于 Fixed，仅高于 Random。

关键 paired 结果：

| 对比 | 指标 | 平均差值 | 正窗口数 |
|---|---|---:|---:|
| Selective-Fixed | Rec H@10 | +0.000156 | 6/9 |
| Selective-Fixed | Rec nDCG@10 | -0.000032 | 4/9 |
| Selective-Fixed | Search H@10 | +0.001000 | 6/9 |
| Selective-Fixed | Search nDCG@10 | +0.000853 | 7/9 |
| Selective-Fixed | Mean nDCG@10 | +0.000411 | 8/9 |
| Selective-Random | Rec nDCG@10 | +0.000237 | 5/9 |
| Selective-Random | Search nDCG@10 | +0.000423 | 5/9 |
| Selective-Random | Mean nDCG@10 | +0.000330 | 7/9 |

## 6. SID 稳定性与 collision

从 W1 到 W9 的累计换码事件数为：

| 方法 | Search suffix 换码事件 | Rec suffix 换码事件 |
|---|---:|---:|
| Selective | 521 | 693 |
| Random | 797 | 808 |

同一 item 可能跨窗口重复出现，因此该数值是事件数而不是 distinct item 数。仅计 R3 新执行的 W2-W9，Selective 为 506/680，Random 为 783/791。

W9 更新完成后的完整 SID collision：

| 方法 | Search duplicate rate | Search collided-item rate | Rec duplicate rate | Rec collided-item rate |
|---|---:|---:|---:|---:|
| Fixed | 0.6200% | 0.8450% | 0.4050% | 0.4350% |
| Selective | **0.4850%** | **0.6900%** | **0.3300%** | **0.3500%** |
| Random | 0.5850% | 0.7950% | 0.3650% | 0.3850% |

多轮 suffix 更新没有造成 collision 累积爆炸。Selective 最终 collision 反而低于固定 t0 mapping，说明 sticky 局部更新至少在当前规模下不会破坏 SID 唯一性。

## 7. 时间成本

W2-W9 的 SID 学习关键路径由累计 UniSAR 更新与两个并行 task adapter 中较慢者组成：

- 8 次更新总关键路径：447.23 秒；
- 平均每窗口：55.90 秒；
- 单窗口范围：39.62-67.51 秒。

下游 T5 增量训练平均每个窗口、每个方法约 620-628 秒。也就是说，当前实现中 SID adapter 更新约占一次完整滚动更新额外时间的 9%。LoRA/adapter 明显降低了 SID 表示更新成本，但系统总成本仍主要来自所有方法都需要执行的下游模型更新。

## 8. 阶段结论

1. 固定 shared prefix、独立选择性更新 Search/Rec suffix 可以连续滚动至少 9 个评估窗口；
2. Selective 在双任务平均 nDCG 上相对 Fixed 为 `+0.000411`，并在 8/9 窗口为正；相对 matched-budget Random 为 `+0.000330`，在 7/9 窗口为正；
3. 主要稳定收益来自 Search：Search nDCG 相对 Fixed 平均 `+0.000853`；
4. Rec nDCG 相对 Fixed 为 `-0.000032`，基本持平但没有形成稳定正收益。W10 尤其表现出 Search 改善、Rec 下降的任务权衡；
5. collision 没有随滚动更新恶化，当前更需要解决的是 Rec selector/adapter 的有效性，而不是继续压低全局 collision；
6. 本阶段只有一个下游 seed。九个连续窗口是相关的时间观测，不能替代独立 seed，因此当前结果是可靠的阶段性证据，但还不是论文最终统计结论。

## 9. 下一步

优先进入 R4，而不是继续扩大模型结构：

1. 对完整 rolling protocol 跑 3 个 seeds，报告按窗口配对的均值、方差和置信区间；
2. 分别评估当期 drift-selected item、实际换码 item、未换码 item，确认全局收益由哪些样本贡献；
3. 针对 Rec 的弱收益检查 selector precision、重复换码和 suffix alignment，避免用 Search 的稳定收益掩盖 Rec 退化；
4. 增加 `full SID retrain` 或 `full task representation retrain` 成本/效果基线，才能完整支持“局部更新更高效”的论文论断；
5. 在自然 Search drift 上重复一次主结论，把 controlled drift 作为机制验证、natural drift 作为外部有效性验证。

## 10. 主要产物

- 滚动编排：`stage3/scripts/run_temporal_rolling_r3.sh`
- 累计 alignment pool：`stage3/scripts/build_rolling_alignment_pools.py`
- Search/Rec mapping 合成：`stage3/scripts/compose_temporal_task_mappings.py`
- task adapter 合并：`stage3/scripts/merge_v13_task_adapters.py`
- 下游 warm-start：`stage3/scripts/run_temporal_incremental_t5.sh`
- 总结脚本：`stage3/scripts/summarize_temporal_rolling_r3.py`
- 汇总目录：`stage3/results/temporal_rolling_r3_summary/`
- 每窗口 SID 与 snapshot：`stage3/data/temporal_rolling_r3/`
- 每窗口下游模型：`stage3/runs/temporal_incremental/R3_*`
