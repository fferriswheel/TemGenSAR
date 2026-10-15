# 阶段5：第一轮 Temporal Update 实验（2026-08-14）

## 1. 实验目的

在 strict 20K Amazon Electronics 数据上，将 `W1` 作为可用的新行为窗口，比较不同更新策略在下一未来窗口 `W2` 的 Search/Rec 表现。本轮主要回答：

1. 仅更新 Rec SID 而不更新下游模型是否有效；
2. 更新下游模型时，动态 SID 是否优于固定 SID；
3. v13b 的局部 SID 更新与 GenSAR-style 全量 SID 重建的成本和可恢复性有何区别；
4. 从头全量重训下游模型能提供怎样的参考上界。

## 2. 时间协议与数据

```text
t0 representation / SID / downstream train: <= 2011-12-31
W1 update:                                2012-01-01 ~ 2012-03-31
W1 validation:                            5,000 Rec + 5,000 Search
W2 test:                                  2012-04-01 ~ 2012-06-30, 5,000 Rec + 5,000 Search
```

W1 新增训练数据：

| Data | Samples |
|---|---:|
| Rec update | 11,912 |
| Search update | 8,826 |
| 动态 SID 发生变化的 item | 2,567 |
| 实际有 W1 Rec 行为的 active item | 6,208 |

评估仍使用 GenSAR 式的 1 positive + 999 negatives candidate ranking，报告 `Hit@10` 和 `nDCG@10`。

## 3. 动态 v13b SID 更新

本轮使用 `v13b Rec adapter update + active-only constrained R2 assignment`：

```text
fixed:  shared prefix, Search suffix, shared codebook, inactive-item Rec SID
update: UniSAR Rec representation -> Rec adapter -> active-item R2 assignment
```

具体性质：

| Property | Result |
|---|---:|
| Prefix 完全保持 | Yes |
| Search SID 完全保持 | Yes |
| Inactive-item Rec SID 完全保持 | Yes |
| 全部 item 的 Rec SID 变更率 | 12.84% |
| Active item 的 Rec SID 变更率 | 41.35% |
| Rec duplicate fraction: t0 -> update | 0.405% -> 0.385% |
| Rec collision groups: t0 -> update | 6 -> 5 |

本轮真正训练的 Rec adapter 参数量为 14,528。代表性时间成本：

| Stage | Wall time |
|---|---:|
| W1 UniSAR representation update | 0:42 |
| Rec adapter replay/stability update | 3:07 |
| SID materialization + constrained assignment | 0:08 |
| SID update total | about 3:58 |

## 4. E0-E5 定义

| Exp | SID | Downstream update | Purpose |
|---|---|---|---|
| E0 | t0 v13b SID | None | 完全不更新 |
| E1 | dynamic v13b Rec SID | None | 只换 SID，不训练下游模型 |
| E2 | t0 v13b SID | W1 incremental, 1 epoch | 只更新下游模型 |
| E3 | dynamic v13b Rec SID | W1 incremental, 1 epoch | 动态 SID + 下游增量更新 |
| E4 | GenSAR faithful W1 from-scratch SID | W1 incremental, 1 epoch | 全量 SID 重建成本对照 |
| E5 | dynamic v13b Rec SID | t0+W1 full-task from scratch, 3 epochs | 全量下游重训参考 |

E2/E3 的下游训练样本、初始 checkpoint、optimizer、epoch 和对齐任务预算完全相同，唯一实质变量是 SID mapping。

E4 不是与 E2/E3 同更新预算的严格公平对照。由于 GenSAR 从头重训 RQ-VAE 后 code label 可置换，Search/Rec SID 都几乎 100% 换号，必须使用全部 W1 Search/Rec 和全 item Index2Item/Item2Index 进行重对齐。因此 E4 的作用是表明“不保留 SID 坐标系”的恢复难度，不能直接解读为 GenSAR 本身的最佳效果。

## 5. W2 最终结果

| Exp | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| E0 No update | 0.0458 | 0.02430 | 0.0592 | 0.03099 |
| E1 SID-only update | 0.0458 | 0.02294 | 0.0592 | 0.03099 |
| E2 Downstream-only | 0.0500 | **0.02718** | **0.0656** | **0.03426** |
| E3 Dynamic SID + incremental | **0.0506** | 0.02652 | 0.0652 | 0.03398 |
| E4 GenSAR full SID rebuild + incremental | 0.0082 | 0.00359 | 0.0082 | 0.00398 |
| E5 Dynamic SID + full retrain | **0.0524** | **0.02721** | **0.0654** | **0.03463** |

相对 E0：

- E2 将 Rec H@10 提高 0.0042，Search H@10 提高 0.0064，说明 W1 下游增量训练有效。
- E3 将 Rec H@10 提高 0.0048，但 Rec nDCG@10 低于 E2 0.00067。
- E2 与 E3 的差异只在千分位，且指标方向不一致；当前单 seed 不支持“动态 SID 已优于固定 SID”的结论。
- E1 说明 SID 变更后不做下游对齐不会自动带来收益，甚至会降低排序质量。
- E4 的崩塌与新 SID 坐标系几乎全量换号相符；1 epoch 的增量训练不足以重学全部 item-token 映射。
- E5 与 E3 最终非常接近；在当前规模上，全量重训没有显示出足以抵消其成本的 W2 优势。

## 6. E5 训练趋势

E5 使用 `t5-small` 从头训练 3 epochs，包含 GenSAR 六任务：SeqRec、PerSrc、QGen、Doc2Query、Item2Index 和 Index2Item。以 W1 Rec/Search 验证集两个 `nDCG@5` 的平均值选择 checkpoint。

| Epoch | W1 Rec nDCG@5 | W1 Search nDCG@5 | Mean |
|---:|---:|---:|---:|
| 1 | 0.03121 | 0.02799 | 0.02960 |
| 2 | 0.03150 | 0.03022 | 0.03086 |
| 3 | 0.03408 | 0.03546 | **0.03477** |

Epoch 3 仍有明显改善，因此不能认为 3 epochs 已达饱和。`best_model_checkpoint` 为 `checkpoint-4338`。

## 7. 成本对比

| Exp/stage | Wall time |
|---|---:|
| E2 downstream incremental | 8:36 |
| E3 downstream incremental | 8:32 |
| E4 GenSAR RQ-VAE + index generation | 7:26 |
| E4 downstream re-alignment | 18:59 |
| E5 full-task downstream retrain | 40:19 |

v13b 的动态 SID 更新约 4 分钟，加 E3 下游增量训练合计约 12.5 分钟。E5 仅下游全量重训就需要 40 分钟。这一轮已显示局部更新的成本优势，但效果优势尚未建立。

## 8. 阶段结论与下一步

当前可以确认：

1. 时间漂移后必须更新下游生成模型，仅替换 SID 不足够。
2. 保持 prefix/Search suffix 和大多数 Rec SID 不变，可以使单轮增量训练正常恢复；全量换号则不能。
3. E3 的时间成本显著小于 E5，但 E3 尚未在总体指标上稳定超过 E2。

下一轮不应直接扩大数据或继续堆叠更复杂 loss，而应先完成：

1. 在 W2 Rec test 上拆分 SID-changed target item 与 unchanged target item，检查收益是否集中在真正更新的 item。
2. 对 E2/E3 增加 3 seeds，报告 mean/std 和 paired significance，避免解读千分位随机差异。
3. 增加 matched-budget random-update 和 drift-selected update，验证 item selector 是否有效。
4. 继续 W2 -> W3 的 test-then-train，统计多轮 cumulative performance drop、SID churn 与 wall time。
5. 后续 Controlled Drift 在 W3/W6/W9 扩展时，同时记录 SID update 和 downstream update 成本，而不只报告最终排序指标。

## 9. 主要产物

- Dynamic SID statistics: `stage3/data/gensar_faithful_20k_rec_w1/v13b_temporal_rec_update/temporal_sid_stats.json`
- E2 log: `stage3/runs/temporal_incremental/E2_downstream_only/train.log`
- E3 log: `stage3/runs/temporal_incremental/E3_dynamic_sid/train.log`
- E4 logs: `stage3/runs/temporal_e4/rqvae.log`, `generate.log`, `t5_incremental/train.log`
- E5 log: `stage3/runs/temporal_e5/full_retrain_dynamic_sid/train.log`
- E5 best checkpoint: `stage3/runs/temporal_e5/full_retrain_dynamic_sid/checkpoint-4338`
