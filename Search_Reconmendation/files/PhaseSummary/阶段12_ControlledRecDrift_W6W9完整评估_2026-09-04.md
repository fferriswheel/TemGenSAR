# 阶段12：Controlled Rec Drift 的 W6/W9 完整评估（2026-09-04）

## 1. 本阶段目标

本阶段继续阶段11的 controlled Rec drift benchmark，回答三个问题：

1. 修正后的 detector 能否在 W6 上稳定改善 Fixed SID；
2. detector 是否优于同预算高支持量选择，而不只是挑中了热门 item；
3. 固定 detector 和更新预算后，结论能否在未参与调参的 W9 漂移点复现。

同时测量 selective Rec-adapter update 相对全 item adapter update 的成本，并验证 prefix、Search suffix、未选 item 和完整 SID collision 的稳定性。

## 2. Benchmark 与时间协议

- 数据：GenSAR-faithful 20K quarterly benchmark，19,999 items。
- 漂移类型：固定 30% user cohort，在同 root+leaf category 内做一对一 item rotation；Search 行为不注入漂移。
- 漂移强度：W3/W6/W9 起始 event rate 分别为 25%/50%/75%。
- W6 实际注入：2013Q2 有 3,258 条替换事件；W6 update 后用 W7 test。
- W9 实际注入：2014Q1 有 5,758 条替换事件；W9 update 后用 W10 test。
- representation 只使用当前窗口截止前已揭示的行为，W10 不参与 W9 SID 生成。
- W6 用于 detector 修正、K 选择和多 seed 验证；W9 只做留出确认，不再调 detector 或 K。

下游统一使用 T5-small，一轮增量训练，训练任务为：

```text
SeqRecWithSrcHis + PerSrcWithRecHis + Item2Index + Index2Item
```

每组使用 1,800-item balanced alignment pool。每次比较内，所有方法的 Rec/Search train、validation、test 和 item-set 文件逐字节一致，并从同一个前序 Fixed checkpoint 开始。

## 3. SID 更新方法

### 3.1 固定结构

```text
[shared prefix: 2 codes | task suffix: 2 codes]
codebook size = [256, 256 | 256, 256]
```

更新 Rec 时：

- shared prefix 固定；
- Search suffix 固定；
- shared codebook 与 Rec codebook 固定；
- 只训练 14,528 个 Rec adapter 参数；
- selected item 加 2 倍 replay item；
- sticky switch ratio 固定为 0.1；
- 未选 item 的 operational SID 必须逐项不变。

### 3.2 对照组

- `Fixed`：不更新 SID，但使用相同窗口数据更新下游模型。
- `Random`：在相同 support bucket 内随机选择 51 个 item。
- `SupportTopK`：按 event/transition 总支持量选择 51 个 item。
- `V1`：阶段11的双侧最小支持量 drift score。
- `V2`：允许单侧 emerging/disappearing，并做 gain/loss 与 counterpart 配额。

### 3.3 Detector V3

V2 在 W6 smoke 中失败。它虽然提高了人工 target coverage，但偏向极低支持量、高相对变化的 tail item；selected item 的 W6 current-support 中位数仅为 10，约四分之一 item 只有 0 到 2 条当前事件。这些 item 即使换码，也缺少足够下游训练信号。

V3 保留 V2 的双向 drift score，但加入当前影响量：

```text
eligible: right_events >= 5
selection_utility = drift_rank_score * sqrt(right_events)
select top-K
```

该改动在 W3 development window 上确定，W6 用于 smoke 与预算选择，W9 不再修改。

## 4. W6 单 seed smoke

W6 update、W7 test，seed 2024。V3 使用原始公共 alignment pool。

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| Fixed | 0.0304 | 0.019331 | 0.0404 | 0.021576 |
| Random | 0.0308 | 0.019863 | 0.0400 | 0.021452 |
| SupportTopK | 0.0310 | 0.019808 | 0.0402 | 0.021531 |
| V1 | 0.0308 | 0.019869 | 0.0400 | 0.021595 |
| V2 | 0.0306 | 0.019398 | 0.0402 | 0.021503 |
| V3 | **0.0316** | **0.020169** | **0.0404** | **0.021732** |

V3 相比 Fixed 的 Rec nDCG@10 提升 0.000839，相对提升 4.34%；相比 SupportTopK 提升 0.000361。Search 未退化，因此 W6 smoke 通过。

## 5. W6 三 seed 结果

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| Fixed | 0.030867 | 0.019476 | 0.039400 | 0.021221 |
| SupportTopK | 0.031200 | 0.019914 | 0.039667 | 0.021322 |
| V3 | **0.031400** | **0.020050** | **0.039733** | **0.021408** |

配对差值：

| 比较 | Rec nDCG@10 mean delta | 正向 seed | Search nDCG@10 mean delta | 正向 seed |
|---|---:|---:|---:|---:|
| V3 - Fixed | +0.000574 | 3/3 | +0.000187 | 3/3 |
| V3 - SupportTopK | +0.000136 | 2/3 | +0.000086 | 2/3 |

因此，“V3 优于 Fixed”在当前三 seed 上稳定；“V3 优于 SupportTopK”只有较弱证据，不能表述为稳定显著优势。

## 6. W6 alignment-budget curve

本实验重新构造一个覆盖所有 K=200 候选的公共 1,800-item alignment pool。Fixed 与每个 V3/Support arm 使用相同数据和起点。

| K | V3 Rec nDCG@10 | Support Rec nDCG@10 | V3-Support | V3 Search nDCG@10 | Support Search nDCG@10 |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.019810 | 0.019930 | -0.000121 | 0.021979 | 0.021778 |
| 51 | **0.020006** | 0.019862 | **+0.000144** | 0.021795 | 0.021788 |
| 100 | 0.019776 | 0.019977 | -0.000201 | 0.022049 | 0.021864 |
| 200 | 0.019864 | 0.019973 | -0.000109 | 0.022003 | 0.022089 |

同协议 Fixed 的 Rec/Search nDCG@10 为 0.019454/0.021829。

预先采用以下选择规则：Rec 为主指标、Search 不明显下降、优先较小预算，并要求 V3 同时胜过 Fixed 和同预算 Support。只有 K=51 满足，因此在查看 W9 结果前固定 `K=51`。

## 7. SID 稳定性与更新成本

| 窗口 | selected | 实际换码 | adapter steps | adapter wall time | 完整 SID duplicate fraction |
|---|---:|---:|---:|---:|---:|
| W6 V3-K51 | 51 | 33 | 45 | 4.60s | 0.400% |
| W9 V3-K51 | 51 | 36 | 45 | 5.67s | 0.395% |

两次更新均满足：prefix 不变、Search suffix 不变、未选 item SID 不变，collision 没有相对初始 mapping 恶化。

W6 全 item Rec-adapter update 作为偏乐观的 full-retokenization 成本下界：

| 方法 | 训练 item | wall time | raw changed items | raw duplicate fraction |
|---|---:|---:|---:|---:|
| V3 selective | 153（51 selected + 102 replay） | 4.60s | 最终 33 | 0.400% |
| Full-item adapter | 19,999 | 25.77s | 17,347 | 12.396% |

在当前硬件与实现下，full-item adapter wall time 为 selective 的约 5.6 倍。它还只是“同一个小 adapter 上训练全部 item”，未包含重新训练完整 RQ-VAE/codebook 的成本，所以应视为全量重建的保守下界，而不是完整 GenSAR retokenization 复刻。

## 8. W9 留出验证

W9 update、W10 test，K=51、detector 参数和 sticky ratio 全部从 W6 固定。四组从共同的 `CRD_W8_Fixed_s2024` checkpoint 开始。

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| Fixed | 0.0288 | 0.018533 | 0.0500 | 0.028408 |
| Random | 0.0304 | 0.019149 | 0.0502 | 0.028481 |
| SupportTopK | **0.0310** | **0.019477** | 0.0506 | 0.028611 |
| V3 | **0.0310** | 0.019464 | **0.0510** | **0.028723** |

V3 相比 Fixed：

- Rec nDCG@10：+0.000931，relative +5.02%；
- Search nDCG@10：+0.000315，relative +1.11%。

V3 相比 Random：Rec/Search 均更高。V3 相比 SupportTopK：Rec 低 0.000013，基本持平但未胜出；Search 高 0.000112。按预设的 Rec 主指标门槛，W9 的“优于 SupportTopK”未通过。

### W9 子集诊断

- V3 与 clean-trajectory 选择重合 43/51，Jaccard 为 0.729，表明 detector 确实响应了注入行为。
- 在 W9 的 5,758 条注入事件中，V3 覆盖 248 条 target 事件，SupportTopK 仅覆盖 27 条；V3 的 source coverage 为 684，Support 为 767。
- W10 的 482 条 injected-target 测试样本上，V3 与 Support 的 nDCG@10 都是 0.005543。
- V3-only 实际换码的 19 个 item 在 W10 对应 83 条 target 样本，但 Fixed/Random/Support/V3 的 nDCG@10 都为 0。

这说明 V3 检测到了更多 emerging target，但当前“一轮下游更新 + 每 item 极少 alignment supervision”没有学会这些 tail SID，因此 detector coverage 没有转化为相对 Support 的收益。

## 9. 当前可以与不可以支持的结论

### 可以支持

1. 固定 prefix 和 Search suffix、只更新少量 Rec suffix 是可行的，且不会导致 collision 恶化。
2. 在相同下游增量训练预算下，V3 selective update 在 W6 三 seed 与 W9 留出窗口均优于 Fixed。
3. Rec-only SID 更新未损伤 Search；本轮 Search 反而小幅正向。
4. selective adapter 的 item/step/wall-time 成本明显低于全 item adapter，并避免大范围换码。

### 尚不能支持

1. 不能声称 drift detector 稳定优于 SupportTopK：W6 为 2/3 seed 正向，W9 Rec 基本打平且略低。
2. 不能声称已充分适应人工注入的 replacement target：W9 target coverage 提高，但实际排序收益没有同步出现。
3. 不能把 5.6 倍直接写成相对“完整重新训练 RQ-VAE”的最终加速比；当前 full-item adapter 只是成本下界。

## 10. 对当前 benchmark 的阶段结论

当前 controlled Rec-drift benchmark 上的既定评估已完成：W6 smoke、detector 修正、budget curve、三 seed、W9 留出和成本上界均已覆盖。

最稳妥的论文叙述是：`impact-aware selective task-local SID update` 能以很小的换码范围稳定优于 Fixed，并保持另一任务；但 detector 相对高支持量启发式的独立价值仍不够强。下一阶段不应继续在 W9 调参，而应优先解决 emerging SID 的下游 alignment/可学习性，或转到第二个包含自然 Rec drift 的 benchmark 做外部验证。

## 11. 主要产物

- W6 smoke：`stage3/results/controlled_rec_w6_smoke_v3/`
- W6 三 seed：`stage3/results/controlled_rec_w6_v3_three_seed/`
- W6 budget curve：`stage3/results/controlled_rec_w6_budget_curve/`
- W9 holdout：`stage3/results/controlled_rec_w9_k51_holdout/`
- W6/W9 SID：`stage3/data/controlled_rec_drift_w3w6w9_v1/W6/`、`W9/`
- W7-W9 representation：`stage3/artifacts/unisar_controlled_rec_W{7,8,9}/`
- 下游模型：`stage3/runs/temporal_incremental/CRD_W6_*`、`CRD_W9_*`
- Detector policy：`stage3/scripts/build_detector_policy_selection.py`
- 汇总脚本：`stage3/scripts/summarize_controlled_rec_w6.py`、`summarize_controlled_rec_w6_budget.py`、`summarize_controlled_rec_w6_three_seed.py`

所有本阶段下游运行的中间 `checkpoint-*` 已清理，只保留最终 best model 与评估结果。
