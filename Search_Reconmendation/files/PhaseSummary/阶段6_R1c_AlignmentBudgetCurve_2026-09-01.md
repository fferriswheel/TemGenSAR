# 阶段6 R1c：Alignment-Budget Curve（2026-09-01）

## 1. 目的与控制变量

固定 R1b drift-selected SID、`sticky ratio=0.1` 和所有下游配置，只改变 `Item2Index`/`Index2Item` 的 `train_data_sample_num`。预算仅使用 W1 validation 的 Rec/Search 平均 nDCG@5 选择，W2 不用于调参。

GenSAR 中预算 `0` 表示使用全部 item，不表示零 alignment。由于当前有 12 个 changed item，`0` 与 `12` 都产生 24 条 semantic/collaborative 样本，因此删除冗余的预算 0 点。

## 2. W1 validation 曲线

| Config budget | Actual examples/task | Steps | Wall time | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 | Mean nDCG@5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 24 | 102 | 5:50 | 0.0672 | 0.04344 | 0.0762 | 0.03825 | 0.03395 |
| 60 | 120 | 103 | 6:15 | 0.0668 | 0.04302 | 0.0828 | 0.04192 | 0.03520 |
| 120 | 240 | 105 | 6:28 | 0.0672 | 0.04295 | 0.0834 | 0.04283 | 0.03568 |
| 600 | 1,200 | 120 | 7:08 | 0.0654 | 0.04231 | 0.0848 | 0.04470 | 0.03630 |
| 1,200 | 2,400 | 139 | 7:41 | 0.0658 | 0.04278 | 0.0860 | 0.04581 | 0.03708 |
| **1,800** | **3,600** | **158** | **8:04** | **0.0672** | **0.04351** | **0.0862** | **0.04675** | **0.03804** |
| 2,567 | 5,134 | 182 | 8:33 | 0.0660 | 0.04346 | 0.0860 | 0.04753 | 0.03896 |

W1 aggregate 指标在 2,567 仍最高，因此曲线并未完全饱和。但 1,800 已达到最大预算 Mean nDCG@5 的 97.6%，并将 optimizer steps 从 182 降到 158（-13.2%）。Rec validation 在 1,800 也不低于 2,567，剩余差距主要来自 Search nDCG。

## 3. 固定选择后的 W2 结果

| Config budget | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---:|---:|---:|---:|---:|
| 12 | 0.0476 | 0.02833 | 0.0586 | 0.03069 |
| 60 | 0.0474 | 0.02845 | 0.0624 | 0.03284 |
| 120 | 0.0468 | 0.02809 | 0.0632 | 0.03335 |
| 600 | 0.0484 | 0.02924 | 0.0634 | 0.03381 |
| 1,200 | 0.0490 | 0.02982 | 0.0638 | 0.03412 |
| **1,800** | **0.0514** | **0.03179** | **0.0640** | **0.03474** |
| 2,567 | 0.0506 | 0.03123 | 0.0610 | 0.03468 |

1,800 在四个 W2 指标上均不劣于 2,567，并明显缓解了 R1b 原配置的 Search H@10 下降。这不用于反向选择预算，只作为 W1 选择 1,800 后的 held-out 验证。

## 4. 结论

后续的 R1b 默认 alignment config budget 改为 **1,800**：

- 实际每个 alignment task 3,600 条样本；
- 相比 2,567 配置减少 13.2% optimizer steps；
- W1 aggregate 保留 97.6%；
- W2 Rec/Search 四个指标均不降。

它是当前单 seed 下的成本性能折中，不是已经被证明可跨 seed/跨窗口固定的常数。下一步应对 1,800 跟 E2/budget-random 运行 3 seeds，并在后续时间窗中保持该值不再调整。

## 5. 产物

- Curve table: `stage3/runs/temporal_r1b/alignment_budget_curve/alignment_budget_curve.csv`
- Curve JSON: `stage3/runs/temporal_r1b/alignment_budget_curve/alignment_budget_curve.json`
- Runs: `stage3/runs/temporal_incremental/R1B_drift_sticky_r01_align*`
- Parser: `stage3/scripts/summarize_alignment_budget_curve.py`
