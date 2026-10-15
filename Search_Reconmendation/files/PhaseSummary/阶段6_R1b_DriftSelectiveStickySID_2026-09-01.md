# 阶段6 R1b：Drift-Selective Sticky SID（2026-09-01）

## 1. 实验目的

R1 发现 active-all 动态 SID 的 code churn 过高，而且在高漂移 item 上显著降低了 Rec 排序质量。R1b 不改变整体 temporal protocol，只修正“哪些 item 可以换码”和“什么时候值得换码”。

比较三种 Rec SID 策略：

```text
E2:     t0 SID，只做 W1 downstream incremental update
Active: 原 E3 active-all dynamic SID
R1b-B:  detector drift-selected + sticky assignment
R1b-C:  相同候选数的 budget-random + sticky assignment
```

## 2. 时间和数据协议

```text
t0:        <= 2011-12-31
W1 update: 2012Q1
W2 test:   2012Q2
```

| Split | Rec | Search |
|---|---:|---:|
| W1 downstream update | 11,912 | 1,000 replay |
| W1 validation | 5,000 | 5,000 |
| W2 test | 5,000 | 5,000 |

下游两组 R1b 使用相同的 t0 checkpoint、random seed、optimizer、1 epoch、182 optimizer steps 和 23,180 条训练样本。Item2Index/Index2Item 的配置预算均为 2,567；GenSAR 会对 semantic/collaborative 两种 index 各生成一条样本，所以每个 alignment task 的实际样本数为 5,134。R1b-B/C 的 downstream 训练预算仍完全相同。

## 3. Drift selector

Item drift 只使用 W1 及其之前信息，不读取 W2 label。对 `C8 -> W1` 同时有足够事件和 transition support 的 209 个 item，按 drift rank score 取 top 30%：

| Statistic | Items |
|---|---:|
| support eligible | 209 |
| drift-selected | 63 |
| budget-random | 63 |

Eligible 规则要求前后窗口都有行为支持，所以排除了 t0-unseen/newly-active cold-start item。Random 组优先按 event support 就近抽样；由于所有 `50+` support item 都被 detector 选中，random pool 无法完全复制该 support bucket，因此它是 matched-budget random，不是严格 support-matched random。

## 4. Sticky hierarchical assignment

共享 prefix 和 Search suffix 全部冻结，只允许 selector 中的 item 更新 Rec suffix。每层量化使用：

```text
cost(i, c) = latent_distance(i, c)
           + switch_ratio * median(old-code distance) * 1[c != old_code]
```

R2 在每个 `[P1,P2,R1]` route 内用 Hungarian assignment，冻结 item 已占用的 code 不得被新分配。已有 collision 可保留，但不引入新 collision。

SID-only sweep 最终选择 `switch_ratio=0.1`：它在低 churn 下仍允许少量 R1 route 更新，而 `>=0.2` 只会改 R2。

| Policy | Selected | SID changed | R1 changed | R2 changed | Duplicate fraction |
|---|---:|---:|---:|---:|---:|
| Drift-selected | 63 | 12 | 2 | 11 | 0.405% |
| Budget-random | 63 | 8 | 0 | 8 | 0.400% |
| Active-all E3 | 6,208 active | 2,567 | 1,193 | 2,555 | - |

Drift-selected 将实际 SID churn 从 2,567 个 item 降到 12 个，且 selector 外 SID 逐 item exact match t0 mapping。SID 层面的 route agreement 没有恶化。

## 5. W2 全量结果

| Method | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| E0 no update | 0.0458 | 0.02430 | 0.0592 | 0.03099 |
| E2 downstream-only | 0.0500 | 0.02718 | **0.0656** | 0.03426 |
| Active-all E3 | **0.0506** | 0.02652 | 0.0652 | 0.03398 |
| R1b budget-random | 0.0504 | 0.02740 | 0.0658 | 0.03465 |
| R1b drift-selected | **0.0506** | **0.03123** | 0.0610 | **0.03468** |

R1b drift-selected 相比 E2：

```text
Rec H@10:     +0.0006, p=0.661
Rec nDCG@10: +0.00405, p<0.0001
```

相比 active-all E3，R1b 保持了相同 Rec H@10，但将 Rec nDCG@10 从 0.02652 提高到 0.03123。这说明 R1 的主要问题确实是过度换码和更新对象错位。

Search H@10 比 E2 低 0.0046，而 nDCG@10 基本持平。Search SID 本身未变，该差异来自共享 T5 在 Rec SID alignment 样本上的参数更新，说明下一轮需要检查 Search replay 比例或 task-local downstream adapter。

## 6. 逐样本诊断

### 6.1 Detector 选中的 W2 target

| Method | N | H@10 | nDCG@10 |
|---|---:|---:|---:|
| E2 | 508 | 0.3071 | 0.1708 |
| Budget-random | 508 | **0.3110** | 0.1762 |
| Drift-selected | 508 | 0.3051 | **0.2053** |

Drift-selected 相比 budget-random：

```text
H@10:    -0.00591, p=0.515
nDCG@10: +0.02910, p<0.0001
```

改进主要表现为将正确 item 提到更前的 beam rank，而不是增加 top-10 coverage。

### 6.2 Drift-selected 且实际换码

12 个换码 item 在 W2 中对应 103 条 target 样本：

| Method | H@10 | nDCG@10 |
|---|---:|---:|
| E2 | 0.4175 | 0.2687 |
| Budget-random | 0.4175 | 0.2924 |
| Drift-selected | **0.4272** | **0.4047** |

Drift-selected 相比 budget-random 的 nDCG@10 提高 0.1122，`p<0.0001`。这与 R1 中 active-all 在 changed+selected 上显著降低 nDCG 的结果相反。

## 7. 成本与实现边界

| Component | Drift-selected | Budget-random |
|---|---:|---:|
| Actual SID changes | 12 | 8 |
| Downstream train samples | 23,180 | 23,180 |
| Actual examples per alignment task | 5,134 | 5,134 |
| Optimizer steps | 182 | 182 |
| Downstream wall time | 8:33 | 8:38 |

本轮为了单独检验 assignment policy，使用已经在 W1 behavior representation 上更新过的 Rec adapter/private latent，然后只对 selector item 执行 sticky SID assignment。因此当前可以声称“SID 换码与下游对齐范围高度稀疏”，但不能声称“LoRA/adapter 反向传播成本也按 63 个 item 稀疏化”。

另一个限制是，12 个 changed item 在配置预算 2,567（每个 alignment task 实际 5,134 条）下会被高频重复。这保证了与 E2/E3 的 downstream 总预算一致，但可能放大 targeted item 上的收益。后续必须增加 alignment-budget curve。

## 8. 阶段结论

R1b 通过了单 seed proof-of-concept：

1. drift selector + old-code switching cost 可将 SID churn 降低约 99.5%；
2. collision 保持在 0.4% 左右，没有以 uniqueness post-hoc 换取效果；
3. 总体 Rec H@10 不降，Rec nDCG@10 显著优于 E2、active-all E3 和 budget-random；
4. 收益确实集中在 detector 选中且实际换码的 item；
5. Search H@10 的退化和小换码集的重复对齐仍是未解问题。

因此下一步不应再调 sticky ratio，而应进入：

```text
R1c-1: alignment sample budget curve
R1c-2: 3 seeds for E2 / random / drift-selected
R1c-3: Search replay ratio or task-local downstream adapter
R2:    selective adapter training and multi-window rolling evaluation
```

## 9. 主要产物

- Sticky sweep: `stage3/data/gensar_faithful_20k_rec_w1/v13b_sticky_sweep_v3/`
- GenSAR export: `stage3/data/gensar_faithful_20k_rec_w1/r1b_export/`
- R1b annotations: `stage3/data/temporal_r1b_analysis/`
- Downstream runs: `stage3/runs/temporal_incremental/R1B_*`
- Instance predictions: `stage3/runs/temporal_r1/R1B_*`
- Subgroup analysis: `stage3/runs/temporal_r1b/`
- Scripts: `materialize_v13b_temporal_rec_sid_sticky.py`, `analyze_temporal_sticky_sweep.py`
