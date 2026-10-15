# 阶段6 R1：Temporal SID 分组诊断（2026-09-01）

## 1. 目的

R1 不修改模型、不重新训练，而是将 W2 Rec test 按 SID 更新状态拆分，检查 E3 相对 E2 的收益是否集中在真正变更 SID 的 item。

比较的三个已有 checkpoint：

```text
E0: t0 SID + no downstream update
E2: t0 SID + W1 downstream incremental update
E3: W1 dynamic Rec SID + identical downstream incremental update
```

E2/E3 仅 SID mapping 不同，下游初始权重、训练样本、optimizer、step 数和评估候选集相同。

## 2. 时间口径修正

本轮复核发现，E0-E5 实际使用的是 quarterly protocol：

```text
t0: <= 2011-12-31
W1 update: 2012Q1
W2 test:   2012Q2
```

旧的《阶段5_第一轮TemporalUpdate实验》曾误写为 2013 半年窗口，现已修正。这是文档标注错误，模型实际读取的 dataset 和 timestamp 没有混用。

原有 item drift CSV 也是半年协议，因此 R1 重新生成了 quarterly `C8 -> W1` item drift，不使用半年分数对当前 W1 分桶。

## 3. 分组定义

W2 Rec test 包含 5,000 条样本，每条保留原始 999 negatives。

| Group | Definition | W2 samples |
|---|---|---:|
| changed | operational Rec suffix 在 t0/W1 mapping 间变化 | 1,520 |
| active-unchanged | W1 有 Rec 事件，但 Rec SID 未变 | 2,097 |
| inactive | W1 没有 Rec 事件 | 1,383 |

其中：

- 2,567 / 19,999 个 item 发生 Rec SID 变更；
- 1,193 个 item 改变 R1，2,555 个 item 改变 R2；
- 1,137 个 changed item 在 t0 Rec train 中无事件；
- W2 changed 样本中，610 条属于 t0 Rec-unseen target。

因此当前 changed 组实际混合了“已有 item 行为漂移”和“W1 新激活/cold-start item”。

## 4. 逐样本评估实现

GenSAR evaluator 原来只保存 aggregate metric。R1 增加了默认关闭的 `save_instance_results` 选项，保存每条样本的 top-10 beam hit vector。

E0/E2/E3 各在完整 W2 Rec test 上重跑一次纯推理，聚合结果与原日志完全一致，之后所有 subgroup 都在同一批逐样本结果上离线计算。

## 5. 主要结果

### 5.1 SID update group

| Group | N | E0 H@10 | E2 H@10 | E3 H@10 | E3-E2 | E0 nDCG@10 | E2 nDCG@10 | E3 nDCG@10 | E3-E2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 5,000 | 0.0458 | 0.0500 | 0.0506 | +0.0006 | 0.02430 | 0.02718 | 0.02652 | -0.00067 |
| Active unchanged | 2,097 | 0.0572 | 0.0782 | 0.0796 | +0.00143 | 0.02916 | 0.04213 | 0.04257 | +0.00044 |
| Changed | 1,520 | 0.0671 | 0.0520 | 0.0533 | +0.00132 | 0.03806 | 0.02966 | 0.02733 | -0.00233 |
| Inactive | 1,383 | 0.0051 | 0.0051 | 0.0036 | -0.00145 | 0.00182 | 0.00180 | 0.00129 | -0.00051 |

E3-E2 在 changed group 的配对 bootstrap（10,000 次）：

```text
H@10:    +0.00132, 95% CI [-0.00197, 0.00461], p=0.523
nDCG@10: -0.00233, 95% CI [-0.00528, 0.00060], p=0.120
```

动态 SID 收益没有集中在 changed item，且 nDCG 方向为负。

### 5.2 t0 Rec seen / unseen

| Group | N | E2 H@10 | E3 H@10 | E2 nDCG@10 | E3 nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Changed, t0 Rec seen | 910 | 0.0868 | 0.0890 | 0.04954 | 0.04565 |
| Changed, t0 Rec unseen | 610 | 0 | 0 | 0 | 0 |

对 t0-unseen item，当前 1 epoch incremental T5 没有产生任何 top-10 命中；SID 更新不能单独解决 downstream cold-start alignment。

### 5.3 Popularity

| Popularity | N | E2 H@10 | E3 H@10 |
|---|---:|---:|---:|
| Head | 2,201 | 0.1127 | 0.1140 |
| Middle | 964 | 0.0021 | 0.0021 |
| Tail | 1,835 | 0 | 0 |

当前 W2 总体指标几乎完全由 head item 贡献，尾部上无法检验 SID 语义变好是否转化为下游收益。

## 6. Drift selector 对齐诊断

Quarterly `C8 -> W1` item detector 要求两个窗口均有至少 5 次事件、3 次 outgoing transition：

| Statistic | Items |
|---|---:|
| C8/W1 union | 8,254 |
| support eligible | 209 |
| top-30% drift selected | 63 |
| selected 且当前 SID changed | 19 |

当前 mapping 生成时并没有使用这个 selector，而是允许所有 W1 active item 重分配。两者的事后对齐度很低：

```text
drift-selected item 中被更新: 19 / 63 = 30.2%
changed item 中属于 drift-selected: 19 / 2567 = 0.74%
```

W2 下游结果：

| Group | N | E2 H@10 | E3 H@10 | E2 nDCG@10 | E3 nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Drift eligible | 855 | 0.2082 | 0.2117 | 0.1156 | 0.1118 |
| Drift selected | 508 | 0.3071 | 0.3130 | 0.1708 | 0.1637 |
| Changed + eligible | 269 | 0.1933 | 0.1896 | 0.1182 | 0.0963 |
| Changed + selected | 154 | 0.3312 | 0.3247 | 0.2032 | 0.1641 |

`changed + selected` 上 E3-E2 `nDCG@10 = -0.03909`，95% CI `[-0.05746, -0.02183]`，paired bootstrap `p < 0.0001`。当前 SID 更新对最有支持度的高漂移 item 显著降低了排序质量。

## 7. Representation / SID topology

UniSAR t0/W1 behavior embedding 诊断：

| Item group | t0 embedding available | Cosine shift, available only | Top-20 neighbor overlap, available only |
|---|---:|---:|---:|
| Active unchanged | 100% | 0.00118 | 0.774 |
| Changed | 63.6% | 0.00137 | 0.754 |
| Inactive | 49.1% | 0.00187 | 0.773 |

对 t0 已有 behavior embedding 的 item，W1 representation 变化实际很小，但 active item 的 SID churn 达 41.35%。这说明离散 code 变更主要由量化边界和重分配触发，而不是连续 representation 发生大幅漂移。

新 SID 对 W1 top-20 behavior neighbor 的边际 suffix-code agreement 有提升，但完整 `[P1,P2,R1]` route agreement 没有提升：

```text
changed suffix agreement: 0.1397 -> 0.1841
changed route agreement:  0.03670 -> 0.03650
```

这意味着 R2 的单层边际聚类变得更一致，但并没有改善实际层级 SID route 与行为拓扑的对齐。

## 8. R1 结论

R1 否定了“当前 active-all assignment 已经能有效应对自然漂移”，但没有否定 task-local dynamic suffix 的总体思路。当前失败来自三个可定位问题：

1. 更新集合是 all active，而不是 drift-selected；
2. constrained R2 assignment 没有 old-code switching cost，微小 latent 变化也可换 code；
3. cold-start/newly-active item 与 existing-item drift 混合，但下游对前者完全没有召回能力。

因此不建议立即对当前 E2/E3 跑多 seed；多 seed 只会更确定地复现一个选择与分配规则不合理的版本。

## 9. 建议的 R1b

在进入 R2 前，先进行一个小型修正实验：

```text
A. Current active-all assignment
B. Drift-selected sticky assignment
C. Matched-budget random sticky assignment
```

Sticky assignment 建议目标：

```text
cost(i, new_code)
= latent_distance(i, new_code)
+ lambda_switch * 1[new_code != old_code]
```

并加入：

- 仅 drift-selected existing item 允许更新；
- t0-unseen/newly-active item 作为独立 cold-start route，不计入 drift update 主结论；
- R1/R2 只在新 code 相对 old code 的距离 margin 超过阈值时改变；
- 先用 SID churn、route topology 和 selector precision 筛选，再跑下游。

R1b 通过后，再进入 E2/E3 三 seed 和 training-budget curve。

## 10. 产物

- Annotation: `stage3/data/temporal_r1_analysis/`
- Quarterly item drift: `stage3/data/temporal_drift_20k_quarterly_v1/rec_item_drift.csv`
- Instance predictions: `stage3/runs/temporal_r1/E0_t0_no_update`, `E2_downstream_only`, `E3_dynamic_sid`
- Subgroup metrics: `stage3/runs/temporal_r1/analysis/subgroup_metrics.csv`
- SID quality: `stage3/runs/temporal_r1/analysis/sid_quality_summary.json`
- Scripts: `prepare_temporal_r1_annotations.py`, `evaluate_temporal_r1_subgroups.py`, `analyze_temporal_r1_sid_quality.py`
