# 阶段11：Controlled Rec Drift 的 W3 初验（2026-09-04）

## 1. 本阶段目标

在保持 Search 数据不变的条件下，于 W3 注入可控 Rec drift，检验“检测漂移 item -> 只更新 Rec adapter/suffix -> 增量训练下游模型”是否优于 Fixed SID 与随机更新 SID。

本轮是 seed 2024 的方法学 smoke test，不作为最终多 seed 结论。

## 2. 实验协议

- 数据：19,999 items 的 GenSAR-faithful 20K quarterly benchmark。
- 漂移：从 W3 开始，对 30% 用户 cohort 的一部分 Rec 事件做同 root+leaf category 内的一对一 item rotation；W3 注入率为 25%。
- W3 实际替换：807 个事件，涉及 640 个 source items 和 640 个 target items。
- W1、W2 使用无注入的 clean trajectory；三组 W3 实验均从同一个 `CRD_Clean_W2_Fixed_s2024` checkpoint 开始。
- W3 用于训练与 validation，W4 用于 test；三组的 Rec/Search train、validation、test 文件哈希完全相同。
- 三组使用完全相同的 1,800-item balanced alignment pool，其中强制包含 Selective 和 Random 的全部 102 个候选 item。
- 三组 prefix 与 Search suffix 完全相同；唯一实验变量是 Rec suffix 更新策略。

## 3. 三组方法

### Fixed

保持 J0 的 prefix、Search suffix 和 Rec suffix 不变，但下游模型仍使用 W3 Rec/Search 数据做同预算增量训练。

### Random

从未被 drift selector 选中的 eligibility 候选中随机选择 51 个 item，单独训练 Rec adapter，再用 sticky ratio 0.1 物化 Rec suffix。它与 Selective 使用相同的候选预算、replay ratio 和 adapter 超参数。

由于高 drift-score item 同时也是高 support item，排除 Selective 候选后没有足够的 `50+` support item，因此 Random 只能近似匹配 support，不能称为严格 support-matched baseline。

### Selective

使用 W2->W3 item-level drift selector：

```text
event_support      = min(left_events, right_events)
transition_support = min(left_transitions, right_transitions)
eligible           = event_support >= 5 and transition_support >= 3
score               = log(1 + event_support)
                      * (0.4 * popularity_log_ratio + 0.6 * transition_js)
```

在 169 个 eligible items 中选择 top 30%，共 51 个。只训练 14,528 个 Rec-adapter 参数，prefix、Search suffix 和 shared codebook 均冻结；训练集由 51 个 selected items 与 102 个 replay items 构成。

## 4. SID 更新结果

| 方法 | 候选 item | 实际换 Rec suffix | Prefix 变化 | Search suffix 变化 | 完整 SID duplicate fraction |
|---|---:|---:|---:|---:|---:|
| Fixed | 0 | 0 | 0 | 0 | 0.4050% |
| Random | 51 | 15 | 0 | 0 | 0.4050% |
| Selective | 51 | 19 | 0 | 0 | 0.4050% |

两种更新均满足未选 item 的 operational SID 完全不变，且没有增加原有 collision。

Selective adapter 在第 1 epoch 取得 best，实际优化 33 steps、3.81 秒；Random adapter 在第 20 epoch 取得 best，90 steps、8.81 秒。

## 5. W4 下游结果

### 整体 5,000 条测试样本

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 | 两任务平均 nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Fixed | 0.0334 | 0.020379 | **0.0652** | 0.035547 | 0.027963 |
| Random | 0.0338 | 0.020590 | 0.0648 | **0.035638** | 0.028114 |
| Selective | **0.0342** | **0.021869** | 0.0644 | 0.035506 | **0.028688** |

Selective 相比 Fixed：

- Rec nDCG@10：+0.001490，relative +7.31%。
- 两任务平均 nDCG@10：+0.000724，relative +2.59%。
- Search nDCG@10：-0.000041，基本保持不变。

Selective 相比 Random：

- Rec nDCG@10：+0.001280，relative +6.21%。
- 两任务平均 nDCG@10：+0.000574，relative +2.04%。

### Rec 子集

| 子集 | 样本数 | Fixed nDCG@10 | Random nDCG@10 | Selective nDCG@10 |
|---|---:|---:|---:|---:|
| drift-eligible target | 639 | 0.116140 | 0.116718 | **0.125646** |
| selective-selected target | 339 | 0.173114 | 0.174203 | **0.191771** |
| selective-recoded target | 160 | 0.207513 | 0.207513 | **0.241592** |
| 人工注入 target | 215 | 0.005996 | 0.005996 | 0.005996 |

因此，当前收益确实集中在 selector 命中并实际换码的 item 上，但人工注入 target 子集尚未受益。

## 6. 关键诊断

当前 selector 使用双侧最小支持量 `min(left,right)`，会系统性排除从低频快速增长的新 target：

| W3 注入角色 | items | eligible items | selected items | 被 selected 覆盖的注入事件 |
|---|---:|---:|---:|---:|
| source | 640 | 83 | 43 | 112 / 807 |
| target | 640 | 11 | 5 | 23 / 807 |

W4 的 215 条注入 target 测试样本中，只有 1 条 target 属于 Selective 候选；19 条样本的 original source 被选择，8 条的 original source 实际换码。换 source SID 无法直接改善 replacement target 的生成，因此三组在注入 target 子集上完全相同。

这说明本轮不能直接表述为“成功适应人工 Rec drift”。更准确的结论是：

1. selective Rec suffix update 能在不损伤 Search、且不增加 collision 的条件下改善被选高支持 item 的排序；
2. 当前 detector 对 emerging/gaining item 不敏感，整体收益尚可能混入 popularity/support selection 效应；
3. 在解决 detector 的方向性偏置前，不应直接扩展到三 seed 或 W6/W9。

## 7. 下一轮建议

优先做 detector v2，而不是立即增加 seed：

1. 将 eligibility 从双侧最小支持改为单侧/总量支持，使 disappearing 和 emerging item 都能进入候选；保留支持置信度但不要求两个窗口都高频。
2. 加入 signed popularity change，并在 gain/loss 两侧分别分配预算，避免 top-K 被下降 source 单侧占满。
3. 增加基于 transition-distribution change 的邻居传播，让被检测 source 的主要新增/流失 transition counterpart 也能进入更新集合；不能使用注入 donor map 作为训练信号。
4. baseline 增加 `SupportTopK`，与 Selective 使用相同预算，专门排除“只因选择了高频 item”带来的收益。
5. detector v2 先只生成 selection 并做 oracle-free coverage 诊断；确认 W4 注入 target 覆盖显著提高后，再跑 Fixed / Random / SupportTopK / Selective 下游。

## 8. 产物

- 汇总：`stage3/results/controlled_rec_w3_seed2024/summary.json`
- 整体指标：`stage3/results/controlled_rec_w3_seed2024/overall_metrics.csv`
- Rec 子集指标：`stage3/results/controlled_rec_w3_seed2024/rec_subgroup_metrics.csv`
- Selective SID：`stage3/data/controlled_rec_drift_w3w6w9_v1/W3/selective_sid/`
- Random SID：`stage3/data/controlled_rec_drift_w3w6w9_v1/W3/random_sid/`
- 下游模型：`stage3/runs/temporal_incremental/CRD_W3_{Fixed,Random,Selective}_s2024/`

中间 checkpoint 已清理，每组只保留最终 best model。

## 9. Detector v2 实现与离线诊断

已新增 `analyze_temporal_item_drift_v2.py`，不覆盖 v1。v2 的主要变化如下：

1. eligibility 从 `min(left,right)` 改为 `max(left,right)`，允许 emerging/disappearing item 进入候选；
2. popularity 使用经过窗口总流量校正的 signed log-rate ratio，而不是只有绝对变化；
3. core budget 在 gain/loss 两侧均衡分配；
4. 预留 30% budget，通过 source-before 与 candidate-after（反方向亦然）的 transition-neighborhood cosine similarity，寻找相反方向 counterpart；
5. 同时输出同预算 `SupportTopK` 控制组，供后续排除 popularity/support 效应；
6. W2->W3 固定总预算 51，确保与 detector v1 的成本一致。

实现与测试：

- `stage3/scripts/analyze_temporal_item_drift_v2.py`
- `stage3/scripts/evaluate_controlled_rec_detector.py`
- `stage3/tests/test_analyze_temporal_item_drift_v2.py`
- 3 个合成单元测试均通过：gain/loss budget、单侧不足时补齐、cross-window counterpart 恢复。

### 固定 51-item budget 的注入事件覆盖

| 注入窗口 | 方法 | source events covered | target events covered |
|---|---|---:|---:|
| W3 | v1 top-51 | 112 | 23 |
| W3 | v2 | 41 | 8 |
| W3 | SupportTopK | 127 | 44 |
| W6 | v1 top-51 | 344 | 29 |
| W6 | v2 | 132 | **132** |
| W6 | SupportTopK | 416 | 42 |
| W9 | v1 top-51 | 688 | 62 |
| W9 | v2 | 148 | **262** |
| W9 | SupportTopK | 767 | 27 |

v2 在较强漂移的 W6/W9 上明显改善 target-side coverage，但 W3 的弱且分散漂移仍不足以从自然波动中识别。

### Clean-vs-injected paired sensitivity

| 窗口 | clean/injected selection overlap | Jaccard | injected-only selected |
|---|---:|---:|---:|
| W3 | 48 / 51 | 0.889 | 3 |
| W6 | 36 / 51 | 0.545 | 15 |
| W9 | 31 / 51 | 0.437 | 20 |

这表明 detector v2 的响应随 controlled drift 强度增强，方向正确；但 W3 不适合立即做下游验证。下一次下游实验应优先选 W6，并使用 Fixed、SupportTopK、v1 和 v2 的同预算对照。

离线诊断结果：`stage3/results/controlled_rec_detector_v2_b51/paired_diagnostic.json`。
