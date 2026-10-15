# 阶段6：R1S Search 漂移与 Selective Adapter（2026-09-02）

## 1. 本轮目标

验证 Rec selective update 的对称版本是否能用于 Search-only drift：

- shared prefix 与 Rec suffix 固定；
- 只训练 Search adapter；
- 只允许漂移检测选中的 item 更新 Search suffix；
- 用 Fixed SID、matched-Random update、drift-Selective update 做公平比较；
- 检查 Search 收益和对 Rec 的跨任务影响。

自然数据的 `C8 -> W1` Search task-level detector 未触发，因此自然场景应执行 no-op。本轮另外构造 persistent controlled Search-only drift，以测试真正的更新路径。

## 2. Controlled Search Drift

注入脚本：`stage3/scripts/inject_controlled_search_drift.py`

- 只用 C8 选择候选，避免查看未来窗口；
- 候选 item 在 C8 至少有 5 次 Search 事件；
- 在相同 shared-prefix 组内交换 dominant query；
- 从 W1 开始持续生效，80% 的目标事件被替换 query，item 不变；
- Rec 事件完全不改。

最终注入 75 个 item，共修改 14,343 条 Search 事件。W1 单窗口修改 1,404 条。任务级检测在 `C8 -> W1` 上触发：query JS、query MMD、item-query centroid 三项均通过 BH 校正。

## 3. Item Drift Detection

脚本：`stage3/scripts/analyze_temporal_search_item_drift.py`

每个 item 使用两类变化：

1. query 分布的 Jensen-Shannon divergence；
2. query BGE centroid 的 cosine shift。

只在左右窗口均至少有 5 条事件时进入候选，再按两个指标的平均 percentile rank 选择 top 30%。`C8 -> W1`：

| 项目 | 数量 |
|---|---:|
| support-eligible | 404 |
| drift-selected | 122 |
| controlled injected | 75 |
| selected ∩ injected | 69 |
| 检测 precision | 56.6% |
| 检测 recall | 92.0% |

自然数据另生成 no-op selection，所有 pair 的 selected item 数为 0。

## 4. Selective Search Adapter

脚本：`stage3/scripts/update_v13b_search_adapter_selective.py`

沿用 v13b/M7b 的 hierarchical Search suffix：

- `S1`：24-d relevance latent，单层 256-code quantizer；
- `S2`：40-d identity latent，单层 256-code quantizer；
- 输入为 frozen common representation、更新后的 query signal 与 frozen shared latent；
- 仅 `search_adapter.*` 可训练，共 117,176 个参数；
- shared encoder/codebook、Search quantizer/decoder、Rec 分支全部冻结。

训练集由 122 个 selected item 和 244 个 low-drift replay item 构成。核心 loss 保留：full/residual reconstruction、relevance reconstruction、identity reconstruction、within-route topology、commitment 和 old-latent stability。

adapter 训练最佳点为 epoch 1：66 optimizer steps，核心训练 5.62 秒。验证结果：

- shared prefix 逐 item 完全不变；
- Rec suffix 逐 item 完全不变；
- selected raw Search code change：16.39%；
- 所有 item raw Search code change：11.82%。

## 5. Operational SID

脚本：`stage3/scripts/materialize_v13b_temporal_search_sid_sticky.py`

采用 prefix-conditional hierarchical sticky assignment：先更新 `S1` relevance code，再在 `[P1,P2,S1]` route 内更新 `S2` identity code。switch ratio 固定为 0.1，与 Rec R1b 的协议一致，不使用 W2 test 调参。

| 统计 | Selective |
|---|---:|
| selected item | 122 |
| operational SID changed | 15 |
| S1 changed | 8 |
| S2 changed | 9 |
| changed ∩ injected | 10 |
| selection 外变化 | 0 |
| duplicate fraction | 0.6200% |

更新继承 t0 已存在的 collision，不引入新 collision。matched-Random 同样选择 122 个 item，并产生 14 个 operational SID change。

## 6. 下游协议

- t0 checkpoint：相同的 `t5-small` checkpoint；
- W1 update：8,826 条 Search update + 1,000 条 t0 Rec replay；
- alignment：Item2Index / Index2Item 各 1,800 条；
- epoch：1；
- W1 validation：Rec 5,000 + Search 5,000；
- W2 test：Rec 5,000 + Search 5,000；
- candidate ranking：1 positive + 999 negatives，beam size 10；
- seeds：42、2024、2025。

Fixed、Random、Selective 除 SID mapping 外，数据、训练任务、预算和超参数均相同。

## 7. 三 Seed W2 结果

均值 ± 标准差：

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| Fixed | 0.04860 ± 0.00040 | 0.02600 ± 0.00012 | 0.07893 ± 0.00168 | 0.04112 ± 0.00106 |
| matched-Random | 0.04667 ± 0.00031 | 0.02556 ± 0.00007 | 0.07487 ± 0.00110 | 0.03918 ± 0.00054 |
| Selective | 0.04853 ± 0.00031 | 0.02597 ± 0.00010 | 0.07873 ± 0.00186 | **0.04148 ± 0.00108** |

Selective 相对 Fixed：

- Search H@10 平均 `-0.00020`，没有改善命中数量；
- Search nDCG@10 平均 `+0.00036`，3/3 seeds 为正；
- Rec H@10 `-0.00007`，Rec nDCG@10 `-0.00003`，可视为基本不变。

Selective 相对 matched-Random：Search H@10 平均 `+0.00387`、Search nDCG@10 平均 `+0.00230`，两项均为 3/3 seeds 正向。

## 8. 换码子集的三 Seed 结果

W2 中目标 item 的 Search SID 实际发生变化的样本固定为 119 条。Selective 相对 Fixed 的结果如下：

| Seed | Fixed H@10 | Selective H@10 | H@10 绝对变化 | Fixed nDCG@10 | Selective nDCG@10 | nDCG 绝对变化 | nDCG 相对提升 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | .33613 | .35294 | +.01681 | .12068 | .15850 | +.03782 | +31.34% |
| 2024 | .35294 | .35294 | .00000 | .12921 | .16054 | +.03133 | +24.25% |
| 2025 | .34454 | .35294 | +.00840 | .12884 | .15991 | +.03107 | +24.12% |
| 三 seed 均值 | .34454 | .35294 | +.00840 | .12625 | .15965 | +.03341 | +26.46% |

三个 seed 的 nDCG@10 都提升；每个 seed 的 paired bootstrap 95% CI 均完全大于 0（10,000 次采样，`p < 1e-4`）。H@10 在两个 seed 上提升、一个 seed 持平。

## 9. Seed 2024 完整子集结果

逐样本 beam result 使用 paired bootstrap（10,000 次）分析：

| W2 Search 子集 | 样本数 | Fixed H@10 / nDCG@10 | Random H@10 / nDCG@10 | Selective H@10 / nDCG@10 |
|---|---:|---:|---:|---:|
| all | 5,000 | .0770 / .03990 | .0754 / .03950 | .0766 / .04025 |
| injected | 413 | .3608 / .21307 | .3172 / .19439 | .3584 / **.22089** |
| drift-selected | 676 | .3491 / .19025 | .3240 / .18082 | .3476 / **.19466** |
| SID changed | 119 | .3529 / .12921 | .2521 / .09280 | .3529 / **.16054** |
| changed ∩ injected | 110 | .3818 / .13979 | .2727 / .10039 | .3818 / **.17368** |

在 SID-changed 子集，Selective 对 Fixed 的 Hit@10 持平，但 nDCG@10 增加 `0.03133`（约 +24.2%，bootstrap `p<0.001`）。说明本轮更新主要把已经进入 top-10 的正确 item 排得更靠前，而不是增加 top-10 命中数。

## 10. 当前判断

1. Search selective adapter 的更新路径已跑通，并满足 prefix/Rec suffix 严格冻结。
2. drift-aware selection 明显优于同预算随机更新，说明收益不是“任意改几个 SID”带来的。
3. 对 Fixed 的全局 H@10 没有提升，是因为 operational SID 只改变 15 个 item；W2 仅 119/5,000 条样本受直接影响。
4. 换码子集的 nDCG 提升稳定且显著，但目前证据仍属于受控 Search drift，不应外推为自然 Search drift 结论。
5. 下一步应做 joint Search+Rec drift，并优先研究在不破坏稳定性的前提下提高 operational change coverage，而不是继续堆叠 loss。

## 11. 关键产物

- 数据：`stage3/data/gensar_faithful_20k_controlled_search_w1`
- quarterly benchmark：`stage3/data/gensar_faithful_20k_quarterly_controlled_search_w1`
- Search adapter：`stage3/data/gensar_faithful_20k_search_w1_controlled/v13b_search_adapter_selective_r2`
- sticky SID sweep：`stage3/data/gensar_faithful_20k_search_w1_controlled/v13b_search_selective_sticky`
- 三 seed 汇总：`stage3/results/temporal_search_r1s_three_seed`
- 子集分析：`stage3/results/temporal_search_r1s_subgroups`
