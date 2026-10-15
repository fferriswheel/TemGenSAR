# 阶段16：MovieLens-25M Temporal Search+Rec 可行性审计（2026-09-07）

## 1. 结论

MovieLens-25M **适合**作为当前工作的第二个长时间跨度 benchmark，但论文口径应限定为：

> natural Rec drift + stationary synthetic Search control + selective task-local SID refresh

它明显优于 KuaiSAR 的地方是同一 item 在相邻窗口中的复现与后续目标覆盖率很高，适合检验“更新 Rec suffix、保持 shared prefix 和 Search suffix 不变”。它不包含真实搜索日志，因此不能用于声称自然 Search drift；Search-only drift 仍应作为 controlled stress test。

推荐先做 18K fixed-catalog pilot，再决定是否扩展 emerging-item/cold-start 轨道。

## 2. 数据与已有工作

### 2.1 官方 MovieLens-25M

官方数据包含：

- 25,000,095 条 rating；
- 162,541 个 user；
- 62,423 个 movie metadata；
- 1,093,351 条带时间戳的 user tag；
- 时间范围为 1995-01-09 至 2019-11-21；
- metadata 只有 title 和 genres，另有 MovieLens ID 到 IMDb/TMDb ID 的链接；
- tag genome 只覆盖 13,816 个 movie，不覆盖整个 catalog。

官方页面：<https://grouplens.org/datasets/movielens/25m/>

本地原始数据已下载到：

`data/movielens25m/raw/ml-25m/`

压缩包 MD5 与官方值一致：`6b51fb2759a8657d3bfcbfc42b592ada`。

### 2.2 已有 joint Search+Rec 设置

Penha et al. (RecSys 2024) 使用 MovieLens rating 构造 Rec，用 genres、user tags 和 genome tags 构造 Search；其 ML 子集约有 20K items、163K users、55K search queries，平均每个 query 对应 6.86 个 relevant items。

Spotify 的 `Semantic IDs for Joint Generative Search and Recommendation`（RecSys 2025 LBR）使用：

- 62,138 movies；
- 文中报告 1.24M user-item interactions；
- 每个 item 由 Gemini-2.0-flash 生成 20 条自然语言 query，10 train / 10 test；
- content/search 表示使用 title、year、description、genres、tags 和 genome tags；
- Search embedding：`all-mpnet-base-v2` + query-item 对比学习；
- Rec embedding：ENMF；
- Multi-task embedding：query-item 与 item-item co-occurrence 联合对比学习；
- SID：RQ-KMeans，2 个 codebook，每层 size 256；
- 下游：`google/flan-t5-base`，joint Search+Rec，3 epochs；
- 评估：Recall@30，5 seeds。

论文：<https://arxiv.org/abs/2508.10478>

### 2.3 当前无法严格复刻的部分

1. 官方 rating 中 `rating >= 4` 有 12.45M 条，和论文报告的 1.24M interactions 相差约 10 倍；论文没有说明该缩减步骤。
2. 官方 MovieLens-25M 不含 description；论文使用的 description 来源没有说明。
3. 论文和 Spotify 官方页面没有链接其 Gemini query、完整预处理代码或数据划分文件；当前检索也未找到作者公开 artifact。
4. 论文的 Rec split 是“每个 user 留最后一次 interaction 测试”，不是本文需要的全局时间窗口 rolling evaluation。

因此后续应称为 **MovieLens-25M 上的 protocol-compatible temporal extension**，不能称为对 Spotify 数据处理的严格复现。

## 3. 本地时序审计

### 3.1 正反馈定义

主口径暂定 `rating >= 4.0`：

- 12,452,811 条 positive interaction；
- 162,342 个 users；
- 40,858 个 active items。

将所有 rating 都当正反馈可作为与已有论文更接近的敏感性实验，但不宜作为唯一主口径，因为低评分不表示正偏好。

### 3.2 推荐初始化和窗口

初始化期：所有 `timestamp < 2015-01-01` 的 positive interaction。

- t0 events：8,721,383；
- t0 users：121,467；
- t0 fixed catalog：18,190 items；
- fixed catalog 中 17,365 个 item 有 user tag，11,939 个有 genome tag。

更新/评估期：`2015-H1 ... 2019-H2`，共 10 个半年窗口。

| 统计量 | 结果 |
|---|---:|
| 每窗 positive events 中位数 | 364,032 |
| 每窗 active users 中位数 | 7,579 |
| 每窗 active items 中位数 | 15,245 |
| fixed catalog 的未来 event 总覆盖 | 87.65% |
| 相邻窗口 item overlap 中位数（all active items） | 68.42% |
| 相邻窗口重复 item 的 event 覆盖中位数 | 97.04% |
| 相邻窗口 support 均 >= 5 的 item 中位数 | 4,394 |
| 至少已有 5 条历史的 event 比例中位数 | 94.94% |

只看 t0 fixed catalog 时，相邻窗口 active-item overlap 约为 74.4%–86.7%，每对窗口有 3,152–4,580 个 item 在两侧 support 都不低于 5。这比 KuaiSAR 更适合评估“选择后能否在下一窗口再次出现并贡献下游指标”。

### 3.3 漂移强度

半年相邻窗口的 item-popularity JS divergence：

- all active catalog：0.0319–0.0489；
- t0 fixed catalog：0.0153–0.0227。

这表明数据存在连续变化，但剔除新 item 后属于中低强度自然漂移，而不是天然的强冲击。是否真正值得换码仍应由 train-only detector 和 downstream utility gate 判断，允许某些窗口输出 `abstain`。

### 3.4 user tag 不能直接作为自然 Search log

2015–2019 每个半年窗通常有约 39K–68K tag events，但 2018-H1 突增至 252,562，其中单个 user 占 71.74%。tag 是用户注释行为，不是 query-click 行为，并且受批量标注影响。因此：

- 可用 tag/genome tag 构造静态 query 或 item metadata；
- 不应把 tag timestamp 直接解释为自然 Search drift；
- Search 的自然漂移结论仍不能由 MovieLens 支撑。

## 4. 推荐 benchmark 协议

### 4.1 Catalog

主轨使用 t0 train-only fixed catalog：2015 年前至少出现一次正反馈的 18,190 个 item。该选择不读取未来标签，且与 RecSys 2024 的约 20K MovieLens catalog 规模接近。

未来新 item 约占 12.35% 的正反馈事件。第一版先从主轨排除，后续单列 `emerging-item/cold-start` 扩展，不把两类问题混在首轮实验中。

### 4.2 Rec 数据

- 输入：目标时刻之前的 positive item SID history，按时间排序并截断到统一最大长度；
- 输出：当前窗口中该 user 的下一 positive item SID；
- SID representation/update 在评估窗口 `W_t` 开始前最多只能使用 `W_{t-1}` 及之前的行为；
- `W_t` 完成评估后，其已揭示标签可以进入下一轮 `W_{t+1}` update；
- 对相同 timestamp 的批量 rating 使用确定性 tie-break，并做 tie-rate 敏感性检查。

### 4.3 Search 数据

推荐两级实现：

1. **Pilot，可完全由官方数据构造**：以 genres、规范化 user tags、top genome tags 形成 query/qrels，复用 RecSys 2024 思路。
2. **论文主实验**：用 title、year、genres、top user tags、top genome tags 为每个 fixed-catalog item 生成 20 条自然语言 query，固定 10 train / 5 validation / 5 test；query bank 在所有时间窗口保持不变。

18,190 items 对应 363,800 条合成 query，明显小于 2025 论文全 catalog 的约 124 万条 query。若后续取得 description，可重新生成增强版并作 ablation；第一版不依赖未公开的 description。

固定 Search query bank 是有意设计的 control：自然 Rec 漂移时，只应改变 Rec suffix；Search 指标用于衡量跨任务遗忘和 prefix 稳定性。

### 4.4 时间划分

建议冻结如下口径：

- `t0`：2015 年前的数据，用于初始 representation、SID 和下游训练；
- `W1`：2015-H1，初始评估和第一次 update；
- `W2-W4`：开发 detector、sticky/alignment budget；
- `W5-W6`：冻结前验证；
- `W7-W10`：2018-H1 至 2019-H2，完全冻结的 temporal test。

每次报告 `W_t update -> W_{t+1} evaluation`，同时保留 Fixed/no-op。不得用 W7-W10 下游结果调整阈值或更新预算。

## 5. 方法与 baseline

首轮应控制下游模型、训练样本、负采样、beam 和 token budget，只改变 SID/update 策略：

1. `Fixed SID / no update`；
2. `Ours selective Rec-suffix LoRA update`；
3. `Matched-random Rec-suffix update`；
4. `SupportTopK Rec-suffix update`；
5. `GenSAR-faithful full representation + SID refresh`；
6. `Spotify Multi-task bi-encoder + RQ-KMeans full refresh`；
7. 可选 oracle：以当前窗口 train-only utility 选择更新 item，不读取下一窗口标签。

SID 主配置沿用当前 `[256,256 | 256,256]`；Spotify 静态 baseline 按原文使用两层 `[256,256]`。如果比较模型参数量，必须为 task-specific token space 补齐等容量控制，避免词表大小成为混杂因素。

## 6. 指标

下游质量：

- Rec：HR@10、nDCG@10、Recall@30；
- Search：HR@10、nDCG@10、Recall@30；
- changed/selected item 子集和 all-item 两种口径；
- 3 seeds，报告 paired mean、95% CI 和逐窗口胜率。

SID 与时序稳定性：

- raw/final collision；
- prefix churn、Rec suffix churn、Search suffix churn；
- selected item 的下一窗口 target coverage；
- Search forgetting；
- wall-clock、GPU hours、trainable parameters、下游 alignment 样本量。

## 7. 风险与 Go/No-Go

### Go

- 5 年、10 个窗口，足以形成比 KuaiSAR 更完整的 rolling story；
- 每个窗口有数十万正反馈，item-level detector support 充足；
- 相邻窗口重复 item 覆盖约 97% 事件，选择性更新具备可观测的后续下游信号；
- 与已有 joint Search+Rec SID 文献使用同源数据，baseline 容易解释。

### 限制

- 没有真实 search log，只能研究 Search retention 或 controlled Search drift；
- rating timestamp 中存在用户集中打分，不能强调长期 user lifecycle；
- fixed catalog 上的自然 drift 不算强，可能有多个窗口合理地选择 no-op；
- 2025 论文的 query、description 和 1.24M interaction 过滤过程不公开，不能宣称严格复现。

### 最终判断

**Go，先做 benchmark builder + 小规模统计/训练 smoke。** 该数据适合作为自然 Rec drift 的外部验证。如果冻结测试窗口中 detector 大多 abstain，则说明 MovieLens 也缺少足够强的自然变化，此时应保留为稳定性/负结果，并继续寻找更强的长时序行为数据，而不是人为调低阈值强制换码。

## 8. 已生成产物

- 审计脚本：`stage3/scripts/audit_movielens25m_temporal_search_rec.py`
- 2015–2019 主审计：`stage3/results/movielens25m_temporal_feasibility_v1/`
- 2010–2019 20-window 对照：`stage3/results/movielens25m_temporal_feasibility_start2010_v1/`
- 原始数据：`data/movielens25m/raw/ml-25m/`

