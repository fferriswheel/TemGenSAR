# 实验细节：协议、完整指标与统计范围

本页用于查阅配置和完整数值。实验动机、对照含义与结论的连续解释见[核心实验](experiments.md)；历史代号见[术语与名称对照](glossary.md)。本页表格中的 Fixed、Selective、P1、P2 等均按该对照解释。

实验记录截至 **2026-09-08**。本页整理已保存的日志与汇总结果，未重新训练或重跑检索评估。指标为 H@10（Hit Rate）与 nDCG@10；绝对差记为 Δ，相对变化按 `100 × (method / baseline − 1)` 计算。

本页缩写口径：Rec 为推荐、Search 为搜索；t0 为更新前的初始模型，W1/W2 为相邻时间窗口；seed 是控制训练随机性的种子。H@10 看目标是否进入前十，nDCG@10 进一步考虑排序位置；“均值 ± 标准差”描述多个训练种子的波动。负候选是与正确 item 一起排序的非目标 item。

更新策略中，Fixed/Frozen 指保持编码但继续训练生成模型；Random 是同预算随机选择；Selective 是根据变化选择；SupportTopK 按当前交互数选择。TASR 是未定义英文全称的历史系列代号，A 为冻结推荐码本的局部更新，B 为刷新推荐码本后对齐。码本是产生离散 token 的向量中心集合。

## 1. 协议与结果范围

| 实验 | 数据与变化 | 测试范围 / 候选协议 | 随机性范围 |
| --- | --- | --- | --- |
| 复现的 GenSAR 静态对照 | Amazon Electronics，19,999 items，仅用训练期行为构建表示 | Rec / Search 各 10,000；999 negatives，beam 10 | 历史单 seed |
| 早期推荐局部选择 | Amazon，当前季度更新 / 下一季度测试，固定搜索编码 | Rec / Search 各 5,000；999 negatives，beam 10 | 增量 seed 2024 / 2025 / 2026，共同 t0 |
| 受控搜索变化与局部更新 | Amazon，持续注入查询关系变化，推荐事件不变 | W1 更新 / W2 测试；两任务各 5,000；999 negatives，beam 10 | 增量 seed 42 / 2024 / 2025，共同 t0 |
| 受控推荐变化与选择策略 | Amazon，30% 用户群体内同类别 item 替换 | W6→W7 与 W9→W10；各任务 5,000；999 negatives，beam 10 | W6 增量 seed 2024 / 2025 / 2026；W9 单 seed 2024 留出 |
| MovieLens 推荐编码更新的严格对照 | 18,191 固定 items；自然推荐变化；固定查询/标签库构造搜索控制任务 | W1→W2；Rec 2,015 / Search 5,000；99 negatives，beam 10 | 增量 seed 42 / 2024 / 2025，共同 t0 |

表中 negatives 是每个目标的负候选数，候选还包括一个正例。它们是采样候选协议下的结果，不能当作全目录检索结果。MovieLens 的 Search 是稳定控制任务，不能用于论证自然 Search 漂移。跨数据集、候选规模或划分方式的绝对指标不直接比较。

## 2. 静态 SID：GenSAR 下游对照

对每个任务，取相同用户最后两条有效事件分别作为验证、测试，称为严格配对的 leave-two-out 划分。比较复现的 GenSAR 编码与本项目多视图共享前缀方案（v13b）；均使用 T5-small、六类任务、三轮训练，按验证集联合 nDCG@5 选择模型。

两组都对发生编码碰撞的 item 做 Sinkhorn 分配调整，原表简写为 CG（collision group，碰撞组）；matched CG 指采用同类处理，不代表最终碰撞率相同。

| SID | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| GenSAR-faithful + CG | 0.0374 | 0.01946 | 0.0534 | 0.02980 |
| v13b + matched CG | **0.0791** | **0.05034** | **0.1240** | **0.07740** |

这是本仓库复现协议中的静态对照，并非直接引用 GenSAR 论文指标，也不是对多种最新方法的多次独立训练比较。每个任务内验证/测试使用相同用户的相邻未来事件；受原始日期粒度限制，部分配对事件同日，顺序仍按原始序列区分。

[协议、划分审计与原始结果记录](../Search_Reconmendation/files/PhaseSummary/实验结果对齐_2026-08-12.md)

## 3. 受控 Search 漂移：换码目标子集约 +26.46%

### 设定

搜索局部更新实验（记录编号 R1S）在相同共享前缀的 item 之间交换主要查询，从 W1 起持续替换相关事件的查询、保持 item 不变，共注入 75 个 item。检测器选中 122 个候选，只训练搜索表示的小模块，固定共享前缀与推荐分支，最终 15 个 item 实际更换搜索 SID。

Fixed、matched-Random、Selective 使用相同 t0、数据、训练任务和预算。下一窗口 5,000 条 Search 测试样本中，**119 条样本的目标属于 Selective 实际换码集合**；所有对比方法在这个相同子集上评估。该子集是基于已发布 mapping 的诊断分组，并非随机样本或推理时的路由条件。

### 总体指标：三 seed 均值

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 0.04860 | 0.02600 | **0.07893** | 0.04112 |
| matched-Random | 0.04667 | 0.02556 | 0.07487 | 0.03918 |
| Selective | 0.04853 | 0.02597 | 0.07873 | **0.04148** |

全体 Search nDCG 的平均绝对增益约 **+0.00036**（按已报告均值约 +0.88%），3/3 seed 为正；Search H@10 没有提升。Rec 指标基本保持。

### 实际换码 target：119 条样本

| 增量 seed | Fixed nDCG@10 | Selective nDCG@10 | 相对提升 |
| ---: | ---: | ---: | ---: |
| 42 | 0.12068 | 0.15850 | +31.34% |
| 2024 | 0.12921 | 0.16054 | +24.25% |
| 2025 | 0.12884 | 0.15991 | +24.12% |
| 三 seed 均值 | **0.12625** | **0.15965** | **约 +26.46%** |

均值绝对差约 +0.03341；表格四舍五入使其与显示均值相减略有差异。26.46% 是原阶段记录的三 seed 汇总值，显示精度下计算约为 26.46%。H@10 均值为 0.34454 → 0.35294，增益主要体现在排序位置。

seed 2024 的机器可读子集结果为 0.1292131783 → 0.1605410968，Δ=0.0313279185，约 **+24.25%**。历史 paired row bootstrap 的 95% CI 为 **[0.02104, 0.04295]**。CSV 中 p=0 表示有限 10,000 次重采样未出现对应反向事件，不能理解为真实概率等于零。其他 seed 的区间结论来自阶段记录，当前 Git 中未包含其完整逐样本预测。

这些结果支持受控 Search 漂移下的局部排序改善；119 条样本、共同 t0、方法定义的换码子集与人工注入方式限制了其外推范围。自然数据未触发 Search detector 时应 no-op。

[阶段6 R1S：完整协议与三 seed 表](../Search_Reconmendation/files/PhaseSummary/阶段6_R1S_SearchSelectiveAdapter_2026-09-02.md) · [seed 2024 子集 CSV](../Search_Reconmendation/stage3/results/temporal_search_r1s_subgroups/subgroup_metrics.csv)

## 4. 受控推荐漂移：收益与选择策略的边界

“漂移 × 交互量”选择策略（原名 V3）在漂移分数上加入当前影响量：`score = drift_rank_score × sqrt(current_events)`，要求当前支持量至少 5，按预算选择 top-K。K=51 在 W9 留出前确定。

### W6 更新 → W7 测试，三 seed 均值

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 0.030867 | 0.019476 | 0.039400 | 0.021221 |
| SupportTopK | 0.031200 | 0.019914 | 0.039667 | 0.021322 |
| V3-K51 | **0.031400** | **0.020050** | **0.039733** | **0.021408** |

V3 相对 Fixed 的 Rec nDCG Δ=+0.000574，约 +2.95%，3/3 seed 正向；相对 SupportTopK 只有 +0.000136，2/3 seed 正向，尚不支持稳定显著优势。

### W9 更新 → W10 测试，seed 2024 留出

| 方法 | Rec nDCG@10 | Search nDCG@10 |
| --- | ---: | ---: |
| Fixed | 0.018533 | 0.028408 |
| matched-Random | 0.019149 | 0.028481 |
| SupportTopK | **0.019477** | 0.028611 |
| V3-K51 | 0.019464 | **0.028723** |

V3 相对 Fixed 的 Rec nDCG 约 +5.02%，但未超过 SupportTopK。V3-only 换码的 19 个 item 在 W10 对应 83 条 target 样本，所有方法的 nDCG@10 均为 0：选中变化 item 不保证新 SID 可被下游有效生成。

[阶段12完整记录](../Search_Reconmendation/files/PhaseSummary/阶段12_ControlledRecDrift_W6W9完整评估_2026-09-04.md) · [W6 三 seed JSON](../Search_Reconmendation/stage3/results/controlled_rec_w6_v3_three_seed/summary.json) · [W9 留出 JSON](../Search_Reconmendation/stage3/results/controlled_rec_w9_k51_holdout/summary.json)

## 5. MovieLens-25M：冻结码本与刷新码本的严格对照

三组均使用 43,882 条当前推荐样本、1,000 条旧搜索回放样本、相同的 150-item 对齐池，两类辅助任务各 150 条；从同一初始模型出发训练一轮，共 353 个优化步骤。额外的“旧推荐 SID→新 SID”迁移任务关闭。蒸馏样本开关随实际换码影响范围变化，是方法定义的一部分；种子 2026 是开发试验，不混入正式三种子的均值。

### 总体指标：均值 ± 标准差

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Frozen + matched update | 0.452275 ± 0.000573 | 0.286285 ± 0.000278 | 0.706400 ± 0.000200 | 0.460273 ± 0.000141 |
| TASR-A | 0.452275 ± 0.000573 | 0.286483 ± 0.000221 | 0.706400 ± 0.000200 | 0.460248 ± 0.000098 |
| TASR-B | 0.452275 ± 0.000573 | **0.286628 ± 0.000190** | 0.706333 ± 0.000306 | 0.460220 ± 0.000149 |

B 相对 Frozen 的 Rec nDCG Δ=**+0.000343559，约 +0.12%**，3/3 seed 正向。历史 seed paired t-test p=0.0303；seed-averaged row bootstrap 95% CI 为 [0.000031, 0.000685]，p=0.0302。B vs A 的差异不显著。三组 Rec H@10 相同，Search 差异很小。

### Rec 漂移子集：三 seed 均值

| 子集 | 样本数 | Frozen nDCG@10 | A nDCG@10 | B nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Detector K150 target | 534 | 0.044424 | **0.045456** | 0.045351 |
| A 实际换码 target | 243 | 0.037395 | 0.039471 | **0.039510** |
| B 实际换码 target | 237 | 0.041079 | **0.043207** | 0.043187 |
| A/B 换码 target 并集 | 326 | 0.029864 | **0.031412** | 0.031397 |

B 在并集上的相对增益约 +5.13%。这些子集的 row-bootstrap 95% CI 仍轻微跨 0，不能仅凭 3/3 seed 正向宣称子集显著。相同 SID、t0 与测试集也限制了 seed 独立性。

### 更新范围与成本

| 方法 | SID 可训练参数 | 实际换码 item | SID 更新训练时间 | 部署完整 SID collision |
| --- | ---: | ---: | ---: | ---: |
| TASR-A | 14,528 | 25 | 15.75 s | 0 |
| TASR-B | 114,176 | 67 | 21.88 s | 0 |

上述是 SID 阶段计时。阶段19开发 pilot 另记录下游训练约 208–211 秒，完整 train / val / test 约 5 分钟/组；这些不是正式三 seed 全流程成本均值，也未覆盖全部表示构建与检测。当前没有证据支持端到端数十倍加速。

[阶段19记录](../Search_Reconmendation/files/PhaseSummary/阶段19_TASR两种时序更新实验_2026-09-08.md) · [指标 CSV](../Search_Reconmendation/stage3/results/movielens25m_tasr_strict_w1_w2_three_seed/model_metrics.csv) · [统计比较 CSV](../Search_Reconmendation/stage3/results/movielens25m_tasr_strict_w1_w2_three_seed/comparisons.csv) · [checkpoint](model-selection.md)

## 6. 设计消融与失败案例

| 实验 | 已观察现象 | 含义 / 证据 |
| --- | --- | --- |
| 多视图前缀（P1）vs 共识前缀（P2） | Rec nDCG 0.4347 → 0.3418；Search 0.4677 → 0.3431 | 共识前缀压缩可能损失任务信息；[阶段17](../Search_Reconmendation/files/PhaseSummary/阶段17_MovieLens25M_P2共识前缀实验_2026-09-07.md) |
| 加入残差重构（P1-R），再加入序列邻域（P1-RN） | 邻域增强版本降低 raw Rec collision，但四指标均值 0.5703，未超过原方案的 0.5706 | 重构或 collision 改善不等于下游收益；[阶段18](../Search_Reconmendation/files/PhaseSummary/阶段18_P1条件残差与Rec信息增强对比_2026-09-08.md) |
| 固定预算下重复 alignment | 换码子集小幅改善后饱和，Search nDCG 下降约 3.4%–4.9% | 增加局部监督会干扰共享模型；[阶段13](../Search_Reconmendation/files/PhaseSummary/阶段13_换码Item下游对齐诊断_2026-09-05.md) |
| Rec task-local LoRA | Search 保持，但换码目标未改善，未换码 Rec 目标退化 | 任务隔离仍不足以解决任务内部干扰；[阶段13](../Search_Reconmendation/files/PhaseSummary/阶段13_换码Item下游对齐诊断_2026-09-05.md) |
| KuaiSAR 自然漂移 | 强制换码退化；保守 gate 避免退化，但最佳总体 Δ 仅 +0.000049，下一窗仅 1 个所选 item 出现为目标 | 作为稳定性与覆盖率诊断，不能当作正向主结果；[阶段15](../Search_Reconmendation/files/PhaseSummary/阶段15_KuaiSAR自然Rec漂移实验_2026-09-05.md) |
| 多窗口 rolling | Rec 后期不稳定，重复换码暴露增加，collision 不是唯一原因 | 需检查累计 mapping / 模型对齐；[阶段8](../Search_Reconmendation/files/PhaseSummary/阶段8_R4_稳定性诊断与优化_2026-09-03.md) |

静态 MovieLens 指标与 W2 时序结果来自不同测试划分，不能将 0.4347 和 0.286628 当作更新前后变化。现有证据支持选择性更新的可行性与局部排序收益；尚需更强的统一基线、多窗口自然漂移和完整独立 seed 验证。[待补实验](roadmap.md)

### 补充结构实验：为什么保留当前前缀？

原方案将内容、行为和搜索三种表示融合到共享前缀。共识前缀变体尝试更强调这些视图共同支持的信息；相同 MovieLens 静态协议下，它使推荐与搜索 nDCG 分别降至 0.3418、0.3431，因此当前保留原多视图方案。现象与任务信息损失的解释一致，但并未单独证明因果机制。

另一个实验固定原前缀，先在后缀训练中显式加入残差重构损失，再引入推荐序列邻域信息。最终版本虽降低了原始推荐 SID 碰撞，其推荐/搜索 nDCG 为 0.4367 / 0.4660，原方案为 0.4347 / 0.4677：两项取舍，四指标均值也未超过原方案。这说明不能只靠重构误差或碰撞率判断 SID 优劣，也为后续单 token 后缀实验提供了验收原则：以排序收益、稳定性与成本共同判断。

## 7. 早期推荐局部更新：公平对照与训练范围

本组对应主文第二节。三组使用共同 t0 seed 2024，增量 seed 为 2024 / 2025 / 2026；固定编码与漂移组使用相同 12 个对齐 item，各组对齐配置预算均为 1,800。均值 ± 标准差如下：

| 策略 | 推荐 H@10 | 推荐 nDCG@10 | 搜索 H@10 | 搜索 nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| 固定编码 | 0.050733 ± 0.000643 | 0.028642 ± 0.000579 | 0.066067 ± 0.001973 | 0.035637 ± 0.000847 |
| 随机局部更新 | 0.050667 ± 0.000306 | 0.027443 ± 0.000207 | 0.066133 ± 0.001102 | 0.034550 ± 0.000401 |
| 漂移驱动局部更新 | 0.050467 ± 0.000833 | 0.030931 ± 0.000745 | 0.066200 ± 0.001908 | 0.035676 ± 0.000826 |

推荐 nDCG 相对固定编码的平均绝对差为 +0.002289，按表中均值计算相对变化约 +7.99%，3/3 种子正向。该实验检验选择和发布策略；不能把三个种子的结果当作后续“只在 189 个 item 上训练 adapter”的三 seed 复验。

后者是另一个单 seed 2024 实验，两组都使用漂移选择，只比较 SID adapter 的梯度训练范围：

| SID adapter 训练范围 | 核心训练时间 | 推荐 nDCG@10 | 搜索 nDCG@10 |
| --- | ---: | ---: | ---: |
| 全部 19,999 个 item | 180.2 s | 0.03179 | 0.03474 |
| 63 个漂移 item + 126 个低变化回放 item | 5.8 s | 0.03148 | 0.03467 |

计时是原运行中的 adapter 核心时间，尚未经过统一进程的重复 profiling，也不包括下游生成模型训练。

[阶段6 R1d：原始表格、控制变量与限制](../Search_Reconmendation/files/PhaseSummary/阶段6_R1d_ThreeSeed与SelectiveAdapter_2026-09-02.md)
