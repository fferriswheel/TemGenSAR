# 后续研究：方法改进与待补实验

本页区分已观察问题、候选改进与验证条件。除明确引用的历史诊断外，以下方案均为**待实现或待验证**，不计入当前方法和实验成绩。

## 1. 优先改进下游兼容性

当前稳定样本蒸馏按整条样本启停。target 未变、仅 history 中出现换码 item 的样本也会失去蒸馏；teacher/student 使用同一输入视图。此外，Search SID 未变时，Rec history 的编码变化仍可能关闭 Search 蒸馏。

建议将受影响情况拆分为 target changed、history changed、两者均变、完全稳定四类，再验证：

- **双视图 teacher/student。** teacher 使用旧 SID 历史，student 使用新 SID 历史。保持语义上下文一致，避免把新编码直接送给只熟悉旧编码的 teacher。
- **按位置保持。** 分别蒸馏稳定前缀和兼容后缀；Search 输出 SID 未变时，检验保留 Search 蒸馏能否减少跨任务遗忘。
- **Item-space 对齐。** 在相同 item 候选集上，将旧、新 SID 的序列概率转为 item 概率，再蒸馏或对比学习，减少 token 名称变化的干扰。
- **区分迁移与适应。** 稳定样本保持旧知识，真实漂移样本用当前监督适应；不应对所有变化目标强制保留旧偏好。

验证时保持 SID mapping、训练样本数、优化步数和总预算不变，同时报告受影响和稳定样本指标。item 概率计算需固定候选、归一化及长度处理；不能把未知真实 target 用作部署路由条件。

## 2. 让 SID 发布考虑下游收益与迁移成本

当前候选主要依据表示/行为漂移与支持量；发布限制关注 switching cost、token 对齐和 collision。它们不能保证候选新 SID 在有限下游训练后更容易生成。

候选决策可形式化为以下待验证目标：

```text
utility(i) = estimated downstream gain(i)
             - λ × migration cost(i)
             - μ × stable-item interference(i)

在更新预算、完整 SID 唯一性和未选 item 冻结约束下，
只发布 utility 足够高的候选；允许全窗口 no-op。
```

下游收益可以通过当前窗口内部留出的 validation 与小预算适配估计，不能查看下一测试窗口。比较旧、新候选时需要等预算适配，否则旧模型对旧 token 的熟悉程度会造成偏置。迁移成本应考虑 target 与 history 暴露、重复换码及跨任务影响，而不只统计改了几个 item。

KuaiSAR 已有保守 utility gate 诊断，但覆盖不足以验证正向收益；它不等同于已经实现上述完整决策目标。[现有诊断](experiments.md)

## 3. 对齐 SID 生成目标与下游目标

当前 tokenizer 的重构、拓扑和 commitment 目标不能直接保证历史条件下的排序质量。优先保留 P1 共享前缀与稳定 Search 分支，研究 Rec suffix 的有限任务监督：

1. 加入 history→item / SID 的条件排序或对比目标，控制几何损失与任务损失的权重。
2. 比较分阶段训练、有限交替优化与可微近似，检查 code collapse、churn 和对稳定 item 的干扰。
3. 若只有换码子集改善，进一步分析未命中问题来自正确 prefix 未进 beam、suffix 选择失败，还是新旧编码监督不足。

可微 SID 与排序目标已有 [DIGER](https://arxiv.org/abs/2601.19711) 等直接相关工作，需在相同预算下比较，避免只以“加入任务 loss”主张方法创新。共享前缀的结构改动应建立在 prefix beam coverage、任务信息保留与 gate 分布诊断上；P2 历史退化不支持直接扩大共识约束。

## 4. 首轮方法验证：2×2 对照

| 组别 | SID 决策 | 下游对齐 | 检验问题 |
| --- | --- | --- | --- |
| S0-A0 | 当前选择与发布 | 当前蒸馏 | 重现现有基点 |
| S0-A1 | 当前选择与发布 | 双视图 / 结构化对齐 | 隔离下游兼容性贡献 |
| S1-A0 | 下游 utility 与迁移成本 | 当前蒸馏 | 隔离 SID 决策贡献 |
| S1-A1 | 下游 utility 与迁移成本 | 改进对齐 | 检验两者能否互补 |

在 TASR-A / B 上分开验证，确保数据、候选池、预算、初始化与 checkpoint 选择规则一致。新方案的额外估计或 look-ahead 成本必须计入总成本。固定开发规则后再进入未来窗口，不能在已反复查看的 W2 上形成新的独立泛化结论。

## 5. 需要补充的实验

| 优先级 | 实验 | 关键控制与报告项 |
| --- | --- | --- |
| P0 | 现有结果可复现 | 干净 Linux 环境、数据构建版本、固定 revision、逐步运行说明；重跑静态和时序关键结果 |
| P0 | 更完整的基线 | Frozen + 等预算下游更新、Random、SupportTopK、full refresh、refresh + Hungarian、DACT；使用同一时间划分、backbone 与候选协议 |
| P0 | 独立的未来窗口 | 预先固定 detector / gate / budget；多个连续窗口，区分首次与重复换码，保留真正未用于开发的测试区间 |
| P0 | 旧新 SID 兼容性消融 | target/history 四类样本；teacher 输入视图、样本与位置 mask、无蒸馏、token vs item-space 对齐 |
| P1 | 完整随机性与不确定性 | 独立表示/SID/t0 初始化与下游 seed；报告均值、方差、效应量及样本层面区间；重复用户/时间相关性需按合适单位重采样 |
| P1 | 漂移强度与预算曲线 | 无漂移、弱到强漂移；等候选数与等实际换码数对照；alignment / replay / 更新参数预算与总体收益 |
| P1 | 更充分的自然数据 | 长时间跨度且 item 跨窗复现充分；联合搜推数据优先；明确真实 Search 与合成 Search 的差别 |
| P1 | 全目录检索与规模 | 在当前 99/999 negatives 之外评估全目录 Recall / nDCG、有效 SID 生成率、覆盖率与 beam 成本 |
| P1 | 端到端效率 | 分列表示、检测、SID 更新、发布、下游训练与推理；峰值显存、wall time、存储增量与不同 catalog 规模 |
| P2 | Prefix 与任务共享结构 | 正确 prefix 的 beam coverage、Search/Rec 信息保留、gate 饱和；在明确瓶颈后比较结构变体 |

其中 full refresh 包含 tokenizer 刷新与公平预算的下游适配；不能以近全目录换码后仅少量对齐的失败结果，替代充分训练的性能上界。

## 6. 验收条件

新增方法应同时报告总体 Rec / Search、共同换码集合和稳定集合、collision / churn / 重复换码、无漂移窗口保持性及全部更新成本。子集收益不能替代总体指标，另一任务保持也需要测量。接受改进应依据预先固定的主指标与保持阈值；现有 26.46% 子集收益和 0.12% 自然漂移总体收益不能混合成一个方法提升数字。
