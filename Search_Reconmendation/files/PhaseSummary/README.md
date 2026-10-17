# 实验记录档案

本目录保留阶段性研究记录，日期、协议与数值用于追踪实验依据。记录中的“当前”“下一步”指该文档形成时的状态；旧计划、服务器路径和开发 pilot 不代表最新默认配置。

当前对外文档：[背景](../../../docs/background.md) · [方法](../../../docs/method.md) · [实验汇总](../../../docs/experiments.md) · [后续研究](../../../docs/roadmap.md)。

本目录的 P1/P2、P1-R/RN、TASR-A/B 是历史方案代号，不是编码长度或性能等级。含义见[名称对照](../../../docs/glossary.md)，具体改了哪些组件见[方法说明](../../../docs/method.md)；其中 TASR 没有在原记录中定义英文全称。

## 主要证据入口

| 主题 | 记录 |
| --- | --- |
| 严格静态 GenSAR 对照与划分审计 | [实验结果对齐](实验结果对齐_2026-08-12.md) |
| Rec 选择性更新与三 seed | [阶段6 R1d](阶段6_R1d_ThreeSeed与SelectiveAdapter_2026-09-02.md) |
| Search 换码子集约 +26.46% | [阶段6 R1S](阶段6_R1S_SearchSelectiveAdapter_2026-09-02.md) |
| 联合漂移与任务局部对齐 | [阶段6 R2b](阶段6_R2b_TaskLocalAlignment联合漂移_2026-09-02.md) |
| 多窗口稳定性 | [阶段8](阶段8_R4_稳定性诊断与优化_2026-09-03.md) |
| Controlled Rec Drift W6 / W9 | [阶段12](阶段12_ControlledRecDrift_W6W9完整评估_2026-09-04.md) |
| 换码 item 下游对齐诊断 | [阶段13](阶段13_换码Item下游对齐诊断_2026-09-05.md) |
| KuaiSAR 与 utility gate 诊断 | [阶段15](阶段15_KuaiSAR自然Rec漂移实验_2026-09-05.md) |
| MovieLens benchmark 审计 | [阶段16](阶段16_MovieLens25M_TemporalSearchRec可行性审计_2026-09-07.md) |
| 共识前缀：限制任务信息进入共享前缀（P2） | [阶段17](阶段17_MovieLens25M_P2共识前缀实验_2026-09-07.md) |
| 显式残差重构，再加入序列邻域（P1-R / P1-RN） | [阶段18](阶段18_P1条件残差与Rec信息增强对比_2026-09-08.md) |
| 冻结推荐码本 vs 刷新码本后对齐，三随机种子（TASR-A / B） | [阶段19](阶段19_TASR两种时序更新实验_2026-09-08.md) |

选定机器可读结果见 [stage3/results](../../stage3/results/)。历史记录中引用的完整数据、模型、逐样本预测或日志不一定随 Git 分发，公开与受限资产范围见[资产访问](../../../docs/storage.md)。
