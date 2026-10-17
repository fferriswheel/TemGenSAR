# 研究与代码导航

## 研究文档

[背景与相关工作](background.md) → [方法与框架](method.md) → [实验结果](experiments.md) → [后续研究](roadmap.md)

首次阅读建议先看实验主文的问题与结论，再按需查看[实验细节](experiment-details.md)。历史代号与技术词汇见[术语与名称对照](glossary.md)。

[模型发布](model-selection.md) · [环境](../environment/README.md) · [资产访问](storage.md) · [源码来源](migration.md)

最新实验为 2026-09-08 的 MovieLens-25M 对照：冻结推荐码本的局部更新与刷新码本后对齐，各用三个随机种子重复训练。它们在源文件中名为 TASR-A/B，TASR 仅是未定义英文全称的历史系列代号。[组件差异](method.md)。受控搜索漂移的换码目标子集 +26.46% 属于早期 Amazon / GenSAR 实验；不同实验的作用与边界见结果页。

## 当前实验代码

| 阶段 | 入口 |
| --- | --- |
| MovieLens 数据与时间窗口 | [build_movielens25m_temporal_benchmark.py](../Search_Reconmendation/stage3/scripts/build_movielens25m_temporal_benchmark.py) |
| 转换 GenSAR 输入 | [build_movielens25m_gensar_temporal.py](../Search_Reconmendation/stage3/scripts/build_movielens25m_gensar_temporal.py) |
| 多视图共享前缀 | [train_multiview_anchor_prefix_v13.py](../Search_Reconmendation/stage3/scripts/train_multiview_anchor_prefix_v13.py) |
| 将任务表示量化为后缀 token 的编码器 | [build_neural_rq_sid_v9_direct_private.py](../Search_Reconmendation/stage3/scripts/build_neural_rq_sid_v9_direct_private.py) |
| 更新前的初始 T5 生成模型 | [run_movielens25m_t0_v13b_smoke.sh](../Search_Reconmendation/stage3/scripts/run_movielens25m_t0_v13b_smoke.sh) |
| 只更新推荐分支的编码模块 | [update_v13b_rec_adapter_selective.py](../Search_Reconmendation/stage3/scripts/update_v13b_rec_adapter_selective.py) |
| token 对齐与部署映射 | [align_and_materialize_rec_suffix.py](../Search_Reconmendation/stage3/scripts/align_and_materialize_rec_suffix.py) |
| 根据换码影响决定哪些样本保留旧模型约束 | [build_affected_rec_adapter_dataset.py](../Search_Reconmendation/stage3/scripts/build_affected_rec_adapter_dataset.py) |
| 相同预算下继续训练 T5，并保持稳定预测 | [run_movielens25m_temporal_distill_t5.sh](../Search_Reconmendation/stage3/scripts/run_movielens25m_temporal_distill_t5.sh) |
| 下一窗口评估与多次随机训练汇总 | [evaluate_sid_refresh_strict.py](../Search_Reconmendation/stage3/scripts/evaluate_sid_refresh_strict.py) · [aggregate_sid_refresh_multiseed.py](../Search_Reconmendation/stage3/scripts/aggregate_sid_refresh_multiseed.py) |
| 生成模型训练器与低秩附加参数支持 | [external/GenSAR/finetune](../external/GenSAR/finetune/) |

脚本名称中的 smoke 是历史命名；共同 t0 的实际角色由训练依赖和[模型注册表](../assets/selected_models.json)确定。入口保留原参数与路径，使用前需按环境说明适配，不能视为一键复现命令。

## 证据档案

[PhaseSummary](../Search_Reconmendation/files/PhaseSummary/) 保留阶段0–19及方法、实验对齐记录；[results](../Search_Reconmendation/stage3/results/)保存选定机器可读汇总。档案中的“下一步”“当前”等表述指对应记录日期；部分计划后来被修正，结论以明确的实验协议与结果来源为准。

早期数据准备入口见 [stage3 README](../Search_Reconmendation/stage3/README.md)。第三方 README 保留其上游说明与署名，来源见[第三方源码](third-party.md)。
