# 研究与代码导航

## 研究文档

[背景与相关工作](background.md) → [方法与框架](method.md) → [实验结果](experiments.md) → [后续研究](roadmap.md)

首次阅读建议先看实验主文的问题与结论，再按需查看[实验细节](experiment-details.md)。历史代号与技术词汇见[术语与名称对照](glossary.md)。

[模型发布](model-selection.md) · [环境](../environment/README.md) · [资产访问](storage.md) · [源码来源](migration.md)

当前最新实验为 2026-09-08 的 MovieLens-25M TASR-A / B 严格三 seed 对照。受控 Search 漂移的换码目标子集 +26.46% 属于早期 Amazon / GenSAR 实验；不同实验的作用与边界见结果页。

## 当前实验代码

| 阶段 | 入口 |
| --- | --- |
| MovieLens 数据与时间窗口 | [build_movielens25m_temporal_benchmark.py](../Search_Reconmendation/stage3/scripts/build_movielens25m_temporal_benchmark.py) |
| 转换 GenSAR 输入 | [build_movielens25m_gensar_temporal.py](../Search_Reconmendation/stage3/scripts/build_movielens25m_gensar_temporal.py) |
| 多视图共享前缀 | [train_multiview_anchor_prefix_v13.py](../Search_Reconmendation/stage3/scripts/train_multiview_anchor_prefix_v13.py) |
| 任务后缀 tokenizer | [build_neural_rq_sid_v9_direct_private.py](../Search_Reconmendation/stage3/scripts/build_neural_rq_sid_v9_direct_private.py) |
| t0 下游模型 | [run_movielens25m_t0_v13b_smoke.sh](../Search_Reconmendation/stage3/scripts/run_movielens25m_t0_v13b_smoke.sh) |
| Rec 选择性更新 | [update_v13b_rec_adapter_selective.py](../Search_Reconmendation/stage3/scripts/update_v13b_rec_adapter_selective.py) |
| token 对齐与部署映射 | [align_and_materialize_rec_suffix.py](../Search_Reconmendation/stage3/scripts/align_and_materialize_rec_suffix.py) |
| 受影响样本与蒸馏 mask | [build_affected_rec_adapter_dataset.py](../Search_Reconmendation/stage3/scripts/build_affected_rec_adapter_dataset.py) |
| matched T5 蒸馏训练 | [run_movielens25m_temporal_distill_t5.sh](../Search_Reconmendation/stage3/scripts/run_movielens25m_temporal_distill_t5.sh) |
| 严格评估与跨 seed 汇总 | [evaluate_sid_refresh_strict.py](../Search_Reconmendation/stage3/scripts/evaluate_sid_refresh_strict.py) · [aggregate_sid_refresh_multiseed.py](../Search_Reconmendation/stage3/scripts/aggregate_sid_refresh_multiseed.py) |
| 下游 trainer / LoRA 支持 | [external/GenSAR/finetune](../external/GenSAR/finetune/) |

脚本名称中的 smoke 是历史命名；共同 t0 的实际角色由训练依赖和[模型注册表](../assets/selected_models.json)确定。入口保留原参数与路径，使用前需按环境说明适配，不能视为一键复现命令。

## 证据档案

[PhaseSummary](../Search_Reconmendation/files/PhaseSummary/) 保留阶段0–19及方法、实验对齐记录；[results](../Search_Reconmendation/stage3/results/)保存选定机器可读汇总。档案中的“下一步”“当前”等表述指对应记录日期；部分计划后来被修正，结论以明确的实验协议与结果来源为准。

早期数据准备入口见 [stage3 README](../Search_Reconmendation/stage3/README.md)。第三方 README 保留其上游说明与署名，来源见[第三方源码](third-party.md)。
