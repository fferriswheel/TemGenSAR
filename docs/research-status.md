# 研究进展导航

本页导航到备份时的研究状态，实验数值以对应记录和汇总表为准。

## 最新工作：MovieLens-25M 与 TASR

- [阶段16：MovieLens-25M benchmark 审计](../Search_Reconmendation/files/PhaseSummary/阶段16_MovieLens25M_TemporalSearchRec可行性审计_2026-09-07.md)
- [阶段17：P2 共识前缀](../Search_Reconmendation/files/PhaseSummary/阶段17_MovieLens25M_P2共识前缀实验_2026-09-07.md)
- [阶段18：P1 条件残差与 Rec 信息](../Search_Reconmendation/files/PhaseSummary/阶段18_P1条件残差与Rec信息增强对比_2026-09-08.md)
- [阶段19：TASR-A / TASR-B 与三 seed](../Search_Reconmendation/files/PhaseSummary/阶段19_TASR两种时序更新实验_2026-09-08.md)
- [严格三 seed 汇总](../Search_Reconmendation/stage3/results/movielens25m_tasr_strict_w1_w2_three_seed/summary.json)

MovieLens benchmark 使用固定 query/tag bank，适合分析自然 Rec drift 和 Search retention；不能据此声称存在自然 Search drift。备份记录中 TASR-B 的 Rec 排序提升可复现，但实际效应很小，Rec H@10 未变；它与 TASR-A 的差异不显著。

## 关键代码

| 工作 | 入口（相对 `Search_Reconmendation/stage3/scripts/`） |
| --- | --- |
| Rec adapter 选择性更新 | `update_v13b_rec_adapter_selective.py` |
| token 对齐与选择性 SID 发布 | `align_and_materialize_rec_suffix.py` |
| matched downstream 数据 | `build_affected_rec_adapter_dataset.py` |
| 下游 T5 蒸馏 | `run_movielens25m_temporal_distill_t5.sh` |
| 严格评估与跨 seed 汇总 | `evaluate_sid_refresh_strict.py`、`aggregate_sid_refresh_multiseed.py` |

下游 GenSAR 的 teacher checkpoint、token KL distillation、`_distill_weight`、LoRA 改动位于根目录 `external/GenSAR/finetune/`；较早的 `Search_Reconmendation/external/GenSAR/` 不包含相同改动。

## 早期实验

[PhaseSummary](../Search_Reconmendation/files/PhaseSummary/) 保留阶段 0–19 及方法、实验对齐记录，包括 controlled Rec drift、KuaiSAR、选择性适配器与多窗口评估。[原 stage3 README](../Search_Reconmendation/stage3/README.md) 描述早期数据和原型准备，阅读当前状态时优先查看阶段19。

备份时的下一步计划是构造 controlled Rec drift severity curve，并与 Frozen / TASR-A / TASR-B / SupportTopK 对照。它是研究计划，不是已经完成的实验结论。
