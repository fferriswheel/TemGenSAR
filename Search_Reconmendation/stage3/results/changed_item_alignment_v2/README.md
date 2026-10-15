# Changed-item-aware Alignment v2: W6 Repeat Curve

## 目的

检验 V3 换码 item 的下游表现是否受 `Item2Index / Index2Item` 对齐监督不足限制。
固定每个 task view 的 alignment 总预算为 1800，只把 33 个实际换码 item 在 Rec alignment 中重复 1、2、4、8 次。

## 公平性控制

- SID、W6 主任务训练数据、W7 验证/测试数据完全相同。
- 初始化均为 `CRD_W5_Fixed_s2024`，T5-small、1 epoch、seed 2024。
- Search alignment 始终是同一组 1800 个唯一 item。
- Rec alignment 总 occurrence 始终为 1800。
- 新公共池覆盖 V3 的全部 33 个实际换码 item；旧池漏掉的 2 个 item 已在所有四臂中统一补齐。
- R2/R4/R8 分别移除 33/99/231 个普通填充 item，换取换码 item 的额外 occurrence。

## W7 结果

| Repeat | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 | Recoded-target nDCG@10 |
|---:|---:|---:|---:|---:|---:|
| R1 | 0.0314 | **0.020111** | **0.0400** | **0.021585** | 0.062610 |
| R2 | 0.0310 | 0.019889 | 0.0386 | 0.020529 | **0.063624** |
| R4 | 0.0314 | 0.019991 | 0.0396 | 0.020857 | **0.063624** |
| R8 | 0.0314 | 0.019995 | 0.0394 | 0.020796 | **0.063624** |

换码目标子集包含 297 条样本、30 个在 W7 出现的换码 item。R2 相对 R1 的子集 nDCG@10 增加 0.001014，但 R4/R8 不再增加，说明收益在两次监督时已经饱和。

未换码目标的 nDCG@10 从 R1 的 0.017427 降至 R2 的 0.017128。简单重采样既减少普通 item 覆盖，也通过共享 T5 参数引入 Rec/Search 任务干扰。

## 结论

R2/R4/R8 均未同时满足预先冻结的三个条件：换码子集改善、整体 Rec 不下降、Search 相对下降不超过 1%。因此不选择 repeat factor，也不在这一配置上继续增加 `old SID -> new SID` bridge。

该实验支持两个判断：

1. 换码 item 的新 SID 学习不足确实是局部瓶颈之一。
2. 仅靠固定预算内重复 alignment 不能解决问题；下一版需要隔离更新梯度或使用更直接、低干扰的映射迁移目标。

机器可读结果见 `w6_repeat_curve/summary.json`。

## Task-local Rec LoRA smoke

在 R1 checkpoint 上冻结全部 T5 参数，仅训练 attention `q/v` 的 rank-8 Rec LoRA：

- trainable parameters：294,912（0.479%）；
- 5,344 条目标或最近 5 个 Rec history 涉及换码 item 的 W6 样本；
- 4,000 条未受影响 replay；
- R2 Rec alignment：1,800 个 occurrence，其中 33 个换码 item 各出现两次；
- Rec 推理开启 adapter，Search 推理关闭 adapter。

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 | 换码目标 nDCG@10 | 未换码目标 nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 0.0314 | **0.020111** | 0.0400 | 0.021585 | 0.062610 | **0.017427** |
| Task-local Rec LoRA | 0.0310 | 0.019736 | 0.0400 | 0.021585 | 0.062610 | 0.017028 |

Search 结果 JSON 与 R1 逐字节相同，证明 task routing 正确隔离了 Search。然而换码目标没有改善，整体下降全部来自未换码目标。因此 task-local 参数隔离仍不够：adapter 虽然只属于 Rec，推理时却会作用于所有 Rec 样本。

下一结构候选应加入 change-conditioned gate，使 Rec adapter 只作用于涉及换码 SID 的输入，并在 alignment 样本上强制开启。当前配置未通过 smoke gate，不扩多 seed。

机器可读结果见 `w6_task_local_rec_adapter/summary.json`。
