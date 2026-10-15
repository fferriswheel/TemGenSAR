# 阶段9：Task-local LoRA 预实验（2026-09-04）

## 1. 目的

检验当前自然窗口上的 Search 收益是否可以通过冻结 Rec SID、隔离下游任务参数来保留，同时消除共享 T5 全量更新造成的任务冲突。

代表窗口为 W3、W5、W9，分别在 W4、W6、W10 测试；seeds 为 42、2024、2025。所有实验共享 seed 2024 的上游 rolling 轨迹，因此属于当前窗口下游训练三 seed，而非完整端到端三 seed。

## 2. Search-only / Rec-SID-frozen 全参数实验

- Search suffix 使用当前窗口 Selective mapping；
- shared prefix 和 Rec suffix 沿用前一窗口 mapping；
- T5 仍全参数更新；
- 保留 unchanged-Rec SID rehearsal/alignment，控制灾难性遗忘；
- alignment budget、epoch、候选集和评估口径与 R4 一致。

| 方法 | Rec nDCG@10 | Search nDCG@10 | mean nDCG@10 |
|---|---:|---:|---:|
| Fixed | 0.023423 | 0.030498 | 0.026960 |
| R4C | 0.022897 | 0.031929 | **0.027413** |
| Search-only / Rec-SID-frozen | 0.022805 | **0.031956** | 0.027380 |

Search-only 相对 Fixed：Search `+0.001458`，Rec `-0.000617`，mean `+0.000420`。冻结 Rec SID 仍不能消除 Rec 退化，说明共享 T5 参数更新仍会造成跨任务干扰。

## 3. Search task-local LoRA

实现方式：

- 冻结原 T5；
- 在 T5 attention 的 `q/v` 上训练 Search LoRA；
- rank 8，alpha 16，dropout 0.05；
- 训练任务为 `PerSrcWithRecHis + SemanticItem2Index + SemanticIndex2Item`；
- Search 推理启用 LoRA，Rec 推理禁用 LoRA；
- 只训练 294,912 参数，占 61,575,168 参数的 `0.479%`；
- 单个 adapter 约 1.2 MB。

三窗口三 seed 结果：

| 方法 | Rec nDCG@10 | Search nDCG@10 | mean nDCG@10 |
|---|---:|---:|---:|
| Search task-local LoRA | 0.021298 | 0.031713 | 0.026505 |

- Rec 三 seed 标准差严格为 0，证明 Search LoRA 没有修改 Rec 路径；
- Search 相对 Fixed 为 `+0.001215`，3/3 seed 为正；
- 但冻结 Rec 下游路径未利用当前窗口行为，Rec 相对 Fixed 为 `-0.002125`；
- 因而双任务 mean 相对 Fixed 为 `-0.000455`。

结论：参数隔离有效，Search LoRA 以 0.479% 参数保留了大部分 Search 收益；但“冻结 Rec SID”不能等价为“Rec 下游模型永不适应新行为”。

## 4. Rec task-local LoRA smoke

新增对称 Rec adapter：

- 训练任务为 `SeqRecWithSrcHis + CollaborativeItem2Index + CollaborativeIndex2Item`；
- Rec SID 保持前一窗口不变；
- 其余 LoRA 配置与 Search 相同。

W3/seed 2024：

| 路径 | Rec nDCG@10 |
|---|---:|
| 冻结 base | 0.022786 |
| Rec LoRA | 0.022527 |
| R4C 全参数更新 | 0.023221 |

当前 Rec LoRA 配置低于冻结 base，因此未继续扩到三 seed。这里遵守预先约定的 smoke gate，避免对失败配置继续消耗算力。

## 5. 当前判断

最终方法不应强制更新两个 adapter，而应为每个任务建立独立候选：

1. freeze：SID 与下游 task adapter 均保持不变；
2. task adapter update：SID 不换码，但 task adapter 用当前行为更新；
3. suffix + task adapter update：只更新发生漂移任务的 suffix 和 adapter；
4. 使用 validation gate 独立决定 Search/Rec 采用哪个候选。

下一步先在 Controlled Rec Drift 上验证：有明确 Rec 漂移时，Rec adapter/suffix update 是否稳定优于 Rec-frozen；随后迁移 KuaiSAR natural drift。

## 6. 产物

- Search-only runner：`stage3/scripts/run_temporal_search_only_three_seed.sh`
- Search LoRA runner：`stage3/scripts/run_temporal_search_lora.sh`
- Search LoRA three-seed runner：`stage3/scripts/run_temporal_search_lora_three_seed.sh`
- Rec LoRA runner：`stage3/scripts/run_temporal_rec_lora.sh`
- LoRA entry：`external/GenSAR/finetune/main_selective_lora.py`
- task-local alignment datasets：`external/GenSAR/finetune/modules/data.py`
- 汇总：`stage3/results/temporal_search_lora_three_seed/`
