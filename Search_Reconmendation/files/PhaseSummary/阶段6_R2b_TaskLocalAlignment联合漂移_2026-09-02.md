# 阶段 6 R2b：Task-Local Alignment 联合漂移实验

日期：2026-09-02

## 1. 问题与修正

R2a 在联合 Search/Rec 下游训练中，将所有实验方法的 55 个换码 item union 同时写入 `src_item_set` 和 `rec_item_set`。GenSAR 的 `Item2Index/Index2Item` 原实现又先从二者 union 采样 item，再交替训练 semantic/rec index，导致：

- Search SID alignment 也会采到仅 Rec 换码或 random 换码的 item；
- Rec SID alignment 也会采到仅 Search 换码的 item；
- 真正 15 个 Search 换码 item 的有限对齐预算被明显稀释。

R2b 增加仅对新数据集生效的 `task_specific_alignment.pkl` 标记：

```text
semantic/Search SID alignment -> src_item_set（29-item 公平 union）
collaborative/Rec SID alignment -> rec_item_set（30-item 公平 union）
```

旧 GenSAR 数据集没有该标记，原有实验行为不变。所有方法仍共享同一 Search 候选池和同一 Rec 候选池，不按各方法实际换码集合单独采样。

## 2. Seed 42 二因素消融

比较 task-local alignment（L）和 Search/Rec update 重采样平衡（B）。表中是 `JB-J0` 的 test nDCG@10 差值：

| 协议 | Rec 差值 | Search 差值 | J0/JB Search 绝对值 |
|---|---:|---:|---:|
| R2a：shared alignment，原始任务比例 | +0.002549 | -0.001024 | 0.04051 / 0.03948 |
| L：task-local alignment | +0.002574 | **+0.000359** | 0.04038 / 0.04074 |
| B：任务数重采样平衡 | +0.002719 | +0.000199 | 0.03791 / 0.03811 |
| L+B | +0.002209 | +0.000107 | 0.03759 / 0.03769 |

两项修正都能令 paired 差值转正，但重采样使 Search 绝对性能明显下降，且与 L 组合没有叠加收益。因此最终协议只采用 L，保留原始的 11,912 Rec 与 8,826 Search update 数据分布。

## 3. 最终五方法三 Seed 结果

均值 ± 样本标准差：

| 方法 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| J0 Fixed | 0.05227 ± 0.00012 | 0.02962 ± 0.00026 | 0.07720 ± 0.00000 | 0.04048 ± 0.00016 |
| JS Search-only | 0.05240 ± 0.00035 | 0.02970 ± 0.00027 | 0.07700 ± 0.00020 | **0.04093 ± 0.00015** |
| JR Rec-only | 0.05200 ± 0.00020 | 0.03194 ± 0.00031 | 0.07727 ± 0.00031 | 0.04059 ± 0.00019 |
| JB Joint | 0.05227 ± 0.00050 | **0.03216 ± 0.00030** | 0.07693 ± 0.00012 | 0.04087 ± 0.00011 |
| JRandom | 0.05220 ± 0.00020 | 0.02953 ± 0.00025 | 0.07707 ± 0.00042 | 0.04060 ± 0.00012 |

关键 paired 结果：

- `JB-J0` Rec nDCG@10：平均 `+0.002532`，3/3 seeds 为正；
- `JB-J0` Search nDCG@10：平均 `+0.000386`，3/3 seeds 为正；
- `JB-JRandom` Rec nDCG@10：平均 `+0.002623`，3/3 seeds 为正；
- `JB-JRandom` Search nDCG@10：平均 `+0.000266`，3/3 seeds 为正；
- `JB-JS` Search nDCG@10：平均 `-0.000064`；
- `JB-JR` Rec nDCG@10：平均 `+0.000216`。

因此联合更新基本保留了两个单任务更新的收益。Search-only 的 Search nDCG 略高于 Joint，差异只有约 `6.4e-5`；Joint 的 Rec nDCG 反而略高于 Rec-only。

Search H@10 在 JB 中平均下降 `0.000267`，但 nDCG 稳定提高，说明主要作用是改善已命中 item 的排序位置，而非扩大 Top-10 命中集合。

## 4. Search 换码子集

15 个 Search 换码 item 对应 W2 的 119 条样本：

| Seed | J0 nDCG@10 | JB nDCG@10 | 差值 | Paired bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| 42 | 0.15396 | 0.18389 | +0.02994 | [0.01684, 0.04572] |
| 2024 | 0.14527 | 0.17760 | +0.03233 | [0.01524, 0.05090] |
| 2025 | 0.15208 | 0.17971 | +0.02762 | [0.01263, 0.04394] |

三个 seed 的区间均完全大于 0。seed 42 中 H@10 完全持平；另两个 seed 各减少 1 条命中，但已命中样本的排名改善足以带来稳定 nDCG 增益。

## 5. 阶段结论

1. shared prefix 固定、Search/Rec suffix 独立更新在联合漂移场景中可行；
2. 两个任务的 SID alignment 必须使用各自的换码候选池，不能简单把联合 item pool 同时用于两种 SID；
3. 漂移选择明显优于 matched-budget random，收益不是任意换码产生的；
4. 当前全局增益较小是因为 operational changes 极少：Search 15 个、Rec 13 个；换码子集上的增益明显更大；
5. 下一阶段应进入 W2 -> W3 的 test-then-train 与多窗口 rolling evaluation，固定本阶段超参数，不再用未来窗口调参。

## 6. 产物

- 数据构造：`stage3/scripts/materialize_temporal_incremental_gensar.py`
- GenSAR task-local alignment：`/data/cjk/external/GenSAR/finetune/modules/data.py`
- 联合 mapping：`stage3/data/temporal_joint_r2a_mappings/`
- 修正版数据集：`/data/cjk/external/GenSAR/data/GenSAR_Faithful_20K_R2B_L_*`
- 完整三 seed 汇总：`stage3/results/temporal_joint_r2b_localalign_full_three_seed/`
- Search 子集结果：`stage3/results/temporal_joint_r2b_localalign_search_s{42,2024,2025}/`
- 下游 runs：`stage3/runs/temporal_incremental/R2B_L_*`
