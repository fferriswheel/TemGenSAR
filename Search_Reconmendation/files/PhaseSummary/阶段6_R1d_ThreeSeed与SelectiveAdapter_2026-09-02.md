# 阶段6 R1d：3-Seed 验证与 Selective Adapter（2026-09-02）

## 1. 本轮目标

本轮完成两件事：

1. 在公平的 alignment item 集合与预算下，对 fixed、budget-random、drift-selected 三种 SID 更新策略做 3-seed 验证。
2. 在 3-seed 结果没有出现系统性问题后，将原先全量 item 的 Rec adapter 更新改为真正的 drift-selective adapter training，并验证 SID 与下游效果。

## 2. 公平实验协议

此前 legacy E2 fixed 数据包含 2,567 个 active item，而 drift-selected 只包含 12 个 changed item，直接比较会混入 alignment item 数量差异。本轮重新构造 fixed-selected 数据，使三种方法满足：

- 相同的 t0 SID 与 t0 checkpoint（seed 2024）；
- 相同的 W1 incremental train、W1 validation 和 W2 test；
- 相同的 alignment config budget：1,800；
- fixed 与 drift-selected 使用相同的 12 个 alignment item；
- incremental training seeds：2024、2025、2026。

这里的 3 seeds 只覆盖增量训练阶段的随机性，不是三个独立 t0 checkpoint 的端到端 3-seed 实验。论文最终实验仍应补充完整的 end-to-end seeds。

## 3. 3-Seed 结果

### 3.1 Mean +/- std

| Method | Val mean nDCG@5 | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Fixed-selected | 0.037248 +/- 0.000270 | 0.050733 +/- 0.000643 | 0.028642 +/- 0.000579 | 0.066067 +/- 0.001973 | 0.035637 +/- 0.000847 |
| Budget-random | 0.035298 +/- 0.000164 | 0.050667 +/- 0.000306 | 0.027443 +/- 0.000207 | 0.066133 +/- 0.001102 | 0.034550 +/- 0.000401 |
| **Drift-selected** | **0.038447 +/- 0.000360** | 0.050467 +/- 0.000833 | **0.030931 +/- 0.000745** | **0.066200 +/- 0.001908** | **0.035676 +/- 0.000826** |

### 3.2 Paired differences

| Comparison | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| Drift - Fixed | -0.000267 | **+0.002289 (3/3 positive)** | +0.000133 | +0.000039 |
| Drift - Random | -0.000200 | **+0.003488 (3/3 positive)** | +0.000067 | +0.001126 |

结论：drift-selected 的主要收益体现在 Rec 排序质量而不是 Hit。Rec nDCG@10 对 fixed 和 random 均在三个种子上一致提升；Rec H@10 基本持平，Search 没有系统性退化。因此可以进入 selective adapter 实现。

## 4. Selective Adapter 实现

### 4.1 参数更新范围

只训练 Rec task adapter，共 14,528 个参数：

- `rec_adapter.task_projection.{weight,bias}`
- `rec_adapter.down.weight`
- `rec_adapter.up.weight`
- `rec_adapter.norm.{weight,bias}`

以下部分冻结：shared encoder/prefix、Search adapter/suffix、两层 Rec quantizer/codebook 和 decoder。因此 shared prefix 与 Search suffix 在参数和 SID 层面都保持不变。

### 4.2 Item 选择与 replay

- C8 -> W1 drift selector 选中 63 个 Rec drift item；
- 从低漂移 item 中固定采样 126 个 replay item（ratio 2.0）；
- 每个 epoch 仅对 189 / 19,999 个 item 计算梯度；
- 全量 item 只用于无梯度评估与最终 SID 导出。

训练保留 reconstruction、topology、commitment 与 stability 四类约束。最优 epoch 为 9，实际训练到 early-stop 共 57 optimizer steps。

### 4.3 SID 生成与 sticky assignment

adapter 是共享函数，即使只用 189 个 item 训练，直接重新量化仍可能改变未选 item。因此 operational SID 继续采用 drift-selected sticky assignment：

- selected item：63；
- 最终 changed item：13；
- selection 外 item 全部保持原 SID；
- duplicate fraction：0.004050，与 t0 基线一致；
- 没有引入新的 collision。

## 5. Selective 与 Full Adapter 对比

### 5.1 SID 阶段成本

| Update | Gradient items | Trainable params | Optimizer steps | Adapter core wall time |
|---|---:|---:|---:|---:|
| Full-item adapter + replay implementation | 19,999 | 14,528 | - | 180.2 s |
| **Selective adapter** | **189** | **14,528** | **57** | **5.8 s** |

当前机器上的 measured core time 约减少 31 倍。这个数字还不是严格的跨机器复杂度结论，但已证明训练开销不必随 item universe 线性增长。

### 5.2 W2 下游结果（seed 2024，alignment budget 1,800）

| SID update | Rec H@10 | Rec nDCG@10 | Search H@10 | Search nDCG@10 |
|---|---:|---:|---:|---:|
| Full adapter drift-selected | 0.0514 | **0.03179** | 0.0640 | **0.03474** |
| **Selective adapter drift-selected** | **0.0516** | 0.03148 | **0.0642** | 0.03467 |

两者差异处于千分位以内。Selective adapter 在大幅减少 SID 更新训练量的同时，保持了 Rec/Search 下游表现，可作为后续多窗口实验的默认更新实现。

## 6. 当前结论与边界

已经支持的结论：

- drift-aware item selection 对 Rec nDCG 的收益跨三个 incremental seeds 稳定；
- 只对 selected + replay item 训练 Rec adapter 是可行的；
- prefix、Search suffix 与 selection 外 operational SID 可以严格冻结；
- selective adapter 的 SID 更新成本显著低于全 item adapter，且单 seed 下游效果等价。

尚不能宣称：

- 3-seed 已覆盖 t0 representation、SID 初始化和 t0 downstream training 的全部随机性；
- alignment budget 1,800 可跨所有窗口固定最优；
- 当前只验证了 Rec drift，Search drift 与双任务同时漂移尚未完成；
- 5.8 s 与 180.2 s 的比值尚未经过同进程、同 profiling 口径的重复计时。

## 7. 主要产物

- 3-seed summarizer：`stage3/scripts/summarize_r1b_three_seed.py`
- 3-seed tables：`stage3/runs/temporal_r1b/three_seed/`
- Selective adapter trainer：`stage3/scripts/update_v13b_rec_adapter_selective.py`
- Selective adapter checkpoint/stats：`stage3/data/gensar_faithful_20k_rec_w1/v13b_rec_adapter_selective_r2/`
- Sticky SID：`stage3/data/gensar_faithful_20k_rec_w1/v13b_selective_adapter_sticky_r01/`
- Downstream run：`stage3/runs/temporal_incremental/R1B_selective_adapter_align1800_s2024/`

