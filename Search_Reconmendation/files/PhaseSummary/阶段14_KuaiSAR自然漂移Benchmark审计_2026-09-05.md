# 阶段14：KuaiSAR 自然漂移 Benchmark 审计（2026-09-05）

## 1. 目标

本阶段评估 KuaiSAR 能否作为 Amazon Electronics controlled drift 之外的第二个真实 Search+Rec temporal benchmark。此阶段先审计数据、时间覆盖、跨任务 item 重叠和自然漂移信号，不在 benchmark 可用性未确定时盲目生成 SID 或训练 T5。

## 2. 数据下载与校验

数据来自 KuaiSAR 官方 Zenodo record 8181109。

| 版本 | 文件 | 压缩大小 | MD5 | 校验 |
|---|---|---:|---|---|
| KuaiSAR-small | `KuaiSAR.zip` | 488,077,651 B | `daea8cbf605db6bd5841740f0e4a12d9` | 通过 |
| KuaiSAR full | `KuaiSAR_v2.zip` | 845,087,902 B | `ed630f47fb013c5b31bb020c483ef262` | 通过 |

本地路径：

- `stage3/raw/kuaisar_small/KuaiSAR_final/`
- `stage3/raw/kuaisar_full/KuaiSAR_v2/`

## 3. Schema 与正反馈定义

### Recommendation

`rec_inter.csv` 包含 user/item/timestamp，以及 `click`、`like`、`follow`、`forward`、`search` 等多行为字段。当前 pilot 将：

```text
click == 1
```

定义为 Rec 正反馈。

### Search

`src_inter.csv` 中一个 search session 对应多个曝光 item，以：

```text
click_cnt > 0
```

定义 query-item 正反馈。未点击曝光可在后续表示学习中作为负例，但本次 catalog 和评估支持量只计正反馈。

### 文本限制

KuaiSAR 不提供可读的 caption/query 原文：

- `caption` 是匿名 token ID 列表；
- `keyword` 也是匿名 token ID 列表；
- item 可读信息主要是四级中/英文类别、item type 等结构化字段。

因此无法直接复用 Amazon 上的 `BGE(title + description)`。后续若使用 KuaiSAR，common/search representation 需用匿名 token 的共现或自监督学习，并向 GenSAR baseline 提供同一组输入信息以保持公平。

## 4. 数据规模

| 统计 | KuaiSAR-small | KuaiSAR full |
|---|---:|---:|
| 时间跨度 | 约 9 天 | 约 18.78 天 |
| users | 25,877 | Rec 25,877 / Search 25,705 |
| Rec rows | 7,493,101 | 14,605,716 |
| Rec positive rows | 3,709,514 | 7,244,890 |
| Search rows | 3,171,231 | 5,059,169 |
| Search positive rows | 370,524 | 587,568 |
| Rec positive items | 1,200,303 | 2,109,640 |
| Search positive items | 296,956 | 461,441 |
| Search/Rec positive item intersection | 14,700 | 28,198 |

虽然 full 数据共有约 689 万 item，在两个任务都出现过正反馈的 item 只有 28,198。这是最合理的联合 catalog 上界。

## 5. 时间窗口审计

审计同时运行 24h 与 72h 窗口。Full 数据可得到：

- 18 个完整 24h 窗口；
- 6 个完整 72h 窗口，记为 W0–W5。

Full 数据的 Search 每日正反馈从高峰 74,965 降到后期 10,248，最低值/中位数为 0.395。这表明时间比较不能只看 raw event count，必须使用概率分布并辅以 persistent-user 口径。

## 6. 自然漂移特征

在 full 数据的 28,198-item 全时段联合 catalog 上：

| 窗口对 | Rec item JS | Rec 双侧至少 5 次的 item | Search item JS | Search 双侧至少 5 次的 item |
|---|---:|---:|---:|---:|
| W0→W1 | 0.524 | 1,025 | 0.808 | 28 |
| W1→W2 | 0.441 | 1,144 | 0.817 | 21 |
| W2→W3 | 0.517 | 765 | 0.855 | 10 |
| W3→W4 | 0.452 | 668 | 0.854 | 10 |
| W4→W5 | 0.505 | 434 | 0.897 | 4 |

一级 category JS 却只有：

- Rec：0.0007–0.0027；
- Search：0.0027–0.0082。

因此 KuaiSAR 中的高 item-level JS 主要体现同类别内容快速轮换和 item lifecycle，而不是宏观类别偏好发生大幅转移。这是真实自然变化，但论文中应表述为 `item-level behavioral/content turnover`，不应无条件表述为用户长期偏好漂移。

## 7. Catalog 方案对比

### 7.1 严格首窗口 catalog

只使用 W0 中两任务的正反馈交集：

- catalog 仅 3,336 item；
- Search positive events 从 W0 的 4,588 降到 W5 的 103。

不适合长期联合评估。

### 7.2 严格初始化期交集 catalog

将 W0–W2 作为 9 天初始化期，只使用这三窗的两任务正反馈交集：

- catalog 为 16,378 item；
- 未来 W3/W4/W5 Search positive events 为 1,350/931/467。

可以用于小型诊断，但 Search 稳定性指标方差较大，不宜作为主 benchmark。

### 7.3 严格任务均衡并集 catalog（最终采用）

为避免交集 catalog 只保留两个任务都热门的 item，同时不使用未来信息，最终从 W0-W2 构造：

```text
Top-15K by Rec positive support
UNION
Top-15K by Search positive support
```

确定性处理同频 tie 后得到 29,784 个 item，其中两侧 Top-15K 的重叠只有 216 个。catalog 在 W2 后冻结，W3-W5 的 item 是否出现不影响 membership。

| Window | Rec positive events | Search positive events |
|---|---:|---:|
| W0 | 206,072 | 21,199 |
| W1 | 521,758 | 35,992 |
| W2 | 308,569 | 13,078 |
| W3 | 138,047 | 5,323 |
| W4 | 57,664 | 3,898 |
| W5 | 28,307 | 2,067 |

W4/W5 按“每用户窗口内第一个目标、history 严格截止到窗口开始前”构造的支持量为：

| Window | Rec targets | Search targets |
|---|---:|---:|
| W4 | 6,072 | 1,737 |
| W5 | 4,786 | 933 |

绝对样本数足以完成自然 Rec drift pilot，但相对全量未来正反馈的覆盖率持续下降。该 benchmark 因而只回答固定 catalog 内 warm-item 稳定性，不回答新 item 冷启动或 open-world 更新。

### 7.4 全时段固定联合 catalog（仅作 transductive 上界）

用全时段两任务正反馈的交集固定 28,198 item：

| Window | Rec positive events | Search positive events |
|---|---:|---:|
| W0 | 48,329 | 7,504 |
| W1 | 106,376 | 14,510 |
| W2 | 54,145 | 5,726 |
| W3 | 51,615 | 5,477 |
| W4 | 36,664 | 5,490 |
| W5 | 23,327 | 3,464 |

这套 catalog membership 使用全时段出现信息，属于 transductive 口径。当前不再把它作为主实验，只保留为覆盖率上界。

## 8. 可用性结论

### 可以做

KuaiSAR full 可作为补充性 `natural Rec-drift + frozen Search stability` benchmark，最终协议为：

```text
fixed catalog: W0-W2 task-balanced Top-15K union, 29,784 items
W0-W2: initialization + detector calibration
W3: Rec update
W4: development evaluation
W4: second Rec update
W5: untouched holdout evaluation
Search SID: fixed
behavior representation: strictly uses revealed windows only
```

### 暂时不能做

1. 不能直接复用 Amazon 的 BGE common/search representation；
2. 不适合在当前支持量下做 item-level Search suffix drift/update；
3. 不能从固定 catalog 的 warm-item 结果推断 open-world item 能力；
4. 不应将高 item JS 单独解释为用户偏好变化。

## 9. 已冻结的实现口径

初始化表示只使用 W0-W2：

- common：caption token、category 和 item type 的 metadata-only 去噪对比编码器，128 维；
- Search：W0-W2 query-item 多正例对比编码器，128 维；
- collaborative：W0-W2 Search/Rec history 的 UniSAR-faithful 表示，64 维；
- GenSAR-faithful：`common + collaborative`；
- GenSAR-signal-matched：`[common, Search] + collaborative`；
- v13b：`common + Search + collaborative` 三 view gated anchor，固定 shared prefix，任务 private suffix。

两种 GenSAR 口径用于区分“遵循原始双 view 结构”和“原始信号完全匹配”。后续所有 SID 均使用 `[256,256 | 256,256]`，并对 GenSAR 与 v13b 使用同样的 collision-group Sinkhorn 导出规则。

## 10. 产物

- 审计脚本：`stage3/scripts/audit_kuaisar_temporal_pilot.py`
- 严格审计结果：`stage3/results/kuaisar_full_temporal_pilot_v2_strict_balanced/`
- 时序 Parquet：`stage3/data/kuaisar_strict_balanced_30k_w0_w5/`
- W0-W2 JSONL：`stage3/data/kuaisar_strict_balanced_30k_init_w0_w2/`
- common encoder：`stage3/artifacts/kuaisar_common_w0_w2_v1/`
- Search encoder：`stage3/artifacts/kuaisar_search_w0_w2_v1/`
- collaborative encoder：`stage3/artifacts/kuaisar_unisar_w0_w2_v1/`
- 数据与表示测试：`stage3/tests/test_audit_kuaisar_temporal_pilot.py`、`test_build_kuaisar_strict_temporal_dataset.py`、`test_export_kuaisar_initialization_jsonl.py`、`test_train_kuaisar_representation.py`
