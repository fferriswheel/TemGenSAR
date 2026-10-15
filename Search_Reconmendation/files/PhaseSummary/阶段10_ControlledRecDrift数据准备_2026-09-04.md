# 阶段10：Controlled Rec Drift 数据准备（2026-09-04）

## 1. 注入定义

- 数据源：完整 20K Amazon Search+Rec benchmark；
- Search 日志、item 静态信息、用户、时间戳、事件数均保持不变；
- 选取 30% 的可用用户 cohort；
- 在相同 metadata `root category + leaf category` 内对 item 做一一轮换；
- 不使用 M7b/v13b SID 分组，避免给本文方法引入先验优势；
- W3/W6/W9 开始的事件替换率依次为 0.25/0.50/0.75，并持久生效。

## 2. 实际注入量

- eligible users：32,590；
- requested affected users：9,689；
- realized affected users：8,495；
- changed events：25,300；
- changed items：13,016；
- W3/W6/W9 stage changed events：4,101 / 10,006 / 11,193。

## 3. Sanity Check

- Search 文件 SHA256 完全相同；
- 所有季度 Rec event/user 数完全相同；
- 注入前 C8→W1、W1→W2 的指标与原始数据完全相同；
- transition JS 相对原始数据的增量：
  - W2→W3：+0.001257；
  - W5→W6：+0.014006；
  - W8→W9：+0.018796；
  - W9→W10：+0.028600。

全局 Rec detector 在原始 Amazon 数据上本来就几乎每个季度触发，因此 `task_trigger=true` 不能单独作为 controlled drift 成功证据。后续应同时报告原始/注入配对差分和 affected-cohort 指标。

## 4. 数据产物

- 注入数据：`stage3/data/gensar_faithful_20k_controlled_rec_w3w6w9_v1/`
- 季度 benchmark：`stage3/data/gensar_faithful_20k_quarterly_controlled_rec_w3w6w9_v1/`
- detector 报告：`stage3/results/controlled_rec_drift_w3w6w9_v1/temporal_drift_report.json`
- 配对 sanity：`stage3/results/controlled_rec_drift_w3w6w9_v1/paired_sanity.json`
- 注入脚本：`stage3/scripts/inject_controlled_rec_drift.py`
- 检查脚本：`stage3/scripts/check_controlled_rec_drift.py`

## 5. 下一步协议

不能直接使用旧 controlled-search rolling 的 W2 checkpoint，否则 Rec-only 实验会携带 Search 注入历史。下一步先用本数据的未注入 W1/W2 构造 clean pre-drift trajectory，再从 W3 开始比较：

1. Fixed SID；
2. matched-budget Random Rec suffix update；
3. selective Rec suffix update；
4. full Rec SID retrain；
5. 每种下游更新使用相同数据、epoch、候选集和 alignment budget；
6. 先 seed 2024 smoke，再对通过 smoke 的方法扩三 seed。
