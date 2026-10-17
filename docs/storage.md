# 模型、数据与资产访问

## 分发范围

| 内容 | 位置与可用性 |
| --- | --- |
| 源码、研究文档、配置与选定汇总指标 | 本 GitHub 仓库 |
| 10 个 MovieLens checkpoint、对应 SID 与 item catalog | [Hugging Face 模型仓库](https://huggingface.co/Ferriswheel1412/TemGenSAR-MovieLens25M)，目前私有，需账号权限 |
| 原始交互、用户映射、预处理数据与 embedding | 未随代码或 HF 模型包分发；复现实验需准备对应数据 |
| 其他历史模型、逐样本预测与完整运行日志 | 历史归档，未作为公开下载资源发布 |

HF 固定 revision、模型目录、配套映射及下载方式见[模型发布](model-selection.md)。固定 query/tag bank 的 Search 控制任务与真实 Search 日志应按具体 benchmark 区分，不能相互替代。

## 数据与运行依赖

| Benchmark | 所需资产 | 构建 / 协议入口 |
| --- | --- | --- |
| Amazon Electronics | Product Search 数据、原始 5-core reviews / metadata、chronological sequences、BGE / 行为表示与 SID | [第三方数据来源](third-party.md)、[历史原型说明](../Search_Reconmendation/stage3/README.md)、[严格静态与时序协议](experiments.md) |
| MovieLens-25M | 原始数据、时间窗口、固定 item catalog、Search query/tag bank、表示与 SID | [benchmark builder](../Search_Reconmendation/stage3/scripts/build_movielens25m_temporal_benchmark.py)、[GenSAR export](../Search_Reconmendation/stage3/scripts/build_movielens25m_gensar_temporal.py) |
| KuaiSAR | 原始联合搜推日志与对应预处理输出 | [严格时序 builder](../Search_Reconmendation/stage3/scripts/build_kuaisar_strict_temporal_dataset.py)、[阶段15协议](../Search_Reconmendation/files/PhaseSummary/阶段15_KuaiSAR自然Rec漂移实验_2026-09-05.md) |

原始数据需从各自发布方按相应条款获取。本仓库尚未提供覆盖全部实验的一键数据准备流程；历史路径、符号链接与模型输入版本需要按[环境说明](../environment/README.md)适配。

## 历史资产清单

以下清单用于来源追踪，不表示文件已公开托管：

- [model_files.tsv](../assets/model_files.tsv)：735 条模型/张量记录；其中 330 个普通 safetensors、393 个普通 pt/pth 等文件和 12 条链接。pt/pth 也可能是 embedding、codebook 或中间张量。
- [backup_manifest.tsv](../assets/backup_manifest.tsv)：10,760 条外置文件/链接记录，包含相对路径、大小、来源与链接目标。
- [source_snapshot.tsv](../assets/source_snapshot.tsv)：354 个初始导入文件的原始哈希；记录导入基点，后续受版本控制的修改以 Git 历史为准。
- [archives.json](../assets/archives.json)：源归档文件名与 SHA-256。

## 从已有归档恢复资产

此入口面向已持有相应归档的研究环境；普通 GitHub clone 不包含这些压缩包。优先通过 HF 获取已发布模型，历史资产可按清单恢复。

```bash
# 列出匹配的历史资产，无需读取归档
python scripts/restore_assets.py \
  --include 'Search_Reconmendation/stage3/runs/movielens25m_temporal/tasr_strict_b_distill_w1_w2_s42/*' \
  --list

# /path/to/archives 和 /path/to/restored 均需替换为实际目录
python scripts/restore_assets.py \
  --backup-dir /path/to/archives \
  --destination /path/to/restored \
  --include 'Search_Reconmendation/stage3/runs/movielens25m_temporal/tasr_strict_b_distill_w1_w2_s42/*' \
  --verify
```

恢复工具需要 Python 3.9+ 与 PATH 中的 zstd。它按清单选定完整包或增量包来源，拒绝覆盖已有文件，并输出逐文件 SHA-256。压缩流顺序读取可能耗时；工具仅恢复普通文件，含链接的选择需缩小范围。完整目录恢复还需处理历史绝对符号链接。

模型、数据、embedding、运行日志与归档由 .gitignore 排除，Git 保存源码、文档及适合分发的汇总结果。源码来源与验证范围见[来源记录](migration.md)。
