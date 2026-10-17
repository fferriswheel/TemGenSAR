# 源码来源与验证记录

本页记录源码和发布资产从哪里来、哪些验证已完成，不承担方法说明。代码只导入研究实现与汇总结果；当前模型访问见[模型发布](model-selection.md)，实验解释见[核心实验](experiments.md)。

## 源码基点

研究快照由 2026-09-07 完整归档与 2026-09-08 postwork 增量归档合并形成，同路径采用较新的增量版本。初始导入提交为 `ce1ddd8511ef3bc863b90dcc3ac91c46ed1b3b68`，包含 354 个原始源码、说明和选定汇总文件；后续修改以 Git 历史为准。

哈希是用于核对文件内容是否变化的摘要。原始哈希见 [source_snapshot.tsv](../assets/source_snapshot.tsv)，归档校验值见 [archives.json](../assets/archives.json)。两份源归档于 2026-10-15 通过 SHA-256 与完整 zstd 压缩流检查。

保留 Search_Reconmendation 与 external/GenSAR 的原层级及不同修改的两份 GenSAR 副本。后期实际使用的旧模型知识保持训练（蒸馏）实现位于根目录 external/GenSAR；上游版本见[第三方来源](third-party.md)。

## 代码、结果与归档边界

Git 中保留源码、说明和汇总指标。模型权重、原始/处理数据、向量表示、逐样本预测、日志、缓存与嵌套 Git 元数据未导入普通 Git 历史。精选 MovieLens 模型随后发布至 Hugging Face，版本与验证记录见[模型发布](model-selection.md)。

[PhaseSummary](../Search_Reconmendation/files/PhaseSummary/) 与早期 stage3 README 是历史研究档案，包含当时的计划、环境路径与阶段性判断。它们用于追踪实验依据；当前方法、结果范围和后续计划分别以 [method](method.md)、[experiments](experiments.md)、[roadmap](roadmap.md) 为入口。

## 已执行检查

初始导入阶段已核对 354 个文件的哈希/大小、全部 230 个原始 Python / shell 源文件及可执行标记，并完成 180 个 Python 与 52 个 Bash 文件的语法检查。恢复工具通过选定恢复、路径穿越拒绝和已有文件保护检查；两个已有数据导出/映射测试模块共 4 项测试通过。

受依赖与环境限制，初始导入阶段未完成全部研究测试或 GPU 训练。后续模型发布完成了文件一致性、SID 约束及代表性 checkpoint 加载/前向检查，历史排名指标未重算。

这些验证限定于来源完整性、语法与产物一致性，不代表干净环境下的端到端复现。每次提交可使用 `python scripts/check_repository.py` 检查 Git 索引中的大文件、运行产物及常见凭据模式。
