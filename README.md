# TemGenSAR

生成式搜索与推荐的研究代码，探索 **shared prefix / task-specific suffix Semantic ID (SID)**、时序漂移检测、选择性 Rec SID 更新，以及下游 T5/LoRA 对齐。

当前代码快照来自 **2026-09-07 完整服务器备份 + 2026-09-08 增量备份**。最新研究记录覆盖 MovieLens-25M、TASR-A / TASR-B 和严格三 seed 评估；早期实验包括 Amazon Electronics、controlled Rec drift 和 KuaiSAR。

## 从哪里开始

- [最新研究进展与入口](docs/research-status.md)
- [模型、数据的存储与恢复](docs/storage.md)
- [环境与运行路径](environment/README.md)
- [完整模型/张量清单](assets/model_files.tsv)
- [大文件与符号链接清单](assets/backup_manifest.tsv)

## 目录

```text
Search_Reconmendation/          原项目目录，保留历史拼写和层级
  stage3/scripts/               数据构建、SID、漂移检测、训练与汇总
  stage3/tests/                 原有测试
  stage3/results/               选定的汇总指标；逐样本结果留在备份
  files/PhaseSummary/           分阶段实验记录
  external/                    早期基线源码
external/GenSAR/                后期实际使用的 GenSAR，含 LoRA / 蒸馏改动
assets/                        文件位置、大小、来源备份和校验值
environment/                   环境说明与服务器依赖参考
scripts/                       仓库检查和按需恢复工具
docs/                          研究导航和存储说明
```

两处 GenSAR 源码有不同的本地改动，不能互相覆盖。上游均来自 [TengShi-RUC/GenSAR](https://github.com/TengShi-RUC/GenSAR)，来源和版本见 [third-party.md](docs/third-party.md)。

## 运行准备

研究环境为 **Linux + Python 3.12 + PyTorch**。历史 shell 脚本使用 Bash、CUDA 和 `/usr/bin/time`，不适合直接在 Windows PowerShell 中运行。

部分脚本使用绝对路径 `/data/cjk/Search_Reconmendation` 和 `/data/cjk/external/GenSAR`。在一个尚未使用的 Linux 工作目录中，把本仓库克隆到 `/data/cjk` 可保持这些路径；其他位置需调整脚本参数或路径。已有 `/data/cjk` 时先参照恢复说明准备单独的恢复目录。

```bash
git clone git@github.com:fferriswheel/TemGenSAR.git /data/cjk
cd /data/cjk
conda create -n temgensar python=3.12
conda activate temgensar
# 根据 CUDA/驱动安装合适的 PyTorch，再安装研究依赖
python -m pip install -r requirements-dev.txt
python -m pytest Search_Reconmendation/stage3/tests -q
```

`requirements.txt` 依据服务器导出的直接依赖整理，并非经过干净环境验证的完整锁文件；较早 GenSAR 的依赖版本不同。见 [环境说明](environment/README.md)。恢复对应数据、SID 映射和模型后，再运行实验入口。

## 代码与大文件

GitHub 保存源码、配置、实验说明和选定的汇总指标。模型权重、训练输入、SID 映射、embedding、逐样本预测、日志、缓存和备份压缩包保存在原备份中。

两份压缩包已于 2026-10-15 通过 SHA-256 校验。`assets/archives.json` 固定文件名和校验值；`backup_manifest.tsv` 标明每个资产应从完整包还是增量包恢复。模型记录中的 `.pt` / `.pth` 也可能是 embedding 或其他张量，须结合文件名和实验记录判断用途。

```bash
# 仅列出资产，不读取压缩包、不写文件
python scripts/restore_assets.py --include 'Search_Reconmendation/stage3/runs/movielens*' --list

# 每次提交前，检查 Git 索引中的大文件和常见凭据模式
git add .
python scripts/check_repository.py
```

后续建议把挑选后的可复用模型放到 Hugging Face 模型仓库；训练数据单独管理，完整备份继续存放在备份盘或对象存储。具体步骤见 [存储说明](docs/storage.md)。
