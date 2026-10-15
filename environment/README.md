# 环境与路径

## 原服务器参考

2026-09-07 的服务器环境导出记录为 Python **3.12.2**、`torch==2.6.0+cu124`、`numpy==1.26.4`、`pandas==2.2.3`、`transformers==5.9.0`。

根目录 `requirements.txt` 选取了研究源码直接使用、且能从服务器导出中确认版本的依赖。`server-pip-freeze.txt` 是较大的共享环境参考，已去掉机器本地的 `file://` 包路径；其中包含与本项目无关的软件，不能当作可移植的完整锁文件。

服务器完整 Conda 导出仍保存在完整备份的 `cjk/_backup_meta_20260907/environment/` 中，没有把整套共享环境复制到仓库。

## 安装

1. 使用独立的 Python 3.12 环境。
2. 按 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/) 选择与驱动兼容的 PyTorch / CUDA 构建。备份使用 2.6.0 / cu124。
3. 在仓库根目录运行 `python -m pip install -r requirements-dev.txt`。
4. 运行 `python -m pip check` 和原有测试，再恢复实验所需资产。

两份 GenSAR 自带的 `requirements.txt` 来自较早上游，指定 `torch==2.3.0`、`transformers==4.45.2` 等不同版本。不要把它们与根目录依赖顺次混装；复现早期基线时使用单独环境。早期 embedding 脚本还使用 `sentence_transformers`，其依赖见相应基线目录。

本次迁移没有在本机安装整个 GPU 环境，也没有重跑训练。Windows 本机的 Python 3.9 环境只用于备份整理、源码检查及能够运行的原有测试；验证范围见 [迁移说明](../docs/migration.md)。

## 原路径

| 原服务器路径 | 本仓库相对路径 |
| --- | --- |
| `/data/cjk/Search_Reconmendation` | `Search_Reconmendation/` |
| `/data/cjk/external/GenSAR` | `external/GenSAR/` |

保留原层级是为了方便对应历史脚本、实验笔记及备份中的绝对符号链接。完整恢复到新位置时，必须处理链接中的原始 `/data/cjk` 路径；原样在 `/data/cjk` 恢复可保持其含义。
