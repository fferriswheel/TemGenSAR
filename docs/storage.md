# 模型、数据与备份

## 本项目的存储划分

| 内容 | 本次位置 | 后续建议 |
| --- | --- | --- |
| 源码、配置、实验说明、汇总指标 | GitHub | 持续用 Git 管理 |
| 最终模型、LoRA、配套 tokenizer / config / SID | 原始备份；首批精选模型已有 HF 私有副本 | 持续记录模型 revision 与源码版本 |
| 原始数据、预处理数据、embedding | 原始备份 | 独立数据存储；可分发部分可用 HF Dataset |
| 所有历史实验和完整服务器备份 | 原始备份 | 备份盘及对象存储，例如 OSS / COS / S3 |

普通 GitHub Git 仓库拒绝超过 **100 MiB** 的文件。Git LFS 把大文件放到额外存储，但仍有单文件及账户配额约束。本项目只 `.safetensors` 就约 72.10 GiB，优先使用独立模型存储。[GitHub 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

Hugging Face 面向模型和数据的托管与分发，支持公开和私有仓库，存储受账户配额和用途要求约束；上传前检查当前剩余容量。[HF 存储说明](https://huggingface.co/docs/hub/storage-limits)。频繁更新的大数据可在对象存储上配合 DVC，Git 中仅记录版本指针；目前仓库没有引入 DVC。[DVC remote storage](https://doc.dvc.org/user-guide/data-management/remote-storage)。

## 已盘点资产

合并增量后的外置项目资产约 **108.69 GiB**，不是两个压缩包大小的简单相加。

- `assets/model_files.tsv`：735 条模型/张量记录，其中 330 个普通 `.safetensors` 文件、393 个普通 `.pt` / `.pth` 等文件，以及 12 条链接。
- `assets/backup_manifest.tsv`：10,760 条外置文件/链接记录，含路径、字节数、来源备份、类型、链接目标。
- `assets/source_snapshot.tsv`：导入的 354 个源码/说明/汇总文件的原始 SHA-256 与来源，用于核对迁移。
- `assets/archives.json`：两份原始压缩包的校验值，已于 2026-10-15 验证。

`.pt` / `.pth` 并不都代表完整模型，也可能是 encoder、embedding、codebook 或中间张量。完整包和增量包中的同一路径，以清单中标记的较新来源为准。

备份位于本地 `D:\EedsBackup`：

```text
cjk_server_backup_20260907.tar.zst
cjk_incremental_20260908_postwork.tar.zst
```

它们仍是完整恢复的主要副本；代码整理没有删除或重写这些备份。

首批精选的 10 个 MovieLens-25M 下游 checkpoint 和配套 SID 已于 2026-10-15 上传到 [私有模型仓库](https://huggingface.co/Ferriswheel1412/TemGenSAR-MovieLens25M)，90 个文件的远端内容哈希验证通过。完整筛选依据、固定 revision 和按需下载代码见 [模型筛选](model-selection.md)。其他历史模型和完整训练数据仍仅保留在原备份中。

## 按需恢复一个模型

在仓库根目录执行。示例恢复最新严格实验的 TASR-B seed 42，`*` 同时选择模型的配置、tokenizer、训练元数据和结果文件：

```powershell
python scripts/restore_assets.py --include 'Search_Reconmendation/stage3/runs/movielens25m_temporal/tasr_strict_b_distill_w1_w2_s42/*' --list
python scripts/restore_assets.py --backup-dir 'D:\EedsBackup' --destination 'D:\EedsBackup\RestoredAssets' --include 'Search_Reconmendation/stage3/runs/movielens25m_temporal/tasr_strict_b_distill_w1_w2_s42/*' --verify
```

需要 Python 3.9+ 和 PATH 中可用的 `zstd`。`--verify` 先校验所需完整压缩包，随后按清单选定来源恢复；压缩流需要顺序读取，选少量文件也可能耗时。工具拒绝覆盖已有文件，输出逐文件 SHA-256 到恢复目录的 `asset_restore_report.json`。

工具只恢复普通文件；若选择中含符号链接或硬链接，它会提示缩小选择范围。完整数据目录往往依赖绝对符号链接，应使用下面的 Linux 恢复方法。

## 完整恢复到 Linux

先把原始压缩包和 `.sha256` 文件原样传到 Linux，不要在 Windows 解开后重新打包。先在独立目录中校验和恢复，避免覆盖已有工作：

```bash
cd /path/to/backups
sha256sum -c cjk_server_backup_20260907.tar.zst.sha256
sha256sum -c cjk_incremental_20260908_postwork.tar.zst.sha256
mkdir -p /data/temgensar-restore
tar --zstd -xpf cjk_server_backup_20260907.tar.zst -C /data/temgensar-restore
tar --zstd -xpf cjk_incremental_20260908_postwork.tar.zst -C /data/temgensar-restore
```

完整恢复会包含原备份里的其他工作文件和私人会话记录；这些内容不属于公开代码仓库。实际执行路径为恢复目录下的 `cjk/`。备份里的绝对链接仍指向 `/data/cjk/...`，需要在 `/data/cjk` 未被占用时将该恢复根目录接入这个位置，或逐一重建链接后再运行。

增量删除清单仅有一个临时 inventory 文件，本次代码整理未应用额外删除；运行实验不依赖它。

## 将最终模型放到 Hugging Face

优先挑选 `movielens25m_temporal/tasr_strict_*_distill_w1_w2_s{42,2024,2025}` 的正式对照、TASR-A、TASR-B，以及对应 t0 模型；开发 seed 2026 和早期 smoke 模型继续留作备份。上传前根据阶段19和汇总表确认所选模型的用途。

1. 按清单恢复选定实验，创建一个干净的发布目录。
2. 仅放入推理需要的权重、tokenizer、模型配置、对应 SID mapping / codebook / item vocabulary，补充模型说明、数据来源、代码 commit、seed 和评估协议。日志、原始交互、逐样本预测不进入发布目录。
3. 通过本机 CLI 登录，创建私有模型仓库，再上传发布目录。不要把 token 写进 Git 或聊天。

```bash
python -m pip install -U huggingface_hub
hf auth login
```

登录后可通过 Python API 上传（替换账号与本地发布目录）：

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = 'YOUR_ACCOUNT/temgensar-tasr-b'
api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type='model',
    folder_path='/path/to/clean-model-release',
    commit_message='Upload reviewed TASR-B model assets',
)
```

上传成功后，在代码仓库新增一条模型记录，保存 `repo_id`、固定的 revision commit、所需文件路径及 SHA-256，让不同代码版本能找回相匹配的权重和 SID。首批已上传模型的记录位于 [`assets/selected_models.json`](../assets/selected_models.json)；上面的 API 代码是后续整理新发布包的通用示例。[HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)、[上传 API](https://huggingface.co/docs/huggingface_hub/guides/upload)。
