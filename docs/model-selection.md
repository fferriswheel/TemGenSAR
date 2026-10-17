# 模型发布

## Hugging Face 版本

MovieLens-25M 模型与配套 SID 已发布至 [Ferriswheel1412/TemGenSAR-MovieLens25M](https://huggingface.co/Ferriswheel1412/TemGenSAR-MovieLens25M)。仓库目前为**私有**，下载需要具有访问权限的 Hugging Face 账号。

| 项目 | 内容 |
| --- | --- |
| 实验快照 | 2026-09-08 |
| 模型上传 | 2026-10-15 |
| 固定 revision | `947652476014c1a04ba6b3d59e27896b7adc804d` |
| 规模 | 10 个 T5-small checkpoint；共 90 个文件，2,608,847,143 bytes（约 2.43 GiB） |
| 注册表 | [selected_models.json](../assets/selected_models.json) |
| 验证 | 远端逐文件大小及 LFS SHA-256 / Git blob 哈希验证通过 |

## Checkpoint 目录

| Hugging Face 路径 | 角色 | 对应 SID |
| --- | --- | --- |
| `models/t0-p1-v13b/` | 共同初始化与 teacher，t0 seed 2026 | `sid/base/` |
| `models/frozen/seed-{42,2024,2025}/` | 固定 SID + 等预算下游更新对照 | `sid/base/` |
| `models/tasr-a/seed-{42,2024,2025}/` | 冻结 codebook 的选择性更新 | `sid/tasr-a/` 与 `sid/base/` |
| `models/tasr-b/seed-{42,2024,2025}/` | Refresh / align / selective materialization | `sid/tasr-b/` 与 `sid/base/` |

TASR-B 为当前主候选，A 为较少 SID 可训练参数的对照。B 相对 Frozen 的全体 Rec nDCG@10 约 +0.12%，与 A 的差异不显著；不能据此宣称 B 有大幅整体优势。详细指标、子集收益和统计范围见[实验结果](experiments.md)。

模型按验证集选择 checkpoint。示例使用 B seed 42，其在 B 的三个 seed 中具有最高 joint validation nDCG@5，未按 W2 测试指标挑选。全部正式 seed 均保留，报告方法性能时使用三 seed 汇总。开发 pilot seed 2026 未混入正式时序均值；t0 seed 2026 是共同初始化。

## 下载示例

安装 `huggingface_hub` 并使用有权限的账号执行 `hf auth login`。按需下载主候选及配套 SID：

```python
from huggingface_hub import snapshot_download

release_dir = snapshot_download(
    repo_id="Ferriswheel1412/TemGenSAR-MovieLens25M",
    revision="947652476014c1a04ba6b3d59e27896b7adc804d",
    token=True,
    local_dir="TemGenSAR-selected",
    allow_patterns=[
        "models/tasr-b/seed-42/*",
        "sid/base/*",
        "sid/tasr-b/*",
        "catalog/*",
        "README.md",
        "validation.json",
        "model_inventory.json",
        "SHA256SUMS",
    ],
)
```

其他模型可替换相应 `models/` 子目录，并使用表中匹配的 SID。完整文件清单与校验值见下载后的模型卡、inventory 和 SHA256SUMS。

## 加载与使用范围

```python
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir = Path(release_dir) / "models/tasr-b/seed-42"
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_dir, local_files_only=True
)

# 下列兼容参数已在 Transformers 4.52.3 上验证。
# 原模型使用 Transformers 5.9.0 保存，extra_special_tokens 为列表。
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=True,
    local_files_only=True,
    extra_special_tokens={},
    additional_special_tokens=[f"<extra_id_{i}>" for i in range(100)],
)
```

模型用于 SID 生成式搜推研究。有效检索还需要对应任务提示、用户上下文、item/SID 映射与约束解码，单独加载权重不能复现排名结果。接口与原始运行入口见[方法](method.md)和[代码导航](research-status.md)。

模型包包含 tokenizer / config、SID checkpoint 与部署映射、GenSAR index、item-only catalog、汇总评估和校验记录。原始交互、用户映射、完整训练输入、逐样本预测及训练日志不包含在模型包内。

## 产物验证与复现边界

- 10 份模型权重的哈希不同；词表大小均为 33,639。
- 已核对 tokenizer、权重结构、6 份 GenSAR index 与原始 MovieLens ID 的对应关系。
- 已检查 18,191-item 部署 catalog 唯一性、A/B 换码 25/67 个，以及 prefix / Search SID 保持不变。
- B seed 42 已在 Python 3.12、PyTorch 2.6.0、Transformers 4.52.3 环境完成加载与一次前向计算，输出有限值。
- 自定义 SID checkpoint 使用受约束的 `weights_only=True` 加载检查；基础 checkpoint 的 NumPy 数组需显式允许已核对的构造器和 dtype。
- 历史排名指标来自保存结果，未重新计算；上述检查不等同于完整 benchmark 复现。

## 其他实验模型

当前 HF 包只覆盖上述 MovieLens 系列。Amazon 静态 / R1S / Controlled Rec Drift、KuaiSAR 及 P1-R / P1-RN 等历史模型尚未包含在该发布中；相应代码与汇总证据仍保留于 GitHub。**约 +26.46% 的 Search 换码子集结果来自 Amazon R1S，使用另一组历史 checkpoint。**

其他历史资产的来源索引见[资产访问](storage.md)。阶段17–19分别记录 [P2 对照](../Search_Reconmendation/files/PhaseSummary/阶段17_MovieLens25M_P2共识前缀实验_2026-09-07.md)、[P1-R / RN 对照](../Search_Reconmendation/files/PhaseSummary/阶段18_P1条件残差与Rec信息增强对比_2026-09-08.md)与[最新时序实验](../Search_Reconmendation/files/PhaseSummary/阶段19_TASR两种时序更新实验_2026-09-08.md)。
