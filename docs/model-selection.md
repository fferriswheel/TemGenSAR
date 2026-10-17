# 模型发布

本页介绍可下载的模型快照（checkpoint，即保存的权重与配置）。这些模型用于预测 item 的语义编码 SID；有效检索还需要配套映射，把生成的编码对应回商品或电影。

## Hugging Face 版本

MovieLens-25M 模型与配套 SID 已发布至 [Ferriswheel1412/TemGenSAR-MovieLens25M](https://huggingface.co/Ferriswheel1412/TemGenSAR-MovieLens25M)。仓库已于 **2026-10-17 公开**，模型与配套 SID 可无需登录下载。

| 项目 | 内容 |
| --- | --- |
| 实验快照 | 2026-09-08 |
| 模型上传 | 2026-10-15 |
| 固定 revision | `803fbdc690e8d80f5f2b4450bb2838c96c26530a` |
| 规模 | 10 个 T5-small checkpoint；共 90 个文件，2,608,847,654 bytes（约 2.43 GiB） |
| 注册表 | [selected_models.json](../assets/selected_models.json) |
| 验证 | 远端逐文件大小及 LFS SHA-256 / Git blob 哈希验证通过 |

公开版本只更新了模型卡与对应校验清单；全部 checkpoint、SID 与评估文件保持不变。原权重版本为 `947652476014c1a04ba6b3d59e27896b7adc804d`，当前下载固定到表中的公开文档版本。

## 模型目录与方法含义

当前比较固定编码、冻结推荐码本更新、刷新推荐码本后对齐三种方案。码本是将连续表示转成离散 token 的向量集合；TASR 是历史系列代号，原记录未定义英文全称，A/B 区分后两种更新方案。t0 表示更新前共同训练好的初始模型。

| Hugging Face 路径 | 角色 | 对应 SID |
| --- | --- | --- |
| `models/t0-p1-v13b/` | 共同初始模型与蒸馏教师；初始训练随机种子 2026 | `sid/base/` |
| `models/frozen/seed-{42,2024,2025}/` | 固定 SID + 等预算下游更新对照 | `sid/base/` |
| `models/tasr-a/seed-{42,2024,2025}/` | 冻结推荐码本与表示解码器，只更新推荐 adapter（小型任务表示网络） | `sid/tasr-a/` 与 `sid/base/` |
| `models/tasr-b/seed-{42,2024,2025}/` | 更新推荐 adapter、码本与表示解码器，对齐新旧 token 后只发布局部换码 | `sid/tasr-b/` 与 `sid/base/` |

刷新码本后对齐（B）为当前主候选，冻结码本（A）所需的 SID 可训练参数较少。B 相对固定编码组的全体推荐 nDCG@10（考虑前十位置的排序指标）约 +0.12%，与 A 的差异不显著。两者核心区别是推荐码本与表示解码器是否允许更新，并非下游是否训练；两组均继续全参数训练 T5。详细指标、子集收益和统计范围见[实验结果](experiments.md)。

每次训练按验证集选保存的模型。示例使用 B 的随机种子 42，它在 B 的三次训练中具有最高搜索/推荐联合验证 nDCG@5，未按未来测试结果挑选。42、2024、2025 三个正式随机种子均保留并用于均值汇总。种子 2026 的时序训练仅用于开发试验；初始模型也用 2026 训练，这是另一角色，不混入正式时序均值。

## 下载示例

安装 `huggingface_hub` 后即可匿名下载。以下示例获取主候选及配套 SID：

```python
from huggingface_hub import snapshot_download

release_dir = snapshot_download(
    repo_id="Ferriswheel1412/TemGenSAR-MovieLens25M",
    revision="803fbdc690e8d80f5f2b4450bb2838c96c26530a",
    token=False,
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

模型包包含文本分词器与配置、SID 模型、最终 item→SID 部署映射、GenSAR 使用的编码索引、只含 item 的目录，以及汇总评估和校验记录。原始交互、用户映射、完整训练输入、逐样本预测及训练日志不包含在模型包内。

## 产物验证与复现边界

- 10 份模型权重的哈希不同；词表大小均为 33,639。
- 已核对 tokenizer、权重结构、6 份 GenSAR index 与原始 MovieLens ID 的对应关系。
- 已检查 18,191-item 部署 catalog 唯一性、A/B 换码 25/67 个，以及 prefix / Search SID 保持不变。
- B seed 42 已在 Python 3.12、PyTorch 2.6.0、Transformers 4.52.3 环境完成加载与一次前向计算，输出有限值。
- 自定义 SID checkpoint 使用受约束的 `weights_only=True` 加载检查；基础 checkpoint 的 NumPy 数组需显式允许已核对的构造器和 dtype。
- 历史排名指标来自保存结果，未重新计算；上述检查不等同于完整 benchmark 复现。

## 其他实验模型

当前 HF 包只覆盖上述 MovieLens 系列。Amazon 静态编码、受控搜索/推荐漂移、KuaiSAR，以及增加残差重构或序列邻域的结构变体等历史模型尚未包含在该发布中；相应代码与汇总证据仍保留于 GitHub。**约 +26.46% 的 Search 换码子集结果来自 Amazon R1S，使用另一组历史 checkpoint。**

其他历史资产的来源索引见[资产访问](storage.md)。阶段17–19分别记录 [共识前缀对照（P2）](../Search_Reconmendation/files/PhaseSummary/阶段17_MovieLens25M_P2共识前缀实验_2026-09-07.md)、[残差重构与序列邻域对照（P1-R / RN）](../Search_Reconmendation/files/PhaseSummary/阶段18_P1条件残差与Rec信息增强对比_2026-09-08.md)与[最新时序实验](../Search_Reconmendation/files/PhaseSummary/阶段19_TASR两种时序更新实验_2026-09-08.md)。
