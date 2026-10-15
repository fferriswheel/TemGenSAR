# 2026-09-08 快照的模型筛选

筛选日期：2026-10-15。依据阶段12–19记录、机器可读结果表、实际训练日志和保留的 best-model 权重；没有重新训练，也没有重新选择测试集上最高分的 seed。

已上传至 [Ferriswheel1412/TemGenSAR-MovieLens25M](https://huggingface.co/Ferriswheel1412/TemGenSAR-MovieLens25M)，**私有仓库**。固定 revision：`947652476014c1a04ba6b3d59e27896b7adc804d`。共 90 个文件、2,608,847,143 bytes（约 2.43 GiB）；远端逐文件大小及 LFS SHA-256 / Git blob 哈希均已验证。

## 首批模型：最新的严格时序对照

| 模型 | 定位 | Rec nDCG@10（3-seed mean） | Search nDCG@10（3-seed mean） | 首批保留 |
| --- | --- | ---: | ---: | --- |
| TASR-B | 最新主候选，refresh / align / selective materialization | 0.286628 | 0.460220 | seeds 42 / 2024 / 2025 |
| TASR-A | 冻结 codebook 的高效版本 | 0.286483 | 0.460248 | seeds 42 / 2024 / 2025 |
| Frozen SID + matched downstream update | 公平对照和回退参照 | 0.286285 | 0.460273 | seeds 42 / 2024 / 2025 |
| P1/v13b t0 | 上述 9 个模型的共同初始化与 teacher | 不与 W2 时序指标直接横比 | 不与 W2 时序指标直接横比 | 原 t0 checkpoint，seed 2026 |

共 **10 个 T5-small checkpoint**。模型、tokenizer、配置、SID tokenizer/checkpoint 和映射约 **2.43 GiB**，另附评估表、模型卡和校验清单。开发 pilot seed 2026 不混入正式三 seed 时序结果；t0 的 seed 2026 是历史共同初始化，需要保留。

TASR-B 相对 Frozen 的 Rec nDCG@10 平均差为 **+0.000344**，相对约 **+0.12%**，3/3 seed 正向；记录中的 seed paired t-test p=0.0303、row bootstrap p=0.0302。三组 Rec H@10 均为 0.452275。Search 差异很小；B 相对 A 的差异不显著。因此 B 是当前最合适的主候选，但证据只支持小幅排序改善，不支持大幅整体提升。

TASR-A 仅更新 14,528 个 SID adapter 参数；TASR-B 更新 114,176 个参数。A 的 SID 更新约 15.75 秒，B 约 21.88 秒。这些是日志中的 **SID 更新阶段** 时间，不是包含下游 T5 训练的总成本。两者最终部署映射均无完整 SID collision，实际换码分别为 25 / 67 个。

正式时序模型来自：

```text
Search_Reconmendation/stage3/runs/movielens25m_temporal/
  tasr_strict_{fixed,a,b}_distill_w1_w2_s{42,2024,2025}/
```

共同 t0 来自：

```text
Search_Reconmendation/stage3/runs/movielens25m_t0/v13b_cg_smoke_seed2026/
```

t0 名称虽然含 `smoke`，日志与正式时序训练的 `initial_model_checkpoint` / `distillation_checkpoint` 均指向该路径；其角色由实际依赖决定。

## 其他候选的取舍

| 系列 | 证据 | 本次处理 |
| --- | --- | --- |
| P1/v13b 静态 SID | 同协议四指标均值 0.5706；Rec/Search nDCG@10 0.4347/0.4677 | 保留对应 t0 初始化；静态结果与 W2 时序结果分开说明 |
| P1-RN | 单 seed Rec nDCG@10 0.4367，Search 0.4660，四指标均值 0.5703 | 次选研究候选；整体未明确优于 P1，暂留原备份 |
| P1-R | Rec nDCG@10 0.4298，低于 P1 的 0.4347 | 暂不上传；collision 改善不等同于下游更好 |
| P2 consensus | 相同静态协议下 Rec/Search nDCG 降约 21%/27% | 排除首批 |
| Controlled Rec Drift V3-K51 | W6 三 seed Rec nDCG +0.000574；W9 留出仍优于 Fixed，但未稳定优于 SupportTopK | 有价值的历史机制验证；列为第二批，不与 MovieLens 绝对指标跨数据集排名 |
| KuaiSAR Detector / utility gate | 强制换码退化；最佳 gate 相对 Fixed 只 +0.000049，目标样本覆盖不足 | 不作为正向主模型上传；留作诊断证据 |

不同数据、负采样和时间协议下的绝对指标不可直接排名。最新、证据较完整的 strict matched MovieLens 对照作为首批；早期有效实验仍完整保留在备份中。

## 配套文件

每个下游 checkpoint 同时保留 `config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json` 和 `model.safetensors`。统一附带：

- P1 anchor / suffix checkpoint、anchor lookup 与原始部署 SID。
- TASR-A adapter、实际部署 mapping、GenSAR semantic/collaborate index。
- TASR-B 对齐后的 tokenizer checkpoint、实际部署 mapping、GenSAR index。
- MovieLens item ID 映射，仅导出 item 部分，排除 user mapping。
- 严格三 seed 汇总、筛选依据、源码 commit 和逐文件 SHA-256。

checkpoint 是 research release：SID 推理、约束解码与评估仍依赖原项目的提示词和数据处理。完整训练数据、用户映射、原始交互、逐样本预测和训练日志不放入模型仓库。

## 本次产物验证

10 份权重的 SHA-256 各不相同；所有 checkpoint 的词表大小为 33,639。已检查全部 tokenizer、权重文件结构、6 份 GenSAR index 与原始 MovieLens ID 的映射、18191-item catalog 的 SID 唯一性、A/B 换码数及 prefix/Search 保持不变的约束。

在本机已有的 Python 3.12 / PyTorch 2.6.0 / Transformers 4.52.3 环境中，TASR-B seed 42 完成完整本地加载和一次前向计算，输出有限值。模型原文件由 Transformers 5.9.0 保存；4.52.3 需要把列表形式的 `extra_special_tokens` 显式转作 `additional_special_tokens`，原始 tokenizer/config 文件保持不变。模型卡记录了加载方法。

四份自定义 SID `.pt` 均使用 `weights_only=True` 检查；基础 suffix checkpoint 中的 NumPy 数组仅加入已核对的 NumPy 构造器/dtype 白名单。历史检索指标未重新计算，验证范围不等同于完整 benchmark 复现。

上传状态、模型角色、源路径、校验值和固定 revision 记录在 [selected_models.json](../assets/selected_models.json)。

## 按需下载主候选

使用有访问权限的账号先执行 `hf auth login`。下面只下载 TASR-B 的预定代表 seed 42，以及相关 SID / catalog；seed 42 不是按 W2 最高分选出。其他正式 seed 仍在仓库中，可将子目录替换为 `seed-2024` / `seed-2025`。

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='Ferriswheel1412/TemGenSAR-MovieLens25M',
    revision='947652476014c1a04ba6b3d59e27896b7adc804d',
    token=True,
    local_dir='TemGenSAR-selected',
    allow_patterns=[
        'models/tasr-b/seed-42/*',
        'sid/base/*',
        'sid/tasr-b/*',
        'catalog/*',
        'README.md',
        'validation.json',
        'model_inventory.json',
    ],
)
```

该 seed 目录中的模型可用 `AutoModelForSeq2SeqLM.from_pretrained(local_path, local_files_only=True)` 加载。tokenizer 的版本兼容参数见下载得到的模型卡；检索效果仍需配套提示词、历史上下文和 SID 约束解码。

## 结果来源

- [阶段19：最新严格三 seed 结果](../Search_Reconmendation/files/PhaseSummary/阶段19_TASR两种时序更新实验_2026-09-08.md)
- [机器可读结果](../Search_Reconmendation/stage3/results/movielens25m_tasr_strict_w1_w2_three_seed/model_metrics.csv)
- [阶段18：P1 / P1-R / P1-RN](../Search_Reconmendation/files/PhaseSummary/阶段18_P1条件残差与Rec信息增强对比_2026-09-08.md)
- [阶段17：P2 对照](../Search_Reconmendation/files/PhaseSummary/阶段17_MovieLens25M_P2共识前缀实验_2026-09-07.md)
- [阶段12：Controlled Rec Drift](../Search_Reconmendation/files/PhaseSummary/阶段12_ControlledRecDrift_W6W9完整评估_2026-09-04.md)
- [阶段15：KuaiSAR](../Search_Reconmendation/files/PhaseSummary/阶段15_KuaiSAR自然Rec漂移实验_2026-09-05.md)
