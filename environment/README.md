# 研究环境与运行路径

本页面向运行实验的环境准备。研究代码把 SID（item 的语义编码）训练和下游生成模型训练分开；需要同时准备依赖、数据与对应编码映射。

## 参考环境

实验服务器环境记录为 Python **3.12.2**、PyTorch **2.6.0+cu124**、NumPy **1.26.4**、pandas **2.2.3**、Transformers **5.9.0**。

根目录 [requirements.txt](../requirements.txt) 整理研究源码直接使用且可从环境导出确认版本的依赖。[server-pip-freeze.txt](server-pip-freeze.txt) 是共享服务器环境参考，包含其他项目的软件，不是可移植的完整锁文件。当前尚未完成干净环境下的全流程安装与训练验证。

## 安装

历史实验入口使用 Linux、Bash、CUDA 与 /usr/bin/time。先建立独立 Python 3.12 环境，并按 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/)安装与驱动匹配的构建，再安装项目依赖：

```bash
git clone https://github.com/fferriswheel/TemGenSAR.git
cd TemGenSAR
conda create -n temgensar python=3.12
conda activate temgensar
# 在此安装匹配 CUDA / 驱动的 PyTorch 构建
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest Search_Reconmendation/stage3/tests -q
```

上述测试命令为验证入口，不表示全部测试已在独立环境通过；部分测试需要数据或额外依赖。源码导入与模型验证范围见[验证记录](../docs/migration.md)。

两份 GenSAR 自带 requirements 来自较早上游，指定 torch 2.3.0、Transformers 4.45.2 等不同版本。复现早期基线时使用独立环境；早期 embedding 脚本的 sentence_transformers 依赖需按对应分支准备。

## 路径与实验入口

| 历史服务器路径 | 仓库目录 |
| --- | --- |
| /data/cjk/Search_Reconmendation | Search_Reconmendation/ |
| /data/cjk/external/GenSAR | external/GenSAR/ |

部分 shell 与 Python 脚本保留上述绝对路径，默认 clone 后不能直接执行全流程。运行前检查所用脚本的 ROOT、GENSAR_ROOT、输入资产和输出路径，将它们适配为实际 checkout 与数据目录。Hugging Face 下载目录也需接入相应模型与 SID 映射参数，确保权重和编码版本匹配。

实验顺序为：数据按时间划分 → 提取训练期向量表示并建立初始 SID → 训练初始生成模型 → 当前窗口检测变化与更新 SID → 各组使用相同预算继续训练生成模型 → 下一窗口评估。脚本中的 t0 指初始阶段。[代码入口](../docs/research-status.md) · [方法](../docs/method.md) · [数据依赖](../docs/storage.md)

## 发布模型的加载验证

刷新推荐码本后对齐的模型（历史代号 TASR-B，TASR 无已定义英文全称），随机种子 42，已在 Python 3.12、PyTorch 2.6.0+cu118、Transformers 4.52.3 下完成权重加载与有限值前向计算。原始模型使用 Transformers 5.9.0 保存，4.52.3 的 tokenizer 兼容参数见[模型发布](../docs/model-selection.md)。加载验证未修改模型文件，也未重跑排名评估。
