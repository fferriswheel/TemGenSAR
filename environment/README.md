# 研究环境与运行路径

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

部分 shell 与 Python 脚本保留上述绝对路径，默认 clone 后不能直接执行全流程。运行前检查所用脚本的 ROOT、GENSAR_ROOT、输入资产和输出路径，将它们适配为实际 checkout 与数据目录。HF 下载目录也需接入相应模型与 SID 参数。

实验顺序为：数据与时间划分 → 训练期表示与 t0 SID → t0 生成模型 → 当前窗口检测与 SID 更新 → matched 下游增量训练 → 下一窗口评估。[代码入口](../docs/research-status.md) · [方法](../docs/method.md) · [数据依赖](../docs/storage.md)

## 发布模型的加载验证

TASR-B seed 42 已在 Python 3.12、PyTorch 2.6.0+cu118、Transformers 4.52.3 下完成权重加载与有限值前向计算。原始模型使用 Transformers 5.9.0 保存，4.52.3 的 tokenizer 兼容参数见[模型发布](../docs/model-selection.md)。加载验证未修改模型文件，也未重跑排名评估。
