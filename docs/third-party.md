# 第三方源码来源

本页列出实际包含的第三方源码。“上游提交”标识所基于的原始版本，目录内还包含本项目修改，因此不代表与上游内容完全相同。OneRec、TIGER 等论文的研究联系见[背景](background.md)，不表示它们的实现已被导入。

| 本仓库目录 | 上游 | 源快照记录的上游提交 |
| --- | --- | --- |
| `external/GenSAR` | https://github.com/TengShi-RUC/GenSAR | `bd1a126e4dbc5a18e30d361fab44b2bbc6274011` |
| `Search_Reconmendation/external/GenSAR` | https://github.com/TengShi-RUC/GenSAR | `bd1a126e4dbc5a18e30d361fab44b2bbc6274011` |
| `Search_Reconmendation/external/Amazon-Product-Search-Datasets` | https://github.com/QingyaoAi/Amazon-Product-Search-Datasets | `64907e59b4ce27738e61607f2d6b16b62dee92ac` |

这是带本地修改的源码快照，该提交仅标识共同上游版本。实际执行的后期副本是 `external/GenSAR`，增量备份更新了其中用旧模型约束新模型的蒸馏训练实现；两份副本分别保留，没有合并或替换。

`Search_Reconmendation/external/Amazon-Product-Search-Datasets/README.md` 保留数据项目的原始说明。原数据及压缩包未上传到本仓库。

仓库保留上游署名与已有说明，未设置覆盖全部第三方内容的统一许可证。第三方代码、模型和数据的使用与再分发需遵循各自的许可及数据使用条款。
