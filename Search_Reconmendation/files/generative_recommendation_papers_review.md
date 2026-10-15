# 生成式推荐四篇论文精读汇总与对比

本文档汇总并扩展前面对四篇生成式推荐论文的阅读讲解：

1. **TIGER**：*Recommender Systems with Generative Retrieval*
2. **Actions Speak Louder than Words**：*Trillion-Parameter Sequential Transducers for Generative Recommendations*
3. **OneRec**：*Unifying Retrieve and Rank with Generative Recommender and Preference Alignment*
4. **PLUM**：*Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations*

这四篇论文构成了一条清晰的发展线索：

```text
传统 ANN/MIPS 召回
    ↓
生成式检索：生成 item Semantic ID
    ↓
工业级行为序列生成建模：retrieval/ranking sequential transduction
    ↓
单阶段列表生成：直接生成最终推荐 session
    ↓
预训练 LLM 适配：SID + CPT + SFT 的工业级生成式召回
```

从学术脉络看，TIGER 证明了“生成 item ID 进行召回”的可行性；Meta 的 HSTU/GR 论文从系统架构上重新表述 retrieval 与 ranking；OneRec 进一步尝试用单个生成模型直接生成最终推荐列表；PLUM 则将预训练大语言模型与 Semantic ID 深度结合，探索工业级 LLM-based generative retrieval。

---

## 目录

- [0. 关键术语预备](#0-关键术语预备)
- [1. TIGER：Recommender Systems with Generative Retrieval](#1-tigerrecommender-systems-with-generative-retrieval)
- [2. Actions Speak Louder than Words：Trillion-Parameter Sequential Transducers](#2-actions-speak-louder-than-wordstrillion-parameter-sequential-transducers)
- [3. OneRec：Unifying Retrieve and Rank with Generative Recommender and Preference Alignment](#3-onerecunifying-retrieve-and-rank-with-generative-recommender-and-preference-alignment)
- [4. PLUM：Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](#4-plumadapting-pre-trained-language-models-for-industrial-scale-generative-recommendations)
- [5. 四篇论文的横向比较](#5-四篇论文的横向比较)
- [6. 四篇论文共同揭示的核心问题](#6-四篇论文共同揭示的核心问题)
- [7. 关键概念复习与问答](#7-关键概念复习与问答)
- [8. 批判性总结与研究方向](#8-批判性总结与研究方向)
- [9. 复习清单](#9-复习清单)

---

# 0. 关键术语预备

## 0.1 ANN

**ANN** 是 **Approximate Nearest Neighbor Search**，即近似最近邻搜索。传统推荐召回中，通常先把用户或查询表示成向量：

$$
\mathbf{q}_u \in \mathbb{R}^d
$$

再把每个物品表示成向量：

$$
\mathbf{v}_i \in \mathbb{R}^d
$$

然后在大规模物品库中寻找与用户向量最相近的候选物品。如果物品库很大，精确计算所有物品的相似度代价极高，因此工业系统通常使用 ANN 加速搜索。

ANN 的本质是：**牺牲少量精确性，换取大规模低延迟检索能力**。

---

## 0.2 MIPS

**MIPS** 是 **Maximum Inner Product Search**，即最大内积搜索。其目标是：

$$
\operatorname{TopK}_{i}\ \mathbf{q}_u^\top \mathbf{v}_i
$$

其中：

- $\mathbf{q}_u$ 是用户或 query embedding；
- $\mathbf{v}_i$ 是候选 item embedding；
- 内积 $\mathbf{q}_u^\top \mathbf{v}_i$ 被用作匹配分数。

MIPS 与 ANN 的关系可以理解为：

```text
MIPS 定义检索目标：找最大内积的 item
ANN 提供加速方法：近似地快速找 top-K
```

TIGER 试图绕开 MIPS/ANN 这一传统范式，直接让模型生成目标物品的 Semantic ID。

---

## 0.3 DLRM 与 LEM

**DLRM** 是 Deep Learning Recommendation Model 的泛称，通常包含大量稀疏特征、稠密特征、交叉特征、embedding table、特征交互模块和多任务输出头。

**LEM** 是 PLUM 文中使用的概念，即 Large Embedding Model。这类模型的绝大多数参数在 embedding table 中，而不是在深层神经网络中。其典型结构是：

```text
巨大 embedding tables + 相对较小的神经网络
```

PLUM 的目标之一是将推荐系统的 scaling 方式从：

$$
\text{scale embedding tables}
$$

转向：

$$
\text{scale neural networks over compact tokens}
$$

---

## 0.4 Semantic ID

**Semantic ID, SID** 是将一个 item 表示为若干离散 codeword token 的序列。例如：

$$
\operatorname{SID}(i)=(c_{i,1},c_{i,2},c_{i,3},c_{i,4})
$$

它与随机 item ID 的区别是：SID 试图让相似 item 共享部分 token，特别是共享前缀。这样，模型生成 token 时不只是记忆一个原子 ID，而是在生成具有语义结构的离散代码。

---

## 0.5 VQ-VAE 与 RQ-VAE

**VQ-VAE** 的核心思想是把连续 latent vector 映射到最近的 codebook vector：

$$
k^*=\arg\min_k \|z_e(x)-e_k\|_2
$$

$$
z_q(x)=e_{k^*}
$$

其训练损失通常包括：

1. 重建损失：

$$
\mathcal{L}_{recon}=\|x-\hat{x}\|^2
$$

2. codebook loss：

$$
\|\operatorname{sg}[z_e(x)]-e_{k^*}\|^2
$$

3. commitment loss：

$$
\beta\|z_e(x)-\operatorname{sg}[e_{k^*}]\|^2
$$

**RQ-VAE** 是 Residual-Quantized VAE。它不是一次性用一个 codeword 近似整个 latent，而是逐层拟合残差：

$$
r_0=z
$$

$$
c_1=\arg\min_k\|r_0-e_k^{(1)}\|^2,
\quad r_1=r_0-e_{c_1}^{(1)}
$$

$$
c_2=\arg\min_k\|r_1-e_k^{(2)}\|^2,
\quad r_2=r_1-e_{c_2}^{(2)}
$$

最终得到：

$$
z\approx e_{c_1}^{(1)}+e_{c_2}^{(2)}+\cdots+e_{c_L}^{(L)}
$$

这正适合构造 Semantic ID：

$$
(c_1,c_2,\ldots,c_L)
$$

---

## 0.6 Generative Retrieval

传统 retrieval 是：

$$
H_u \rightarrow \mathbf{q}_u \rightarrow \operatorname{ANN/MIPS} \rightarrow \{i_1,\ldots,i_K\}
$$

生成式 retrieval 是：

$$
H_u \rightarrow \text{autoregressive model} \rightarrow \operatorname{SID}(i) \rightarrow i
$$

即模型直接生成候选 item 的离散标识符。

---

## 0.7 CPT、SFT 与 DPO

- **CPT**：Continued Pre-training，继续预训练。PLUM 使用 CPT 让 LLM 学会 SID、视频元数据和用户行为序列之间的关系。
- **SFT**：Supervised Fine-Tuning，有监督微调。PLUM 用用户真实点击视频的 SID 作为目标进行 generative retrieval 微调。
- **DPO**：Direct Preference Optimization。OneRec 使用 DPO，根据 Reward Model 选择的 chosen/rejected session 对生成模型进行偏好对齐。

---

# 1. TIGER：Recommender Systems with Generative Retrieval

## 1.1 论文定位

TIGER 是生成式推荐中非常关键的一篇论文。它的核心问题是：**能否不再通过 ANN/MIPS 做向量检索，而是让模型直接生成目标 item 的 Semantic ID？**

传统序列推荐通常是：

$$
(i_1,i_2,\ldots,i_n) \rightarrow \mathbf{q}_u
$$

然后在 item embedding 库中做 MIPS：

$$
\operatorname{TopK}_{i}\ \mathbf{q}_u^\top \mathbf{v}_i
$$

TIGER 则改为：

$$
\operatorname{SID}(i_1),\operatorname{SID}(i_2),\ldots,\operatorname{SID}(i_n)
\rightarrow
\operatorname{SID}(i_{n+1})
$$

最后通过 lookup table 将生成的 SID 映射回 item。

因此，TIGER 的本质是把推荐召回从：

```text
向量相似度搜索问题
```

改写成：

```text
离散语义 ID 的自回归生成问题
```

---

## 1.2 整体框架

TIGER 包含两个主要阶段：

### 阶段一：Semantic ID generation

物品内容特征，例如 title、brand、category、price 等，先经过 Sentence-T5 或 BERT 得到连续语义 embedding：

$$
x_i \in \mathbb{R}^d
$$

然后用 RQ-VAE 量化成离散代码：

$$
\operatorname{SID}(i)=(c_{i,1},c_{i,2},c_{i,3})
$$

为了避免多个 item 映射到同一个 SID，TIGER 会额外添加一个 collision token，最终 SID 长度为 4：

$$
(c_{i,1},c_{i,2},c_{i,3},c_{i,4})
$$

### 阶段二：Generative recommendation

用户历史 item 序列被替换成 SID token 序列，输入到 Transformer encoder-decoder 中。模型的输出是下一 item 的 SID。

整体流程为：

```text
Item metadata
    ↓
Pre-trained text encoder
    ↓
Dense semantic embedding
    ↓
RQ-VAE quantization
    ↓
Semantic ID
    ↓
Seq2Seq Transformer
    ↓
Next item Semantic ID
    ↓
Item lookup
```

---

## 1.3 Semantic ID 的作用

Semantic ID 的最大价值是让相似物品共享 token。例如：

$$
(10,21,35)
$$

应该比：

$$
(10,23,32)
$$

更接近：

$$
(10,21,40)
$$

因为前者与目标共享更多前缀 token。

这相当于把 item ID 从原子符号变成了有层级语义结构的 token tuple。这样模型学习一个 codeword 时，实际上是在学习一组相似物品的共享统计规律。

---

## 1.4 RQ-VAE 在 TIGER 中的细节

TIGER 使用 RQ-VAE 对 item content embedding 进行残差量化。设 encoder 输出：

$$
z=E(x)
$$

初始残差：

$$
r_0=z
$$

第 $d$ 层 codebook 为：

$$
C_d=\{e_k^{(d)}\}_{k=1}^{K}
$$

第 $d$ 层选择最近 codeword：

$$
c_d=\arg\min_k\|r_d-e_k^{(d)}\|^2
$$

更新残差：

$$
r_{d+1}=r_d-e_{c_d}^{(d)}
$$

最终 SID 为：

$$
(c_0,c_1,\ldots,c_{m-1})
$$

TIGER 的具体实现中：

- Sentence-T5 产生 768 维 item embedding；
- RQ-VAE encoder 中间层大小为 512、256、128；
- latent dimension 为 32；
- 使用 3 层 residual quantization；
- 每层 codebook size 为 256；
- 额外第 4 个 token 用于 collision handling。

---

## 1.5 训练目标

给定用户历史：

$$
(i_1,i_2,\ldots,i_n)
$$

每个 item 替换为 SID：

$$
\operatorname{SID}(i_t)=(c_{t,1},c_{t,2},c_{t,3},c_{t,4})
$$

输入序列为所有历史 item 的 SID token 拼接：

$$
(c_{1,1},\ldots,c_{1,4},c_{2,1},\ldots,c_{n,4})
$$

模型输出下一 item 的 SID：

$$
(c_{n+1,1},c_{n+1,2},c_{n+1,3},c_{n+1,4})
$$

训练目标是 next-token prediction，即最大化下一 item SID token 的条件概率。

---

## 1.6 user ID token 是否有语义？

TIGER 输入中还加入 user ID token。需要注意：**user ID 本身没有自然语义信息**。它不像 item title 或 category 那样直接包含内容语义。

但 user ID token 仍有意义，因为模型可以在训练中学习一个 user-specific embedding，用来捕获长期个性化偏好。

形式上，没有 user ID 时：

$$
(i_1,\ldots,i_n)\rightarrow i_{n+1}
$$

加入 user ID 后：

$$
(u,i_1,\ldots,i_n)\rightarrow i_{n+1}
$$

这能区分两个短期行为相似但长期偏好不同的用户。

不过，TIGER 的消融结果显示 user ID 的提升较小。Beauty 数据集上：

| 设置 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| No user information | 0.04458 | 0.0302 | 0.06479 | 0.0367 |
| With user ID | 0.0454 | 0.0321 | 0.0648 | 0.0384 |

这说明 TIGER 的主要收益不是来自 user ID，而是来自 Semantic ID 和生成式检索框架。

对于新用户，user ID embedding 没有训练历史，因而不能直接提供个性化知识。TIGER 对 **item cold-start** 更友好，而不是天然解决 **user cold-start**。

---

## 1.7 推理与 invalid ID 问题

TIGER 推理时使用 beam search 生成 SID。由于 SID 组合空间很大，模型可能生成不存在于 item corpus 中的 SID。

若每层 codebook size 为 256、SID 长度为 4，则理论组合空间为：

$$
256^4 = 4,294,967,296
$$

约为 42.9 亿，而实验数据集只有约 1–2 万 item。因此，有效 SID 只占极小比例。

TIGER 观察到 top-10 预测中 invalid ID 比例约为 0.1% 到 1.6%；top-20 中约为 0.3% 到 6%。常见处理方式是增大 beam size 并过滤无效 SID。

这是生成式检索相对传统 ANN/MIPS 的一个工程风险：**ANN 返回的一定是真实 item；生成模型可能生成不存在的 ID。**

---

## 1.8 实验结果

TIGER 在 Amazon Review 的三个数据集上评估：Beauty、Sports and Outdoors、Toys and Games。主要指标为 Recall@K 和 NDCG@K。

TIGER 相比 GRU4Rec、Caser、HGN、SASRec、BERT4Rec、FDSA、S³-Rec、P5 等基线整体更优。

关键结果如下：

| Dataset | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Sports and Outdoors | 0.0264 | 0.0181 | 0.0400 | 0.0225 |
| Beauty | 0.0454 | 0.0321 | 0.0648 | 0.0384 |
| Toys and Games | 0.0521 | 0.0371 | 0.0712 | 0.0432 |

消融实验显示：

| ID 类型 | 结论 |
|---|---|
| Random ID | 性能最差，说明仅仅生成离散 ID 不够 |
| LSH SID | 有一定语义，但弱于 RQ-VAE |
| RQ-VAE SID | 表现最好，说明可学习语义量化很关键 |

Beauty 数据集上：

| ID 类型 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Random ID | 0.0296 | 0.0205 | 0.0434 | 0.0250 |
| LSH SID | 0.0379 | 0.0259 | 0.0533 | 0.0309 |
| RQ-VAE SID | 0.0454 | 0.0321 | 0.0648 | 0.0384 |

---

## 1.9 冷启动与多样性

### item cold-start

TIGER 通过 item content 生成 SID，因此新 item 只要有内容特征，就能被映射到已有 SID 空间中。论文用 Beauty 数据集移除 5% 测试物品模拟 unseen items，并发现 TIGER 在一定设置下优于 Semantic KNN baseline。

关键原因是：

$$
\text{new item content} \rightarrow \text{SID prefix shared with seen items}
$$

即使新 item 没有交互历史，也可以通过语义前缀进入候选空间。

### 多样性控制

TIGER 可通过 decoding temperature 控制生成结果多样性。Beauty 数据集上：

| Temperature | Entropy@10 | Entropy@20 | Entropy@50 |
|---|---:|---:|---:|
| T=1.0 | 0.76 | 1.14 | 1.70 |
| T=1.5 | 1.14 | 1.52 | 2.06 |
| T=2.0 | 1.38 | 1.76 | 2.28 |

这说明生成式推荐可以通过 decoding 策略调节推荐分布。

---

## 1.10 TIGER 的局限

1. **数据规模较小**：实验数据集 item 数量约 1–2 万，远小于真实工业库。
2. **推理成本较高**：beam search 自回归生成通常比 ANN/MIPS 更贵。
3. **invalid ID 问题**：可能生成无效 SID。
4. **SID 依赖内容质量**：若 item metadata 噪声大或内容贫乏，SID 质量受影响。
5. **user cold-start 未根本解决**：user ID embedding 对新用户没有直接作用。

---

## 1.11 TIGER 的核心结论

TIGER 的贡献可以概括为：

```text
用 RQ-VAE 将 item 内容压缩成 Semantic ID；
用 Transformer 自回归生成下一 item 的 Semantic ID；
用生成式检索替代传统 ANN/MIPS 召回。
```

它奠定了后续生成式推荐工作的基本范式。

---

# 2. Actions Speak Louder than Words：Trillion-Parameter Sequential Transducers

## 2.1 论文定位

这篇论文来自 Meta，关注的是工业级推荐系统如何从传统 DLRM 转向 **Generative Recommenders, GRs**。其标题 “Actions Speak Louder than Words” 强调：在大规模推荐中，用户行为序列本身是一种强大的建模模态，往往比文本内容更重要。

它与 TIGER 的关系是：

- TIGER 关注如何生成 item Semantic ID 做 retrieval；
- 这篇论文关注如何把 retrieval 和 ranking 都重写为 sequential transduction tasks，并设计适合工业场景的 HSTU 架构。

它的核心目标不是简单提出一个推荐模型，而是重构工业推荐系统的基础建模范式。

---

## 2.2 从 DLRM 到 Generative Recommenders

传统 DLRM 使用大量 heterogeneous features：

- categorical features：item ID、creator ID、user ID、language、city 等；
- numerical features：CTR、watch ratio、decayed counters 等；
- cross features 和人工统计特征；
- feature interaction modules：FM、DCN、DIN、MMoE、PLE 等。

GR 的想法是把这些复杂特征统一成一个时间序列。

对于 categorical features：

```text
慢变化类别特征压缩后并入主行为序列
```

对于 numerical features：

```text
不直接保留所有统计量，而让模型从原始行为序列和 target-aware attention 中学习这些聚合规律
```

所以 GR 的核心转变是：

$$
\text{人工聚合特征与交叉特征}
\quad\rightarrow\quad
\text{模型从长行为序列中自动学习}
$$

---

## 2.3 Sequential transduction formulation

论文将推荐任务表述为 sequential transduction。给定输入 token 序列：

$$
x_0,x_1,\ldots,x_{n-1}
$$

输出：

$$
y_0,y_1,\ldots,y_{n-1}
$$

其中 $y_i$ 可以为空 $\varnothing$。

设 $\Phi_i$ 表示系统展示给用户的内容，例如视频、图片、帖子；$a_i$ 表示用户对该内容的动作，例如 like、skip、completion、share。

---

## 2.4 Retrieval 的 formulation

Retrieval 的目标是：

$$
p(\Phi_{i+1}\mid u_i)
$$

即根据当前用户状态 $u_i$ 预测下一个值得推荐的内容。

论文将 retrieval 写成：

$$
x_i=(\Phi_i,a_i)
$$

$$
y_i=\Phi_i'
$$

其中：

$$
\Phi_i'=
\begin{cases}
\Phi_i, & a_i \text{ is positive}\\
\varnothing, & \text{otherwise}
\end{cases}
$$

也就是说，负反馈 item 仍然进入历史序列，但通常不作为生成目标。

这与传统只保留正反馈 item 的 SASRec/GRU4Rec 不同。GR 显式建模：

$$
(\Phi_0,a_0),(\Phi_1,a_1),\ldots
$$

而不是只建模：

$$
\Phi_0,\Phi_1,\ldots
$$

---

## 2.5 Ranking 的 formulation

Ranking 阶段给定候选 $\Phi_i$，预测用户动作 $a_i$：

$$
p(a_i\mid H_u,\Phi_i)
$$

论文通过 interleaving content and action，让模型在 causal autoregressive 结构中仍然具有 target-aware 能力：

$$
\Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_i
\rightarrow a_i
$$

因此，ranking 与 retrieval 可被统一写成序列转导任务，但二者条件概率不同：

| 任务 | 条件概率 | 含义 |
|---|---|---|
| Retrieval | $p(\Phi\mid H_u)$ | 找候选内容 |
| Ranking | $p(a\mid H_u,\Phi)$ | 给定候选预测用户动作 |

这也回答了一个关键问题：**ranking 阶段的 $\Phi_i$ 不是凭空来的，它仍然来自 retrieval candidate source。**

---

## 2.6 Generative training

传统工业推荐训练常是 impression-level：每个候选 item 都构造一条训练样本，因此同一用户历史会被重复编码多次。

GR 改为 generative training：一次输入较长用户序列，在多个位置同时产生监督信号。这样 encoder 计算可以被多个目标共享。

论文给出的复杂度直观变化为：

$$
O(N^3d+N^2d^2)
\quad\rightarrow\quad
O(N^2d+Nd^2)
$$

即减少一个 $O(N)$ 因子。

---

## 2.7 HSTU 架构

论文提出 **HSTU, Hierarchical Sequential Transduction Unit**，用于替代标准 Transformer 以及 DLRM 中复杂的特征提取、特征交互和表示变换模块。

HSTU 每层包含三步。

### 1. Pointwise projection

$$
U(X),V(X),Q(X),K(X)=\operatorname{Split}(\phi_1(f_1(X)))
$$

### 2. Spatial aggregation

$$
A(X)V(X)=\phi_2(Q(X)K(X)^T+r^{p,t}_{ab})V(X)
$$

### 3. Pointwise transformation

$$
Y(X)=f_2(\operatorname{Norm}(A(X)V(X))\odot U(X))
$$

其中 $U(X)$ 是 gate，$r^{p,t}_{ab}$ 是包含位置和时间信息的相对注意力偏置。

HSTU 的设计意图是用一个统一模块替代 DLRM 的：

```text
Feature extraction
Feature interaction
Representation transformation
```

---

## 2.8 Pointwise aggregated attention

HSTU 不使用标准 softmax attention：

$$
\operatorname{softmax}(QK^T)V
$$

而采用 pointwise aggregated attention。原因是推荐中用户行为的数量本身就是偏好强度信号。

例如：

```text
用户 A 看过某类内容 2 次
用户 B 看过某类内容 200 次
```

softmax attention 更关注相对分布，可能削弱绝对强度信息；pointwise attention 更能累积相关证据。

合成实验中：

| 架构 | HR@10 | HR@50 |
|---|---:|---:|
| Transformers | 0.0442 | 0.2025 |
| HSTU with softmax | 0.0617 | 0.2496 |
| HSTU pointwise | 0.0893 | 0.3170 |

这支持了 pointwise aggregated attention 在非平稳推荐词表中的优势。

---

## 2.9 Stochastic Length

推荐历史序列很长，full self-attention 成本为：

$$
O(n^2)
$$

HSTU 使用 **Stochastic Length, SL** 降低训练成本。其思想是：长历史用户在训练时不总是使用完整序列，而是随机选择子序列。

背后的假设是：用户长期行为具有冗余性和多尺度重复性，不必每次都看完整历史。

实验显示，在 4096 或 8192 序列长度下，即使大幅增加 sparsity，主要任务的 NE 退化很小。

---

## 2.10 M-FALCON

ranking 阶段通常需要对大量候选 item 打分。朴素 target-aware 实现需要对每个候选单独跑一遍：

$$
O(mn^2d)
$$

其中 $m$ 是候选数，$n$ 是历史长度。

M-FALCON 的思想是：

1. 将多个候选同时拼到用户历史后面；
2. 修改 attention mask，使候选之间不能互相 attend；
3. 一次 forward pass 得到多个候选的预测；
4. 使用 microbatch 和 KV cache 进一步摊薄计算。

复杂度近似从：

$$
O(mn^2d)
$$

变为：

$$
O((n+b_m)^2d)
$$

cached forward pass 可进一步近似为：

$$
O(b_md^2+b_mnd)
$$

M-FALCON 使 target-aware GR ranking 在工业场景中具备可服务性。

---

## 2.11 实验结果

### 公开数据集

HSTU 在 MovieLens 和 Amazon Books 上优于 SASRec。Amazon Books 上，HSTU-large 相对 SASRec 的 NDCG@10 提升达 65.8%。

### 工业 retrieval

| 方法 | HR@100 | HR@500 | Online E-Task | Online C-Task |
|---|---:|---:|---:|---:|
| DLRM | 29.0% | 55.5% | 0% | 0% |
| GR interactions only | 35.6% | 61.7% | — | — |
| GR new source | 36.9% | 62.4% | +6.2% | +5.0% |

### 工业 ranking

| 方法 | E-Task NE | C-Task NE | Online E-Task | Online C-Task |
|---|---:|---:|---:|---:|
| DLRM | 0.4982 | 0.7842 | 0% | 0% |
| GR | 0.4845 | 0.7645 | +12.4% | +4.4% |

### scaling law

论文观察到 GR 指标随 training compute 呈现近似 power-law 变化：

$$
HR@100 = 0.15 + 0.0195\ln C
$$

$$
HR@500 = 0.395 + 0.0212\ln C
$$

$$
NE = 0.549 - 5.3\times 10^{-3}\ln C
$$

这表明 GR 相比传统 DLRM 有更强的 compute scaling 潜力。

---

## 2.12 局限

1. 工业结果难以完全复现；
2. GR 仍需候选源支持 ranking；
3. HSTU/M-FALCON 工程复杂度高；
4. pointwise attention 的优势可能依赖具体推荐场景；
5. scaling law 需要更多跨平台证据。

---

## 2.13 核心结论

这篇论文的核心贡献是：

```text
把工业推荐系统中的 retrieval 和 ranking 统一表述为 sequential transduction；
提出 HSTU 处理高基数、非平稳、长序列推荐数据；
通过 Stochastic Length 和 M-FALCON 解决训练与推理效率问题；
展示工业级 scaling law 与线上收益。
```

---

# 3. OneRec：Unifying Retrieve and Rank with Generative Recommender and Preference Alignment

## 3.1 论文定位

OneRec 是更激进的生成式推荐路线。它的目标是：**不再把生成式模型只作为 retrieval source，而是用一个端到端生成模型直接生成最终推荐列表，从而统一 retrieve 与 rank。**

传统 cascade ranking 系统是：

```text
Video corpus ~10^10
    ↓ retrieval
Candidates ~10^5
    ↓ coarse ranking
Candidates ~10^3
    ↓ fine ranking
Final list ~10^2
```

OneRec 试图改成：

```text
User history
    ↓
Encoder-Decoder generative model
    ↓
Recommended session list
```

---

## 3.2 为什么要统一 retrieve 和 rank？

传统多阶段系统中，每一阶段只负责从接收到的 candidates 中选 top-k。如果 retrieval 阶段漏掉了高质量 item，后续 ranker 再强也无法补救。

OneRec 认为 cascade pipeline 的问题是：

1. 各阶段独立优化，目标不完全一致；
2. 上游阶段限制下游性能上限；
3. 需要大量工程规则组合不同阶段输出；
4. point-wise 生成或召回难以保证最终列表的整体连贯性与多样性。

因此，它提出 single-stage generative recommendation。

---

## 3.3 输入与输出

用户侧输入是正向历史行为序列：

$$
H_u=\{v_1^h,v_2^h,\ldots,v_n^h\}
$$

其中 $v_i^h$ 是用户有效观看或互动的视频，例如 likes、follows、shares。

输出不是单个 item，而是一个 session：

$$
S=\{v_1,v_2,\ldots,v_m\}
$$

论文中 session 通常包含 5 到 10 个短视频，实验中设置：

$$
n=256,\quad m=5
$$

这与 TIGER 的根本区别是：

| 模型 | 输出 |
|---|---|
| TIGER | next item SID |
| OneRec | session-wise item list |

---

## 3.4 Balanced Identifier Quantization

OneRec 也需要将视频变成可生成 token。它没有直接使用 TIGER 的 RQ-VAE，而使用 **Residual K-Means Quantization**，并引入 **Balanced K-means**。

每个视频有多模态 embedding：

$$
e_i\in\mathbb{R}^d
$$

残差量化过程为：

$$
r_i^1=e_i
$$

第 $l$ 层选择最近中心：

$$
s_i^l=\arg\min_k \|r_i^l-c_k^l\|_2^2
$$

残差更新：

$$
r_i^{l+1}=r_i^l-c_{s_i^l}^l
$$

最终视频 identifier 为：

$$
(s_i^1,s_i^2,\ldots,s_i^L)
$$

实验中：

$$
K=8192,\quad L=3
$$

因此一个视频可以表示为：

```text
<a_9><b_3><c_1>
```

---

## 3.5 Balanced K-means 的意义

普通 K-means 可能出现 code 分布极不均衡，即少数 cluster 很大、很多 cluster 很小。生成式模型会因此偏向少数高频 code。

Balanced K-means 要求每个 cluster 包含相同数量的视频：

$$
w=\frac{|V|}{K}
$$

算法核心是：每个 centroid 从未分配样本中选择最近的 $w$ 个视频，然后更新中心。

其目标是避免 hourglass phenomenon，提高 code 使用均衡性。

---

## 3.6 Session-wise list generation

OneRec 最关键的创新是 **session-wise generation**。

传统 point-wise generation 是：

$$
H_u\rightarrow v_{t+1}
$$

若要生成列表，需要多次生成或用规则组合。

OneRec 直接学习：

$$
H_u\rightarrow S=\{v_1,\ldots,v_m\}
$$

这使模型可以学习列表内部结构，例如：

- 相邻视频的连贯性；
- 列表内多样性；
- 推荐顺序；
- 用户在一个 session 内的连续观看行为。

论文通过筛选 high-quality sessions 构造训练数据。标准包括：

1. session 中用户实际观看视频数量不少于 5；
2. 总观看时长超过阈值；
3. 用户有点赞、收藏、分享等互动行为。

因此 OneRec 本质上是在做：

$$
\text{imitate high-value recommendation sessions}
$$

---

## 3.7 模型结构

OneRec 使用 T5-style encoder-decoder。

Encoder 输入用户历史行为序列：

$$
H=Encoder(H_u)
$$

Decoder 以自回归方式生成目标 session 的 semantic ID tokens。

为了扩大容量，OneRec 在 decoder FFN 中使用 Sparse Mixture-of-Experts：

$$
H_{t}^{l+1}=\sum_{i=1}^{N_{MoE}}g_{i,t}FFN_i(H_t^l)+H_t^l
$$

其中只有 top-$K_{MoE}$ 个 expert 被激活。

实验设置：

$$
N_{MoE}=24,\quad K_{MoE}=2
$$

在线推理时，只有约 13% 参数被激活。

---

## 3.8 Next-token prediction loss

训练时，在每个 item SID 前加入 BOS token：

$$
\bar{S}=\{s_{BOS},s_1^1,\ldots,s_1^L,s_{BOS},s_2^1,\ldots,s_m^L\}
$$

训练目标是 next-token prediction：

$$
\mathcal{L}_{NTP}
= -\sum_{i=1}^{m}\sum_{j=1}^{L}\log P(s_i^{j+1}\mid \text{previous tokens};\Theta)
$$

这与语言模型训练形式一致，只是 token 是视频 identifier。

---

## 3.9 Reward Model

OneRec 进一步引入 Reward Model：

$$
R(u,S)
$$

它给定用户 $u$ 和候选 session $S$，输出 session reward。

RM 训练 ground truth 来自真实线上日志中的 session 反馈标签，例如：

- swt：session watch time；
- vtr：view probability；
- wtr：follow probability；
- ltr：like probability。

每个 item 先得到 target-aware representation：

$$
e_i=v_i\odot u
$$

session 内 item 通过 self-attention 交互：

$$
h_f=SelfAttention(hW_s^Q,hW_s^K,hW_s^V)
$$

然后多个 tower 预测不同 reward：

$$
\hat{r}_{swt},\hat{r}_{vtr},\hat{r}_{wtr},\hat{r}_{ltr}
$$

训练损失为多目标 BCE。

---

## 3.10 IPA：Iterative Preference Alignment

OneRec 将 DPO 引入推荐列表生成。但推荐系统不同于 NLP：一个用户请求通常只能展示一个结果，不能同时得到真实 chosen/rejected 对。

OneRec 的做法是：用当前模型 $M_t$ 通过 beam search 生成 $N$ 个 session：

$$
S_u^n\sim M_t(H_u),\quad n=1,\ldots,N
$$

实验中：

$$
N=128
$$

Reward Model 打分：

$$
r_u^n=R(u,S_u^n)
$$

最高分作为 chosen：

$$
S_u^w=S_u^{\arg\max_n r_u^n}
$$

最低分作为 rejected：

$$
S_u^l=S_u^{\arg\min_n r_u^n}
$$

DPO 损失为：

$$
\mathcal{L}_{DPO}
=-\log\sigma\left(
\beta\log\frac{M_{t+1}(S_u^w\mid H_u)}{M_t(S_u^w\mid H_u)}
-
\beta\log\frac{M_{t+1}(S_u^l\mid H_u)}{M_t(S_u^l\mid H_u)}
\right)
$$

总损失：

$$
\mathcal{L}=\mathcal{L}_{NTP}+\lambda\mathcal{L}_{DPO}
$$

为了降低成本，仅约 1% 数据进入 DPO alignment。

---

## 3.11 工业部署

OneRec 部署包括：

1. distributed training system；
2. online serving system；
3. DPO sample server。

推理优化包括：

- KV cache decoding；
- float16 quantization；
- beam size = 128；
- MoE 稀疏激活。

---

## 3.12 实验结果

### Offline

OneRec 与 SASRec、BERT4Rec、FDSA、TIGER 以及多种 DPO 变体比较。

关键结果：

| 模型 | max swt | max vtr | max wtr | max ltr |
|---|---:|---:|---:|---:|
| TIGER-1B | 0.1368 | 0.6776 | 0.00758 | 0.0579 |
| OneRec-1B | 0.1529 | 0.7013 | 0.00919 | 0.0660 |
| OneRec-1B+IPA | 0.1933 | 0.7646 | 0.00992 | 0.1203 |

论文强调：

- OneRec-1B 相比 TIGER-1B，在 maximum swt 上提升 1.78%，maximum ltr 上提升 3.36%；
- OneRec-1B+IPA 相比 OneRec-1B，在 maximum swt 上提升 4.04%，maximum ltr 上提升 5.43%。

### Online A/B

| 模型 | Total Watch Time | Average View Duration |
|---|---:|---:|
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| OneRec-1B+IPA | +1.68% | +6.56% |

OneRec 在快手主场景线上获得明显观看时长提升。

---

## 3.13 局限

1. **RM 偏差问题**：IPA 的 preference pair 来自 Reward Model，而非真实同时对比反馈。
2. **离线评估闭环**：RM 既用于构造偏好，也用于部分离线评估，可能形成偏差闭环。
3. **多目标难题**：论文承认 likes 等互动指标仍有局限。
4. **可控性问题**：端到端生成列表后，业务规则、安全、多样性和探索约束如何稳定注入仍是工程难点。
5. **标题中的“unifying retrieve and rank”有一定宣传性**：它更像绕过传统 rank，而不是显式建模 $p(a\mid H,\Phi)$。

---

## 3.14 核心结论

OneRec 的核心贡献是：

```text
从 point-wise item generation 推进到 session-wise list generation；
用一个端到端生成模型直接输出推荐 session；
用 Reward Model + IPA/DPO 进行偏好对齐；
在真实短视频主场景中获得线上收益。
```

---

# 4. PLUM：Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations

## 4.1 论文定位

PLUM 的核心问题是：**如何将预训练大语言模型适配到工业级生成式推荐，尤其是 YouTube 级别的视频召回？**

与 TIGER 和 OneRec 从推荐数据训练生成模型不同，PLUM 从预训练 LLM 出发，例如 Gemini-1.5 MoE 系列模型，通过 SID、continued pre-training 和 retrieval SFT 进行适配。

PLUM 的框架是：

```text
Pre-trained LLM
    ↓
Expand vocabulary with SID tokens
    ↓
Continued pre-training on user behavior + video metadata
    ↓
Supervised fine-tuning for generative retrieval
    ↓
Beam search generates next-video SIDs
```

---

## 4.2 为什么要做 PLUM？

传统 YouTube 生产召回模型属于 Large Embedding Models。它们大部分参数在巨大 embedding table 中，而 neural network 本体较小。

PLUM 的目标是改变这一点：

| 模型 | 参数主要位置 |
|---|---|
| LEM | 大量 item/user embedding tables |
| PLUM | 大规模神经网络主体 |

论文报告，LEM 的神经网络只占总参数的 0.4%，而 PLUM 的神经网络占 90%。这说明 PLUM 试图把推荐系统 scaling 方式从 embedding table scaling 转向 neural network scaling。

---

## 4.3 PLUM 的三阶段

PLUM 包含：

1. **Item tokenization with SID-v2**；
2. **Continued pre-training, CPT**；
3. **Task-specific fine-tuning for generative retrieval**。

---

## 4.4 SID-v2：多模态与行为信号增强的 Semantic ID

PLUM 认为 TIGER 的 SID 存在三个限制：

1. 依赖单一内容 embedding；
2. 缺少用户行为协同信号；
3. 层级结构与 SID 空间利用率仍可改进。

因此提出 SID-v2。

---

## 4.5 Fused Multi-Modal Content Representation

每个视频可能有多个 embedding：

$$
\{x_m\}_{m=1}^{M}
$$

每个模态通过独立 encoder：

$$
z_m=E_m(x_m)
$$

拼接后投影：

$$
\tilde{z}=[z_1,\ldots,z_M]
$$

$$
z=Project(\tilde{z})
$$

然后将 $z$ 输入 residual quantizer。

这使视频 SID 同时吸收文本、视觉、音频等信息。

---

## 4.6 Multi-resolution codebooks

TIGER 使用每层相同大小 codebook。PLUM 改为多分辨率 codebook：

$$
K_l=\frac{2048}{2^{l-1}}
$$

前几层分辨率高，更具区分性；后几层分辨率低，编码低熵 residual。

这有助于提高 SID 空间利用率，避免巨大稀疏组合空间。

---

## 4.7 Progressive masking

PLUM 用 progressive masking 强化层级结构。随机选择前若干层 codebook 参与训练：

$$
m_l=1_{l<r},\quad r\in[1,L]
$$

最终量化向量为：

$$
\hat{z}=\sum_{l=1}^{L}m_le_l^*
$$

这迫使 SID 前缀本身具有语义意义，而不是只有完整 SID 才能重建 item。

---

## 4.8 Co-occurrence contrastive regularization

PLUM 认为用户行为共现是推荐相似性的关键来源。两个视频即使内容不同，如果经常被用户连续观看，也应在推荐意义上接近。

因此，SID-v2 加入 co-occurrence contrastive loss：

$$
\mathcal{L}_{con}
= -\sum_i \log
\frac{\exp(sim(p_i,p_i^+))}
{\sum_j\exp(sim(p_i,p_j))}
$$

其中 $p_i^+$ 是与视频 $i$ 共现的视频表示。

这使 SID 同时反映：

```text
content similarity + behavioral co-occurrence similarity
```

---

## 4.9 SID-v2 总损失

PLUM 的 SID 训练损失为：

$$
\mathcal{L}=\mathcal{L}_{recon}+\mathcal{L}_{rq}+\mathcal{L}_{con}
$$

其中：

$$
\mathcal{L}_{recon}=\sum_{m=1}^{M}\|x_m-\hat{x}_m\|^2
$$

$\mathcal{L}_{rq}$ 是 RQ-VAE 量化损失，$\mathcal{L}_{con}$ 是共现对比损失。

---

## 4.10 Continued Pre-training

CPT 的目标是让 LLM 理解 SID 这种新模态，并将 SID 与自然语言、视频元数据和用户行为对齐。

CPT 数据包含两部分。

### 1. User behavior data

示例 schema：

```text
wh = <sid_1> <channel_name> <watch_ratio> <watch_time>
     <hours_since_final_watch> <sid_2> ... || <sid_n>
```

### 2. Video metadata corpus

示例：

```text
Video <sid> has title (en): <video_title>
The topics in video <sid> are: <topics>
```

还包括 description、ASR captions、channel name、synthetic data 等。

CPT 数据比例：

```text
50% user behavior data
50% video metadata corpus
```

训练规模：

```text
1 million steps
batch size = 16
approximately 260B tokens
```

---

## 4.11 CPT 的意义

CPT 同时完成两类对齐：

1. **SID-text alignment**：让模型知道 SID 对应什么视频语义；
2. **SID-behavior alignment**：让模型知道用户历史 SID 序列如何预测后续观看。

Appendix 中的 in-context learning 示例显示，LLM-initialized CPT 模型能根据 SID few-shot prompt 输出合理自然语言主题，例如 “psychology and mind”；随机初始化模型则输出不连贯字符串。这说明 CPT 后 SID 被纳入 LLM 的语义空间。

---

## 4.12 Generative Retrieval SFT

SFT 阶段专门优化 retrieval。输入 prompt 包含：

```text
watch history | user features | context video features
```

watch history 是按时间排序的视频观看序列，每个观看由 SID tokens 和其他 feature tokens 构成。

目标是用户实际点击视频的 SID：

$$
[sid_1,\ldots,sid_L]
$$

训练损失为：

$$
\mathcal{L}_{SFT}
= -\sum_{t=1}^{L}r(user,v_{click})
\log P(sid_t\mid Context_{user},History_{user},sid_{<t})
$$

实际训练中，由于成本高，PLUM 根据 handcrafted reward 采样训练样本，然后对采样后的样本等权训练。

---

## 4.13 推理

推理时使用 beam search 解码多个 SID 序列：

$$
\hat{s}^{(1)},\ldots,\hat{s}^{(K)}
$$

每个 SID 映射到真实视频：

$$
\hat{s}^{(k)}\rightarrow v^{(k)}
$$

生成的候选视频进入 YouTube 的候选池。

论文报告 SFT 后 hallucination rate 低于 5%，SID-to-video 映射唯一性较高。

---

## 4.14 PLUM vs 生产 LEM

PLUM 使用 900M activated-parameter Gemini-1.5 MoE 模型，在 Long Form Video 和 Shorts 上评估。

### Recommendation quality

Table 2 报告 PLUM 相对 LEM 的比值：

| Metric | LFV | Shorts |
|---|---:|---:|
| Effective Vocab Size | 2.60× | 13.24× |
| CTR | 1.42× | 1.33× |
| WT/View | 0.72× | 1.13× |
| WF/View | 1.32× | 1.03× |

其中 Effective Vocab Size 是覆盖 95% impressions 所需 unique videos 数量。PLUM 显著扩大了推荐内容覆盖面。

需要注意：LFV 的 WT/View 为 0.72×，说明长视频单次观看时长不如 LEM，存在指标 trade-off。

### Live experiments

PLUM 作为额外 candidate source 加入系统，与 LEM+ 对照：

| Metric | LFV | Shorts |
|---|---:|---:|
| Engaged Users | +0.07% | +0.28% |
| Panel CTR | +0.76% | +4.96% |
| Views | +0.80% | +0.39% |
| Satisfaction | +0.06% | +0.39% |

这说明 PLUM 能作为新召回源为现有生产系统提供增量价值。

### Sample efficiency

PLUM 900M MoE 每天约训练 250M examples，而传统 LEM 每天训练数十亿 examples。PLUM 单样本成本更高，但整体训练 FLOPs 小于 LEM 的 0.55×。

---

## 4.15 SID-v2 消融

| SID 模型 | SID Uniqueness | VID Recall@10 |
|---|---:|---:|
| SIDv1 baseline | 94.0% | 12.3% |
| SIDv2 full | 96.7% | 14.4% |
| Ablate Multi-Resolution | 94.8% | 13.2% |
| Ablate Multi-Embedding | 96.9% | 12.8% |
| Ablate Co-occurrence | 91.8% | 12.6% |

结果说明：

- 完整 SID-v2 明显优于 SIDv1；
- co-occurrence 对 SID uniqueness 和 recall 都非常关键；
- 多模态 embedding 与多分辨率 codebook 也有独立贡献。

---

## 4.16 CPT 与 LLM 初始化消融

| 模型 | LLM init | CPT | Recall@10 |
|---|---:|---:|---:|
| R1 | No | No | 0.19 |
| R2 | Yes | No | 0.23 |
| CR1 | No | Yes | 0.27 |
| CR2 | Yes | Yes | 0.28 |

结论：

1. CPT 是最大贡献来源；
2. LLM 初始化也有稳定增益；
3. 最优组合是 LLM init + CPT + SFT；
4. PLUM 不是 zero-shot LLM 推荐，而是大规模领域适配。

---

## 4.17 Scaling study

PLUM 使用 Gemini-1.5 MoE family：

| Model | Activated params | Learning rate | Global batch size |
|---|---:|---:|---:|
| MoE-110M | 110M | $1\times10^{-4}$ | 25,600 |
| MoE-370M | 370M | $7\times10^{-5}$ | 15,360 |
| MoE-900M | 900M | $5\times10^{-5}$ | 7,680 |
| MoE-3B | 3B | $2\times10^{-5}$ | 3,584 |

实验设置：

- YouTube 生产数据；
- 候选库为 billions-scale 视频；
- 输入长度 1536 tokens；
- 训练连续 7 天数据，评估第 8 天；
- 使用 1024 Google v6e TPUs。

主要观察：

1. training/evaluation loss 随 Iso-FLOPS 增加呈 power-law-like 下降；
2. Recall@10 随 compute 增加继续提升，饱和迹象较弱；
3. 固定 compute budget 下，最优模型大小会随 compute 增加而变大；
4. MoE-3B 未超过 MoE-900M，可能因 batch size 和训练样本不足。

作者总结：compute-optimal generative retrieval 需要同步扩大：

$$
\text{model size} + \text{training examples}
$$

---

## 4.18 PLUM 的局限

1. 主要解决 retrieval，不是完整 ranking 系统；
2. 线上实验是作为候选源增量接入，而非完全替代生产系统；
3. LFV 的 WT/View 低于 LEM，存在指标 trade-off；
4. invalid SID 和 collision 仍需处理；
5. CPT 成本极高，260B tokens 级别难以复现；
6. reward-based sampling 可能继承 exposure bias 与 feedback loop。

---

## 4.19 核心结论

PLUM 的核心贡献是：

```text
将 SID 作为新模态注入预训练 LLM；
通过 SID-v2 提升 item tokenization 质量；
通过 CPT 对齐 SID、自然语言和用户行为；
通过 retrieval SFT 让 LLM 生成下一个视频 SID；
在 YouTube 生产系统中作为无大 embedding table 的新召回源提供增量价值。
```

---

# 5. 四篇论文的横向比较

## 5.1 总览表

| 维度 | TIGER | Actions Speak Louder | OneRec | PLUM |
|---|---|---|---|---|
| 核心目标 | 生成 item SID 做召回 | 将 retrieval/ranking 表述为 sequential transduction | 单阶段生成最终推荐 session | 适配预训练 LLM 做工业级生成式召回 |
| 主要任务 | next-item retrieval | retrieval + ranking | session-wise recommendation | candidate generation / retrieval |
| 输出 | 单个 item Semantic ID | content token 或 action token | 推荐视频列表 | 下一视频 SID |
| item 表示 | RQ-VAE SID | 高基数 content/action tokens | Balanced K-means SID | SID-v2 |
| 模型 | Seq2Seq Transformer | HSTU | Encoder-decoder + MoE | Gemini-1.5 MoE decoder-only LLM |
| 是否从预训练 LLM 出发 | 否 | 否 | 否 | 是 |
| 是否做列表生成 | 否 | 不作为主线 | 是 | 否 |
| 是否做偏好对齐 | 否 | 否 | IPA + DPO | 否，主要 reward-sampled SFT |
| 工业部署 | 否，公开数据实验 | Meta 工业部署 | 快手主场景 A/B | YouTube 多 surface retrieval |
| 主要创新 | Semantic ID generative retrieval | HSTU + SL + M-FALCON + scaling law | session-wise generation + IPA | SID-v2 + CPT + LLM adaptation |

---

## 5.2 范式演化

### TIGER

```text
把 retrieval 从 ANN/MIPS 改成生成 SID
```

### Actions Speak Louder

```text
把 retrieval 和 ranking 都写成用户行为序列转导
```

### OneRec

```text
不再分 retrieve/rank，直接生成推荐列表
```

### PLUM

```text
把预训练 LLM 通过 SID 和 CPT 适配成工业级生成式召回模型
```

---

## 5.3 item tokenization 对比

| 维度 | TIGER | OneRec | PLUM |
|---|---|---|---|
| 输入表示 | 文本/metadata embedding | 多模态视频 embedding | 多模态 embedding |
| 量化方法 | RQ-VAE | residual balanced K-means | RQ-VAE-style SID-v2 |
| 是否均衡 code 分布 | 部分依赖训练 | 明确 Balanced K-means | 通过 multi-resolution 改善 |
| 是否用行为信号 | 主要不用 | 间接来自训练 session | co-occurrence contrastive loss |
| 是否关注 SID-text 对齐 | 否 | 否 | 是，CPT 强化 |

---

## 5.4 retrieval 实现对比

| 论文 | Retrieval 如何实现 |
|---|---|
| TIGER | beam search 生成单个 item SID，再 lookup item |
| Actions Speak Louder | 学习 $p(\Phi\mid H_u)$，可用 MIPS/ANN、beam search 或 hierarchical retrieval；论文重点不在具体召回后端 |
| OneRec | 直接生成 session list，弱化传统 retrieval/ranking 区分 |
| PLUM | decoder-only LLM beam search 生成多个视频 SID，作为候选源加入系统 |

---

## 5.5 ranking 处理方式对比

| 论文 | Ranking 方式 |
|---|---|
| TIGER | 不涉及精排，只做召回 |
| Actions Speak Louder | 显式建模 $p(a\mid H,\Phi)$，用 M-FALCON 加速大量候选打分 |
| OneRec | 通过 session generation 和 RM/DPO 直接优化列表质量，绕过传统 ranking |
| PLUM | 本文主要做 retrieval，后续 ranking 仍由生产系统处理 |

---

## 5.6 训练目标对比

| 论文 | 训练目标 |
|---|---|
| TIGER | next item SID token prediction |
| Actions Speak Louder | sequential transduction：next content/action prediction |
| OneRec | session-wise next-token prediction + DPO preference alignment |
| PLUM | CPT next-token prediction + reward-sampled retrieval SFT |

---

## 5.7 用户行为建模对比

| 论文 | 用户行为使用方式 |
|---|---|
| TIGER | 用户历史 item SID 序列；可加入 user ID token |
| Actions Speak Louder | 用户 action 是核心模态，显式 interleave content/action |
| OneRec | 使用正向历史行为，并学习 high-value session |
| PLUM | CPT 中建模 watch history，SFT 中用 watch history + context features 预测 next watch |

Actions Speak Louder 最强调 user actions；PLUM 则把 user actions 作为 LLM 的新训练语料；OneRec 更关注高质量 session；TIGER 更关注 item SID 语义结构。

---

## 5.8 LLM 作用对比

| 论文 | LLM 角色 |
|---|---|
| TIGER | 无预训练 LLM，只是 seq2seq Transformer |
| Actions Speak Louder | 受 LLM scaling 启发，但模型是推荐专用 HSTU |
| OneRec | 使用生成式模型和 MoE，但不是通用 LLM 适配 |
| PLUM | 直接从预训练 LLM/Gemini MoE 出发，做推荐领域适配 |

---

## 5.9 工业化程度对比

| 论文 | 工业化程度 |
|---|---|
| TIGER | 公开数据集验证，方法开创性强 |
| Actions Speak Louder | Meta 工业级部署，强调 1.5T 参数和线上收益 |
| OneRec | 快手主场景 A/B，主张单阶段生成 |
| PLUM | YouTube 生产环境 retrieval source，强调无大 embedding table |

---

## 5.10 主要风险对比

| 论文 | 主要风险 |
|---|---|
| TIGER | invalid SID、beam search 成本、规模验证不足 |
| Actions Speak Louder | 工程复杂、复现困难、ranking/retrieval 仍需系统协同 |
| OneRec | RM 偏差、DPO 伪偏好、端到端可控性难 |
| PLUM | CPT 成本高、仍需后续 ranker、指标 trade-off、SID hallucination |

---

# 6. 四篇论文共同揭示的核心问题

## 6.1 生成式推荐到底生成什么？

四篇论文中的“生成”含义不同：

| 论文 | 生成对象 |
|---|---|
| TIGER | item SID |
| Actions Speak Louder | content token 或 action token |
| OneRec | 推荐 session list |
| PLUM | next video SID |

因此，不能简单把它们都归为同一种 generative recommendation。它们分别位于不同系统层级。

---

## 6.2 生成式 retrieval 是否取代 ANN/MIPS？

TIGER 和 PLUM 最直接挑战 ANN/MIPS：模型生成 SID 后 lookup item，不需要传统向量索引。

但实际工业系统中：

- Actions Speak Louder 仍承认 retrieval 可以用 MIPS、beam search 或 hierarchical retrieval；
- PLUM 作为 candidate source 加入现有系统；
- OneRec 试图直接生成最终列表，但仍需解决服务可控性。

因此目前更准确的判断是：

```text
生成式 retrieval 正在成为重要候选源，尚未普遍完全取代 ANN/MIPS 和多阶段系统。
```

---

## 6.3 item tokenization 是核心瓶颈

TIGER、OneRec、PLUM 都表明：item 如何被 tokenized 对生成式推荐性能至关重要。

好的 SID 需要满足：

1. 唯一性高；
2. 层级结构明确；
3. 与内容语义一致；
4. 与用户行为相似性一致；
5. code 分布均衡；
6. 便于 beam search；
7. 对新 item 可泛化。

PLUM 的 SID-v2 可视为 TIGER RQ-VAE SID 的工业强化版本。

---

## 6.4 行为数据比文本更关键，但文本仍有价值

Actions Speak Louder 强调：actions speak louder than words。工业推荐中，用户行为序列通常比纯内容文本更有预测力。

PLUM 则给出平衡观点：

- 用户行为用于建模偏好；
- 视频元数据和 LLM 知识用于理解 SID 和内容语义；
- CPT 连接二者。

因此更合理的结论是：

```text
行为信号是推荐预测的核心；
文本和多模态内容是泛化、冷启动和语义理解的重要补充。
```

---

## 6.5 生成式推荐的工程代价

生成式模型带来的问题包括：

1. 自回归 decoding 延迟；
2. beam search 成本；
3. invalid SID/hallucination；
4. SID collision；
5. 多样性与准确性 trade-off；
6. 线上安全、规则、探索和多目标约束注入困难；
7. 需要大规模 CPT 或 RL/DPO infrastructure。

这些问题决定了生成式推荐不是简单“把推荐变成语言模型”即可。

---

# 7. 关键概念复习与问答

## 7.1 ANN 和 MIPS 的区别是什么？

MIPS 是目标：

$$
\arg\max_i \mathbf{q}^T\mathbf{v}_i
$$

ANN 是加速方法：在大规模向量库中近似搜索最近邻或最大内积项。

---

## 7.2 user ID 有语义吗？

user ID 本身没有自然语义，但其 embedding 可以通过历史行为学习个性化偏好。

对新用户，旧 user ID embedding 没有直接作用。因此 user ID 不解决 user cold-start。TIGER 更主要解决 item cold-start。

---

## 7.3 Retrieval 阶段的 $\Phi_i$ 从哪里来？

在 Actions Speak Louder 中：

- retrieval 负责生成或检索候选 $\Phi_i$；
- ranking 给定候选 $\Phi_i$ 后预测用户动作 $a_i$。

因此 ranking 的候选并非凭空产生，仍来自 retrieval source。

---

## 7.4 OneRec 的 Reward Model ground truth 是什么？

OneRec 的 RM ground truth 来自真实线上日志的 session-level 用户反馈，例如 session watch time、view probability、follow probability、like probability 等。

DPO 的 chosen/rejected pair 不是用户真实同时比较，而是 RM 对模型生成 session 打分后构造的伪偏好对。

---

## 7.5 PLUM 的 ground truth 是什么？

PLUM 的 retrieval SFT ground truth 是用户日志中实际点击或观看的下一个视频，将其映射成 SID token 序列。

SFT 目标是让模型在给定用户历史和上下文时生成该点击视频的 SID。

---

# 8. 批判性总结与研究方向

## 8.1 批判性总结

四篇论文共同说明生成式推荐正在从学术概念进入工业系统。但它们解决的问题不同：

- TIGER 是生成式 SID retrieval 的概念验证；
- HSTU/GR 是工业推荐系统序列化与 scaling 的架构探索；
- OneRec 是端到端推荐列表生成的系统尝试；
- PLUM 是预训练 LLM 适配工业召回的实践。

不能简单判断哪一篇“最好”。它们位于推荐系统不同层级。

---

## 8.2 可能的研究方向

### 方向一：更好的 Semantic ID 训练

问题包括：

- 如何融合内容、多模态、行为、时效性；
- 如何处理动态 item corpus；
- 如何避免 frequent retraining；
- 如何设计可控、可解释的层级 SID；
- 如何降低 collision 和 hallucination。

### 方向二：受约束 decoding

生成式 retrieval 需要保证生成有效 SID，可研究：

- trie-constrained decoding；
- prefix-aware beam search；
- semantic prefix fallback；
- hybrid dense-sparse generation。

### 方向三：生成式 retrieval 与 ANN/MIPS 的混合系统

可能路线：

```text
generative model produces semantic prefixes
    ↓
ANN/MIPS within prefix bucket
    ↓
ranker reranks candidates
```

或：

```text
ANN produces candidates
    ↓
generative model reranks or diversifies
```

### 方向四：listwise generative recommendation

OneRec 展示了 session-wise generation 的潜力，但仍需解决：

- 多目标优化；
- 安全与业务约束；
- 长期用户价值；
- exploration/exploitation；
- diversity/fairness；
- controllability。

### 方向五：LLM-based recommendation foundation model

PLUM 表明 SID 和 CPT 可以让 LLM 学会推荐任务。未来问题包括：

- ranking 是否也能用 PLUM-style LLM；
- personalized search 是否能统一；
- LLM 是否能同时生成 SID 和自然语言解释；
- 是否能形成跨 domain 的推荐 foundation model。

---

# 9. 复习清单

## 9.1 TIGER 应掌握

- 什么是 Semantic ID；
- RQ-VAE 如何逐层量化残差；
- 为什么 RQ-VAE SID 优于 Random ID 和 LSH SID；
- TIGER 如何训练和推理；
- invalid ID、cold-start、多样性控制是什么。

## 9.2 Actions Speak Louder 应掌握

- DLRM 与 GR 的区别；
- retrieval 和 ranking 如何写成 sequential transduction；
- HSTU 的三个子模块；
- pointwise attention 为什么适合推荐；
- Stochastic Length 与 M-FALCON 的作用；
- 线上 retrieval/ranking 结果与 scaling law。

## 9.3 OneRec 应掌握

- 为什么 cascade ranking 存在阶段割裂；
- Balanced K-means 如何构造视频 identifier；
- session-wise generation 与 point-wise generation 的区别；
- Reward Model 如何训练；
- IPA/DPO 如何构造 chosen/rejected pair；
- 线上 A/B 结果与局限。

## 9.4 PLUM 应掌握

- LEM 与 PLUM 的 scaling 差异；
- SID-v2 的三项核心改进；
- CPT 如何对齐 SID、文本和用户行为；
- retrieval SFT 如何训练；
- PLUM 相比 LEM 的线上增量价值；
- CPT 与 LLM initialization 的消融结论；
- scaling study 的主要发现。

---

# 10. 最终综合判断

这四篇论文共同展示了生成式推荐的四个层次：

```text
1. 生成 item ID：TIGER
2. 生成行为序列：Actions Speak Louder / HSTU
3. 生成推荐列表：OneRec
4. 用预训练 LLM 生成推荐候选：PLUM
```

从理论上看，生成式推荐的核心优势在于：

- 用离散 token 表达 item；
- 让模型直接生成候选；
- 共享相似 item 的知识；
- 支持更灵活的 decoding 和多样性控制；
- 可能绕开传统 embedding table scaling 的瓶颈。

从工程上看，它仍面临：

- 自回归推理成本；
- invalid SID；
- 约束解码；
- 动态 item corpus；
- 多目标对齐；
- 线上可控性；
- 大规模训练和部署成本。

因此，较为稳健的结论是：

> 生成式推荐不是简单替代传统推荐系统，而是在召回、行为建模、列表生成和 LLM 适配等层面逐步重构推荐系统。短期内，它更可能作为高质量候选源、长序列行为建模器或列表生成模块与现有系统混合部署；长期看，它可能推动推荐系统从 embedding-table-centric 架构转向 token-and-neural-network-centric 架构。

