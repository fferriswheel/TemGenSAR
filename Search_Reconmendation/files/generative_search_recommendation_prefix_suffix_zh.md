# 生成式 Search & Recommendation：研究现状总结与 Prefix/Suffix Semantic ID 方案设计

## 1. Search 与 Recommendation 融合的现有研究概览

近年来，生成式推荐系统开始探索将 **搜索（Search）** 与 **推荐（Recommendation）** 统一建模。其核心目标是：用一个统一模型同时处理 query-driven retrieval，即由显式查询驱动的搜索任务，以及 context-driven recommendation，即由用户历史、上下文和隐式偏好驱动的推荐任务。

在传统工业系统中，search 和 recommendation 往往是两套相对独立的系统：搜索系统主要依赖用户输入的 query、文本匹配和相关性建模；推荐系统则主要依赖用户历史行为、协同过滤信号和上下文特征。生成式检索和 Semantic ID 的兴起，使得两类任务有可能被统一为：

$$
\text{context} \rightarrow \text{generate item identifiers}
$$

区别只在于 context 中是否包含显式 query，以及模型需要优化的是 query relevance 还是 user preference。

---

### 1.1 RecSys 2025: Semantic IDs for Joint Generative Search & Recommendation

**研究内容：**  
该方向的代表性工作研究了如何为 search 和 recommendation 构造统一的 Semantic ID。其核心问题是：如果同一个 item 同时服务于搜索和推荐任务，那么它的离散标识符是否应该由 search-specific embedding 构造，还是由 recommendation-specific embedding 构造，抑或由二者联合训练得到？

**方法：**  
该工作构造了一个统一的 item embedding space，用于同时服务 search 和 recommendation。具体做法是通过多任务训练的 bi-encoder 融合两类信号：

- search 侧：query–item relevance 或 query–item click signal；
- recommendation 侧：user–item interaction 或 item–item co-occurrence signal。

然后基于该统一 embedding space 生成 Semantic IDs，使生成式模型可以在同一套 item token space 中处理 search 和 recommendation。

**主要贡献：**

1. 指出 task-specific Semantic IDs 会产生任务冲突：  
   - search-oriented IDs 可能更适合 query relevance；
   - recommendation-oriented IDs 可能更适合 user behavior prediction；
   - 但二者在联合模型中可能互相损害。

2. 证明 multi-task embedding 生成的 Semantic IDs 可以在 search 和 recommendation 之间取得更好的折中。

3. 为“统一生成式搜索与推荐”提供了一个较清晰的实验范式。

**局限性：**

该类方法虽然实现了统一 Semantic ID space，但通常仍然是 **单一 ID 序列**。也就是说，同一个 item 对 search 和 recommendation 使用完全相同的 token 序列，并没有显式区分 shared semantics 和 task-specific semantics。

---

### 1.2 Spotify Research / GenSAR 类工作

**研究内容：**  
这类研究尝试将用户行为、搜索 query 和上下文信息统一输入到一个 generative retrieval model 中，使模型既能完成 recommendation，也能完成 search。

**核心思想：**

$$
\text{task token} + \text{query or user history} + \text{context}
\rightarrow
\text{item Semantic ID}
$$

例如：

$$
<search>, q, h_u \rightarrow SID_i
$$

$$
<recommend>, h_u \rightarrow SID_i
$$

其中 \(q\) 表示搜索 query，\(h_u\) 表示用户历史行为。

**主要贡献：**

1. 证明 search 和 recommendation 在生成式框架下可以共享模型参数。
2. 说明 search 的语义信号可以补充 recommendation 中的内容理解能力。
3. 说明 recommendation 的行为信号可以补充 search 中对个性化偏好的理解。
4. 推动了 query-driven 和 query-free 两类任务的统一建模。

**局限性：**

这类工作通常依赖自建数据或工业内部数据。目前并没有形成一个社区公认的标准 benchmark。因此，不同论文之间的结果往往不完全可比。

---

### 1.3 SynerGen: Personalized Search 与 Recommendation 的统一生成模型

**研究内容：**  
SynerGen 类工作试图使用一个 decoder-only Transformer 同时支持 personalized search 和 recommendation。相比只统一 retrieval，它进一步尝试在 retrieval 和 ranking 层面都进行统一建模。

**方法特点：**

- 输入可以包含 query、用户行为序列、上下文 item、用户特征等；
- 模型通过自回归方式生成或评分候选 item；
- 训练目标可能同时包括 retrieval objective 与 ranking objective；
- 通过多任务学习让 search 和 recommendation 互相增强。

**主要贡献：**

1. 把 search 和 recommendation 的融合从候选召回推进到更广义的个性化信息获取。
2. 强调显式 query intent 与隐式 user preference 的联合建模。
3. 提出一个更接近“统一信息访问模型”的系统方向。

**关键问题：**

当用户当前 query 与长期偏好冲突时，模型应该如何权衡？

例如，用户长期喜欢摇滚音乐，但当前搜索 “sleep meditation”。此时模型应该更尊重即时搜索意图，还是混入长期偏好？这是 search–recommendation 融合中非常核心的问题。

---

### 1.4 UniSearch / 工业端到端生成式搜索系统

**研究内容：**  
UniSearch 类工作从搜索系统出发，尝试用端到端生成式框架替代传统搜索中的 recall、pre-ranking、ranking 级联结构。

**方法特点：**

- 通过生成式模型直接生成相关 item 的 Semantic IDs；
- 使用 reward model 或 preference optimization 对齐真实用户反馈；
- 在搜索任务中引入类似推荐系统中的用户行为和上下文信息；
- 强调从多阶段 pipeline 转向 unified generative architecture。

**主要贡献：**

1. 说明生成式方法不只适用于 recommendation，也适用于 search。
2. 展示了 query + user context + item generation 的工业可行性。
3. 为 search 和 recommendation 的进一步融合提供了系统基础。

**与 OneRec 的关系：**

OneRec 是：

$$
\text{user history} \rightarrow \text{recommended session list}
$$

UniSearch 类方法则是：

$$
\text{query + user context} \rightarrow \text{search result list}
$$

二者结合后的自然扩展是：

$$
\text{query + user history + context}
\rightarrow
\text{personalized generated result list}
$$

这正是 unified search–recommendation 的核心方向。

---

## 2. 当前研究的共同主题

现有 Search–Recommendation 融合研究大致呈现出以下共同趋势。

### 2.1 Multi-task Learning

大多数方法都将 search 和 recommendation 视为两个相关但不同的任务，通过 task token、multi-task loss 或共享模型参数进行联合训练。

典型形式为：

$$
\mathcal{L}
=
\lambda_s \mathcal{L}_{search}
+
\lambda_r \mathcal{L}_{rec}
$$

其中：

- \(\mathcal{L}_{search}\)：优化 query–item relevance；
- \(\mathcal{L}_{rec}\)：优化 user–item interaction 或 next-item prediction；
- \(\lambda_s, \lambda_r\)：控制两类任务的权重。

---

### 2.2 Semantic IDs / Item Tokenization

生成式模型无法直接生成一个高维 item embedding，因此需要将 item 表示为离散 token 序列，即 Semantic ID：

$$
item_i \rightarrow SID_i = (s_i^1, s_i^2, \ldots, s_i^L)
$$

现有方法通常通过以下方式构造 Semantic ID：

- RQ-VAE；
- residual quantization；
- hierarchical clustering；
- multi-task bi-encoder embedding + quantization；
- multi-modal embedding + quantization；
- co-occurrence-aware contrastive learning。

---

### 2.3 Co-occurrence-aware Modeling

推荐中的 item similarity 并不等同于内容语义相似。两个 item 可能文本上不相似，但用户经常一起点击、一起观看或连续消费，因此在推荐意义上相似。

因此，越来越多工作开始将 co-occurrence signal 融入 item representation 或 SID learning：

$$
\text{content similarity}
+
\text{behavioral co-occurrence}
\rightarrow
\text{recommendation-aware Semantic ID}
$$

这类方法试图让 Semantic ID 同时具备：

1. 内容语义；
2. 用户行为语义；
3. 协同过滤结构；
4. 下游生成式召回友好的层级 token 结构。

---

### 2.4 Unified Generation

当前趋势是用同一个 generative backbone 支持多种任务：

$$
<search>, q, h_u \rightarrow SID
$$

$$
<recommend>, h_u \rightarrow SID
$$

$$
<rank>, h_u, item \rightarrow action
$$

这种统一形式有助于减少系统复杂度，并让不同任务共享语义和行为知识。

---

### 2.5 Benchmark Gap

目前尚无社区公认、开源、长期维护的 **joint search & recommendation benchmark**。

已有工作可能会基于 MovieLens-25M、Amazon Reviews、query–item pairs 或工业日志自建联合实验协议，但这些更像是论文内部的实验设置，而不是标准 benchmark。

因此，该方向仍存在明显的 benchmark 建设空间。

---

## 3. 当前 Multi-task Semantic ID 的局限

现有 multi-task Semantic ID 大多采用单一统一 token 序列：

$$
SID_i = (s_i^1, s_i^2, \ldots, s_i^L)
$$

这个 SID 同时服务于 search 和 recommendation。其优势是简单、共享性强，但也存在几个问题。

### 3.1 Search 与 Recommendation 的相似性定义不同

Search 关心的是：

$$
\text{query relevance}
$$

Recommendation 关心的是：

$$
\text{user preference / behavioral relevance}
$$

这两个目标并不总是一致。

例如：

- 搜索任务中，“wireless earbuds” 应该匹配标题、描述、属性高度相关的商品；
- 推荐任务中，用户购买 wireless earbuds 后，可能更需要推荐 phone case 或 charger，这些 item 与 query 语义并不相同，但行为上高度相关。

因此，将两类信号压缩到同一个 token 序列中，可能导致任务冲突。

---

### 3.2 统一 ID 难以解释任务冲突

如果一个 item 的统一 SID 在 search 上表现好、recommendation 上表现差，我们很难判断问题来自：

- 内容语义 token 不佳；
- 行为语义 token 不佳；
- search signal 与 rec signal 冲突；
- quantization 本身造成信息损失；
- 模型没有学好 task conditioning。

显式结构化的 ID 可以让错误分析更清楚。

---

### 3.3 缺少模块化更新能力

工业系统中的 recommendation signal 往往变化很快，例如用户兴趣、item popularity、co-occurrence pattern 都会随时间漂移。

而 search 侧的内容语义可能相对稳定。

如果 search 和 recommendation 共用完全相同的 SID，那么一旦推荐行为信号发生漂移，更新 SID 可能会影响 search 任务。反之亦然。

---

## 4. Proposed Design: Shared Prefix / Task-Specific Suffix Semantic ID

为缓解当前 multi-task Semantic ID 的冲突，我们提出一种 **Shared Prefix + Task-Specific Suffix Semantic ID** 设计。

核心思想是将 item ID 拆成两部分：

$$
SID = [\text{Shared Prefix} \mid \text{Task-Specific Suffix}]
$$

其中：

- Shared Prefix 表示任务无关的通用内容语义；
- Task-Specific Suffix 表示 search 或 recommendation 的任务特定语义。

---

## 5. SID 结构设计

### 5.1 基本形式

对于每个 item \(i\)，定义：

$$
SID_i^{search}
=
[p_i^1, p_i^2, \ldots, p_i^P
\mid
s_i^1, s_i^2, \ldots, s_i^S]
$$

$$
SID_i^{rec}
=
[p_i^1, p_i^2, \ldots, p_i^P
\mid
r_i^1, r_i^2, \ldots, r_i^R]
$$

其中：

- \(p_i^1, \ldots, p_i^P\)：shared prefix tokens；
- \(s_i^1, \ldots, s_i^S\)：search-specific suffix tokens；
- \(r_i^1, \ldots, r_i^R\)：recommendation-specific suffix tokens。

也可以写成：

$$
SID_i^t = [P_i \mid Z_i^t]
$$

其中 \(t \in \{search, rec\}\)。

---

### 5.2 Shared Prefix 的语义

Shared Prefix 用于表示 item 的通用语义，应尽量 task-agnostic。

它可以包含：

- 标题语义；
- 文本描述；
- 图像或视频视觉内容；
- 音频或 ASR 内容；
- 类别、品牌、主题等相对稳定属性；
- 长期不变的 item identity 信息。

目标是让 prefix 表示：

$$
\text{what the item is}
$$

而不是：

$$
\text{how users interact with it in a specific task}
$$

---

### 5.3 Recommendation Suffix 的语义

Recommendation suffix 用于表示 item 在推荐场景中的行为语义，例如：

- item–item co-occurrence；
- user–item collaborative filtering signal；
- sequential transition pattern；
- session-level co-consumption；
- complementarity / substitutability；
- popularity-adjusted behavioral clusters。

它表示：

$$
\text{how this item behaves in recommendation}
$$

例如：

- 用户看完 A 后常看 B；
- 用户买了耳机后常买保护壳；
- 用户刷完某类短视频后常继续刷另一类视频。

这些关系未必是纯内容相似，但对推荐非常重要。

---

### 5.4 Search Suffix 的语义

Search suffix 用于表示 item 在搜索场景中的 query relevance。

它可以来自：

- query–item click logs；
- query–item relevance labels；
- BM25 / dense retrieval teacher；
- search session logs；
- query reformulation data；
- attribute matching signals。

它表示：

$$
\text{how this item matches explicit queries}
$$

例如：

- item 是否匹配 query 中的核心实体；
- item 是否满足 query intent；
- item 是否覆盖 query 的属性约束；
- item 在搜索日志中是否经常被点击。

---

## 6. 构造流程

### 6.1 Step 1: 多模态内容编码

对每个 item 提取多模态 embedding：

$$
x_i^{text}, x_i^{image}, x_i^{audio}, x_i^{meta}
$$

通过各自 encoder 得到：

$$
z_i^{text} = E_{text}(x_i^{text})
$$

$$
z_i^{image} = E_{image}(x_i^{image})
$$

$$
z_i^{audio} = E_{audio}(x_i^{audio})
$$

然后拼接并投影：

$$
z_i =
\text{Proj}([z_i^{text}, z_i^{image}, z_i^{audio}, z_i^{meta}])
$$

该表示用于生成 shared prefix。

---

### 6.2 Step 2: Shared Prefix Quantization

对 \(z_i\) 使用 residual quantization 或 RQ-VAE：

$$
p_i^1 = Q_1(z_i)
$$

$$
r_i^2 = z_i - e_{p_i^1}
$$

$$
p_i^2 = Q_2(r_i^2)
$$

继续递归，得到：

$$
P_i = (p_i^1, p_i^2, \ldots, p_i^P)
$$

为了保证 prefix 有稳定层级结构，可以加入：

- multi-resolution codebook；
- progressive masking；
- prefix reconstruction loss；
- prefix entropy regularization。

---

### 6.3 Step 3: Recommendation Suffix Learning

构造 recommendation behavior representation \(b_i^{rec}\)。

可以基于：

1. item–item co-occurrence matrix；
2. user sequence transition graph；
3. session co-watch / co-click graph；
4. item2vec / graph embedding；
5. contrastive learning over positive co-occurring pairs。

例如，对共现 item pair \((i, j^+)\)，使用 InfoNCE loss：

$$
\mathcal{L}_{cooc}
=
-\log
\frac{
\exp(\text{sim}(b_i, b_{j^+}) / \tau)
}{
\sum_{j \in \mathcal{B}}
\exp(\text{sim}(b_i, b_j) / \tau)
}
$$

然后将 \(b_i^{rec}\) 量化为 recommendation suffix：

$$
R_i = (r_i^1, r_i^2, \ldots, r_i^R)
$$

---

### 6.4 Step 4: Search Suffix Learning

构造 search representation \(b_i^{search}\)。

可以通过 query–item relevance 数据训练 bi-encoder：

$$
q = E_q(\text{query})
$$

$$
d_i = E_d(\text{item})
$$

优化：

$$
\mathcal{L}_{search}
=
-\log
\frac{
\exp(q^\top d_i / \tau)
}{
\sum_{j \in \mathcal{B}}
\exp(q^\top d_j / \tau)
}
$$

得到 search-aware item representation：

$$
b_i^{search} = d_i
$$

再量化为 search suffix：

$$
S_i = (s_i^1, s_i^2, \ldots, s_i^S)
$$

---

### 6.5 Step 5: SID 拼接

最终得到两套 task-conditioned SID：

Recommendation:

$$
SID_i^{rec} = [P_i \mid R_i]
$$

Search:

$$
SID_i^{search} = [P_i \mid S_i]
$$

也可以为 joint task 构造：

$$
SID_i^{joint} = [P_i \mid R_i \mid S_i]
$$

但这种形式会增加序列长度和推理成本，需要根据任务设计选择。

---

## 7. 训练目标设计

### 7.1 Prefix Reconstruction Loss

保证 shared prefix 保留 item 内容语义：

$$
\mathcal{L}_{prefix-recon}
=
\|x_i - \hat{x}_i\|^2
$$

如果是多模态输入：

$$
\mathcal{L}_{prefix-recon}
=
\sum_m \|x_i^m - \hat{x}_i^m\|^2
$$

---

### 7.2 Recommendation Co-occurrence Loss

保证 recommendation suffix 捕捉行为共现：

$$
\mathcal{L}_{rec-cooc}
=
-\log
\frac{
\exp(\text{sim}(g_i^{rec}, g_{j^+}^{rec})/\tau)
}{
\sum_j \exp(\text{sim}(g_i^{rec}, g_j^{rec})/\tau)
}
$$

其中 \(j^+\) 是与 item \(i\) 共现的正样本。

---

### 7.3 Search Query Alignment Loss

保证 search suffix 捕捉 query relevance：

$$
\mathcal{L}_{search-align}
=
-\log
\frac{
\exp(E_q(q)^\top g_i^{search}/\tau)
}{
\sum_j \exp(E_q(q)^\top g_j^{search}/\tau)
}
$$

---

### 7.4 Prefix-Suffix Disentanglement Loss

为了避免 suffix 重复编码 prefix 中已有的内容语义，可以加入去相关约束：

$$
\mathcal{L}_{dis}
=
\|C(P_i, R_i)\|_F^2
+
\|C(P_i, S_i)\|_F^2
$$

其中 \(C(\cdot, \cdot)\) 表示 cross-covariance。

也可以使用 adversarial loss，让 prefix 难以预测 task label，而 suffix 能预测 task label：

$$
\mathcal{L}_{task}
=
\mathcal{L}_{task-cls}(R_i, rec)
+
\mathcal{L}_{task-cls}(S_i, search)
-
\gamma \mathcal{L}_{task-cls}(P_i, task)
$$

---

### 7.5 Generative Retrieval Loss

统一生成式模型训练时，输入 task token：

Search:

$$
<search>, q, h_u \rightarrow SID_i^{search}
$$

Recommendation:

$$
<recommend>, h_u \rightarrow SID_i^{rec}
$$

训练损失：

$$
\mathcal{L}_{gen}
=
-\sum_t
\log P(sid_t \mid context, sid_{<t})
$$

总损失：

$$
\mathcal{L}
=
\lambda_1 \mathcal{L}_{prefix-recon}
+
\lambda_2 \mathcal{L}_{rec-cooc}
+
\lambda_3 \mathcal{L}_{search-align}
+
\lambda_4 \mathcal{L}_{dis}
+
\lambda_5 \mathcal{L}_{gen}
$$

---

## 8. 模型训练方式

### 8.1 两阶段训练

第一阶段：SID tokenizer training

- 训练 shared prefix；
- 训练 rec suffix；
- 训练 search suffix；
- 得到 item 到 SID 的映射表。

第二阶段：Generative model training

- 使用固定 SID；
- 训练生成式模型根据 task input 生成对应 SID；
- 支持 search 和 recommendation 多任务训练。

---

### 8.2 端到端联合训练

也可以将 tokenizer 与 generative model 联合训练，但工程复杂度更高。

优点：

- SID 更直接适配生成任务；
- tokenizer 可根据下游 loss 调整。

缺点：

- 离散量化不可导；
- SID mapping 变化会破坏训练稳定性；
- 工业部署中难以频繁更新。

更推荐先使用两阶段训练。

---

## 9. 推理设计

### 9.1 Recommendation Inference

输入：

$$
<recommend>, h_u
$$

生成：

$$
[p_i^1, \ldots, p_i^P \mid r_i^1, \ldots, r_i^R]
$$

映射回 item：

$$
SID_i^{rec} \rightarrow item_i
$$

---

### 9.2 Search Inference

输入：

$$
<search>, q, h_u
$$

生成：

$$
[p_i^1, \ldots, p_i^P \mid s_i^1, \ldots, s_i^S]
$$

映射回 item：

$$
SID_i^{search} \rightarrow item_i
$$

---

### 9.3 Hybrid Inference

如果同时存在 query 和推荐上下文，可以采用三种策略：

1. **Search-first generation**

$$
<search>, q, h_u \rightarrow SID^{search}
$$

2. **Recommendation-first generation**

$$
<recommend>, q, h_u \rightarrow SID^{rec}
$$

3. **Joint generation**

$$
<joint>, q, h_u \rightarrow [P_i \mid S_i \mid R_i]
$$

第三种最灵活，但计算成本最高。

---

## 10. 预期优势

### 10.1 缓解任务冲突

现有 multi-task Semantic ID 将 search 和 recommendation 信号压缩到同一 token 序列中，容易产生冲突。Prefix/suffix 结构可以将通用语义与任务特定语义分离。

---

### 10.2 提高可解释性

如果生成错误，可以分析是 prefix 错误还是 suffix 错误：

- prefix 错误：item 内容语义理解错误；
- rec suffix 错误：行为共现建模错误；
- search suffix 错误：query relevance 建模错误。

这比单一统一 SID 更容易诊断。

---

### 10.3 支持模块化更新

当 recommendation 行为分布漂移时，可以更新 rec suffix；当 search query 分布变化时，可以更新 search suffix；而 shared prefix 可以保持稳定。

这有助于降低工业系统中的 SID refresh 成本。

---

### 10.4 改善冷启动

对于新 item，虽然没有充分行为信号，但仍可根据内容生成 shared prefix。缺失的 task suffix 可以通过：

- 内容相似 item 的 suffix 迁移；
- query matching 估计 search suffix；
- co-occurrence graph bootstrapping 估计 rec suffix；
- fallback 到 prefix-only retrieval。

---

### 10.5 支持统一模型

一个生成式模型可以通过 task token 控制输出不同 suffix：

$$
<search> \rightarrow search\ suffix
$$

$$
<recommend> \rightarrow rec\ suffix
$$

这比为 search 和 recommendation 分别训练两套模型更简洁。

---

## 11. 可能风险与挑战

### 11.1 SID 序列变长

Prefix + suffix 会增加 token length，导致生成成本变高。

可选解决方案：

- prefix 2–3 tokens；
- suffix 1–2 tokens；
- multi-resolution codebook；
- constrained decoding；
- shared prefix cache。

---

### 11.2 Task-specific suffix 数据稀疏

Search logs 和 recommendation logs 的覆盖不同，有些 item 可能只有推荐行为，没有搜索行为，或反之。

解决方案：

- suffix distillation；
- graph propagation；
- semi-supervised quantization；
- prefix-to-suffix prediction。

---

### 11.3 Prefix 与 suffix 信息泄漏

如果 prefix 已经编码了大量行为信号，suffix 的作用会变弱；如果 suffix 编码过多内容语义，任务分离也会失败。

需要引入 disentanglement regularization。

---

### 11.4 Evaluation 难度

由于没有标准 benchmark，需要自建联合评估协议。推荐至少同时报告：

- Search Recall@K / NDCG@K；
- Recommendation Recall@K / NDCG@K；
- joint score；
- task interference metric；
- valid SID rate；
- SID collision rate；
- diversity / coverage。

---

## 12. 实验设计建议

### 12.1 Baselines

推荐比较以下 baselines：

1. content-only SID；
2. recommendation-only SID；
3. search-only SID；
4. unified multi-task SID；
5. shared-prefix + rec-suffix；
6. shared-prefix + search-suffix；
7. shared-prefix + dual-suffix；
8. task prompt only, no suffix separation。

---

### 12.2 Ablation Studies

建议做以下消融：

- 去掉 shared prefix；
- 去掉 rec suffix；
- 去掉 search suffix；
- 去掉 co-occurrence loss；
- 去掉 query alignment loss；
- 不使用 disentanglement loss；
- suffix 长度变化；
- prefix 长度变化；
- shared-prefix/task-suffix 不同 codebook 大小。

---

### 12.3 Evaluation Metrics

Search:

$$
Recall@K,\quad NDCG@K,\quad MRR@K
$$

Recommendation:

$$
Recall@K,\quad NDCG@K,\quad HR@K
$$

Generation-specific:

$$
ValidSIDRate,\quad CollisionRate,\quad Coverage,\quad Entropy@K
$$

Task conflict:

$$
\Delta_{search} = Metric_{search}^{joint} - Metric_{search}^{single}
$$

$$
\Delta_{rec} = Metric_{rec}^{joint} - Metric_{rec}^{single}
$$

如果 shared-prefix/task-suffix 方法能同时提高或减少两个任务的负迁移，则说明设计有效。

---

## 13. Benchmark 现状与构造建议

### 13.1 当前是否有标准 benchmark？

目前没有广泛公认的、类似 MovieLens 或 MS MARCO 那样的 joint search & recommendation 标准 benchmark。

现有论文通常使用：

- MovieLens-25M 构造推荐任务；
- 合成或外部 query 构造搜索任务；
- 工业 search logs + recommendation logs；
- 自建 query–item relevance 数据；
- 多任务训练和多指标评估。

因此，该方向仍缺少统一数据集、统一切分方式和统一 leaderboard。

---

### 13.2 可行的 Benchmark 构造

一个可复现 benchmark 可以包含：

1. Recommendation data:
   - user history；
   - next-item labels；
   - user–item interactions。

2. Search data:
   - query–item pairs；
   - query click logs；
   - relevance labels。

3. Shared item corpus:
   - item metadata；
   - title / description；
   - category；
   - multi-modal features if available。

4. Unified item IDs:
   - 所有任务共享 item universe。

训练样本：

$$
<recommend>, h_u \rightarrow item_i
$$

$$
<search>, q \rightarrow item_i
$$

$$
<personalized-search>, q, h_u \rightarrow item_i
$$

这三类任务可以构成一个较完整的联合评估框架。

---

## 14. 总结

现有 Search–Recommendation 融合研究已经证明：

1. search 与 recommendation 可以在生成式检索框架下统一；
2. Semantic ID 是统一建模中的核心组件；
3. search signal 和 recommendation signal 既互补又冲突；
4. 现有 multi-task Semantic ID 多为单一统一 ID，缺少显式结构分离；
5. 目前没有社区标准化 benchmark。

因此，**Shared Prefix + Task-Specific Suffix Semantic ID** 是一个有潜力的研究方向。它的核心优势在于：

- 将通用 item 语义与任务特定语义分离；
- 缓解 search 与 recommendation 的目标冲突；
- 提高可解释性；
- 支持模块化更新；
- 更适合冷启动和持续学习；
- 为统一生成式 search–recommendation 模型提供结构化 item tokenization。

一个可行的研究题目可以是：

> **Shared-Prefix and Task-Specific-Suffix Semantic IDs for Unified Generative Search and Recommendation**

或中文表述为：

> **面向统一生成式搜索与推荐的共享前缀—任务后缀语义标识符学习**

