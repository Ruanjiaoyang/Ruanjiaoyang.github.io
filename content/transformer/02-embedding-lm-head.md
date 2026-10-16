---
title: "Embedding 与 LM Head：模型的入口和出口"
title_en: "Embeddings and the LM Head: The Model's Entrance and Exit"
series: "Transformer 架构"
order: 2
date: 2026-10-16
tags: [Transformer, Embedding, LM Head, Weight Tying]
summary_en: >
  The embedding table turns token ids into vectors and the LM head turns the final hidden state back into
  a distribution over the vocabulary. This post explains why dense embeddings are needed, shows that a
  lookup is a one-hot matrix product and what that means for gradients, derives the softmax cross-entropy
  gradient, analyzes weight tying (benefits, costs, and why large models often untie), quantifies the cost
  of large vocabularies in parameters, FLOPs and logit memory, covers vocabulary parallelism and
  stabilizers such as z-loss and logit soft-capping, and distinguishes the LM head from MTP and draft heads.
---

## 0. 入口和出口

在原始 Transformer（第 0 篇）中，一个 token 进出模型要经过两个对称的部件：

- **入口：嵌入层**（embedding）。把分词器给出的整数 id 变成 $d$ 维向量；
- **出口：LM Head**（语言模型头）。把最后一层的隐藏向量变成词表上每个 token 的分数，再经过 softmax 得到下一个 token 的概率。

原始 Transformer 让两者共享同一个矩阵，并把嵌入乘以 $\sqrt d$。这一篇讲清楚这两个部件在做什么、为什么可以共享、什么时候不该共享，以及词表变大之后它们为什么成了显著的成本。

## 1. 嵌入层

### 1.1 定义

对词表大小为 $V$、隐藏维度为 $d$ 的模型，嵌入矩阵为

$$
E\in\mathbb R^{V\times d}
$$

第 $i$ 行 $E[i]$ 是 id 为 $i$ 的 token 的向量。嵌入就是查表：

$$
x_t = E[\text{token}_t]
$$

一个 $[B, L]$ 的 id 张量经过嵌入层变成 $[B, L, d]$ 的向量张量。

### 1.2 查表就是乘以 one-hot

把 token $i$ 写成 one-hot 行向量 $e_i\in\{0,1\}^{1\times V}$（第 $i$ 位为 1），则

$$
e_iE = E[i]
$$

所以嵌入层在数学上就是一个**没有偏置的线性层**，输入是 one-hot 向量。实现时不会真的做这个矩阵乘法（$V$ 维的 one-hot 乘以 $V\times d$ 的矩阵太浪费），而是直接按索引取行。这也说明嵌入层几乎不消耗计算，它的成本在**存储**和**访存**上。

**梯度是稀疏的.** 由于 $x_t = e_iE$，损失对 $E$ 的梯度是

$$
\frac{\partial\mathcal L}{\partial E} = \sum_t e_{i_t}^\top\frac{\partial\mathcal L}{\partial x_t}
$$

只有这个批次中**出现过的 token 对应的行**会收到梯度，其余行的梯度为 0。这带来两个后果：

- 罕见 token 很少被更新，嵌入学得不好（第 1 篇的 glitch token 是极端情况）；
- 优化器状态（如 Adam 的一阶、二阶矩）对没出现的行也会因为权重衰减或动量继续变化，这是一些实现对嵌入层使用不同优化设置的原因。

### 1.3 为什么需要稠密的嵌入

直接把 one-hot 向量送进网络行不行？有两个问题：

- **计算角度**：one-hot 向量的维度是 $V$（几万到几十万），后续每一层的权重都要乘以这个维度，开销无法接受。嵌入把它压到 $d$（几百到几千）维；
- **学习角度**：任意两个 one-hot 向量互相正交、距离相同，无法表示词与词之间的语义关系。稠密嵌入可以把相似的 token 放在空间中相近的位置，比如 `apple` 和 `banana` 都是水果，它们的向量也会相似。网络在一个 token 上学到的东西可以通过相近的向量泛化到相似的 token 上。

嵌入是和整个模型一起端到端训练的。早期的 word2vec、GloVe 单独预训练词向量，再交给下游模型；今天的语言模型直接从随机初始化开始，与其他参数一起学习。

### 1.4 尺度：为什么原始 Transformer 要乘以 $\sqrt d$

常见的初始化让嵌入矩阵的元素方差约为 $1/d$（例如 $\mathcal N(0, 1/d)$），这样每个嵌入向量的范数约为 1，每个分量约为 $1/\sqrt d$ 的量级。原始 Transformer 接下来要把**正弦位置编码**加上去，而位置编码的每个分量在 $[-1, 1]$ 之间，范数约为 $\sqrt{d/2}$。

如果直接相加，位置编码的量级比内容嵌入大得多，内容信息会被淹没。乘以 $\sqrt d$ 后，嵌入每个分量回到 $O(1)$ 的量级，与位置编码相当。

现代模型大多使用 RoPE（作用在注意力内部的 Q、K 上，不加在嵌入上），这个缩放的原始动机就不存在了，所以 Llama 等模型不再乘以 $\sqrt d$。但也有模型保留了这个缩放（例如 Gemma 系列对嵌入乘以 $\sqrt d$），主要是为了配合它们的初始化和归一化方案，让残差流在开始时有合适的尺度。

## 2. LM Head

### 2.1 定义

最后一层（以及最后的归一化）之后得到隐藏向量 $h_t\in\mathbb R^d$。LM Head 是一个线性层：

$$
z_t = W_{\text{out}}h_t,\qquad W_{\text{out}}\in\mathbb R^{V\times d}
$$

$z_t\in\mathbb R^V$ 称为 **logits**，第 $i$ 个分量是 token $i$ 的未归一化分数。经过 softmax 得到下一个 token 的分布：

$$
p(x_{t+1} = i\mid x_{\le t}) = \frac{\exp(z_{t,i})}{\sum_{j=1}^V\exp(z_{t,j})}
$$

现代大模型的 LM Head 通常不带偏置。

**几何解读.** $z_{t,i} = w_i^\top h_t$，其中 $w_i$ 是 $W_{\text{out}}$ 的第 $i$ 行。所以 LM Head 是在问：**隐藏向量 $h_t$ 与每个 token 的"输出向量" $w_i$ 有多对齐？** 对齐得越好，该 token 的概率越高。整个模型的工作，就是把上下文加工成一个指向"正确的下一个 token"的输出向量的方向。

### 2.2 交叉熵损失与它的梯度

训练目标是最小化正确 token $y$ 的负对数概率：

$$
\mathcal L = -\log p(y\mid x_{\le t}) = -z_y + \log\sum_j\exp(z_j)
$$

对 logits 求导：

$$
\frac{\partial\mathcal L}{\partial z_i} = -\mathbb 1[i = y] + \frac{\exp(z_i)}{\sum_j\exp(z_j)} = p_i - \mathbb 1[i = y]
$$

写成向量：$\nabla_z\mathcal L = p - e_y$，即**预测分布减去 one-hot 目标**。由此可以读出两件事：

- 正确 token 的 logit 被往上推（梯度 $p_y - 1<0$），其他所有 token 的 logit 按各自的预测概率被往下压；
- 对 $W_{\text{out}}$ 的梯度是 $(p - e_y)h_t^\top$：正确 token 的输出向量 $w_y$ 被拉向 $h_t$，其他 token 的输出向量按概率被推离 $h_t$。

注意这里与嵌入层的区别：**LM Head 的每一行在每一步都会收到梯度**（只要概率不为 0），梯度是稠密的；嵌入层只有出现的 token 收到梯度。

## 3. 权重共享（Weight Tying）

### 3.1 做法

令

$$
W_{\text{out}} = E
$$

输入嵌入与 LM Head 使用同一个矩阵。于是 $z_{t,i} = E[i]^\top h_t$：**下一个 token 的分数就是隐藏向量与该 token 的输入嵌入的内积。** 原始 Transformer、GPT-2、T5 等都使用了权重共享（Press & Wolf, 2017; Inan et al., 2017）。

### 3.2 好处

- **省参数**：少了一个 $V\times d$ 的矩阵。对小模型非常可观（见第 4 节的表）；
- **语义空间一致**：同一个 token 的输入表示和输出表示是同一个向量。"语义相近的 token 输入向量相近"这个性质，直接变成"语义相近的 token 在输出时得分也相近"，这是一个合理的归纳偏置；
- **罕见 token 学得更好**：1.2 节说过嵌入层的梯度是稀疏的，而 LM Head 的梯度是稠密的。共享之后，每个 token 的向量在每一步都通过输出端得到训练信号，罕见 token 不再只靠自己出现的少数几次被更新。Press & Wolf 的实验中，共享权重在语言建模上降低了困惑度。

### 3.3 代价

- **两个角色要用同一组向量**：输入嵌入要表达"这个 token 是什么"，输出向量要表达"在什么上下文之后会出现这个 token"。两者相关但不完全相同。例如，一个 token 作为输入时需要携带它自己的语义，作为输出时需要和"预测它的那种上下文表示"对齐。共享强迫二者一致，限制了表达自由度；
- **尺度冲突**：输出端希望 logits 有合适的动态范围，输入端希望嵌入与残差流的尺度相匹配。共享时两者的需求要同时满足，这也是一些共享权重的模型对嵌入乘以缩放因子、或对 logits 除以 $\sqrt d$ 的原因。

### 3.4 现代模型怎么选

经验上的趋势是：**小模型多数共享，大模型多数不共享。**

- 小模型中嵌入参数占比很高（可达三成），共享能省下大量参数，换来的容量可以用在 Transformer 层上；
- 大模型中嵌入参数占比很小，省下的参数微不足道，而解除共享带来的表达自由度更有价值。

例如 GPT-2、Qwen2 的 0.5B 和 1.5B 等小模型共享权重；Llama 2、Llama 3 的 8B 及以上、Qwen2 的 7B 及以上不共享。但这不是绝对规律，Gemma 系列各尺寸都共享。所以不能简单认为某一种做法一定更先进，它取决于模型规模和整体设计。

## 4. 大词表的代价

第 1 篇提到词表在变大：从 3 万到 15 万甚至 25 万。嵌入层和 LM Head 因此成为显著的成本。

### 4.1 参数量

嵌入和 LM Head 各有 $Vd$ 个参数。几个例子（按公开配置估算）：

| 模型 | $V$ | $d$ | 是否共享 | 嵌入 + LM Head 参数 | 占总参数 |
|---|---|---|---|---|---|
| GPT-2 small | 50257 | 768 | 共享 | 约 39M | 约 31% |
| Qwen2-0.5B | 151936 | 896 | 共享 | 约 136M | 约 28% |
| Llama 2 7B | 32000 | 4096 | 不共享 | 约 262M | 约 4% |
| Llama 3 8B | 128256 | 4096 | 不共享 | 约 1.05B | 约 13% |

Llama 3 把词表从 3.2 万扩到 12.8 万，嵌入相关参数增加了 7 亿多，占比从 4% 升到 13%。

### 4.2 计算量

嵌入层是查表，几乎没有计算。但 LM Head 是一个真正的矩阵乘法：每个 token 需要约 $2Vd$ 次浮点运算。相对于每个 token 的总前向计算约 $2N_{\text{params}}$（第 0 篇第 6 节），LM Head 的占比约为

$$
\frac{2Vd}{2N_{\text{params}}}
$$

对 Llama 3 8B 约为 $128256\times 4096/8\times 10^9\approx 6.6\%$。对小模型或大词表模型，这个比例更高。

**MoE 模型中尤其明显.** MoE 模型的总参数很大，但每个 token 只激活一小部分专家（第 25 篇）。而 LM Head 是**稠密**的，每个 token 都要算完整个词表的 logits。所以激活参数少，并不意味着输出层的计算也变成了稀疏计算。在激活参数只有几十亿的 MoE 模型里，LM Head 可能占到每个 token 计算量的相当大一部分。

### 4.3 Logits 的显存

训练时需要存储所有位置的 logits 来计算损失和梯度，形状是 $[B, L, V]$。例如 $B = 8$、$L = 4096$、$V = 128{,}000$，用 FP32 存储需要

$$
8\times 4096\times 128000\times 4\ \text{bytes}\approx 16.8\ \text{GB}
$$

这还只是一份，反向传播还要存它的梯度。所以大词表模型的训练框架通常会：

- **分块计算交叉熵**：沿序列维度分块，每块算完 logits 和损失后立即释放；
- **融合 LM Head 与损失**：把矩阵乘法、softmax 和交叉熵写成一个算子，不把完整的 logits 写回显存。

### 4.4 词表并行

在多卡训练和推理中，单张卡可能放不下或算不动完整的 LM Head。**词表并行**（vocabulary parallelism）把 $W_{\text{out}}$ 按词表维度切到 $P$ 张卡上，每张卡负责 $V/P$ 个 token 的 logits。

难点在 softmax：它需要所有 token 的 logits。做法是：

1. 每张卡算出自己那部分 logits 的最大值，做一次 all-reduce 取全局最大值（数值稳定）；
2. 每张卡算 $\sum\exp(z_j - z_{\max})$，再做一次 all-reduce 求和，得到全局归一化常数；
3. 正确 token 的 logit 只在一张卡上，那张卡负责算损失的那一项。

这样不需要把完整的 logits 收集到一张卡上，通信量只有几个标量。嵌入层也可以按同样方式切分，每张卡只查自己负责的那部分 id，再求和合并。

## 5. 输出层的数值稳定

LM Head 的 logits 可能在训练中不断增大，导致 softmax 过于尖锐、数值溢出或训练不稳定。两种常见的处理方式：

**z-loss**（PaLM 使用）。记 softmax 的归一化常数 $Z = \sum_j\exp(z_j)$，加一个辅助损失

$$
\mathcal L_z = \lambda\,(\log Z)^2,\qquad \lambda\approx 10^{-4}
$$

它鼓励 $\log Z$ 接近 0，也就是让 logits 的整体水平不要漂移得太大。softmax 对 logits 整体加一个常数不变，所以这个约束不改变预测分布本身，只是消除了一个自由度，让数值更稳定。

**Logit soft-capping**（Gemma 2 使用）。用 $\tanh$ 把 logits 平滑地限制在 $[-c, c]$ 内：

$$
z\leftarrow c\cdot\tanh(z/c)
$$

$|z|$ 远小于 $c$ 时几乎不变，接近 $c$ 时被压住。

这两种方法和第 23 篇的 QK-Norm 思路相同：注意力的 logits 和输出的 logits 都是"内积过大导致 softmax 饱和"的潜在来源。

## 6. LM Head 与 MTP / Draft Head 的区别

近年出现了一些"额外的输出头"，容易和 LM Head 混淆。

**标准 LM Head** 负责预测**下一个** token：

$$
h_t\ \longrightarrow\ x_{t+1}
$$

**MTP**（Multi-Token Prediction，如 DeepSeek-V3）、**Medusa**、**EAGLE** 等方法会额外预测更远的未来 token：

$$
h_t\ \longrightarrow\ \hat x_{t+1},\ \hat x_{t+2},\ \dots
$$

它们的用途是：

- **训练时作为辅助目标**：MTP 让模型在每个位置多预测几个未来 token，增加训练信号的密度，并鼓励隐藏状态对更远的未来做规划；
- **推理时提出候选**：在投机解码中，这些头快速给出接下来几个 token 的猜测，再由主模型一次前向验证。

但它们**不替代**标准 LM Head。最终输出的每个 token 都要经过目标模型的标准 LM Head 验证或生成，额外的头只负责提出候选或提供辅助训练信号。部署时，MTP 模块可以直接丢弃，模型照常工作；也可以保留下来用于投机解码加速。

## 7. 一个参考实现

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, scale=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) / math.sqrt(d_model))
        self.scale = math.sqrt(d_model) if scale else 1.0   # 原始 Transformer: scale=True

    def forward(self, ids):                                 # ids: (B, L)
        return F.embedding(ids, self.weight) * self.scale   # 按索引取行: (B, L, d)


class LMHead(nn.Module):
    def __init__(self, vocab_size, d_model, tied_embedding=None):
        super().__init__()
        if tied_embedding is not None:
            self.weight = tied_embedding.weight             # 权重共享：同一个 Parameter
        else:
            self.weight = nn.Parameter(torch.randn(vocab_size, d_model) / math.sqrt(d_model))

    def forward(self, h):                                   # h: (B, L, d)
        return h @ self.weight.T                            # logits: (B, L, V)


def lm_loss(logits, targets, z_loss_coef=0.0):
    """标准交叉熵，可选 z-loss"""
    logits = logits.float()                                 # 在 FP32 下算 softmax
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    if z_loss_coef > 0:
        log_z = torch.logsumexp(logits, dim=-1)             # log Z
        loss = loss + z_loss_coef * (log_z ** 2).mean()
    return loss


vocab, d = 1000, 64
embed = TokenEmbedding(vocab, d)
head = LMHead(vocab, d, tied_embedding=embed)
assert head.weight is embed.weight                          # 共享：两处是同一个张量
```

共享权重时，嵌入层和 LM Head 的梯度会累加到同一个参数上：一部分来自输入端（稀疏，只有出现的 token），一部分来自输出端（稠密，所有 token）。

## 8. 小结

- **嵌入层**是一个以 one-hot 为输入的线性层，实现为查表。它把高维稀疏的 id 压成低维稠密向量，让语义相近的 token 在空间中相近。它的梯度是稀疏的，只有出现的 token 被更新。
- **LM Head** 计算隐藏向量与每个 token 的输出向量的内积，得到 logits。交叉熵的梯度是"预测分布减去 one-hot 目标"，所有 token 都收到梯度。
- **乘以 $\sqrt d$** 是为了让嵌入与正弦位置编码的量级匹配；使用 RoPE 的模型大多不再需要它。
- **权重共享**省参数、让输入输出语义一致、让罕见 token 得到稠密的训练信号；代价是两个角色被迫使用同一组向量。小模型多共享，大模型多不共享。
- **大词表**让嵌入和 LM Head 在参数、计算、logits 显存上都成为显著成本；LM Head 在 MoE 模型中也是稠密计算。分块交叉熵、算子融合和词表并行是常见的工程解法。
- **z-loss** 和 **logit soft-capping** 控制输出 logits 的数值范围。
- **MTP 和 draft head** 预测更远的 token，用于辅助训练或投机解码，但不替代标准 LM Head。

下一篇进入位置编码：从原始 Transformer 的正弦编码和 BERT、GPT-2 的可学习编码讲起。

## 参考文献

- Mikolov et al. *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013.
- Press & Wolf. *Using the Output Embedding to Improve Language Models.* EACL 2017.
- Inan, Khosravi & Socher. *Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling.* ICLR 2017.
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019.
- Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.* 2022.
- Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size.* 2024.
- Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* 2024.
- Li et al. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML 2024.
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024.
