---
title: "绝对位置编码：正弦函数与可学习向量"
title_en: "Absolute Positional Encodings: Sinusoidal and Learned"
series: "Transformer 架构"
order: 3
date: 2026-10-16
tags: [Transformer, 位置编码, Positional Encoding]
summary_en: >
  Self-attention is permutation-equivariant, so position must be injected explicitly. This post proves
  that property, lists what a good positional encoding should provide, then analyzes the two absolute
  schemes: the original sinusoidal encoding (its frequencies, the rotation identity that makes relative
  offsets linear, the distance-only and direction-blind dot product) and the learned position embeddings
  of BERT and GPT-2. It expands the attention score into content and position terms, discusses adding
  versus concatenating, and explains why absolute encodings extrapolate poorly.
---

## 0. 为什么需要位置编码

### 0.1 注意力看不到顺序

第 0 篇 2.4 节提到，自注意力是**置换等变**的。这里完整写一遍。设输入 $X\in\mathbb R^{L\times d}$，$P$ 是任意一个 $L\times L$ 置换矩阵（把序列打乱）。打乱后的输入是 $PX$，于是

$$
Q' = PXW_Q = PQ,\qquad K' = PK,\qquad V' = PV
$$

$$
\mathrm{softmax}\!\left(\frac{Q'K'^\top}{\sqrt{d_k}}\right)V' = \mathrm{softmax}\!\left(\frac{PQK^\top P^\top}{\sqrt{d_k}}\right)PV
$$

$PSP^\top$ 是对矩阵 $S$ 的行和列做同一个置换。逐行 softmax 与这种置换可交换：$\mathrm{softmax}(PSP^\top) = P\,\mathrm{softmax}(S)\,P^\top$。再利用 $P^\top P = I$：

$$
= P\,\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)P^\top P V = P\cdot\mathrm{Attention}(Q,K,V)
$$

输出只是跟着输入一起被打乱，每个 token 得到的向量与它在序列中的位置无关。FFN、LayerNorm、残差都是逐位置独立的，也不引入顺序信息。所以**一个没有位置编码的 Transformer 编码器，把 "狗咬人" 和 "人咬狗" 看成同一个词袋。**

（解码器有因果掩码，掩码本身是位置相关的：位置 $i$ 能看到 $i$ 个 token。所以严格说，带因果掩码的模型可以从"能看到多少个 token"间接推断位置。第 7 篇讨论的 NoPE 就利用了这一点。但对编码器和双向注意力，位置编码是必需的。）

### 0.2 好的位置编码应该提供什么

RNN 按顺序逐个处理 token，位置信息天然包含在计算过程中。注意力是全局并行的，必须另外注入。理想的位置编码希望做到：

1. **区分绝对位置**：每个位置有唯一的表示；
2. **表达相对位置**：两个 token 相距多远、谁在前谁在后，最好能被模型直接利用。语言中很多规律是相对的（形容词通常紧挨在名词前面），而不是绝对的（"第 37 个词"本身没有意义）；
3. **对不同长度的序列一致**：同样的相对距离在长句和短句中应该有同样的含义；
4. **能外推**：在比训练时更长的序列上仍然有效。

这一篇讲的两种**绝对位置编码**主要满足第 1 条，对第 2 到 4 条做得都不够好。后面几篇的相对位置偏置、RoPE 及其扩展，就是逐条改进的结果。

## 1. 正弦位置编码

### 1.1 定义

原始 Transformer 为每个位置 $pos$ 生成一个 $d$ 维向量，第 $2i$ 和 $2i+1$ 维分别是

$$
PE_{(pos,\,2i)} = \sin(\omega_i\,pos),\qquad
PE_{(pos,\,2i+1)} = \cos(\omega_i\,pos),\qquad
\omega_i = \frac{1}{10000^{2i/d}}
$$

$i = 0,1,\dots,d/2-1$。位置编码直接**加**到嵌入上：

$$
h^{(0)}_{pos} = \sqrt d\cdot E[\text{token}_{pos}] + PE_{pos}
$$

位置编码的维度与嵌入相同，所以可以相加。它不含任何可学习参数。

### 1.2 多频率：像一个连续的二进制计数器

$d/2$ 对维度各用一个频率 $\omega_i$，从 $\omega_0 = 1$ 几何递减到 $\omega_{d/2-1}\approx 1/10000$。对应的波长 $2\pi/\omega_i$ 从 $2\pi\approx 6.3$ 个位置增长到约 $6.3$ 万个位置。

可以类比二进制计数：

```
位置   第 0 位  第 1 位  第 2 位  第 3 位
 0       0       0       0       0
 1       1       0       0       0
 2       0       1       0       0
 3       1       1       0       0
 4       0       0       1       0
 ...
```

低位翻转得快，高位翻转得慢，所有位合起来唯一确定一个数。正弦编码是这种思想的连续、平滑版本：高频维度（$i$ 小）随位置快速振荡，区分相邻位置；低频维度（$i$ 大）变化缓慢，区分相距很远的位置。

**为什么同时用 sin 和 cos.** 单独一个 $\sin(\omega pos)$ 在一个周期内会有两个位置取到相同的值，无法区分。$(\sin\omega pos,\ \cos\omega pos)$ 是单位圆上的一个点，在一个周期内一一对应。这对正弦余弦也是下一节线性关系成立的关键。

### 1.3 相对位置是一个线性变换

论文选择正弦函数的主要理由是：**对任意固定的偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。** 对第 $i$ 对维度，由和角公式

$$
\begin{aligned}
\sin\omega_i(pos+k) &= \sin\omega_i pos\cos\omega_i k + \cos\omega_i pos\sin\omega_i k \\
\cos\omega_i(pos+k) &= \cos\omega_i pos\cos\omega_i k - \sin\omega_i pos\sin\omega_i k
\end{aligned}
$$

写成矩阵形式：

$$
\begin{pmatrix}\sin\omega_i(pos+k)\\ \cos\omega_i(pos+k)\end{pmatrix}
=
\underbrace{\begin{pmatrix}\cos\omega_i k & \sin\omega_i k\\ -\sin\omega_i k & \cos\omega_i k\end{pmatrix}}_{R_i(k)}
\begin{pmatrix}\sin\omega_i pos\\ \cos\omega_i pos\end{pmatrix}
$$

$R_i(k)$ 是一个旋转矩阵，只依赖偏移 $k$，与 $pos$ 无关。把所有维度对拼起来，$PE_{pos+k} = R(k)\,PE_{pos}$，$R(k)$ 是分块对角的旋转矩阵。

**含义**：从位置 $pos$ 移动 $k$ 步，等于在每个二维子空间里旋转一个固定角度 $\omega_ik$。论文推测这让模型容易学会"关注相对位置 $k$ 处的 token"：只需要学到一个对应 $R(k)$ 的线性变换。这个"旋转"的观察，正是第 5 篇 RoPE 的出发点。

### 1.4 两个位置编码的内积只依赖距离

计算两个位置编码的内积：

$$
PE_p^\top PE_q = \sum_{i=0}^{d/2-1}\big(\sin\omega_ip\sin\omega_iq + \cos\omega_ip\cos\omega_iq\big) = \sum_{i=0}^{d/2-1}\cos\big(\omega_i(p-q)\big)
$$

利用了 $\sin a\sin b + \cos a\cos b = \cos(a-b)$。结果**只依赖距离 $p-q$**，与绝对位置无关。而且当 $|p-q|$ 增大时，各项余弦的相位逐渐错开，总和整体上倾向于减小（但不是单调的，会有波动）。

这看起来很好，但有两个问题：

- **不区分方向**：余弦是偶函数，$\cos(\omega(p-q)) = \cos(\omega(q-p))$。所以 $PE_p^\top PE_q$ 无法区分"$q$ 在 $p$ 前面 3 个位置"和"$q$ 在 $p$ 后面 3 个位置"；
- **进入注意力后被破坏**：注意力里算的不是 $PE_p^\top PE_q$，而是经过投影后的内积，见 1.5 节。

### 1.5 进入注意力之后

位置编码加在嵌入上之后，注意力分数是

$$
\begin{aligned}
s_{pq} &= (x_p + PE_p)\,W_QW_K^\top\,(x_q + PE_q)^\top \\
&= \underbrace{x_pW_QW_K^\top x_q^\top}_{\text{内容-内容}} + \underbrace{x_pW_QW_K^\top PE_q^\top}_{\text{内容-位置}} + \underbrace{PE_pW_QW_K^\top x_q^\top}_{\text{位置-内容}} + \underbrace{PE_pW_QW_K^\top PE_q^\top}_{\text{位置-位置}}
\end{aligned}
$$

（这里把 $x$、$PE$ 写成行向量。）

- **位置-位置**项：中间夹了一个任意矩阵 $W_QW_K^\top$，不再是 1.4 节的纯余弦和，也就不再只依赖 $p-q$。模型要通过学习 $W_Q,W_K$ 才能把它变回相对位置的函数；
- **内容-位置**和**位置-内容**两项：把内容和绝对位置混在一起，引入了与相对位置无关的噪声。

也就是说，正弦编码本身具有良好的相对位置结构，但"加在输入上、再经过任意投影"的用法让这个结构只能被间接利用。这个四项展开是理解后续工作的关键：Transformer-XL 和 T5（第 4 篇）直接改写这几项，RoPE（第 5 篇）则让投影之后的内积严格只依赖相对位置。

### 1.6 已知的缺点

1. **外推性差**：理论上正弦函数对任何位置都有定义，论文也希望它能外推到更长序列。但实践中，当推理长度超过训练长度时，低频维度会出现训练中从未见过的相位组合，模型的表现明显下降。Press et al.（2022，ALiBi）的实验显示，正弦编码的模型在超过训练长度后困惑度迅速上升；
2. **相对位置只能间接利用**：见 1.5 节；
3. **高维度区分度低**：$i$ 越大，频率越低。在常见的序列长度内（几千个位置），最后一些维度的相位几乎不变，相邻位置在这些维度上几乎无法区分。这些维度的作用主要是区分很远的位置，但如果训练长度覆盖不到它们的一个周期，它们也学不到什么有用的东西。

## 2. 可学习的绝对位置编码

### 2.1 做法

BERT 和 GPT-2 换了一个更直接的办法：把位置编码也当作参数学出来。设最大长度为 $L_{\max}$，位置嵌入矩阵为

$$
P\in\mathbb R^{L_{\max}\times d}
$$

第 $pos$ 行就是位置 $pos$ 的编码，与 token 嵌入相加：

$$
h^{(0)}_{pos} = E[\text{token}_{pos}] + P[pos]
$$

BERT 的 $L_{\max} = 512$，GPT-2 为 1024。BERT 还额外加了一个 segment embedding，用来区分句子对中的第一句和第二句。

### 2.2 优缺点

- **优点**：完全数据驱动，不预设任何函数形式。原始 Transformer 论文的消融实验中，可学习编码与正弦编码的效果几乎相同；
- **缺点：完全不能外推**。超过 $L_{\max}$ 的位置根本没有对应的向量。要处理更长的序列，只能扩展 $P$（新的行随机初始化或从已有位置复制），再在长序列上继续训练；
- **靠后的位置学得不好**：训练数据中长序列较少时，大位置编号出现的次数少，对应的位置向量训练得不充分（与第 2 篇中罕见 token 的嵌入是同一个问题）；
- **同样只能间接表达相对位置**：1.5 节的四项展开对可学习编码同样成立。

有研究可视化了 GPT-2 等模型学到的位置嵌入，发现它们自发地呈现出类似正弦曲线的平滑周期结构，说明正弦编码的设计确实符合模型"想要"的形式。

## 3. 加法还是拼接

把位置信息**加**到 token 嵌入上，会不会把两种信息混在一起、造成失真？为什么不**拼接**？

**拼接在线性投影之后等价于加法.** 设拼接 $[x;\,p]$ 后经过一个线性层 $W = \begin{pmatrix}W_1\\ W_2\end{pmatrix}$：

$$
[x;\,p]\,W = xW_1 + pW_2
$$

这就是"先各自投影，再相加"。而 Transformer 的每一层都先对输入做线性投影（$W_Q,W_K,W_V$）。所以把 $p$ 直接加到 $x$ 上，相当于把 $W_2$ 约束为 $W_1$。拼接能多提供的自由度有限，却要付出维度增加的代价。

**加法为什么不会严重失真.** 高维空间中，随机方向几乎相互正交。经验上，训练好的模型中 token 嵌入和位置嵌入大致占据不同的子空间，两者的余弦相似度很低。如果两类信息位于近似正交的子空间中，相加之后仍然可以被后续的线性投影分别读出。可以这样理解：$d$ 维向量的表示能力远大于单个 token 的语义所需，嵌入实际只用了其中一部分方向，位置信息可以放在剩余的方向上。加入位置信息也让同一个 token 在不同位置有不同的表示，这正是我们想要的。

## 4. 实现

```python
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """原始 Transformer：固定的正弦位置编码，加在嵌入上"""
    def __init__(self, d_model, max_len=8192, base=10000.0):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)        # (max_len, 1)
        i = torch.arange(0, d_model, 2, dtype=torch.float)                  # 2i = 0, 2, 4, ...
        omega = base ** (-i / d_model)                                      # ω_i = 1 / base^{2i/d}
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * omega)
        pe[:, 1::2] = torch.cos(pos * omega)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):                                                   # x: (B, L, d)
        return x + self.pe[: x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    """BERT / GPT-2：可学习的位置嵌入，最长 max_len"""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):                                                   # x: (B, L, d)
        L = x.size(1)
        assert L <= self.pe.num_embeddings, "超过最大长度：可学习位置编码无法外推"
        pos = torch.arange(L, device=x.device)
        return x + self.pe(pos)


# 验证 1.4 节：两个正弦位置编码的内积只依赖距离
pe = SinusoidalPositionalEncoding(64).pe
print(torch.allclose(pe[10] @ pe[13], pe[100] @ pe[103]))   # True：距离都是 3
print(torch.allclose(pe[10] @ pe[13], pe[13] @ pe[10]))     # True：不区分方向
```

## 5. 小结

- 自注意力是**置换等变**的，本身看不到顺序，必须显式注入位置信息。
- **正弦位置编码**用一组几何递减的频率编码位置，像一个连续的二进制计数器。偏移 $k$ 等于在每个二维子空间里旋转固定角度，两个位置编码的内积只依赖距离。
- 但它加在输入上，经过 $W_QW_K^\top$ 之后，注意力分数分解为内容-内容、内容-位置、位置-内容、位置-位置四项，相对位置结构只能被间接利用；内积也不区分方向。实践中外推性很差，高维度在常见长度内区分度低。
- **可学习位置编码**（BERT、GPT-2）完全数据驱动，效果与正弦相当，但无法处理超过最大长度的序列，靠后的位置学得不充分。
- **相加**不会严重失真：拼接后的线性投影等价于相加，且位置与内容大致位于不同的子空间。

下一篇讲第一种直接作用在注意力分数上的方案：T5 的相对位置偏置。

## 参考文献

- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Gehring et al. *Convolutional Sequence to Sequence Learning.* ICML 2017.
- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Radford et al. *Language Models are Unsupervised Multitask Learners.* 2019.
- Wang & Chen. *What Do Position Embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding.* EMNLP 2020.
- Ke, He & Liu. *Rethinking Positional Encoding in Language Pre-training.* ICLR 2021.
- Press, Smith & Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022.
