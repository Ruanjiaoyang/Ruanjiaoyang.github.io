---
title: "相对位置编码：从 Shaw 到 T5 的位置偏置"
title_en: "Relative Positional Encoding: From Shaw et al. to the T5 Bias"
series: "Transformer 架构"
order: 4
date: 2026-10-16
tags: [Transformer, 位置编码, 相对位置, T5, ALiBi]
summary_en: >
  Instead of adding position to the input, relative schemes inject the offset i - j directly into the
  attention computation. Starting from the four-term expansion of the attention score, this post covers
  Shaw et al.'s relative key embeddings, Transformer-XL's decomposition, and in detail the T5 relative
  position bias: a learned scalar per head and distance bucket, with exact buckets for short distances and
  logarithmic buckets for long ones. It walks through the bucketing function, shows a reference
  implementation, and closes with ALiBi's fixed linear biases and the trade-offs of bias-based schemes.
---

## 0. 从四项展开说起

第 3 篇 1.5 节把绝对位置编码下的注意力分数展开成四项：

$$
s_{ij} = \underbrace{x_iW_QW_K^\top x_j^\top}_{\text{内容-内容}} + \underbrace{x_iW_QW_K^\top p_j^\top}_{\text{内容-位置}} + \underbrace{p_iW_QW_K^\top x_j^\top}_{\text{位置-内容}} + \underbrace{p_iW_QW_K^\top p_j^\top}_{\text{位置-位置}}
$$

问题在于后三项都依赖**绝对位置** $p_i, p_j$，而我们更关心的是**相对位置** $i - j$。既然如此，何不直接改写注意力的计算，让位置只以 $i-j$ 的形式出现？这就是**相对位置编码**的思路：

- 不再把位置向量加到输入上；
- 在计算注意力时，根据 $i - j$ 注入位置信息。

这样做天然满足第 3 篇 0.2 节的第 2、3 条要求：同样的相对距离在任何位置、任何长度的序列中含义相同。这一篇按时间顺序介绍三种做法，重点是 T5 的位置偏置，最后介绍思路相近的 ALiBi。

## 1. Shaw et al.（2018）：相对位置的键向量

最早的相对位置编码（Shaw, Uszkoreit & Vaswani, 2018）为每个相对距离学习一个向量 $a_{ij}^K\in\mathbb R^{d_k}$，加到键上：

$$
s_{ij} = \frac{q_i^\top\big(k_j + a_{ij}^K\big)}{\sqrt{d_k}}
$$

$a_{ij}^K$ 只依赖 $i - j$，并且把距离截断在 $[-k_{\max}, k_{\max}]$ 内：

$$
a_{ij}^K = w^K_{\mathrm{clip}(j-i,\,-k_{\max},\,k_{\max})}
$$

一共只需要学习 $2k_{\max}+1$ 个向量。截断的理由是：精确的相对位置在远距离时用处不大，"很远"就够了；而且截断后可以泛化到训练时没见过的长度。论文也对值向量加了类似的 $a_{ij}^V$，但发现它的作用不大。

缺点是实现开销：$a_{ij}^K$ 对每对 $(i,j)$ 都不同，$q_i^\top a_{ij}^K$ 这一项不能直接写成一次大矩阵乘法，需要专门的计算技巧。

## 2. Transformer-XL（2019）：改写四项展开

Transformer-XL（Dai et al., 2019）直接从四项展开出发，逐项替换：

$$
s_{ij} = \underbrace{x_iW_QW_{K,E}^\top x_j^\top}_{\text{(a) 内容-内容}} + \underbrace{x_iW_QW_{K,R}^\top R_{i-j}^\top}_{\text{(b) 内容-相对位置}} + \underbrace{u\,W_{K,E}^\top x_j^\top}_{\text{(c) 全局内容偏置}} + \underbrace{v\,W_{K,R}^\top R_{i-j}^\top}_{\text{(d) 全局位置偏置}}
$$

- 把键一侧的绝对位置 $p_j$ 换成相对位置的正弦编码 $R_{i-j}$（不可学习）；
- 把查询一侧的绝对位置 $p_i W_Q$ 换成两个可学习的向量 $u, v$：查询的位置不应该影响"我该关注什么内容"，所以用一个与位置无关的全局向量代替；
- 内容和位置用不同的键投影 $W_{K,E}$、$W_{K,R}$。

(a) 是纯内容匹配；(b) 是"这个查询内容偏好多远的距离"；(c) 是"某些键内容无论如何都更重要"；(d) 是"全局上偏好多远的距离"。Transformer-XL 需要这种相对编码，是因为它在段与段之间复用缓存的隐藏状态，绝对位置编码在跨段时会冲突。

这个设计表达力很强，但计算复杂，(b) 项同样需要专门的实现技巧。

## 3. T5 相对位置偏置

T5（Raffel et al., 2020）把上面的思路**简化到极致**：只保留一个与内容无关的**标量偏置**，直接加到注意力分数上。

### 3.1 定义

对第 $h$ 个头：

$$
e_{ij}^{(h)} = \frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_k}} + r_{ij}^{(h)}
$$

$$
\alpha_{ij}^{(h)} = \mathrm{softmax}_j\big(e_{ij}^{(h)}\big)
$$

- $q_i^\top k_j/\sqrt{d_k}$ 是**内容相关性**：内容上应该关注谁；
- $r_{ij}^{(h)}$ 是一个**标量**，只依赖 $i$ 和 $j$ 的相对距离：位置上应该关注谁。

模型不再给输入加任何位置向量。对照四项展开，T5 只保留了 (a) 内容-内容，再加上一个类似 (d) 的纯位置项，而且这个位置项直接用一个可学习的标量表示。

（实际的 T5 实现中，注意力分数没有除以 $\sqrt{d_k}$，这个缩放被合并进了 $W_Q$ 的初始化里。这不影响这里的讨论。）

### 3.2 偏置从哪里来：查表

模型维护一个可学习的偏置表

$$
R\in\mathbb R^{H\times B}
$$

$H$ 是头数，$B$ 是**相对位置桶**（relative position bucket）的数量，T5 取 $B = 32$。对第 $h$ 个头：

$$
r_{ij}^{(h)} = R_{h,\ b(i - j)}
$$

$b(\cdot)$ 把相对距离映射到桶的编号。整条链路是：

$$
i - j\ \longrightarrow\ \text{桶编号 } b(i-j)\ \longrightarrow\ r_{ij} = R_{h,b}\ \longrightarrow\ \frac{q_i^\top k_j}{\sqrt{d_k}} + r_{ij}\ \longrightarrow\ \mathrm{softmax}
$$

$R$ 是训练中学习的参数表，$r_{ij}$ 是根据距离从表中查出来的一个数。每个头有自己的一行，所以不同的头可以学到不同的位置偏好：有的头偏好紧邻的词，有的头偏好较远的词，有的头几乎不在乎距离。

T5 的偏置表只有 $H\times 32$ 个参数，并且**在所有层之间共享**（每层的同一个头使用同一组偏置）。

### 3.3 分桶：近处精细，远处粗糙

T5 不为每个距离单独学一个参数，而是把距离分组。分桶的原则是：**近距离划分得细，远距离划分得粗。** 理由和 Shaw 的截断一样：相邻一两个词的精确距离很重要，而 100 个词和 110 个词的区别几乎没有意义。

以 T5 编码器（双向注意力）为例，设 $n = i - j$（查询位置减去键位置），32 个桶的分配如下：

1. **先按方向分成两半**：$n>0$（键在查询之前）和 $n\le 0$（键在查询之后，包括自己）各用 16 个桶。这让偏置**区分方向**，弥补了正弦编码内积不区分方向的缺陷；
2. **每个方向内，前一半精确**：$|n| = 0, 1, \dots, 7$ 各占一个桶；
3. **后一半按对数划分**：$|n|$ 从 8 到 127 按对数尺度分到剩下的 8 个桶中，距离越大，一个桶覆盖的范围越宽；
4. **超过最大距离的都归入最后一个桶**：$|n|\ge 128$ 共用一个桶。

对数桶的编号计算为

$$
b = 8 + \left\lfloor\frac{\log(|n|/8)}{\log(128/8)}\times 8\right\rfloor,\qquad 8\le|n|<128
$$

$|n| = 8$ 时 $b = 8$，$|n|$ 接近 128 时 $b$ 接近 16，取 $\min(b, 15)$ 保证不越界。大致上，距离 8 到 11 一个桶、11 到 16 一个桶、……、90 到 128 一个桶，每个桶覆盖的距离范围大约是前一个的 $16^{1/8}\approx 1.41$ 倍。

**解码器**（因果注意力）只能看到之前的 token，$n\ge 0$，所以不需要按方向分，32 个桶全部用于一个方向：$0\sim 15$ 精确，$16\sim 127$ 对数，$\ge 128$ 归入最后一个桶。

### 3.4 为什么能处理训练之外的长度

所有 $|n|\ge 128$ 的距离都映射到同一个桶，所以无论序列多长，偏置都有定义，不存在"没见过的位置编号"。模型在训练中已经学会了"很远就是很远"，这个知识在更长的序列上依然适用。这是一个很强的先验，但它让 T5 的偏置可以直接用在比训练更长的序列上。

当然，能"处理"不等于效果好：所有远处的 token 得到同样的位置偏置，模型无法区分 200 和 2000 的距离，远距离的位置分辨率完全依赖内容。

### 3.5 实现

```python
import math
import torch
import torch.nn as nn


def relative_position_bucket(rel_pos, bidirectional=True, num_buckets=32, max_distance=128):
    """
    rel_pos: 键位置 - 查询位置，即 j - i，形状任意的整数张量
    返回: 同形状的桶编号，取值 [0, num_buckets)
    """
    n = -rel_pos                                   # n = i - j，正数表示键在查询之前
    bucket = torch.zeros_like(n)
    if bidirectional:
        num_buckets //= 2                          # 两个方向各一半
        bucket += (n < 0).long() * num_buckets     # 键在查询之后：用后一半的桶
        n = n.abs()
    else:
        n = n.clamp(min=0)                         # 因果：只有 n >= 0
    max_exact = num_buckets // 2
    is_small = n < max_exact                       # 近距离：每个距离一个桶
    # 远距离：对数划分
    n_float = n.float().clamp(min=1)
    large = max_exact + (
        torch.log(n_float / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).long()
    large = large.clamp(max=num_buckets - 1)       # 超过 max_distance 归入最后一个桶
    bucket += torch.where(is_small, n, large)
    return bucket


class T5RelativeBias(nn.Module):
    def __init__(self, n_heads, num_buckets=32, max_distance=128, bidirectional=True):
        super().__init__()
        self.bias = nn.Embedding(num_buckets, n_heads)   # 偏置表 R，转置存储: (B, H)
        self.num_buckets, self.max_distance = num_buckets, max_distance
        self.bidirectional = bidirectional

    def forward(self, Lq, Lk, device=None):
        q_pos = torch.arange(Lq, device=device)[:, None]
        k_pos = torch.arange(Lk, device=device)[None, :]
        buckets = relative_position_bucket(k_pos - q_pos, self.bidirectional,
                                           self.num_buckets, self.max_distance)   # (Lq, Lk)
        return self.bias(buckets).permute(2, 0, 1)        # (H, Lq, Lk)，加到每个头的注意力分数上


# 用法：scores = q @ k.transpose(-2, -1) / sqrt(d_k) + rel_bias(Lq, Lk)[None]
b = relative_position_bucket(torch.tensor([0, -1, -7, -8, -20, -127, -500, 3]))
print(b.tolist())   # [0, 1, 7, 8, 10, 15, 15, 19]：n=-3（键在后 3 位）落在后一半 16+3
```

在一个 T5 模型里，偏置只在第一层计算一次，之后各层复用，所以开销很小。

### 3.6 优缺点

**优点：**

- 位置信息直接作用在注意力分数上，且只依赖相对距离，避免了绝对位置编码中内容与位置的混杂；
- 区分方向；
- 参数极少（$H\times 32$），远距离归入同一个桶，可以处理任意长度；
- 不同的头可以学到不同的距离偏好。

**缺点：**

- **远距离分辨率为零**：128 以外的所有距离被视为相同，对需要精确长距离位置的任务不利；
- **计算效率**：偏置是一个 $L\times L$ 的矩阵（每个头一个），需要显式加到分数矩阵上。这和高度优化的注意力内核（如 FlashAttention，第 20 篇）结合时不如不带偏置的版本高效，长序列时尤其明显；
- **与 KV Cache 的配合**：解码时每一步都要为新的查询位置重新查表计算偏置，虽然不难，但不如 RoPE 那样把位置信息直接写进缓存的键里方便；
- **外推效果一般**：Press et al.（2022）的实验中，T5 偏置在超出训练长度时比正弦编码和 RoPE 外推得好，但速度明显更慢。

## 4. ALiBi：不学习的线性偏置

ALiBi（Attention with Linear Biases, Press et al., 2022）把 T5 的偏置进一步简化成一个**固定的、不需要学习的**线性函数：

$$
e_{ij}^{(h)} = \frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_k}} - m_h\cdot(i - j),\qquad j\le i
$$

距离越远，惩罚越大，注意力分数线性下降。每个头有一个固定的斜率 $m_h$，取一个几何数列。对 $H = 8$ 个头：

$$
m_h = 2^{-h},\qquad h = 1,\dots,8,\quad\text{即}\quad \tfrac12,\ \tfrac14,\ \dots,\ \tfrac1{256}
$$

一般地取 $m_h = 2^{-8h/H}$。斜率大的头只关注附近的 token，斜率小的头可以看得很远，各个头形成从局部到全局的不同"视野"。

ALiBi 不给输入加任何位置向量，没有任何位置参数。它的主要卖点是**外推**："短训练、长测试"（Train Short, Test Long）：在长度 1024 上训练的模型，在 2048 甚至更长的序列上困惑度仍然稳定，效果优于正弦编码、RoPE 和 T5 偏置。直观原因是线性惩罚对任何距离都有定义，而且远处 token 的权重会被持续压低，模型在更长的序列上看到的注意力模式与训练时类似。

ALiBi 被 BLOOM、MPT 等模型采用。但它的"外推"在一定程度上来自对远距离注意力的强烈抑制：很远的 token 几乎得不到注意力，模型实际上更像是在一个有效窗口内工作。对于需要从很长上下文中精确检索信息的任务，这种偏置可能成为限制。后来主流模型大多选择了 RoPE 加长度扩展的路线（第 5、6 篇）。

## 5. 对比

| | 位置信息加在哪里 | 形式 | 参数 | 区分方向 | 外推 |
|---|---|---|---|---|---|
| 正弦（原始 Transformer） | 输入 | 固定向量 | 无 | 内积不区分 | 差 |
| 可学习（BERT、GPT-2） | 输入 | 可学习向量 | $L_{\max}\times d$ | 可学习 | 不能 |
| Shaw et al. | 键（及值） | 可学习向量，截断 | $(2k_{\max}+1)\times d_k$ | 是 | 截断后可以 |
| Transformer-XL | 注意力分数（四项改写） | 正弦 + 可学习 $u,v$ | 少量 | 是 | 一般 |
| T5 偏置 | 注意力分数 | 可学习标量，分桶 | $H\times 32$ | 是 | 一般，较慢 |
| ALiBi | 注意力分数 | 固定线性惩罚 | 无 | 只用于因果 | 好（偏向局部） |

## 6. 小结

- 绝对位置编码的问题在于位置以绝对形式进入四项展开中的三项；**相对位置编码**直接在注意力计算中注入 $i - j$。
- **Shaw et al.** 为每个截断后的相对距离学习一个向量加到键上；**Transformer-XL** 逐项改写四项展开，用相对正弦编码和全局向量 $u, v$ 替换绝对位置。
- **T5** 只保留一个标量偏置：每个头、每个距离桶一个可学习的数，加到注意力分数上。分桶近处精细（0 到 7 或 0 到 15 精确）、远处对数、128 以外归为一桶，区分方向，参数极少，在各层之间共享。
- 核心理解：**$QK^\top$ 决定"内容上应该关注谁"，$r_{ij}$ 调整"位置上应该关注谁"。**
- **ALiBi** 用固定的、按头几何递减的线性惩罚代替可学习偏置，外推性好，但偏向局部注意力。
- 偏置类方法的共同代价是需要一个显式的 $L\times L$ 偏置矩阵，和高效注意力内核结合不够方便。

下一篇的 RoPE 换了一个思路：不改注意力分数，而是旋转 $q$ 和 $k$ 本身，让它们的内积**严格**只依赖相对位置，同时位置信息可以直接写进 KV Cache。

## 参考文献

- Shaw, Uszkoreit & Vaswani. *Self-Attention with Relative Position Representations.* NAACL 2018.
- Dai et al. *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.* ACL 2019.
- Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 2020.
- Press, Smith & Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022.
- BigScience Workshop. *BLOOM: A 176B-Parameter Open-Access Multilingual Language Model.* 2022.
