---
title: "Transformer 整体结构：从输入到输出走一遍"
title_en: "The Transformer, End to End"
series: "Transformer 架构"
order: 0
date: 2026-10-16
tags: [Transformer, LLM, 系列导读]
summary_en: >
  A walk through the original encoder-decoder Transformer, following one batch of tokens from the
  tokenizer to the output distribution: embeddings and positional encodings, encoder and decoder blocks,
  the three kinds of attention and their masks, the output projection, training with teacher forcing and
  autoregressive inference. It then counts parameters and FLOPs, compares encoder-only, decoder-only and
  encoder-decoder models, shows what a modern decoder-only LLM block looks like, and maps every component
  to the post in this series that covers it.
---

## 0. 这个系列在讲什么

今天几乎所有大语言模型都建立在 Transformer（Vaswani et al., 2017, *Attention Is All You Need*）之上。但"Transformer"这个词指的已经不是 2017 年那个具体的模型，而是一个**骨架**：分词、嵌入、位置信息、注意力、归一化、前馈网络、残差连接、输出层、解码。过去几年里，几乎每一个部件都被替换或改造过：

- 位置编码从正弦函数变成了 RoPE，又发展出长度扩展、部分旋转、去掉位置编码等做法；
- 注意力从多头注意力变成了 GQA、MLA，又出现了稀疏注意力、线性注意力和二者的混合；
- 归一化从 Post-LN 变成了 Pre-RMSNorm，再加上 QK-Norm；
- 前馈网络从 ReLU 变成了 SwiGLU，再变成 MoE；
- 残差连接从固定的 $x + F(x)$ 开始变成可学习的跨层路由。

这个系列把骨架拆开，**每个部件一篇，每个变体也单独一篇**，讲清楚它解决了什么问题、怎么推导、代价是什么。这一篇是导读：先把原始 Transformer 从输入到输出完整走一遍，建立整体图景，再说明现代大语言模型的一个 block 长什么样，最后给出全系列的篇目地图。

## 1. 原始 Transformer 的整体结构

原始 Transformer 是一个用于机器翻译的 **Encoder-Decoder**（编码器-解码器）模型：输入一句英文，输出一句德文。

```
源序列（英文）                         目标序列（德文，右移一位）
     │                                        │
  分词器                                   分词器
     │                                        │
 Token Embedding × √d_model            Token Embedding × √d_model
     +                                        +
 位置编码                                  位置编码
     │                                        │
 ┌───▼──────────────────┐            ┌───────▼────────────────────┐
 │ Encoder Block × N    │            │ Decoder Block × N          │
 │  多头自注意力         │            │  带因果掩码的多头自注意力     │
 │  Add & Norm          │            │  Add & Norm                │
 │  前馈网络 FFN         │  ───K,V──▶ │  交叉注意力（Q 来自解码器）   │
 │  Add & Norm          │            │  Add & Norm                │
 └──────────────────────┘            │  前馈网络 FFN               │
                                     │  Add & Norm                │
                                     └───────┬────────────────────┘
                                             │
                                    线性层（LM Head）
                                             │
                                          Softmax
                                             │
                                  下一个 token 的概率分布
```

原论文的基础配置（Transformer-base）：

| 超参数 | 记号 | 取值 |
|---|---|---|
| 层数（编码器、解码器各） | $N$ | 6 |
| 隐藏维度 | $d_{\text{model}}$ | 512 |
| 注意力头数 | $h$ | 8 |
| 每个头的维度 | $d_k = d_v = d_{\text{model}}/h$ | 64 |
| 前馈网络中间维度 | $d_{\text{ff}}$ | 2048 |
| Dropout | $P_{\text{drop}}$ | 0.1 |

下面跟着一批数据，从输入走到输出。记批大小为 $B$，序列长度为 $L$，词表大小为 $V$。

## 2. 输入：从文本到向量

### 2.1 分词

模型不能直接处理字符串。**分词器**（tokenizer）把文本切成 token，再把每个 token 映射成词表中的整数 id：

```
"Attention is all you need"  →  ["Att", "ention", " is", " all", " you", " need"]  →  [2103, 1463, 318, 477, 345, 761]
```

现代模型普遍使用**子词**（subword）粒度的分词，比如 BPE：常见词保持完整，生僻词被拆成若干片段。这样既不会因为词表外的词而失败，序列也不会像按字符切分那样过长。原始 Transformer 在英德翻译上使用约 37000 个 token 的共享 BPE 词表。分词的细节见第 1 篇。

输出形状：整数张量 $[B, L]$。

### 2.2 Token Embedding

**嵌入矩阵** $E\in\mathbb R^{V\times d_{\text{model}}}$ 的第 $i$ 行是 id 为 $i$ 的 token 的向量表示。嵌入就是查表：

$$
x_t = E[\text{token}_t]\in\mathbb R^{d_{\text{model}}}
$$

原论文把嵌入乘以 $\sqrt{d_{\text{model}}}$ 再使用。原因是嵌入初始化的尺度较小，而接下来要加上的正弦位置编码每个分量在 $[-1,1]$ 之间，乘以 $\sqrt{d_{\text{model}}}$ 让两者的量级相当，位置信息不至于盖过内容信息。原论文还让编码器嵌入、解码器嵌入和输出层共享同一个矩阵（权重共享）。这些见第 2 篇。

输出形状：$[B, L, d_{\text{model}}]$。

### 2.3 位置编码

注意力本身对输入的顺序不敏感：把序列中的 token 打乱，每个 token 得到的注意力输出只是跟着重排，数值不变（第 8 篇会证明这个"置换等变性"）。所以必须显式注入位置信息。原始 Transformer 使用固定的**正弦位置编码**，直接加到嵌入上：

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
h^{(0)}_t = \sqrt{d_{\text{model}}}\,E[\text{token}_t] + PE_t
$$

位置编码是整个系列中变体最多的部件之一，第 3 到 7 篇专门讨论。

## 3. 编码器

编码器由 $N$ 个相同结构的 block 堆叠而成，每个 block 有两个子层。

### 3.1 多头自注意力

**自注意力**让序列中的每个 token 从所有 token（包括自己）那里收集信息。对输入 $X\in\mathbb R^{L\times d_{\text{model}}}$，先用三个矩阵投影出查询、键、值：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

然后计算**缩放点积注意力**：

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

$QK^\top$ 的第 $(i,j)$ 个元素衡量 token $i$ 有多"关注" token $j$；softmax 按行归一化成权重；再用权重对 $V$ 加权平均。除以 $\sqrt{d_k}$ 是为了让点积的方差不随维度增大，避免 softmax 饱和。

**多头**的意思是把 $d_{\text{model}}$ 维拆成 $h$ 个 $d_k$ 维的子空间，每个头独立计算注意力，最后拼接起来再投影：

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,\qquad
\text{head}_i = \mathrm{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})
$$

不同的头可以学习不同类型的关系（比如句法依存、指代、相邻位置）。编码器的自注意力是**双向**的：每个 token 都能看到整个源序列。唯一需要的掩码是 **padding mask**：批内句子长短不一，补齐用的 padding 位置不应被关注。

注意力的推导、$\sqrt{d_k}$ 的来由和多头的实现见第 8 篇。

### 3.2 前馈网络

注意力只做了"信息交换"：每个 token 的输出是其他 token 的值向量的加权平均，本质上是线性组合。**前馈网络**（FFN）对每个位置**独立地**做一次非线性变换：

$$
\mathrm{FFN}(x) = \max(0,\ xW_1 + b_1)\,W_2 + b_2,\qquad W_1\in\mathbb R^{d_{\text{model}}\times d_{\text{ff}}},\ W_2\in\mathbb R^{d_{\text{ff}}\times d_{\text{model}}}
$$

先升维到 $d_{\text{ff}} = 4d_{\text{model}}$，过 ReLU，再降回来。可以这样理解分工：注意力负责"从别的 token 那里取信息"，FFN 负责"对取到的信息做加工"。FFN 占了模型大部分参数，也被认为是存储事实性知识的主要位置。现代模型中它被替换为 SwiGLU 和 MoE，见第 24 到 27 篇。

### 3.3 残差连接与层归一化

每个子层都包在一个"残差 + 归一化"结构里（图中的 Add & Norm）：

$$
x \leftarrow \mathrm{LayerNorm}\big(x + \mathrm{Sublayer}(x)\big)
$$

- **残差连接** $x + \mathrm{Sublayer}(x)$ 让梯度可以绕过子层直接回传，使深层网络可训练；
- **层归一化**（LayerNorm）把每个 token 的向量归一化到均值 0、方差 1（再乘以可学习的缩放、加上偏移），控制激活值的尺度。

注意顺序是**先加残差，再归一化**，这叫 **Post-LN**。现代模型几乎都改成了 **Pre-LN**：$x\leftarrow x + \mathrm{Sublayer}(\mathrm{Norm}(x))$，训练更稳定。归一化见第 21 到 23 篇，残差见第 28 到 30 篇。

一个编码器 block 的完整计算：

$$
\begin{aligned}
x' &= \mathrm{LayerNorm}\big(x + \mathrm{MultiHeadSelfAttn}(x)\big) \\
x'' &= \mathrm{LayerNorm}\big(x' + \mathrm{FFN}(x')\big)
\end{aligned}
$$

每个 block 的输入输出形状都是 $[B, L, d_{\text{model}}]$，所以可以任意堆叠。$N$ 层之后得到编码器输出 $H_{\text{enc}}\in\mathbb R^{B\times L_{\text{src}}\times d_{\text{model}}}$，它是源序列的上下文表示。

## 4. 解码器

解码器也是 $N$ 个 block，每个 block 有**三个**子层。

### 4.1 带因果掩码的自注意力

解码器要逐个生成目标序列。训练时整个目标序列一次性输入，但位置 $t$ 不能看到位置 $t$ 之后的 token，否则就是"偷看答案"。做法是在 softmax 之前，把注意力分数矩阵的上三角（$j>i$）设为 $-\infty$：

$$
\mathrm{scores}_{ij} =
\begin{cases}
\dfrac{q_i^\top k_j}{\sqrt{d_k}}, & j\le i \\[6pt]
-\infty, & j > i
\end{cases}
$$

softmax 之后这些位置的权重为 0。这个掩码叫**因果掩码**（causal mask）。它让一次前向传播就能同时计算所有位置的预测，而每个位置只依赖它之前的 token。

### 4.2 交叉注意力

第二个子层是**交叉注意力**：查询来自解码器，键和值来自编码器的输出：

$$
Q = X_{\text{dec}}W_Q,\quad K = H_{\text{enc}}W_K,\quad V = H_{\text{enc}}W_V
$$

这是源序列与目标序列交流的唯一通道：解码器在生成每个词时，去源句中"查找"相关的部分。翻译中的词对齐就是在这里学到的。交叉注意力只需要源序列的 padding mask，不需要因果掩码，因为源序列在生成前就已全部已知。

### 4.3 前馈网络

与编码器相同。每个子层同样是 Post-LN 的残差结构。

一个解码器 block 的完整计算：

$$
\begin{aligned}
y' &= \mathrm{LayerNorm}\big(y + \mathrm{MaskedSelfAttn}(y)\big) \\
y'' &= \mathrm{LayerNorm}\big(y' + \mathrm{CrossAttn}(y', H_{\text{enc}})\big) \\
y''' &= \mathrm{LayerNorm}\big(y'' + \mathrm{FFN}(y'')\big)
\end{aligned}
$$

### 4.4 三种注意力的对照

| | Q 来源 | K、V 来源 | 掩码 | 作用 |
|---|---|---|---|---|
| 编码器自注意力 | 源序列 | 源序列 | padding | 理解源句 |
| 解码器自注意力 | 目标序列 | 目标序列 | 因果 + padding | 理解已生成的部分 |
| 交叉注意力 | 目标序列 | 编码器输出 | 源序列 padding | 从源句取信息 |

## 5. 输出：从向量到概率

解码器最后一层的输出 $h_t\in\mathbb R^{d_{\text{model}}}$ 经过一个线性层（**LM Head**）映射到词表大小的 **logits**，再过 softmax：

$$
z_t = h_tW_{\text{out}}^\top,\quad W_{\text{out}}\in\mathbb R^{V\times d_{\text{model}}},\qquad
p(y_{t+1}\mid y_{\le t}, x) = \mathrm{softmax}(z_t)
$$

原论文中 $W_{\text{out}}$ 与嵌入矩阵 $E$ 共享。

### 5.1 训练：Teacher Forcing

训练时，目标序列右移一位作为解码器输入（开头加起始符 `<s>`），模型在每个位置预测下一个 token。损失是所有位置的交叉熵之和：

$$
\mathcal L = -\sum_{t=1}^{L_{\text{tgt}}}\log p_\theta(y_t\mid y_{<t}, x)
$$

无论模型在位置 $t-1$ 预测得对不对，位置 $t$ 的输入都用**真实的** $y_{t-1}$。这叫 **teacher forcing**。加上因果掩码，整个序列的所有位置可以在一次前向传播中并行训练，这是 Transformer 相对 RNN 最大的训练效率优势。

原论文的其他训练细节：

- 优化器 Adam，$\beta_1 = 0.9$，$\beta_2 = 0.98$，$\epsilon = 10^{-9}$；
- 学习率先线性预热再按步数的平方根倒数衰减：$\text{lr} = d_{\text{model}}^{-0.5}\cdot\min\big(\text{step}^{-0.5},\ \text{step}\cdot\text{warmup}^{-1.5}\big)$，warmup 为 4000 步。Post-LN 对预热非常依赖，第 22 篇会解释原因；
- 标签平滑 $\epsilon_{ls} = 0.1$：把目标分布从 one-hot 改成"正确 token 0.9，其余平分 0.1"。这会让困惑度变差，但提升了 BLEU。

### 5.2 推理：自回归生成

推理时没有真实的目标序列，只能**自回归**地生成：

1. 编码器对源序列做一次前向，得到 $H_{\text{enc}}$（只算一次）；
2. 解码器输入 `<s>`，得到第一个 token 的分布，按某种策略选出 $y_1$；
3. 把 $y_1$ 接到输入后面，再预测 $y_2$；
4. 重复，直到生成结束符 `</s>` 或达到最大长度。

每一步只新增一个 token，但朴素实现会把整个前缀重新算一遍。**KV Cache** 把历史 token 的键和值缓存下来，每一步只计算新 token，见第 9 篇。"按某种策略选出"指的是贪心、束搜索、Top-p 采样等解码方法，见第 31 篇。

训练和推理的这种不对称（训练并行、推理串行）是自回归 Transformer 的基本特征，也是推理优化的主要战场。

## 6. 参数量与计算量

### 6.1 参数量

忽略偏置和归一化参数（它们很少），对隐藏维度 $d$、$d_{\text{ff}} = 4d$：

| 部件 | 参数量 |
|---|---|
| 一个多头注意力（$W_Q,W_K,W_V,W_O$） | $4d^2$ |
| 一个 FFN（$W_1, W_2$） | $2\cdot d\cdot 4d = 8d^2$ |
| 编码器 block（自注意力 + FFN） | $12d^2$ |
| 解码器 block（自注意力 + 交叉注意力 + FFN） | $16d^2$ |
| 嵌入（共享） | $Vd$ |

对 Transformer-base（$d = 512$，$N = 6$，$V\approx 37000$）：编码器 $6\times 12\times 512^2\approx 18.9$M，解码器 $6\times 16\times 512^2\approx 25.2$M，嵌入约 $19$M，合计约 $63$M，与论文报告的 65M 吻合。

**多头不增加参数**：$h$ 个头各自的 $d\times d_k$ 投影拼起来正好是一个 $d\times d$ 矩阵。头数只改变计算的划分方式。

对 decoder-only 模型（没有交叉注意力），每层约 $12d^2$ 参数，总参数量近似为

$$
N_{\text{params}}\approx 12\,n_{\text{layer}}\,d^2
$$

例如 GPT-3 175B：$n_{\text{layer}} = 96$，$d = 12288$，$12\times 96\times 12288^2\approx 174$B，几乎完全吻合。这个公式很适合快速估算。注意 FFN 占了三分之二的参数。

### 6.2 计算量

一次矩阵乘法 $[m\times n]\cdot[n\times p]$ 约需 $2mnp$ 次浮点运算（乘加各算一次）。对一个 token 做一次前向传播：

- 所有线性层：每个参数参与一次乘加，约 $2N_{\text{params}}$；
- 注意力分数与加权求和：$QK^\top$ 和 $AV$ 对长度为 $L$ 的上下文各需约 $2Ld$，每层 $4Ld$。

所以每个 token 的前向计算约为

$$
C_{\text{forward}}\approx 2N_{\text{params}} + 4\,n_{\text{layer}}\,L\,d
$$

训练时反向传播约为前向的两倍，所以常用的估算是**每个 token 训练约 $6N_{\text{params}}$ 次浮点运算**（上下文不太长时注意力项可忽略）。

### 6.3 复杂度随序列长度的变化

对长度为 $L$ 的整个序列：

- 线性层（投影和 FFN）：$O(Ld^2)$，与长度**线性**相关；
- 注意力的分数矩阵：$O(L^2d)$，与长度**平方**相关，而且要存储 $L\times L$ 的矩阵。

当 $L$ 小于几倍 $d$ 时线性层占主导；长上下文时注意力的平方项成为瓶颈。这就是第 12 到 20 篇各种注意力变体（稀疏、线性、混合、IO 优化）要解决的问题。

## 7. 三种架构家族

原始 Transformer 是 encoder-decoder，后来分化出三条路线：

| | Encoder-only | Decoder-only | Encoder-Decoder |
|---|---|---|---|
| 代表 | BERT、RoBERTa | GPT 系列、Llama、Qwen、DeepSeek | 原始 Transformer、T5、BART |
| 注意力 | 双向 | 因果（单向） | 编码器双向 + 解码器因果 + 交叉 |
| 预训练目标 | 掩码语言模型（MLM） | 下一个 token 预测 | 去噪 / 序列到序列 |
| 擅长 | 理解：分类、抽取、检索表示 | 生成，以及把一切任务转成生成 | 输入输出差异大的转换：翻译、摘要 |

**为什么今天的大语言模型几乎都是 decoder-only？** 常见的解释有几条：

- **目标统一**：下一个 token 预测可以直接用海量无标注文本训练，所有任务（问答、翻译、代码、推理）都能写成"给定前文，续写"；
- **训练效率**：因果掩码下每个位置都产生一个训练信号，一条序列有 $L$ 个预测目标；MLM 只在被掩码的 15% 位置上算损失；
- **推理友好**：KV Cache 对因果注意力天然成立，生成时每步只需计算一个新 token；双向注意力中新 token 会改变所有旧 token 的表示，无法这样缓存；
- **规模化验证**：GPT-3 之后，decoder-only 在扩展规律（scaling law）和涌现能力上被大量验证，生态和工程优化也都围绕它展开。

这些都是经验性的解释，不是严格的理论结论。encoder-decoder 在输入远长于输出的任务上仍有效率优势，双向注意力的表示能力也仍被用于检索和嵌入模型。

## 8. 现代 decoder-only LLM 的一个 block

把原始 Transformer 和今天主流大语言模型（以 Llama 系列为代表）的一个 block 对照：

| 部件 | 原始 Transformer（2017） | 现代 LLM 常见做法 | 本系列 |
|---|---|---|---|
| 分词 | BPE，约 37K 词表 | 字节级 BPE，10 万到 25 万词表 | 第 1 篇 |
| 嵌入与输出 | 共享权重，乘 $\sqrt{d}$ | 小模型共享、大模型多不共享 | 第 2 篇 |
| 位置编码 | 正弦，加在嵌入上 | RoPE，作用在 Q、K 上；长度扩展；部分层或部分维度不用 | 第 3–7 篇 |
| 注意力 | 多头注意力（MHA） | GQA 或 MLA；长上下文用稀疏 / 线性 / 混合注意力 | 第 8–20 篇 |
| 归一化 | Post-LayerNorm | Pre-RMSNorm，加 QK-Norm | 第 21–23 篇 |
| FFN | ReLU，$d_{\text{ff}} = 4d$ | SwiGLU，或 MoE | 第 24–27 篇 |
| 残差 | $\mathrm{Norm}(x + F(x))$ | $x + F(\mathrm{Norm}(x))$；新模型尝试可学习的跨层连接 | 第 28–30 篇 |
| 偏置项 | 有 | 线性层大多去掉偏置 | — |
| 解码 | 束搜索 | 温度 + Top-p 采样 | 第 31 篇 |

写成公式，一个现代 block 是

$$
\begin{aligned}
h &= x + \mathrm{Attn}\big(\mathrm{RMSNorm}(x)\big) \\
x_{\text{next}} &= h + \mathrm{FFN}_{\text{SwiGLU}}\big(\mathrm{RMSNorm}(h)\big)
\end{aligned}
$$

其中 Attn 内部对 Q、K 施加 RoPE（以及可能的 QK-Norm），用 GQA 共享 K、V 头。所有 block 之后再接一个 RMSNorm 和 LM Head。骨架和 2017 年一样，但几乎每个部件都换了实现。

## 9. 篇目地图

**基础**

| # | 文章 | 核心问题 |
|---|---|---|
| 1 | 分词器：word / char / subword 与 BPE | 文本怎么变成 token？为什么大模型数不清单词里的字母？ |
| 2 | Embedding 与 LM Head | 嵌入和输出层为什么可以共享？大词表的代价在哪里？ |

**位置编码**

| # | 文章 | 核心问题 |
|---|---|---|
| 3 | 正弦与可学习的绝对位置编码 | 为什么用正余弦？绝对位置编码为什么不能外推？ |
| 4 | T5 相对位置偏置 | 如何把相对距离直接加到注意力分数上？ |
| 5 | RoPE 旋转位置编码 | 如何用绝对位置的操作得到只依赖相对位置的内积？ |
| 6 | 长度扩展：PI、调大 RoPE base、YaRN、DCA | 训练 4K 的模型怎么用到 128K？ |
| 7 | Partial RoPE、Decoupled RoPE 与 NoPE | 位置信息是否需要占据所有维度、所有层？ |

**注意力**

| # | 文章 | 核心问题 |
|---|---|---|
| 8 | 缩放点积注意力与多头注意力 | 为什么除以 $\sqrt{d_k}$？自注意力和交叉注意力有何不同？ |
| 9 | KV Cache 与 Prefill / Decode | 为什么缓存 K、V 而不缓存 Q？为什么解码是访存瓶颈？ |
| 10 | MQA 与 GQA | 如何用共享 K、V 头压缩缓存？ |
| 11 | MLA | 如何把 K、V 压成低维隐变量，又不增加计算？ |
| 12 | 滑动窗口与局部 / 全局交错注意力 | 只看附近的 token 够不够？ |
| 13 | NSA | 压缩、选择、滑窗三路如何组合成可训练的稀疏注意力？ |
| 14 | DSA 与 CSA / HCA | 用轻量索引器挑出相关 token；先压缩再稀疏 |
| 15 | 线性注意力 | 去掉 softmax 后，注意力如何变成循环状态更新？ |
| 16 | Gated DeltaNet | 如何在写入新记忆时修正旧记忆，并按需遗忘？ |
| 17 | KDA | 让不同特征通道有不同的遗忘速度 |
| 18 | Lightning Attention | 块内与块间分开计算的线性注意力 |
| 19 | 混合注意力架构 | 线性层与全注意力层按什么比例混合？ |
| 20 | FlashAttention | 不改变结果，如何通过 IO 优化大幅加速注意力？ |

**归一化**

| # | 文章 | 核心问题 |
|---|---|---|
| 21 | LayerNorm、BatchNorm 与 RMSNorm | 为什么 NLP 不用 BatchNorm？去掉均值为什么不影响效果？ |
| 22 | Pre-Norm 与 Post-Norm | 归一化放在哪里，对梯度传播意味着什么？ |
| 23 | QK-Norm | 为什么要单独控制 Q、K 的长度？ |

**FFN 与 MoE**

| # | 文章 | 核心问题 |
|---|---|---|
| 24 | FFN 与激活函数：ReLU、GELU、SwiGLU | FFN 在做什么？门控为什么有效？ |
| 25 | MoE 基础与负载均衡 | 稀疏激活如何扩大参数而不增加计算？专家如何均衡？ |
| 26 | DeepSeekMoE：细粒度专家与共享专家 | 如何让专家更专业化？ |
| 27 | LatentMoE | 在低维空间里做专家计算 |

**残差连接**

| # | 文章 | 核心问题 |
|---|---|---|
| 28 | 残差连接与 Pre-Norm 稀释 | 残差为什么能训练深网络？越深为什么每层贡献越小？ |
| 29 | Hyper-Connections 与 mHC | 把一条残差流扩展成多条，并保持稳定 |
| 30 | Attention Residuals | 沿深度方向做注意力，选择读取哪些层 |

**解码**

| # | 文章 | 核心问题 |
|---|---|---|
| 31 | 采样策略 | 贪心、束搜索、Top-k、Top-p、温度、Best-of-N 各适合什么场景？ |

## 10. 怎么读

- **第一次系统学习**：按顺序读第 1、2、3、5、8、9、10、21、22、24、25、28、31 篇，这是现代大语言模型的主干。
- **关注长上下文与推理效率**：第 6、7、9、10、11、12 到 20 篇。
- **关注前沿架构**：第 7、11、14、16、17、19、27、29、30 篇，它们大多来自 2024 年之后的模型。
- **前置知识**：线性代数（矩阵乘法、内积、秩）、概率（softmax、交叉熵）和基本的神经网络训练（反向传播、梯度）。

每篇会给出推导，以及关键部件的 PyTorch 参考实现。

## 参考文献

- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Radford et al. *Language Models are Unsupervised Multitask Learners.* 2019.
- Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 2020.
- Brown et al. *Language Models are Few-Shot Learners.* NeurIPS 2020.
- Kaplan et al. *Scaling Laws for Neural Language Models.* 2020.
- Touvron et al. *LLaMA: Open and Efficient Foundation Language Models.* 2023.
