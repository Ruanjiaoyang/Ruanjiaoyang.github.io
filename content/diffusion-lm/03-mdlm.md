---
title: "MDLM 与 MD4：掩码扩散就是加权的掩码语言模型"
title_en: "MDLM and MD4: Masked Diffusion as a Weighted Masked Language Model"
series: "扩散语言模型"
order: 3
date: 2026-10-16
tags: [扩散语言模型, 掩码扩散, MDLM, MD4]
summary_en: >
  MDLM and MD4 show that plain absorbing-state (masked) diffusion, with the right parameterization and a
  continuous-time objective, is far stronger than earlier results suggested. This post derives the
  interpolating forward process and its posterior, the SUBS parameterization (zero masking probability and
  carry-over unmasking), the Rao-Blackwellized discrete-time bound and its continuous-time limit, which is
  a weighted average of masked-language-modeling losses and is invariant to the noise schedule. It proves
  that SEDD's absorbing loss reduces to exactly the same objective, then covers efficient ancestral
  sampling with caching, semi-autoregressive generation, and MD4's generalized state-dependent schedules.
---

## 0. 三条路线汇合到同一个目标

前两篇里，离散扩散有两种写法：

- **D3PM**：离散时间，网络预测 $x_0$ 的分布，代入真实后验，用 ELBO 训练；
- **SEDD**：连续时间，网络预测概率比值（concrete score），用 score entropy 训练。

两者都在**吸收态**（[MASK]）噪声下效果最好。SEDD 第 5.4 节还留下了一个线索：吸收态下真实比值等于"时间的标量函数 × 与时间无关的条件分布"，所以学比值和学"被 mask 的位置是什么 token"本质上是同一件事。

2024 年年中，三篇工作几乎同时沿着这个线索给出了清晰的答案：

- **MDLM**（Masked Diffusion Language Models，Sahoo et al., NeurIPS 2024）：从 D3PM 出发，设计了一个简单的参数化（SUBS），推出连续时间下极其简洁的损失，并配合现代的训练技巧，把掩码扩散的困惑度推到新的水平；
- **MD4**（Simplified and Generalized Masked Diffusion，Shi et al., NeurIPS 2024）：独立推出了同样的连续时间 ELBO，并把它推广到每个 token 有不同掩码速率的情形；
- **RADD**（Ou et al., 2024）：从 SEDD 出发，证明网络根本不需要时间输入。这是下一篇的内容。

它们共同的结论可以浓缩成一句话：**掩码扩散的训练目标，就是对不同掩码比例下的 BERT 式掩码语言模型损失做加权平均。** 这一篇以 MDLM 为主线完整推导这个结论，然后说明它和 SEDD 的关系，最后介绍 MD4 的推广。

## 1. 记号

MDLM 使用**列向量**和**连续时间** $t\in[0,1]$：

- 每个 token 是 $K$ 个类别之一，用 one-hot **列向量**表示。所有 one-hot 向量的集合记为 $\mathcal V = \{x\in\{0,1\}^K : \sum_{i=1}^K x_i = 1\}$；
- 第 $K$ 个类别是特殊的 [MASK] token，它的 one-hot 向量记为 $m$；
- $\mathrm{Cat}(\cdot\,;\pi)$ 是类别概率为 $\pi\in\Delta^K$ 的分类分布，$\Delta^K$ 是 $K$ 维概率单纯形；
- $\mathbf 1$ 是全 1 向量；$\langle a,b\rangle$ 是内积，$a\odot b$ 是逐元素乘积；
- 干净数据记为 $x$，$t$ 时刻的加噪变量记为 $z_t$；$t=0$ 无噪声，$t=1$ 噪声最强；$s<t$ 表示比 $t$ 更早（噪声更少）的时刻。

**噪声调度** $\alpha_t\in[0,1]$ 是关于 $t$ 严格递减的函数，$\alpha_0\approx 1$，$\alpha_1\approx 0$。它表示 $t$ 时刻保留原始信息的比例。

三篇文章的记号对照如下（都对应"保留原 token 的概率"）：

| | D3PM | SEDD | MDLM |
|---|---|---|---|
| 时间 | 离散 $t=1,\dots,T$ | 连续 $t\in[0,T]$ | 连续 $t\in[0,1]$ |
| 保留概率 | $\bar\alpha_t$ | $e^{-\bar\sigma(t)}$ | $\alpha_t$ |
| 干净数据 / 加噪变量 | $x_0$ / $x_t$ | $x_0$ / $x_t$ | $x$ / $z_t$ |

## 2. 插值离散扩散

MDLM 先定义一个比掩码更一般的前向过程：在干净数据 $x$ 和一个固定的先验分布 $\mathrm{Cat}(\cdot\,;\pi)$ 之间做插值。这是高斯扩散 $x_t = \alpha_t x + \sigma_t\epsilon$ 的离散对应物。

### 2.1 边际分布

$$
q(z_t\mid x) = \mathrm{Cat}\big(z_t;\ \alpha_t x + (1-\alpha_t)\pi\big)
$$

读法：以概率 $\alpha_t$ 保持原样，以概率 $1-\alpha_t$ 从先验 $\pi$ 中重新采样。$\pi$ 取均匀分布时就是均匀扩散，取 $\pi = m$ 时就是掩码扩散。

### 2.2 两个时刻之间的转移

对 $s<t$：

$$
q(z_t\mid z_s) = \mathrm{Cat}\big(z_t;\ \alpha_{t|s}\,z_s + (1-\alpha_{t|s})\pi\big),\qquad \alpha_{t|s} = \frac{\alpha_t}{\alpha_s}
$$

**验证.** 先从 $x$ 到 $z_s$，再从 $z_s$ 到 $z_t$，两步复合后 $z_t$ 的分布应当等于 $q(z_t\mid x)$。把 $z_s$ 的分布 $\alpha_s x + (1-\alpha_s)\pi$ 代入转移（转移对 $z_s$ 是线性的）：

$$
\alpha_{t|s}\big(\alpha_s x + (1-\alpha_s)\pi\big) + (1-\alpha_{t|s})\pi
= \alpha_t x + \big(\alpha_{t|s} - \alpha_t + 1 - \alpha_{t|s}\big)\pi
= \alpha_t x + (1-\alpha_t)\pi
$$

第一个等号用了 $\alpha_{t|s}\alpha_s = \alpha_t$，并且 $\pi$ 是一个分布，"从 $\pi$ 重新采样"之后再插值仍然是 $\pi$。所以 $\alpha_{t|s}=\alpha_t/\alpha_s$ 正是让两步复合一致的选择。读法：从 $s$ 到 $t$，有 $1-\alpha_{t|s}$ 比例的概率质量被转移到先验 $\pi$ 上。

### 2.3 后验

和 DDPM、D3PM 一样，ELBO 中需要真实后验 $q(z_s\mid z_t,x)$。由贝叶斯公式和马尔可夫性：

$$
q(z_s\mid z_t, x) = \frac{q(z_t\mid z_s)\,q(z_s\mid x)}{q(z_t\mid x)}
$$

逐项写出（把每一项都看作关于 $z_s$ 的 $K$ 维向量）：

- $q(z_t\mid z_s)$ 作为 $z_s$ 的函数：$\mathrm{Cat}(z_t;\alpha_{t|s}z_s + (1-\alpha_{t|s})\pi) = \alpha_{t|s}\langle z_t, z_s\rangle + (1-\alpha_{t|s})\langle z_t,\pi\rangle$。对所有可能的 $z_s$ 排成向量，就是 $\alpha_{t|s}z_t + (1-\alpha_{t|s})\mathbf 1\pi^\top z_t$；
- $q(z_s\mid x)$ 作为 $z_s$ 的函数：向量 $\alpha_s x + (1-\alpha_s)\pi$；
- 分母 $q(z_t\mid x) = \alpha_t z_t^\top x + (1-\alpha_t)z_t^\top\pi$，是个标量。

于是

$$
q(z_s\mid z_t, x) = \mathrm{Cat}\left(z_s;\ \frac{\big[\alpha_{t|s}z_t + (1-\alpha_{t|s})\mathbf 1\pi^\top z_t\big]\odot\big[\alpha_s x + (1-\alpha_s)\pi\big]}{\alpha_t z_t^\top x + (1-\alpha_t)z_t^\top\pi}\right)
$$

这就是 D3PM 后验公式 $\mathrm{Cat}\big(x_tQ_t^\top\odot x_0\bar Q_{t-1}/(x_0\bar Q_t x_t^\top)\big)$ 在插值型转移矩阵下的具体形式。一般情况下它不算简单，但对掩码扩散可以大幅化简。

## 3. 掩码扩散

### 3.1 前向掩码过程

取 $\pi = m$：

$$
q(z_t\mid x) = \mathrm{Cat}\big(z_t;\ \alpha_t x + (1-\alpha_t)m\big)
$$

每个 token 以概率 $\alpha_t$ 保持原样，以概率 $1-\alpha_t$ 变成 [MASK]。一旦变成 [MASK]，在之后的所有时刻都保持 [MASK]（吸收态）：$q(z_t\mid z_s = m) = \mathrm{Cat}(z_t;\alpha_{t|s}m + (1-\alpha_{t|s})m) = \mathrm{Cat}(z_t; m)$。

### 3.2 化简后验

把 $\pi=m$ 代入 2.3 节的公式，按 $z_t$ 是否为 [MASK] 分两种情况。

**情况一：$z_t\ne m$。** 此时 $z_t$ 一定等于 $x$（[MASK] 不会变回普通 token，普通 token 也不会变成其他普通 token）。$\pi^\top z_t = m^\top z_t = 0$，第一个方括号是 $\alpha_{t|s}z_t$，只在 $z_t$ 那一位非零。所以 $z_s$ 必然等于 $z_t$：

$$
q(z_s\mid z_t, x) = \mathrm{Cat}(z_s;\ z_t)
$$

**情况二：$z_t = m$。** 第一个方括号是 $\alpha_{t|s}m + (1-\alpha_{t|s})\mathbf 1$：在 $m$ 位为 1，在其他位为 $1-\alpha_{t|s}$。与第二个方括号 $\alpha_s x + (1-\alpha_s)m$ 逐元素相乘：

- $m$ 位：$1\cdot(1-\alpha_s) = 1-\alpha_s$；
- $x$ 位：$(1-\alpha_{t|s})\alpha_s = \alpha_s - \alpha_t$；
- 其他位：0。

分母是 $\alpha_t\langle m,x\rangle + (1-\alpha_t) = 1-\alpha_t$（$x$ 是干净 token，$\langle m,x\rangle=0$）。所以

$$
q(z_s\mid z_t, x) =
\begin{cases}
\mathrm{Cat}(z_s;\ z_t), & z_t\ne m \\[6pt]
\mathrm{Cat}\left(z_s;\ \dfrac{(1-\alpha_s)\,m + (\alpha_s - \alpha_t)\,x}{1-\alpha_t}\right), & z_t = m
\end{cases}
$$

**读法.** 如果 $t$ 时刻没被 mask，那么更早的 $s$ 时刻也一定没被 mask，而且就是它本身。如果 $t$ 时刻被 mask 了，那么在 $s$ 时刻：

- 以概率 $\dfrac{\alpha_s - \alpha_t}{1-\alpha_t}$ 还没被 mask，是原始 token $x$（它恰好在 $(s,t]$ 之间被 mask）；
- 以概率 $\dfrac{1-\alpha_s}{1-\alpha_t}$ 已经是 [MASK]（它在 $s$ 之前就被 mask 了）。

两个概率之和为 1。直观上，"在 $(s,t]$ 内被 mask"占"在 $t$ 之前被 mask"的比例，正是 $\frac{\alpha_s-\alpha_t}{1-\alpha_t}$。

## 4. SUBS 参数化

### 4.1 用网络代替 $x$

反向过程需要 $p_\theta(z_s\mid z_t)$。和 D3PM 的 $x_0$ 参数化一样，引入去噪网络

$$
x_\theta(z_t, t):\ \mathcal V\times[0,1]\to\Delta^K
$$

输出对干净 token 的预测分布，然后把它代入真实后验：

$$
p_\theta(z_s\mid z_t) = q\big(z_s\mid z_t,\ x = x_\theta(z_t,t)\big)
$$

### 4.2 两个"替换"

MDLM 的参数化叫 **SUBS**（substitution），因为它在网络输出上做了两处替换，把真实后验的已知结构直接写进模型。

**1. 零掩码概率（Zero Masking Probabilities）.** 干净数据里不会出现 [MASK]，$\langle x, m\rangle = 0$。所以网络也不应该把概率分给 [MASK]：把 [MASK] 对应的 logit 设为 $-\infty$，保证

$$
\langle x_\theta(z_t,t),\ m\rangle = 0
$$

**2. 保留已揭开的 token（Carry-Over Unmasking）.** 由情况一，未被 mask 的 token 在反向过程中不会改变。所以如果 $z_t\ne m$，网络直接复制输入：

$$
x_\theta(z_t, t) = z_t,\qquad z_t\ne m
$$

两处替换之后，反向过程为

$$
p_\theta(z_s\mid z_t) =
\begin{cases}
\mathrm{Cat}(z_s;\ z_t), & z_t\ne m \\[6pt]
\mathrm{Cat}\left(z_s;\ \dfrac{(1-\alpha_s)\,m + (\alpha_s-\alpha_t)\,x_\theta(z_t,t)}{1-\alpha_t}\right), & z_t = m
\end{cases}
$$

**为什么这样做有用.** 不加这两处替换，$p_\theta$ 仍然是一个合法的反向过程，但它要靠训练去"学会"两件我们本来就知道的事：不要输出 [MASK]、不要改已经揭开的 token。D3PM 的做法相当于没有这两处替换，而下一节会看到，加上替换之后 ELBO 中的 KL 项可以**解析地**化成一个非常简单的形式。

## 5. Rao-Blackwellized 似然界

### 5.1 离散时间的 ELBO

把 $[0,1]$ 分成 $T$ 段，记 $t(i) = i/T$，$s(i) = (i-1)/T$。和 DDPM 一样，负 ELBO 分成三项：

$$
\mathcal L_{\text{NELBO}} = \underbrace{\mathbb E_q\big[-\log p_\theta(x\mid z_{t(0)})\big]}_{\mathcal L_{\text{recons}}} + \underbrace{\mathcal L_{\text{diffusion}}}_{\text{逐步 KL}} + \underbrace{D_{\mathrm{KL}}\big(q(z_{t(T)}\mid x)\,\|\,p_\theta(z_{t(T)})\big)}_{\mathcal L_{\text{prior}}}
$$

当 $\alpha_0 = 1$ 时 $z_0 = x$，重构项为 0；当 $\alpha_1 = 0$ 时 $z_1$ 一定是 [MASK]，与先验 $p_\theta(z_1) = \mathrm{Cat}(\cdot\,;m)$ 完全一致，先验项也为 0。所以只剩中间的扩散项

$$
\mathcal L_{\text{diffusion}} = \sum_{i=1}^{T}\mathbb E_{q(z_{t(i)}\mid x)}\Big[D_{\mathrm{KL}}\big(q(z_{s(i)}\mid z_{t(i)},x)\,\big\|\,p_\theta(z_{s(i)}\mid z_{t(i)})\big)\Big]
$$

### 5.2 每一项 KL 的闭式解

固定一步 $s<t$，按 $z_t$ 分情况。

**$z_t\ne m$：** 真实后验和 $p_\theta$ 都是 $\mathrm{Cat}(z_s; z_t)$，KL 为 0。

**$z_t = m$：** 两个分布只在 $m$ 和"非 mask token"上有质量：

| $z_s$ | 真实后验 $q$ | 模型 $p_\theta$ |
|---|---|---|
| $m$ | $\frac{1-\alpha_s}{1-\alpha_t}$ | $\frac{1-\alpha_s}{1-\alpha_t}$ |
| $x$（真实 token） | $\frac{\alpha_s-\alpha_t}{1-\alpha_t}$ | $\frac{\alpha_s-\alpha_t}{1-\alpha_t}\langle x_\theta, x\rangle$ |
| 其他 token $v$ | 0 | $\frac{\alpha_s-\alpha_t}{1-\alpha_t}\langle x_\theta, v\rangle$ |

$m$ 那一行两边相等（这是零掩码概率带来的），KL 中这一项为 $\log 1 = 0$。真实后验在"其他 token"上为 0，不贡献 KL。只剩 $x$ 那一行：

$$
D_{\mathrm{KL}} = \frac{\alpha_s-\alpha_t}{1-\alpha_t}\log\frac{1}{\langle x_\theta(z_t,t),\ x\rangle} = -\frac{\alpha_s-\alpha_t}{1-\alpha_t}\log\langle x_\theta(z_t,t),\ x\rangle
$$

两种情况可以合并成一个式子：$z_t\ne m$ 时，由保留已揭开 token，$x_\theta(z_t,t) = z_t = x$，$\log\langle x_\theta, x\rangle = \log 1 = 0$，自动为 0。因此

$$
\mathcal L_{\text{diffusion}} = \sum_{i=1}^T\mathbb E_q\left[\frac{\alpha_{t(i)} - \alpha_{s(i)}}{1-\alpha_{t(i)}}\log\big\langle x_\theta(z_{t(i)}, t(i)),\ x\big\rangle\right]
$$

注意 $\alpha_t<\alpha_s$，所以系数为负，乘上 $\log\langle\cdot\rangle\le 0$ 得到非负的损失。**每一项就是被 mask 位置上的交叉熵，乘一个只依赖时间的权重。**

### 5.3 不做替换会怎样

作为对比，如果网络可以把概率分给 [MASK]（即没有零掩码概率这一处替换），记 $\mu = \langle x_\theta, m\rangle$。代入 2.3 节的一般后验，模型在 $z_t = m$ 时给出：

- $z_s = m$ 的概率：$\dfrac{\alpha_s\mu + 1-\alpha_s}{\alpha_t\mu + 1-\alpha_t}$；
- $z_s = x$ 的概率：$\dfrac{(\alpha_s-\alpha_t)\langle x_\theta, x\rangle}{\alpha_t\mu + 1-\alpha_t}$。

与真实后验做 KL，得到

$$
\frac{\alpha_s-\alpha_t}{1-\alpha_t}\log\frac{\alpha_t\mu + (1-\alpha_t)}{(1-\alpha_t)\langle x_\theta, x\rangle}
+ \frac{1-\alpha_s}{1-\alpha_t}\log\frac{(1-\alpha_s)\big(\alpha_t\mu + (1-\alpha_t)\big)}{(1-\alpha_t)\big(\alpha_s\mu + (1-\alpha_s)\big)}
$$

这个式子更复杂，含有比值的对数，数值上也不稳定（$\alpha$ 接近 0 或 1 时容易溢出）。当 $\mu = 0$ 时，第二项的对数里分子分母相等，第一项的 $\alpha_t\mu$ 消失，就退化为 5.2 节的结果。

### 5.4 为什么叫 Rao-Blackwellization

Rao-Blackwell 定理说的是：对一个估计量，把能解析算出的部分先求条件期望，得到的新估计量方差不会更大。这里的对应关系是：

- 原始 ELBO 里的一部分随机性（模型在 [MASK] 上的概率、已揭开 token 是否被改动）在 SUBS 下可以**解析地**确定；
- 把这部分"先算掉"，损失只剩被 mask 位置的交叉熵，蒙特卡洛估计的方差更小，界也更紧。

所以 MDLM 把这个化简后的界称为 Rao-Blackwellized 似然界。

## 6. 连续时间极限

### 6.1 取 $T\to\infty$

记 $\Delta = 1/T$，$s = t - \Delta$。当 $\Delta\to 0$ 时

$$
\frac{\alpha_t - \alpha_s}{1-\alpha_t} = \frac{\alpha_t - \alpha_{t-\Delta}}{1-\alpha_t}\approx\frac{\alpha_t'\,\Delta}{1-\alpha_t}
$$

求和 $\sum_i(\cdot)\Delta$ 变成积分，得到**连续时间的 NELBO**：

$$
\mathcal L^\infty_{\text{NELBO}} = \mathbb E_q\int_0^1\frac{\alpha_t'}{1-\alpha_t}\log\big\langle x_\theta(z_t,t),\ x\big\rangle\,dt
$$

$\alpha_t'<0$，$\log\langle\cdot\rangle\le 0$，所以被积函数非负。论文还说明，对于掩码扩散，离散时间的界随 $T$ 增大而变紧，因此连续时间的版本是其中最紧的。这和 VDM 对高斯扩散得出的结论一致。

### 6.2 对噪声调度不变

做变量代换 $\gamma = \log(1-\alpha_t)$。则

$$
d\gamma = \frac{-\alpha_t'}{1-\alpha_t}\,dt
$$

$t$ 从 0 到 1 时，$\gamma$ 从 $\log(1-\alpha_0) = -\infty$ 增加到 $\log(1-\alpha_1) = 0$。于是

$$
\mathcal L^\infty_{\text{NELBO}} = -\,\mathbb E_q\int_{-\infty}^{0}\log\big\langle x_\theta(z_\gamma,\gamma),\ x\big\rangle\,d\gamma
$$

$z_\gamma$ 的分布只依赖 $\gamma$（每个 token 以概率 $1-\alpha = e^\gamma$ 被 mask）。只要 $\alpha_0 = 1$、$\alpha_1 = 0$，**损失的值就和 $\alpha_t$ 的具体函数形式无关**。不同调度只是在 $\gamma$ 轴上重新分配采样点，会影响蒙特卡洛估计的方差和训练动态，但不影响优化的目标本身。这对应连续高斯扩散里"ELBO 只依赖 SNR 两端的值"的结论。

### 6.3 线性调度下的样子

最常用的是线性调度 $\alpha_t = 1-t$。此时 $\alpha_t' = -1$，$1-\alpha_t = t$：

$$
\mathcal L^\infty_{\text{NELBO}} = \mathbb E_{t\sim\mathcal U[0,1]}\,\mathbb E_{z_t}\left[\frac1t\,\big(-\log\langle x_\theta(z_t,t),\ x\rangle\big)\right]
$$

$t$ 正好是每个 token 被 mask 的概率。所以：**随机选一个掩码比例 $t$，按这个比例随机 mask，在被 mask 的位置上算交叉熵，再乘以 $1/t$。**

这里的 $1/t$ 和前两篇得到的权重完全一致：D3PM 在互信息调度下的吸收态损失是按 $1/t$ 加权的 MLM，SEDD 在 log-linear 调度下 log 项的权重也是 $1/t$。三条路线到了这里汇合。

$1/t$ 的直觉：掩码比例为 $t$ 时，期望有 $tL$ 个位置被 mask，对它们的交叉熵求和再除以 $t$，相当于把"每个被 mask 位置的平均损失"乘以序列长度 $L$。所以损失在不同掩码比例之间是"按位置平均"的，不会因为 mask 得多就自动占更大比重。

## 7. 掩码扩散语言模型

### 7.1 推广到序列

对长度为 $L$ 的序列 $x^{1:L}$，做两个假设：

1. 前向加噪在各位置之间**独立**；
2. 反向过程在给定整个 $z_t^{1:L}$ 的条件下，在各位置之间**独立分解**：$p_\theta(z_s^{1:L}\mid z_t^{1:L}) = \prod_l p_\theta(z_s^l\mid z_t^{1:L})$。

注意第二条中每个位置的分布都以**整个序列** $z_t^{1:L}$ 为条件，网络 $x_\theta^l(z_t^{1:L},t)$ 是一个双向的 Transformer，第 $l$ 个输出是对位置 $l$ 的预测。在这两个假设下，序列的 NELBO 就是各位置之和：

$$
\mathcal L^\infty_{\text{NELBO}} = \mathbb E_q\int_0^1\frac{\alpha_t'}{1-\alpha_t}\sum_{l=1}^L\log\big\langle x_\theta^l(z_t^{1:L},t),\ x^l\big\rangle\,dt
$$

由保留已揭开 token，求和中未被 mask 的位置贡献为 0，实际只在被 mask 的位置上计算。

### 7.2 与 BERT 的联系

BERT 的 MLM 损失是在固定掩码比例（15%）下，对被 mask 位置算交叉熵。MDLM 的损失是**对所有掩码比例做加权平均的 MLM 损失**。区别只有两点：掩码比例是随机的、覆盖从 0 到 1 的整个范围；不同比例按 $\frac{-\alpha_t'}{1-\alpha_t}$ 加权。

这建立了扩散模型与 encoder-only 模型之间的直接联系：**一个用这种方式训练的 BERT，就是一个合法的生成模型**，可以计算似然上界，也可以从全 [MASK] 开始采样。

### 7.3 训练上的关键细节

MDLM 强调，掩码扩散过去表现不佳，有相当一部分原因在工程上而不是方法上：

1. **分词**：使用大词表（如 GPT-2 或 BERT 的 3 万到 5 万词表），而不是小词表（如 8k）。论文发现这对捕捉长距离依赖很重要，也是先前一些离散扩散基线吃亏的地方。
2. **数值稳定性**：直接实现 5.2 节的简化损失，只在被 mask 的位置上算交叉熵，而不是像 D3PM 那样具体化转移矩阵、在概率空间里做乘除。
3. **网络结构**：使用带旋转位置编码（RoPE）的 Diffusion Transformer（DiT），用自适应 LayerNorm 注入时间，取代 D3PM 中的 T5 结构。
4. **低差异采样器**：一个批次里如果独立采样 $t$，可能恰好都集中在某一段，导致 ELBO 估计方差很大。低差异采样把 $[0,1]$ 分成 $B$ 段（$B$ 为批大小），第 $i$ 个样本取 $t_i = \frac{i-1+u}{B}$，其中所有样本共用一个 $u\sim\mathcal U[0,1]$。这样每个 $t_i$ 的边缘仍是均匀分布（无偏），但一个批次总能覆盖整个时间区间，方差显著降低。

### 7.4 训练流程

1. 采样一个批次的序列 $x^{1:L}$，用低差异采样得到每个样本的 $t$；
2. 每个位置独立地以概率 $1-\alpha_t$ 替换成 [MASK]，得到 $z_t^{1:L}$；
3. 网络前向，得到每个位置的 logits，把 [MASK] 的 logit 设为 $-\infty$；
4. 在被 mask 的位置上计算交叉熵 $-\log\langle x_\theta^l, x^l\rangle$，乘以 $\frac{-\alpha_t'}{1-\alpha_t}$（线性调度下为 $1/t$），求和并反向传播。

## 8. MDLM 与 SEDD 吸收态是同一个目标

SEDD 学习比值 $s_\theta$，MDLM 学习去噪分布 $x_\theta$。SEDD 第 5.4 节推出，吸收态下被 mask 的位置 $i$ 上真实比值为

$$
\frac{p_t(\text{第 }i\text{ 位换成 }v)}{p_t(x_t)} = \frac{\alpha_t}{1-\alpha_t}\,p_0\big(x^i = v\mid x_t^{\text{UM}}\big)
$$

这里把 SEDD 的 $e^{-\bar\sigma(t)}$ 写成了 $\alpha_t$。这提示一种参数化：

$$
s_\theta(x_t,t)_{i,v} = \frac{\alpha_t}{1-\alpha_t}\,x_\theta^i(x_t,t)_v
$$

即用一个输出概率分布的网络 $x_\theta$ 乘以已知的标量来表示比值。把它代入 SEDD 的吸收态损失（SEDD 一文 5.4 节，这次保留常数项 $K$）。记 $r = \frac{\alpha_t}{1-\alpha_t}$，对一个被 mask 的位置，括号内为

$$
\sum_{v}s_v - r\log s_{x^i} + K(r)
= r\sum_v x_{\theta,v} - r\log r - r\log x_{\theta,x^i} + r\log r - r
$$

由于 $x_\theta$ 是概率分布，$\sum_v x_{\theta,v} = 1$，第一项为 $r$，与最后的 $-r$ 抵消；$-r\log r$ 与 $+r\log r$ 抵消。只剩

$$
-r\log x_\theta^i(x_t,t)_{x^i}
$$

前面还有 SEDD 的权重 $\sigma(t)$。由 $\alpha_t = e^{-\bar\sigma(t)}$ 得 $\alpha_t' = -\sigma(t)\alpha_t$，于是

$$
\sigma(t)\cdot r = \sigma(t)\frac{\alpha_t}{1-\alpha_t} = \frac{-\alpha_t'}{1-\alpha_t}
$$

这正是 MDLM 的权重。**所以在这个参数化下，SEDD 吸收态的 $\mathcal L_{\text{DWDSE}}$ 与 MDLM 的连续时间 NELBO 逐项相等，连常数都一致。** 两者给出的是同一个似然上界。

两者的区别只在于参数化：SEDD 直接输出（对数）比值，没有"$\sum_v$ 为常数"这个约束，网络要自己学会这个结构；MDLM 输出一个归一化的分布，结构由 softmax 自动保证。这也部分解释了 MDLM 在同等设置下略好于 SEDD。

## 9. 采样

### 9.1 祖先采样

生成长度为 $L$ 的序列：

1. 初始化 $z_1^l = m$，对所有 $l$；
2. 把 $[0,1]$ 分成 $T$ 步。在每一步，给定 $z_t^{1:L}$，对每个位置独立地从 $p_\theta(z_s^l\mid z_t^{1:L})$ 采样：
   - 已揭开的位置保持不变；
   - 被 mask 的位置以概率 $\frac{\alpha_s - \alpha_t}{1-\alpha_t}$ 揭开，揭开时从 $x_\theta^l(z_t^{1:L}, t)$ 中采样一个 token；否则保持 [MASK]。
3. 到 $t=0$ 时所有位置都已揭开。

这个过程和 SEDD 的 Tweedie $\tau$-leaping 在吸收态下完全相同（见 SEDD 一文 6.4 节）。

### 9.2 缓存：没有新 token 被揭开就不用重算

保留已揭开 token 带来一个实用的加速。如果在某一步里**没有任何位置被揭开**，那么 $z_s^{1:L} = z_t^{1:L}$，序列没有变化。$T$ 很大时这种情况很常见：每一步揭开的期望数量是 $\frac{\alpha_s-\alpha_t}{1-\alpha_t}\times(\text{剩余 mask 数})$，步长很小时大多是 0。

如果去噪网络**不以时间 $t$ 为条件**，那么输入不变时输出也不变，可以直接复用上一步缓存的 $x_\theta(z_t^{1:L})$，跳过这一步的网络调用。这样网络调用次数不超过"序列发生变化的步数"，最多 $L$ 次，与 $T$ 无关。$T$ 可以取得很大（离散化误差很小），而计算量几乎不增加。

这要求网络不依赖时间。下一篇 RADD 会证明，掩码扩散的最优去噪器本来就与 $t$ 无关（这在第 8 节的比值分解中已经可以看出：$p_0(x^i=v\mid x_t^{\text{UM}})$ 里没有 $t$），所以去掉时间条件不损失表达能力。相比之下，SEDD 直接参数化的比值带有 $\frac{\alpha_t}{1-\alpha_t}$ 这个随时间变化的因子，网络必须以时间为条件，因此不能这样缓存。

### 9.3 半自回归生成

训练时的上下文长度是固定的 $L$。要生成任意长度的文本，MDLM 用了一种**半自回归**（SAR）方法：

1. 先用上面的过程生成一段长度为 $L$ 的序列 $\tilde x^{1:L}$；
2. 取它的后 $L-L'$ 个 token，$\tilde x^{L'+1:L}$，作为下一轮的**前缀**，放在新序列的位置 $1$ 到 $L-L'$；位置 $L-L'+1$ 到 $L$ 初始化为 [MASK]；
3. 运行反向过程。由保留已揭开 token，前缀位置自动保持不变，只有后 $L'$ 个位置被生成；
4. 把新生成的 $L'$ 个 token 接到之前的序列后面，得到长度 $L+L'$ 的序列；
5. 重复，每次把窗口向后移动 $L'$。

块内仍然是并行的扩散生成，块与块之间是自回归的。这个思路后来在"块扩散"（Block Diffusion）等工作中被系统化，成为扩散语言模型支持变长生成和 KV 缓存的常用方案。

## 10. MD4：同一个 ELBO 的另一种写法与推广

MD4（Shi et al., 2024）与 MDLM 同期独立完成，结论高度一致，侧重点有所不同。

### 10.1 相同的连续时间 ELBO

MD4 从连续时间马尔可夫链出发，同样推出了掩码扩散的 ELBO 是带权重的交叉熵积分：

$$
\mathcal L_\infty = \int_0^1\frac{\alpha_t'}{1-\alpha_t}\,\mathbb E_{q(z_t\mid x)}\left[\sum_{l:\,z_t^l = m}(x^l)^\top\log\mu_\theta^l(z_t,t)\right]dt
$$

$\mu_\theta$ 对应 MDLM 的 $x_\theta$。MD4 的几个补充观点：

- 它强调这个 ELBO 可以写成**对掩码比例的积分**，并给出了与 SEDD 的 score entropy、与先前连续时间离散扩散的损失之间的对应关系，指出先前一些工作的损失之所以表现较差，部分原因是参数化和数值实现；
- 它同样指出，在 $\alpha_0=1,\alpha_1=0$ 的前提下，ELBO 对调度不变，调度的选择影响的是方差和样本质量；
- 它讨论了前向过程的时间反转，给出了与 MDLM 相同的反向转移和采样过程。

### 10.2 GenMD4：按 token 设定掩码速率

MD4 的主要推广是**状态相关的掩码调度**。在 MDLM 里所有 token 的掩码概率都是同一个 $1-\alpha_t$。但不同 token 的"难度"不同：功能词容易从上下文推出，内容词更难。直觉上，可以让容易的 token 晚一点被 mask（反向时早一点被揭开），让难的 token 早一点被 mask。

GenMD4 让每个 token 类别 $v$ 有自己的调度 $\alpha_{t,v}$，例如多项式形式

$$
\alpha_{t,v} = 1 - t^{w_v}
$$

其中 $w_v>0$ 是可学习的参数。前向过程中，干净 token $x^l = v$ 在 $t$ 时刻以概率 $1-\alpha_{t,v}$ 被 mask。ELBO 相应地变成每个 token 用自己的权重 $\frac{-\alpha_{t,v}'}{1-\alpha_{t,v}}$。

因为 $w_v$ 决定了哪些位置被 mask，而 mask 是离散采样，ELBO 对 $w_v$ 的梯度不能直接反向传播。MD4 用 REINFORCE 估计这部分梯度，并用 leave-one-out 基线降低方差。

论文报告 GenMD4 在困惑度上比 MD4 有进一步改善。代价是调度不再与 token 无关，第 6.2 节的"调度不变性"不再成立，反向采样时也需要知道每个位置的 token 类别才能确定揭开速率（揭开前并不知道，需要用模型预测来处理）。

### 10.3 实验

MD4 在 OpenWebText 上训练 GPT-2 规模的模型，零样本困惑度优于同期的 SEDD 等离散扩散模型；在像素级图像建模（把每个像素值当作一个类别）上，也取得了当时离散扩散模型中领先的 bits/dim 结果。这说明掩码扩散不只是语言模型的技巧，而是一类通用的离散生成模型。

## 11. 实验与局限

**MDLM 的实验.** 论文在 LM1B 和 OpenWebText 上训练，与自回归模型、SEDD、D3PM 等对比：

- 在 OpenWebText 上，MDLM 的困惑度上界约为 23，SEDD 约为 24，同规模、同训练量的自回归模型约为 17.5（具体数值取决于训练 token 数，以论文为准）；
- 消融实验表明，SUBS 参数化、连续时间目标、低差异采样、大词表和现代结构各自都有贡献，合起来把掩码扩散与自回归模型的差距缩小了一大截；
- 在生成质量上，配合缓存的祖先采样在相同网络调用次数下优于 SEDD；
- 论文还在 DNA 序列建模上验证了 MDLM，显示这一框架在生物序列上同样有效。

**局限.**

- **与自回归模型仍有差距**：困惑度上界与同规模 AR 模型仍有明显差距（部分来自"上界"与"精确似然"之间的差距本身）；
- **并行采样的独立性近似**：每一步内被揭开的多个位置是独立采样的，步数少时可能出现不一致（例如同一个词被揭开两次）；
- **已揭开的 token 不能修改**：保留已揭开 token 既是优点也是限制，早期的错误无法在后续步骤中纠正。后续的 remasking 类采样器（如 ReMDM）专门处理这个问题；
- **评测的微妙之处**：Zheng et al.（2024）指出，常用的 Gumbel-max 类别采样在低精度（如 float32）下会截断概率分布，使生成困惑度偏低、多样性下降，比较不同模型的生成质量时需要统一用高精度采样。

## 12. 小结

- **插值离散扩散** $q(z_t\mid x) = \mathrm{Cat}(\alpha_t x + (1-\alpha_t)\pi)$ 是高斯扩散的离散对应物；取 $\pi = m$ 就是掩码扩散，此时后验极其简单：未 mask 的保持不变，被 mask 的以 $\frac{\alpha_s-\alpha_t}{1-\alpha_t}$ 的概率回到原 token。
- **SUBS 参数化**把两条已知结构写进网络：不预测 [MASK]，已揭开的 token 原样复制。这使 ELBO 中的 KL 解析地化成被 mask 位置上的加权交叉熵（Rao-Blackwellization）。
- **连续时间 NELBO** 是 $\int_0^1\frac{\alpha_t'}{1-\alpha_t}\log\langle x_\theta,x\rangle\,dt$，对调度形式不变；线性调度下就是按 $1/t$ 加权的 MLM 损失，与 D3PM、SEDD 推出的权重一致。
- **与 SEDD 等价**：把比值参数化为 $\frac{\alpha_t}{1-\alpha_t}x_\theta$，SEDD 吸收态损失与 MDLM 损失逐项相等。
- **采样**：祖先采样从全 [MASK] 开始逐步揭开；网络不依赖时间时可以缓存，网络调用次数不超过序列长度；半自回归方式可以生成任意长度。
- **MD4** 独立推出了同样的 ELBO，并把掩码速率推广为与 token 类别有关、可学习的调度。

第 9.2 节用到了一个还没证明的结论：掩码扩散的最优去噪器与时间无关。下一篇 RADD 会从 SEDD 的比值出发严格证明这一点，并进一步说明吸收态扩散的似然与任意顺序自回归模型之间的关系。

## 参考文献

- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS 2021.
- Kingma et al. *Variational Diffusion Models.* NeurIPS 2021.
- Peebles & Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023.
- Lou, Meng & Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* ICML 2024.
- Sahoo et al. *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024.
- Shi et al. *Simplified and Generalized Masked Diffusion for Discrete Data.* NeurIPS 2024.
- Ou et al. *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data.* 2024.
- Zheng et al. *Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling.* 2024.
- Arriola et al. *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.* ICLR 2025.
- Wang et al. *Remasking Discrete Diffusion Models with Inference-Time Scaling.* 2025.
