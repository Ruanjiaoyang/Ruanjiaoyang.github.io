---
title: "DDPM：一步步加噪，再一步步去噪"
title_en: "DDPM: Add Noise Step by Step, Then Learn to Remove It"
series: "从 GAN 到 Diffusion"
order: 3
date: 2026-10-16
tags: [生成模型, Diffusion, DDPM]
summary_en: >
  DDPM defines a fixed Markov chain that gradually turns data into Gaussian noise and trains a
  network to reverse it. This post derives, step by step, the closed-form marginal q(x_t|x_0),
  the tractable posterior q(x_{t-1}|x_t,x_0) by completing the square, the full variational bound
  and its decomposition into L_T, L_{t-1} and L_0, and how the Gaussian KL terms collapse to the
  simple noise-prediction loss. It then covers the training and sampling algorithms, the choice of
  reverse variance, noise schedules, network architecture, and the epsilon, x_0 and v parameterizations.
---

扩散模型（Diffusion Models）是一类生成模型，核心思想是：**对数据逐步添加噪声，直到它变成纯噪声；再学习这个过程的逆过程，把噪声一步步还原为数据。**

今天所用的扩散模型大多源自 2020 年的 DDPM（Denoising Diffusion Probabilistic Models，Ho et al.）。它对更早的扩散概率模型（Sohl-Dickstein et al., 2015，*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*）做了关键的简化，第一次让扩散模型的图像质量追上了 GAN。

DDPM 用**变分推断**来建模，这是因为扩散模型本身也是一个**隐变量模型**。和上一篇的 VAE 对比：

| | VAE | DDPM |
|---|---|---|
| 隐变量 | 一个 $z$，维度远低于数据 | $T$ 个 $x_1,\dots,x_T$，**每个都和数据同维** |
| 编码器（推断过程） | 可学习的 $q_\phi(z\mid x)$ | **固定**的加噪过程 $q(x_{1:T}\mid x_0)$，没有参数 |
| 解码器（生成过程） | 一步 $p_\theta(x\mid z)$ | $T$ 步 $p_\theta(x_{t-1}\mid x_t)$，共享同一个网络 |
| 训练目标 | ELBO | ELBO，化简后变成噪声回归 |

所以 DDPM 可以看成一个**层数很多、每层隐变量和数据同维、编码器固定不学**的分层 VAE。下面的推导会把上一篇的 ELBO 原样再走一遍。

整个过程可以用一条链来表示：

```
             前向过程 q（固定，逐步加噪）
   x_0 ──────▶ x_1 ──────▶ x_2 ──── ··· ────▶ x_{T-1} ──────▶ x_T ≈ N(0, I)
  数据                                                          纯噪声
   x_0 ◀────── x_1 ◀────── x_2 ◀─── ··· ◀──── x_{T-1} ◀────── x_T
             反向过程 p_θ（学习，逐步去噪）
```

## 1. 前向过程：加噪

### 1.1 马尔可夫链

前向扩散过程对数据逐步添加高斯噪声，最终把它变成随机噪声。对原始数据 $x_0\sim q(x_0)$，这个过程由 $T$ 步组成，构成一条**马尔可夫链**（每一步只依赖上一步）。联合分布为：

$$
q(x_{1:T}\mid x_0) = \prod_{t=1}^{T} q(x_t\mid x_{t-1})
$$

每一步的转移是一个**条件高斯分布**：

$$
q(x_t\mid x_{t-1}) = \mathcal{N}\big(x_t;\; \sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big)
$$

$\{\beta_t\}_{t=1}^T$ 是每一步加入噪声的**方差**，也就是**噪声强度**，取值在 0 到 1 之间，控制噪声添加的速度。通常让它逐步增加，$\beta_1<\beta_2<\cdots<\beta_T$。DDPM 原文取 $T=1000$，$\beta_t$ 从 $10^{-4}$ 线性增长到 $0.02$。

**为什么均值要乘 $\sqrt{1-\beta_t}$？** 如果每一步只是加噪声 $x_t = x_{t-1}+\sqrt{\beta_t}\,\epsilon$，方差会一直累加，越来越大。乘上 $\sqrt{1-\beta_t}$ 后，若 $x_{t-1}$ 的方差为 1，则

$$
\mathrm{Var}(x_t) = (1-\beta_t)\cdot 1 + \beta_t = 1
$$

方差保持不变。也就是说，每一步都把信号**缩小一点**，再**补上一点噪声**，总"能量"不变。第 6 篇会看到，这对应 SDE 框架下的 **VP（Variance Preserving）SDE**。

### 1.2 任意时刻的闭式解

用**重参数化**把一步转移写成：

$$
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_{t}, \qquad \epsilon_{t}\sim\mathcal{N}(0,I)
$$

定义

$$
\alpha_t = 1-\beta_t, \qquad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s
$$

递推展开：

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t} \\
&= \sqrt{\alpha_t}\Big(\sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-1}\Big) + \sqrt{1-\alpha_t}\,\epsilon_{t} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t}}_{\text{两个独立高斯之和}} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\;\bar\epsilon_{t-2} \\
&\;\;\vdots \\
&= \sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_1}\,x_0 + \sqrt{1-\alpha_t\alpha_{t-1}\cdots\alpha_1}\;\epsilon \\
&= \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\;\epsilon, \qquad \epsilon\sim\mathcal{N}(0,I)
\end{aligned}
$$

第四行用到了"两个独立的零均值高斯之和仍是高斯，方差相加"：

$$
\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1-\alpha_t\alpha_{t-1}
$$

每往回展开一步都是同样的操作，最终得到

$$
q(x_t\mid x_0) = \mathcal{N}\big(x_t;\; \sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t)\, I\big)
$$

这条式子非常重要：**任意时间步的 $x_t$ 都可以由 $x_0$ 和 $\beta$ 序列直接算出，不必逐步迭代。** 训练时，我们只要随机挑一个 $t$，从 $x_0$ 一步采出 $x_t$ 即可。

### 1.3 最终收敛到标准高斯

$\bar\alpha_t$ 是一串小于 1 的数的连乘，随 $t$ 单调递减。只要 $\beta_t$ 不趋于 0，就有 $\bar\alpha_t\to 0$，于是

$$
\lim_{t\to\infty} q(x_t\mid x_0) = \mathcal{N}(0, I), \qquad \lim_{t\to\infty} q(x_t) = \mathcal{N}(0, I)
$$

第二个式子说的是：不论 $x_0$ 来自什么分布，充分加噪后的边际分布都是标准高斯，完全丧失原始数据的信息。按原文的线性调度，$\bar\alpha_{1000}\approx 4\times10^{-5}$，$x_T$ 中信号的系数 $\sqrt{\bar\alpha_T}$ 只有约 $0.006$，已经几乎是纯噪声。

一个有用的量是**信噪比**（signal-to-noise ratio）：

$$
\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}
$$

它随 $t$ 单调递减，从很大（几乎是干净数据）降到接近 0（几乎是纯噪声）。后面会看到，DDPM 的损失权重可以很自然地用 SNR 表达。

## 2. 反向过程：去噪

### 2.1 用神经网络近似反向一步

前向过程把数据变成噪声，反向过程就是**去噪**。如果知道每一步反向的真实分布 $q(x_{t-1}\mid x_t)$，就可以从一个随机噪声 $x_T\sim\mathcal{N}(0,I)$ 出发，逐步去噪，最终生成一个真实样本。**所以反向过程就是生成数据的过程。**

问题是 $q(x_{t-1}\mid x_t)$ 算不出来。由贝叶斯公式

$$
q(x_{t-1}\mid x_t) = \frac{q(x_t\mid x_{t-1})\,q(x_{t-1})}{q(x_t)}
$$

其中 $q(x_{t-1}) = \int q(x_{t-1}\mid x_0)\,q(x_0)\,dx_0$ 需要对整个数据分布积分。所以我们用**神经网络**来学习它。反向过程同样定义为一条马尔可夫链，转移分布是由神经网络参数化的高斯：

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)
$$

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\big(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\big)
$$

- **初始分布**：$p(x_T) = \mathcal{N}(0, I)$；
- **转移分布**：均值 $\mu_\theta(x_t,t)$ 和方差 $\Sigma_\theta(x_t,t)$ 由神经网络预测。所有时间步共享同一个网络，时间步 $t$ 作为额外输入。

扩散模型的训练目标就是训练这个网络，因为它最终构成了数据的生成模型。

**为什么反向一步可以用高斯来近似？** 真实的 $q(x_{t-1}\mid x_t)$ 一般不是高斯。但当 $\beta_t$ 足够小时，前向一步只加了一点点噪声，反向一步也只需要"挪动一点点"；在这么小的尺度上，任何光滑分布都可以用高斯很好地近似（Feller, 1949 给出了连续扩散过程的这一性质）。这也是为什么扩散模型需要**很多小步**：步子越大，单个高斯越难描述反向一步，这正是下一篇 DDIM 跳步时要面对的问题。

### 2.2 关键：加上 $x_0$ 之后，后验可以精确计算

$q(x_{t-1}\mid x_t)$ 算不出来，但如果**额外给定 $x_0$**，后验 $q(x_{t-1}\mid x_t, x_0)$ 是可以精确计算的。由贝叶斯公式：

$$
q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t\mid x_{t-1}, x_0)\;q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

由马尔可夫性，$q(x_t\mid x_{t-1},x_0) = q(x_t\mid x_{t-1})$。右边三项都是已知的高斯：

- $q(x_t\mid x_{t-1}) = \mathcal{N}\big(\sqrt{\alpha_t}\,x_{t-1},\; \beta_t I\big)$
- $q(x_{t-1}\mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_{t-1}}\,x_0,\; (1-\bar\alpha_{t-1}) I\big)$
- $q(x_t\mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t) I\big)$

直观上，$x_0$ 在这里是一个**参考点**：已知起点 $x_0$ 和当前位置 $x_t$，就能推断出中间的 $x_{t-1}$ 大概在哪。训练时我们手上有 $x_0$，所以可以用这个可解的后验来监督网络。

### 2.3 配方求后验

高斯密度为 $\mathcal{N}(x;\mu,\sigma^2)\propto\exp\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$。把三项代入（为书写简洁按标量写，向量情形逐维相同）：

$$
q(x_{t-1}\mid x_t, x_0) \propto \exp\Big(-\frac12\Big[\frac{(x_t-\sqrt{\alpha_t}\,x_{t-1})^2}{\beta_t} + \frac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}\,x_0)^2}{1-\bar\alpha_{t-1}} - \frac{(x_t-\sqrt{\bar\alpha_t}\,x_0)^2}{1-\bar\alpha_t}\Big]\Big)
$$

我们关心的是它作为 $x_{t-1}$ 的函数长什么样。展开方括号，按 $x_{t-1}$ 的幂次整理：

$$
\Big[\cdots\Big] = \Big(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}\Big)x_{t-1}^2 - 2\Big(\frac{\sqrt{\alpha_t}}{\beta_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\,x_0\Big)x_{t-1} + C(x_t,x_0)
$$

$C(x_t,x_0)$ 是与 $x_{t-1}$ 无关的项，包括最后那个 $q(x_t\mid x_0)$ 带来的项，归一化时会被吸收掉。指数部分是 $x_{t-1}$ 的二次函数，所以**后验仍是高斯**。对照 $\frac{(x-\mu)^2}{\sigma^2} = \frac{1}{\sigma^2}x^2 - \frac{2\mu}{\sigma^2}x + \cdots$，二次项系数是方差的倒数，一次项系数是 $\mu/\sigma^2$ 的 $-2$ 倍。

**方差。** 二次项系数通分：

$$
\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}} = \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1-\bar\alpha_{t-1})} = \frac{\alpha_t - \bar\alpha_t + 1 - \alpha_t}{\beta_t(1-\bar\alpha_{t-1})} = \frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}
$$

所以后验方差为

$$
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

**均值。** 均值等于方差乘以一次项系数（去掉 $-2$）：

$$
\tilde\mu_t(x_t,x_0) = \tilde\beta_t\Big(\frac{\sqrt{\alpha_t}}{\beta_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\,x_0\Big)
= \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0
$$

于是

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\big(x_{t-1};\; \tilde\mu_t(x_t,x_0),\; \tilde\beta_t I\big)
$$

$\tilde\mu_t$ 是 $x_t$ 和 $x_0$ 的加权组合。

### 2.4 用噪声表示后验均值

由前向闭式解 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，可以反解出

$$
x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\Big(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\Big)
$$

代入 $\tilde\mu_t$ 的第二项：

$$
\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\cdot\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}
= \frac{\beta_t}{\sqrt{\alpha_t}\,(1-\bar\alpha_t)}\,x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\,\epsilon
$$

（用了 $\sqrt{\bar\alpha_t} = \sqrt{\alpha_t}\sqrt{\bar\alpha_{t-1}}$。）$x_t$ 的总系数为

$$
\frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\sqrt{\alpha_t}\,(1-\bar\alpha_t)} = \frac{1-\bar\alpha_t}{\sqrt{\alpha_t}\,(1-\bar\alpha_t)} = \frac{1}{\sqrt{\alpha_t}}
$$

所以

$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big)
$$

这个形式非常有启发性：**后验均值就是"从 $x_t$ 里减掉一部分噪声，再放大一点"。** $x_t$ 是已知的，唯一未知的是加进去的噪声 $\epsilon$。这暗示了网络该学什么：**去预测这个噪声。**

## 3. 目标函数：变分下界

扩散模型的训练目标，说到底是**让模型学会去噪**：前向过程在每一步给 $x_0$ 加噪，生成一系列中间变量 $x_t$；反向过程要从 $x_t$ 逐步去除噪声，直到恢复 $x_0$。下面从最大似然出发，严格推出这个目标。

### 3.1 最小化负对数似然

我们希望模型的生成分布 $p_\theta(x_0)$ 尽可能逼近真实数据分布 $q(x_0)$，即最小化负对数似然：

$$
\begin{aligned}
L &= \mathbb{E}_{q(x_0)}\big[-\log p_\theta(x_0)\big] \\
&= -\,\mathbb{E}_{q(x_0)}\Big[\log \int p_\theta(x_{0:T})\,dx_{1:T}\Big] \\
&= -\,\mathbb{E}_{q(x_0)}\Big[\log \int q(x_{1:T}\mid x_0)\,\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T}\Big] \\
&= -\,\mathbb{E}_{q(x_0)}\Big[\log \mathbb{E}_{q(x_{1:T}|x_0)}\Big[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big]\Big]
\end{aligned}
$$

第二行是对所有隐变量 $x_{1:T}$ 积分得到边际似然；第三行同乘同除前向过程 $q(x_{1:T}\mid x_0)$，它在这里扮演 VAE 中编码器的角色。积分在高维上算不出来，需要一个下界（对负对数似然来说是上界）。

### 3.2 Jensen 不等式

Jensen 不等式说的是：若 $f$ 是**凸函数**，则

$$
f\Big(\frac1n\sum_{i=1}^n x_i\Big) \le \frac1n\sum_{i=1}^n f(x_i), \qquad \text{期望形式：}\; f\big(\mathbb{E}[x]\big)\le\mathbb{E}\big[f(x)\big]
$$

直观上，凸函数在两点之间的弦总在函数图像上方。若 $f$ 是**凹函数**，不等号反向。

$\log$ 是凹函数，所以 $\log\mathbb{E}[\cdot]\ge\mathbb{E}[\log(\cdot)]$；等价地，$-\log$ 是凸函数，$-\log\mathbb{E}[\cdot]\le\mathbb{E}[-\log(\cdot)]$。用它可以给负对数似然构造一个**上界**：

$$
\begin{aligned}
L &= -\,\mathbb{E}_{q(x_0)}\Big[\log \mathbb{E}_{q(x_{1:T}|x_0)}\Big[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big]\Big] \\
&\le -\,\mathbb{E}_{q(x_0)}\Big[\mathbb{E}_{q(x_{1:T}|x_0)}\Big[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big]\Big] \\
&= \mathbb{E}_{q(x_{0:T})}\Big[\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}\Big] =: L_\text{VLB}
\end{aligned}
$$

$L_\text{VLB}$ 称为变分下界（variational lower bound），严格来说它是负对数似然的上界，也就是负 ELBO。最小化 $L_\text{VLB}$ 就是在最大化 ELBO。这和上一篇 VAE 的推导二完全一样，只是隐变量从一个 $z$ 变成了一串 $x_{1:T}$。

### 3.3 把 VLB 拆成逐步的 KL

下面把 $L_\text{VLB}$ 一步步拆开。先代入前向过程和反向过程的乘积形式：

$$
\begin{aligned}
L_\text{VLB}
&= \mathbb{E}_{q}\Big[\log\frac{\prod_{t=1}^{T} q(x_t\mid x_{t-1})}{p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)}\Big] \\
&= \mathbb{E}_{q}\Big[-\log p(x_T) + \sum_{t=1}^{T}\log\frac{q(x_t\mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)}\Big] \\
&= \mathbb{E}_{q}\Big[-\log p(x_T) + \sum_{t=2}^{T}\log\frac{q(x_t\mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)} + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\Big]
\end{aligned}
$$

这里 $\mathbb{E}_q$ 是 $\mathbb{E}_{q(x_{0:T})}$ 的简写。最后一步把 $t=1$ 这一项单独拿出来，原因马上会看到。

**关键一步：把前向转移"翻转"。** 对 $t\ge2$，由马尔可夫性和贝叶斯公式，

$$
q(x_t\mid x_{t-1}) = q(x_t\mid x_{t-1},x_0) = \frac{q(x_{t-1}\mid x_t, x_0)\;q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}
$$

这样就把前向的 $q(x_t\mid x_{t-1})$ 换成了我们在 2.2 节算出的后验 $q(x_{t-1}\mid x_t,x_0)$。（$t=1$ 时这么做会出现 $q(x_0\mid x_0)$，没有意义，所以上面把它单独拿了出来。）代入：

$$
\begin{aligned}
L_\text{VLB}
&= \mathbb{E}_{q}\Big[-\log p(x_T) + \sum_{t=2}^{T}\log\Big(\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}\cdot\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}\Big) + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\Big] \\
&= \mathbb{E}_{q}\Big[-\log p(x_T) + \sum_{t=2}^{T}\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)} + \sum_{t=2}^{T}\log\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)} + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\Big]
\end{aligned}
$$

**望远镜求和。** 第三项中相邻的分子分母依次相消：

$$
\sum_{t=2}^{T}\log\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)} = \log\frac{q(x_2\mid x_0)}{q(x_1\mid x_0)}+\log\frac{q(x_3\mid x_0)}{q(x_2\mid x_0)}+\cdots+\log\frac{q(x_T\mid x_0)}{q(x_{T-1}\mid x_0)} = \log\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)}
$$

代回去，$q(x_1\mid x_0)$ 和最后一项的分子相消：

$$
\begin{aligned}
L_\text{VLB}
&= \mathbb{E}_{q}\Big[-\log p(x_T) + \sum_{t=2}^{T}\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)} + \log\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)} + \log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}\Big] \\
&= \mathbb{E}_{q}\Big[\log\frac{q(x_T\mid x_0)}{p(x_T)} + \sum_{t=2}^{T}\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)} - \log p_\theta(x_0\mid x_1)\Big]
\end{aligned}
$$

**写成 KL 散度。** 每个 $\log$ 比值在相应的条件分布下取期望，就是一个 KL 散度：

$$
L_\text{VLB} = \mathbb{E}_{q}\Big[\underbrace{\mathrm{KL}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T} + \sum_{t=2}^{T}\underbrace{\mathrm{KL}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} \underbrace{-\log p_\theta(x_0\mid x_1)}_{L_0}\Big]
$$

三类项各自的含义：

- **$L_T$：先验匹配项。** 衡量加噪到最后的分布与标准高斯先验之间的差距。前向过程是固定的，$p(x_T)$ 也是固定的标准高斯，**这一项不含任何可学习参数，是一个常数**，训练时可以忽略。只要 $\bar\alpha_T$ 足够接近 0，它也接近 0。
- **$L_0$：重构项。** 最后一步从 $x_1$ 还原 $x_0$ 的对数似然，作用和 VAE 的重构项一样。DDPM 原文为它设计了一个离散化的解码器：图像像素是 $\{0,\dots,255\}$ 中的整数，把高斯密度在每个像素值对应的区间上积分，得到一个合法的离散概率。在后面的简化目标里，它被当作 $t=1$ 的情形一并处理。
- **$L_{t-1}$：去噪匹配项。** 真正需要优化的部分。它要求**网络的每一步去噪分布，都贴近给定 $x_0$ 时可以精确计算的真实后验**。

这里可以看到 2.2 节那个可解后验的作用：没有它，KL 的第一个参数是算不出来的 $q(x_{t-1}\mid x_t)$；有了它，每一项都是两个高斯之间的 KL，有闭式解。

### 3.4 两个高斯的 KL

两个一维高斯之间的 KL 散度为

$$
\mathrm{KL}\big(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12
$$

现在 KL 的两边分别是

- $q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\big(\tilde\mu_t(x_t,x_0),\; \tilde\beta_t I\big)$
- $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\big(\mu_\theta(x_t,t),\; \Sigma_\theta(x_t,t)\big)$

DDPM 把反向方差**固定**为 $\Sigma_\theta(x_t,t) = \sigma_t^2 I$，不让网络学（$\sigma_t^2$ 的取法见 4.3 节）。这样两个高斯的方差都和 $\theta$ 无关，KL 中只有均值差这一项依赖 $\theta$：

$$
L_{t-1} = \mathbb{E}_{q}\Big[\frac{1}{2\sigma_t^2}\,\big\|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\big\|^2\Big] + C
$$

也就是说，**训练网络等价于让它预测真实后验的均值**。

### 3.5 预测噪声：$\epsilon$ 参数化

网络可以直接输出 $\mu_\theta$，但 2.4 节已经给出了更好的选择。真实后验均值是

$$
\tilde\mu_t(x_t,x_0) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big)
$$

$x_t$ 是网络的输入，已知；未知的只有 $\epsilon$。所以把 $\mu_\theta$ 参数化成**完全相同的形式**，只是把真实噪声 $\epsilon$ 换成网络预测的噪声 $\epsilon_\theta(x_t,t)$：

$$
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)
$$

两者相减，$x_t$ 抵消，只剩噪声之差：

$$
\tilde\mu_t - \mu_\theta = \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\,\big(\epsilon_\theta(x_t,t) - \epsilon\big)
$$

代入 3.4 节的式子，并把 $x_t$ 写成 $x_0$ 和 $\epsilon$ 的函数：

$$
L_{t-1} = \mathbb{E}_{x_0\sim q(x_0),\,\epsilon\sim\mathcal{N}(0,I)}\Big[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\,\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\|^2\Big] + C
$$

现在网络的任务很清楚了：**看到加噪后的 $x_t$ 和时间步 $t$，预测当初加进去的噪声 $\epsilon$。** 这就是"去噪扩散概率模型"名字的由来。

### 3.6 简化目标 $L_\text{simple}$

DDPM 的作者对上式做了进一步简化：**把前面的权重系数直接去掉**，并对 $t$ 均匀采样：

$$
L_\text{simple}(\theta) = \mathbb{E}_{t\sim\mathcal{U}\{1,\dots,T\},\;x_0\sim q(x_0),\;\epsilon\sim\mathcal{N}(0,I)}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\|^2\Big]
$$

$t=1$ 时这一项就对应 $L_0$（忽略离散化），$t>1$ 时对应去掉权重的 $L_{t-1}$。

**为什么去掉权重反而更好？** 看看被去掉的权重长什么样。取 $\sigma_t^2=\tilde\beta_t$，权重化简为

$$
\frac{\beta_t^2}{2\tilde\beta_t\,\alpha_t\,(1-\bar\alpha_t)} = \frac{\beta_t}{2\,\alpha_t\,(1-\bar\alpha_{t-1})}
$$

$t$ 很小时，$1-\bar\alpha_{t-1}$ 接近 0，这个权重**非常大**。而小 $t$ 对应的任务是"给几乎干净的图像去掉一点点噪声"，这是最容易的去噪任务。原始权重让这些简单任务在损失中占了主导。

去掉权重后，小 $t$ 的项被相对降权，网络能**把注意力集中在大 $t$ 的困难去噪任务上**，也就是从很嘈杂的输入中恢复出整体结构。实验上这显著提升了样本质量（CIFAR-10 上 FID 3.17）。

代价是 $L_\text{simple}$ 不再是严格的似然上界，所以 DDPM 的对数似然指标并不突出。这个"样本质量与似然"的取舍，后来由 Improved DDPM 部分解决（见 5.3 节）。

**用 SNR 看权重。** 上面那个权重有一个很漂亮的形式。如果改用"预测 $x_0$"的写法，$\|x_0-\hat x_0\|^2 = \frac{1-\bar\alpha_t}{\bar\alpha_t}\|\epsilon-\epsilon_\theta\|^2$，可以验证在 $\sigma_t^2=\tilde\beta_t$ 时 $L_{t-1}$ 等于

$$
L_{t-1} = \frac12\big(\mathrm{SNR}(t-1) - \mathrm{SNR}(t)\big)\,\big\|x_0 - \hat x_0(x_t,t)\big\|^2
$$

也就是说，真正的 ELBO 按"相邻两步之间信噪比下降了多少"来给每一步加权。后来的 VDM（Variational Diffusion Models）正是从这个形式出发，把扩散模型推广到连续时间，并让噪声调度也可以学习。各种扩散模型的"损失加权"之争，本质上都是在选择不同的 SNR 函数作为权重。

### 3.7 再看一遍：DDPM 是一个分层 VAE

把推导和上一篇对照：

| | VAE | DDPM |
|---|---|---|
| 隐变量 | $z$ | $x_{1:T}$ |
| 近似后验（编码器） | $q_\phi(z\mid x)$，可学 | $q(x_{1:T}\mid x_0)$，固定 |
| 先验 | $p(z)=\mathcal{N}(0,I)$ | $p(x_T)=\mathcal{N}(0,I)$ |
| 解码器 | $p_\theta(x\mid z)$ | $\prod_t p_\theta(x_{t-1}\mid x_t)$ |
| ELBO 中的 KL 正则 | $\mathrm{KL}(q_\phi(z\mid x)\,\|\,p(z))$，需要学 | $L_T$，常数 |
| ELBO 中的重构 | $\mathbb{E}\log p_\theta(x\mid z)$ | $L_0$ 加上所有 $L_{t-1}$ |

DDPM 选择了一个"更弱"的模型：编码器不学习，隐变量也不压缩维度。但正是这个限制带来了好处：

- 编码器固定，就不存在 VAE 里"KL 项和重构项拉扯"的问题，也不存在后验坍缩；
- 前向过程是线性高斯的，所以 $q(x_t\mid x_0)$ 和 $q(x_{t-1}\mid x_t,x_0)$ 都有闭式解，ELBO 的每一项都能精确计算；
- 把一个困难的生成问题拆成了 $T$ 个简单的去噪问题，每个都只需要"挪动一点点"。

## 4. 训练与采样

### 4.1 训练算法

DDPM 原文的训练算法（Algorithm 1）：

1. **repeat**
2. 　从数据集中采样 $x_0\sim q(x_0)$
3. 　采样时间步 $t\sim\mathcal{U}\{1,\dots,T\}$
4. 　采样噪声 $\epsilon\sim\mathcal{N}(0,I)$
5. 　对 $\nabla_\theta\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\|^2$ 做一步梯度下降
6. **until** 收敛

训练本质上就是一个**回归问题**，非常简单、稳定，没有 GAN 那样的对抗博弈，也没有 VAE 那样的两项拉扯。每个 batch 里不同样本可以取不同的 $t$，一个网络同时学会所有噪声水平下的去噪。

```python
# 预先算好 alphas_bar[t]，t = 0..T-1
def ddpm_loss(model, x0):
    b = x0.size(0)
    t = torch.randint(0, T, (b,), device=x0.device)
    eps = torch.randn_like(x0)
    ab = alphas_bar[t].view(b, 1, 1, 1)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps     # 一步采出 x_t
    return F.mse_loss(model(x_t, t), eps)            # 预测噪声
```

### 4.2 采样算法

DDPM 原文的采样算法（Algorithm 2）：

1. 采样 $x_T\sim\mathcal{N}(0,I)$
2. **for** $t = T,\dots,1$ **do**
3. 　若 $t>1$，采样 $z\sim\mathcal{N}(0,I)$；否则 $z=0$
4. 　$\displaystyle x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big) + \sigma_t z$
5. **end for**
6. **return** $x_0$

第 4 步就是从 $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(\mu_\theta,\sigma_t^2 I)$ 中采样：前一项是均值 $\mu_\theta$（注意 $1-\alpha_t=\beta_t$），后一项是重参数化加入的噪声。最后一步 $t=1$ 不加噪声，直接输出均值。

```python
@torch.no_grad()
def ddpm_sample(model, shape):
    x = torch.randn(shape)
    for t in reversed(range(T)):
        eps = model(x, torch.full((shape[0],), t))
        mean = (x - betas[t] / (1 - alphas_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + sigmas[t] * torch.randn_like(x) if t > 0 else mean
    return x
```

采样过程有一个直观的理解：每一步，网络先估计出 $x_t$ 中的噪声，减掉一部分，得到一个稍微干净一点的图像，再加上一点新的随机噪声。加入随机噪声这一步很重要：它让采样过程保持随机性，类似第 5 篇要讲的 Langevin 动力学。

采样需要走完全部 $T$ 步，每步一次网络前向。$T=1000$ 时，生成一张图就要跑 1000 次网络，比 GAN 的一次前向慢几个数量级。**这是 DDPM 最大的缺点**，下一篇的 DDIM 专门解决这个问题。

### 4.3 反向方差 $\sigma_t^2$ 怎么取

DDPM 尝试了两种取法：

- $\sigma_t^2 = \beta_t$：如果数据 $x_0\sim\mathcal{N}(0,I)$，这是最优的选择；
- $\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$：如果数据集中在一个确定的点上，这是最优的选择。

这两者分别对应反向过程熵的上界和下界，实验中样本质量差不多。$t$ 较大时两者几乎相等（$\bar\alpha_{t-1}\approx\bar\alpha_t\approx 0$），只在 $t$ 很小时有差别。

### 4.4 生成过程中的"由粗到细"

观察 DDPM 采样过程中不同时刻的 $\hat x_0$（用 $x_t$ 和 $\epsilon_\theta$ 反推的干净图像估计），会发现：

- $t$ 很大时，$\hat x_0$ 只有模糊的整体轮廓和颜色布局；
- $t$ 中等时，物体的形状、结构逐渐清晰；
- $t$ 很小时，只是在补充纹理、边缘等高频细节。

也就是说，扩散模型的生成是**由粗到细**的：先决定"画什么"，再决定"怎么画细节"。这也解释了为什么大 $t$ 的去噪任务更重要（3.6 节），以及为什么后来的 guidance 可以只在部分时间步施加（第 7 篇）。

## 5. 实践细节与改进

### 5.1 网络结构

$\epsilon_\theta(x_t,t)$ 的唯一硬性要求是**输入和输出同维**。DDPM 用的是一个改造过的 **U-Net**：

- **编码器-解码器结构**：先逐步下采样提取特征，再逐步上采样恢复分辨率，同分辨率的层之间有跳跃连接，既有全局视野，又保留细节；
- **残差块**：每个分辨率上有若干残差块，用 GroupNorm 归一化；
- **自注意力**：在 $16\times16$ 等较低分辨率的特征图上加入自注意力层，建模长距离依赖；
- **时间步嵌入**：$t$ 先经过类似 Transformer 位置编码的正弦嵌入，再经过 MLP，加到每个残差块的特征上。这样同一个网络能根据 $t$ 知道当前的噪声水平，调整去噪的力度。

后来的 DiT（Diffusion Transformer）把 U-Net 换成了 Transformer：把（隐空间的）图像切成 patch，当作 token 处理，时间步和条件通过自适应 LayerNorm（adaLN）注入。DiT 显示出很好的扩展性，成为 Sora、Stable Diffusion 3 等新一代模型的骨干。

### 5.2 噪声调度

$\beta_t$ 的取法称为**噪声调度**（noise schedule），它决定了信息被破坏的速度。

**线性调度**（DDPM 原文）：$\beta_t$ 从 $10^{-4}$ 线性增长到 $0.02$。问题是对低分辨率图像（如 $32\times32$、$64\times64$），信息被破坏得太快：后面相当多的时间步里 $x_t$ 已经几乎是纯噪声，这些步对训练和采样的贡献很小，是一种浪费。

**余弦调度**（Improved DDPM）：直接定义 $\bar\alpha_t$，

$$
\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\Big(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\Big)
$$

$s=0.008$ 是一个小偏移，防止 $t=0$ 附近的 $\beta_t$ 过小；$\beta_t = 1-\bar\alpha_t/\bar\alpha_{t-1}$，并截断到不超过 $0.999$，防止 $t=T$ 附近出现奇异。余弦调度让 $\bar\alpha_t$ 在中间段近似线性下降，在两端变化平缓，信息被更均匀地破坏。

后来的研究发现，**最合适的调度和分辨率有关**：分辨率越高，相邻像素越相关，同样的噪声对图像的破坏越小，需要更多的噪声（把 SNR 整体向下平移）。此外，很多调度在 $t=T$ 时 $\bar\alpha_T$ 并不严格为 0，训练时 $x_T$ 仍残留少量信号，而采样时却从纯噪声开始，导致训练和推理不一致（例如模型生成不出很暗或很亮的图像）。修正办法是重新缩放调度，使终点的 SNR 恰好为 0（zero terminal SNR）。

### 5.3 学习方差：Improved DDPM

DDPM 固定方差，Improved DDPM（Nichol & Dhariwal, 2021）让网络额外输出一个向量 $v$，在 $\beta_t$ 和 $\tilde\beta_t$ 之间插值（在对数域）：

$$
\Sigma_\theta(x_t,t) = \exp\big(v\log\beta_t + (1-v)\log\tilde\beta_t\big)
$$

由于 $L_\text{simple}$ 不依赖方差，方差部分用原始的 $L_\text{VLB}$ 来训练，两者混合：

$$
L_\text{hybrid} = L_\text{simple} + \lambda\,L_\text{VLB}, \qquad \lambda = 0.001
$$

（$L_\text{VLB}$ 中对均值的梯度被截断，只更新方差。）学习方差的好处有两个：对数似然显著改善；更重要的是，采样步数大幅减少时（例如 50 步），学到的方差能让样本质量下降得少得多。

Improved DDPM 还发现 $L_\text{VLB}$ 的不同项梯度方差差别很大，于是对 $t$ 做重要性采样（按每项损失的历史均方根加权抽样），以降低直接优化 $L_\text{VLB}$ 时的方差。

### 5.4 三种预测目标

网络不一定要预测噪声 $\epsilon$。由 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，已知 $x_t$ 时，$x_0$ 和 $\epsilon$ 知道一个就能算出另一个，所以可以有几种等价的参数化：

**预测噪声 $\epsilon$**（DDPM 默认）：

$$
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

**预测干净数据 $x_0$**：网络直接输出 $\hat x_0$，再代入 $\tilde\mu_t(x_t,\hat x_0)$ 得到均值。

**预测速度 $v$**（Salimans & Ho, 2022）：定义

$$
v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0
$$

则

$$
x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,v, \qquad \epsilon = \sqrt{1-\bar\alpha_t}\,x_t + \sqrt{\bar\alpha_t}\,v
$$

几何上，若令 $\sqrt{\bar\alpha_t}=\cos\phi$、$\sqrt{1-\bar\alpha_t}=\sin\phi$，则 $x_t = \cos\phi\,x_0 + \sin\phi\,\epsilon$ 是 $(x_0,\epsilon)$ 平面上的一个旋转，$v = \frac{dx_t}{d\phi}$ 就是这个旋转的"速度"，这也是名字的由来。

这三种参数化在数学上可以互相转换，但数值性质不同：

- **$\epsilon$ 预测**在 $t$ 很大时有问题：此时 $x_t$ 几乎全是噪声，预测 $\epsilon$ 很容易（几乎就是 $x_t$ 本身），但由它反推的 $\hat x_0$ 要除以一个接近 0 的 $\sqrt{\bar\alpha_t}$，误差被急剧放大。
- **$x_0$ 预测**在 $t$ 很小时有问题：此时 $x_t\approx x_0$，网络容易学成恒等映射。
- **$v$ 预测**在两端都表现良好，在高噪声端更稳定。它被渐进蒸馏、Imagen Video、Stable Diffusion 2 等很多后续模型采用。

后来的 Flow Matching / Rectified Flow 用 $x_t = (1-t)\,x_0 + t\,\epsilon$ 的直线插值，预测速度 $\epsilon - x_0$，和 $v$ 预测是同一类思路。

## 6. 小结

DDPM 用三步把一个看似复杂的生成问题，变成了一个极其简单的回归问题：

1. **前向过程有闭式解**：$q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I)$，可以从 $x_0$ 一步采出任意 $x_t$；
2. **给定 $x_0$ 时反向后验可解**：$q(x_{t-1}\mid x_t,x_0) = \mathcal{N}(\tilde\mu_t,\tilde\beta_t I)$，于是 ELBO 拆成 $L_T$（常数）、$L_{t-1}$（逐步的高斯 KL）和 $L_0$（重构）；
3. **高斯 KL 化简后就是预测噪声**：$L_\text{simple} = \mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t)\|^2$。

| 方面 | DDPM 的选择 |
|---|---|
| 训练 | 均匀采样 $t$，回归加入的噪声，简单稳定 |
| 采样 | 从纯噪声出发走满 $T$ 步，每步一次网络前向 |
| 优点 | 样本质量高、多样性好、没有模式崩溃、训练稳定 |
| 缺点 | 采样慢；$L_\text{simple}$ 下似然指标一般 |

还有一条暗线，会在后面几篇反复出现。DDPM 预测的噪声和**分布的 score**（对数密度的梯度）只差一个系数：

$$
\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

所以"预测噪声"其实就是"估计 score"。第 5 篇会从 score 的角度重新出发，殊途同归地走到同一个模型；第 6 篇会把这一切放到连续时间的随机微分方程里统一起来。但在那之前，下一篇先解决 DDPM 最实际的问题：**怎样不重新训练，把 1000 步采样压到几十步。**

## 参考文献

- Feller. *On the Theory of Stochastic Processes, with Particular Reference to Applications.* Berkeley Symposium 1949.
- Sohl-Dickstein et al. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML 2015.
- Ho, Jain & Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Nichol & Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
- Kingma, Salimans, Poole & Ho. *Variational Diffusion Models.* NeurIPS 2021.
- Salimans & Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022.（$v$-prediction）
- Peebles & Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023.（DiT）
- Hoogeboom, Heek & Salimans. *simple diffusion: End-to-end diffusion for high resolution images.* ICML 2023.（调度与分辨率）
- Lin et al. *Common Diffusion Noise Schedules and Sample Steps are Flawed.* WACV 2024.（zero terminal SNR）
- Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023.
