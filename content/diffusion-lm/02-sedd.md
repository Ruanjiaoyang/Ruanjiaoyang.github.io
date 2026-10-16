---
title: "SEDD：离散空间里的 score 是概率比值"
title_en: "SEDD: Score Entropy and the Ratios of the Data Distribution"
series: "扩散语言模型"
order: 2
date: 2026-10-16
tags: [扩散语言模型, 离散扩散, SEDD, Score Entropy, CTMC]
summary_en: >
  In continuous diffusion the score is the gradient of the log density, which does not exist on a
  discrete vocabulary. SEDD replaces it with the concrete score, the vector of probability ratios
  p_t(y)/p_t(x), and learns it with score entropy, a Bregman-divergence loss that respects positivity.
  This post sets up continuous-time discrete diffusion (CTMCs), derives the reverse-time rate matrix,
  the implicit and denoising forms of score entropy, the ELBO that turns the loss into a perplexity bound,
  the closed-form transition kernels for uniform and absorbing noise, the discrete Tweedie formula behind
  Tweedie tau-leaping, and conditional infilling with an unconditional model.
---

## 0. 从 D3PM 到 SEDD：离散空间的 score 是什么

上一篇的 D3PM 把扩散搬到了离散空间：用转移矩阵代替高斯噪声，网络预测干净数据的分布 $p_\theta(x_0\mid x_t)$，再代入真实后验得到反向一步。这条路线（论文里称为**均值预测**，mean prediction）可以工作，但在文本上和自回归模型之间一直有明显的似然差距。

回头看连续扩散，它真正的理论核心不是"预测 $x_0$"，而是 **score**：

$$
\nabla_x \log p_t(x)
$$

有了 score，反向 SDE、概率流 ODE、似然计算、条件生成都能统一推出来（见上一个系列的 Score-based 和 SDE 两篇）。所以一个自然的问题是：**离散空间里的 score 应该是什么？** 离散空间没有梯度，$\nabla_x$ 无从谈起。

SEDD（Score Entropy Discrete Diffusion，Lou, Meng & Ermon, ICML 2024）给出的答案是：

- 离散空间里与 score 对应的量是**概率比值** $\dfrac{p_t(y)}{p_t(x)}$，称为 **concrete score**；
- 学习这个比值不能直接照搬分数匹配的 $\ell^2$ 损失，因为比值必须是正数。SEDD 为此设计了新的损失 **score entropy**；
- score entropy 可以写成"去噪"形式，训练和 DDPM 一样简单；它还能给出对数似然的上界，因此可以直接和 GPT-2 比较困惑度。

这篇论文获得了 ICML 2024 的最佳论文奖。它让人第一次看到，离散扩散在语言建模上可以和同规模的自回归模型站在同一水平线上。

**记号提醒。** D3PM 用的是"行向量乘矩阵"的约定，$[Q_t]_{ij} = q(x_t = j\mid x_{t-1}=i)$，每一**行**和为 1。SEDD 沿用连续时间马尔可夫链的文献习惯，用"矩阵乘列向量"的约定：$Q_t(y,x)$ 表示从 $x$ 跳到 $y$ 的速率，每一**列**和为 0。下文全部使用 SEDD 的约定。另外，D3PM 中的 $\bar Q_t$ 表示转移矩阵的连乘；为了不和它混淆，本文把 SEDD 论文中的反向速率矩阵 $\bar Q_t$ 记作 $\tilde Q_t$。

## 1. 连续时间的离散扩散

### 1.1 状态空间与概率向量

先考虑一个有限状态空间 $\mathcal{X} = \{1,\dots,N\}$，比如单个 token 的词表。$\mathcal X$ 上的概率分布可以写成一个**概率质量向量** $p\in\mathbb{R}^N$，各元素非负且和为 1。第 5 节再把它推广到长度为 $d$ 的序列 $\mathcal X = \{1,\dots,N\}^d$。

### 1.2 前向过程：速率矩阵与线性 ODE

D3PM 使用离散时间 $t=1,\dots,T$，每一步乘一个转移矩阵。SEDD 改用**连续时间**：分布族 $p_t$ 按照一个线性常微分方程演化

$$
\frac{dp_t}{dt} = Q_t\,p_t,\qquad p_0 = p_{\text{data}}
$$

其中 $Q_t\in\mathbb R^{N\times N}$ 是**扩散矩阵**（也叫速率矩阵、生成元），满足两个条件：

1. **非对角元非负**：$Q_t(y,x)\ge 0$ 对 $y\ne x$，表示单位时间内从 $x$ 跳到 $y$ 的"速率"；
2. **每一列和为 0**：$Q_t(x,x) = -\sum_{y\ne x}Q_t(y,x)$。对角元是"离开 $x$ 的总速率"取负号。

第二个条件保证总概率守恒：

$$
\frac{d}{dt}\sum_y p_t(y) = \mathbf 1^\top Q_t\,p_t = 0
$$

因为 $\mathbf 1^\top Q_t = 0$（每列和为零）。

**这个过程在样本层面是什么样子？** 取一个很小的时间步 $\Delta t$，用 Euler 法离散化 ODE：$p_{t+\Delta t}\approx (I + \Delta t\,Q_t)\,p_t$。对照"转移矩阵乘概率向量"的形式，单个样本的转移概率就是

$$
p(x_{t+\Delta t}=y\mid x_t = x) = \delta_{xy} + Q_t(y,x)\,\Delta t + O(\Delta t^2)
$$

读法：在 $\Delta t$ 这么短的时间内，样本大概率停在原地；以大约 $Q_t(y,x)\Delta t$ 的概率跳到某个 $y\ne x$。这样一个过程叫**连续时间马尔可夫链**（CTMC）。它和 D3PM 的关系是：把 D3PM 的每一步转移矩阵写成 $I + \Delta t\,Q_t$ 并让 $\Delta t\to 0$，就得到 CTMC。

**噪声调度。** 实践中通常取 $Q_t = \sigma(t)\,Q$：$Q$ 是一个固定的基础矩阵，$\sigma(t)>0$ 是噪声强度。定义累积噪声

$$
\bar\sigma(t) = \int_0^t \sigma(s)\,ds
$$

因为不同时刻的 $Q_t$ 都是同一个 $Q$ 乘标量，它们彼此可交换，ODE 的解可以直接写成矩阵指数：

$$
p_t = \exp\big(\bar\sigma(t)\,Q\big)\,p_0
$$

于是**从 $x_0$ 到 $x_t$ 的转移概率**就是这个矩阵指数的第 $x_0$ 列：

$$
p_{t|0}(x_t\mid x_0) = \exp\big(\bar\sigma(t)\,Q\big)(x_t,\,x_0)
$$

这对应 DDPM 的 $q(x_t\mid x_0)$ 和 D3PM 的 $\mathrm{Cat}(x_0\bar Q_t)$：训练时可以从 $x_0$ 一步采出任意时刻的 $x_t$。只要 $\bar\sigma(T)$ 足够大，$p_T$ 就接近 $Q$ 决定的极限分布 $p_{\text{base}}$（比如均匀分布，或者全是 [MASK]）。

### 1.3 逆向过程：速率乘以概率比值

连续扩散里有一个重要结论：前向 SDE 有一个显式的逆时间 SDE，其中唯一未知的量是 score。离散的 CTMC 也有完全对应的结论：

$$
\frac{dp_{T-t}}{dt} = \tilde Q_{T-t}\,p_{T-t}
$$

逆向速率矩阵的非对角元为

$$
\tilde Q_t(y,x) = \frac{p_t(y)}{p_t(x)}\,Q_t(x,y),\qquad y\ne x
$$

对角元照例取 $\tilde Q_t(x,x) = -\sum_{y\ne x}\tilde Q_t(y,x)$。

**推导。** 用"概率流守恒"来理解最直接。在 $[t, t+\Delta t]$ 内，前向过程中从 $y$ 跳到 $x$ 的概率质量是

$$
p_t(y)\,Q_t(x,y)\,\Delta t
$$

逆着时间看，这正是从 $x$ 跳回 $y$ 的那部分质量。逆向过程在 $x$ 处的质量是 $p_t(x)$，所以逆向从 $x$ 跳到 $y$ 的速率是

$$
\tilde Q_t(y,x) = \frac{p_t(y)\,Q_t(x,y)}{p_t(x)}
$$

这就是上式。也可以直接验证：把它代入 $\frac{d}{dt}p_{T-t} = -\dot p_{T-t}$ 两边，用 $\sum_{y}Q_t(y,x)=0$ 整理，两边相等。

注意 $Q_t(x,y)$ 的下标顺序：前向是"从 $y$ 到 $x$"，逆向是"从 $x$ 到 $y$"，正好反过来。

**结论：只要知道比值 $p_t(y)/p_t(x)$，就能把前向过程倒过来跑。** 前向的 $Q_t$ 是我们自己设计的，是已知的，所以学习逆向过程完全归结为学习这些比值。

### 1.4 Concrete score：离散空间的 score

比值向量

$$
\left[\frac{p_t(y)}{p_t(x)}\right]_{y\ne x}
$$

称为 **concrete score**（Meng et al., 2022）。为什么说它是 score 的离散版本？

在离散结构上，可以把"梯度"定义为差分：$\nabla f(x)_y = f(y) - f(x)$，其中 $y$ 取遍 $x$ 的邻居。连续情形下 score 可以写成 $\nabla_x\log p = \nabla_x p / p$。照搬到离散情形：

$$
\frac{\nabla p(x)_y}{p(x)} = \frac{p(y) - p(x)}{p(x)} = \frac{p(y)}{p(x)} - 1
$$

所以"归一化的梯度"就是比值减 1，比值本身就承载了 score 的全部信息。

两者的作用也完全对应：

| | 连续扩散 | 离散扩散（SEDD） |
|---|---|---|
| 前向过程 | SDE $dx = f\,dt + g\,dw$ | CTMC $\frac{dp_t}{dt} = Q_t p_t$ |
| 逆向过程需要的未知量 | $\nabla_x\log p_t(x)$ | $p_t(y)/p_t(x)$ |
| 逆向过程 | 漂移项加上 $-g^2\nabla_x\log p_t$ | 速率乘以 $p_t(y)/p_t(x)$ |
| 学习目标 | 分数匹配 | score entropy（本文） |

还有一个和连续 score 相同的好性质：**比值和归一化常数无关**。若 $p(x) = \tilde p(x)/Z$，则 $p(y)/p(x) = \tilde p(y)/\tilde p(x)$，$Z$ 被消掉了。这意味着网络不需要输出一个归一化的分布，只需要输出相对大小。

## 2. 怎么学这些比值：已有方法的问题

目标是用网络 $s_\theta(x,t)$ 去逼近 concrete score：

$$
s_\theta(x,t)_y \approx \frac{p_t(y)}{p_t(x)},\qquad y\ne x
$$

在 SEDD 之前，大致有三类做法。

**均值预测（Mean Prediction）。** 这是 D3PM 和 Campbell et al.（2022）的做法：不直接参数化比值，而是学习去噪分布 $p_{0|t}(x_0\mid x_t)$，再间接得到逆向过程。问题有两个：其一，论文认为这是在绕路学一个更难的对象；其二，D3PM 的离散时间损失在连续时间极限下会出问题，必须做近似，这些近似在实践中影响效果。

**比率匹配（Ratio Matching，Hyvärinen, 2007）。** 通过最大似然学习每个维度上的条件边际概率。它偏离了标准分数匹配的框架，需要特殊而昂贵的网络结构，实证表现一般不如均值预测。

**Concrete Score Matching（CSM，Meng et al., 2022）。** 把分数匹配的 Fisher 散度直接推广过来，用平方误差回归比值：

$$
\mathcal L_{\text{CSM}} = \frac12\,\mathbb E_{x\sim p_t}\left[\sum_{y\ne x}\left(s_\theta(x,t)_y - \frac{p_t(y)}{p_t(x)}\right)^2\right]
$$

这看起来是最自然的推广，但实践中训练很困难。原因在于**$\ell^2$ 损失和"比值必须为正"不兼容**：

- 真实比值可能非常小（接近 0），也可能非常大（跨好几个数量级）。平方误差对所有尺度一视同仁，对小比值的拟合几乎不施加压力；
- 网络输出跑到 0 或负数时，平方损失的惩罚只是有限值，而逆向过程中的速率 $s_\theta\cdot Q$ 必须非负，负值没有意义；
- 论文的实验中，CSM 的训练会发散或明显落后。

所以需要一个"知道比值是正数"的损失。

## 3. Score Entropy

### 3.1 定义

**定义（Score Entropy）.** 给定分布 $p$、非负权重 $w_{xy}\ge 0$ 和比值网络 $s_\theta(x)_y>0$，

$$
\mathcal L_{\text{SE}} = \mathbb E_{x\sim p}\left[\sum_{y\ne x} w_{xy}\left(s_\theta(x)_y - \frac{p(y)}{p(x)}\log s_\theta(x)_y + K\!\left(\frac{p(y)}{p(x)}\right)\right)\right]
$$

其中

$$
K(a) = a(\log a - 1)
$$

是一个只依赖真实比值的**归一化常数**，它不影响优化，作用是让损失的最小值恰好为 0。

先从直觉上拆开看括号里的三项。记真实比值为 $r = p(y)/p(x)$，网络输出为 $s$：

- **$-r\log s$**：形式上就是交叉熵 $-\sum p_{\text{true}}\log p_{\text{pred}}$，用真实比值加权网络输出的对数，鼓励 $s$ 变大；
- **$s$**：标准交叉熵针对的是概率，有"和为 1"的约束。这里的 $s$ 是任意正数，没有这个约束。如果只有 $-r\log s$，最优解会让 $s\to\infty$。线性项 $s$ 正好起到惩罚作用：$s$ 越大，这一项线性增长，把 $s$ 拉回来；
- **$K(r)$**：常数，保证最小值为 0。

两股力量在 $s = r$ 处平衡：对 $s$ 求导，$1 - r/s = 0$。这就是 score entropy 能恢复真实比值的原因。

"entropy"这个名字来自它和交叉熵的关系：它把交叉熵从概率单纯形推广到了任意正实数。

### 3.2 为什么非负：一个 Bregman 散度

把括号里的单项写成二元函数

$$
\ell(r, s) = s - r\log s + r\log r - r
$$

这正是函数 $F(u) = u\log u$ 的 **Bregman 散度**：

$$
D_F(r, s) = F(r) - F(s) - F'(s)(r - s)
= r\log r - s\log s - (\log s + 1)(r - s)
= r\log r - r\log s - r + s
$$

和 $\ell(r,s)$ 完全一致。$D_F$ 也就是正数上的**广义 KL 散度**（不要求和为 1 的 KL）。由于 $F(u)=u\log u$ 严格凸，Bregman 散度满足：

$$
D_F(r,s)\ge 0,\quad\text{等号当且仅当 } s = r
$$

论文里的表述是用 $F=-\log$ 的 Bregman 散度：$D_{-\log}(s, r) = -\log s + \log r + \frac{s - r}{r}$，乘以 $r$ 后正好等于 $\ell(r,s)$。两种说法等价，用 $u\log u$ 的版本可以不带额外的权重 $r$，看起来更直接。

因此 score entropy 有三个直接的性质：**非负、关于 $s$ 是凸的、在 $s = r$ 处取到 0**。

### 3.3 一致性

**命题（一致性）.** 设 $p$ 全支撑（每个状态的概率都大于 0），$w_{xy}>0$。在样本无限、模型容量足够的极限下，最小化 $\mathcal L_{\text{SE}}$ 得到的最优参数满足

$$
s_{\theta^*}(x)_y = \frac{p(y)}{p(x)}
$$

此时 $\mathcal L_{\text{SE}} = 0$。

**证明。** $\mathcal L_{\text{SE}}$ 是若干项 $p(x)\,w_{xy}\,D_F\big(p(y)/p(x),\ s_\theta(x)_y\big)$ 之和。每一项的系数都大于 0，每个 $D_F$ 都非负，且仅在 $s = r$ 处为 0。模型容量足够时每一项可以独立取到 0，因此最优解就是逐项取真实比值。$\square$

### 3.4 梯度的自动缩放

这是相对 CSM 最关键的改进。取 $w_{xy}=1$，对单个输出 $s=s_\theta(x)_y$ 求导：

$$
\frac{\partial\,\ell}{\partial s} = 1 - \frac{r}{s} = \frac{s - r}{s},
\qquad
\frac{\partial\,\ell_{\text{CSM}}}{\partial s} = s - r
$$

所以

$$
\nabla_{s}\mathcal L_{\text{SE}} = \frac{1}{s}\,\nabla_{s}\mathcal L_{\text{CSM}}
$$

score entropy 的梯度是 CSM 梯度乘以 $1/s$：

- 当 $s$ 很小时，梯度被放大。$s\to 0$ 时 $-r\log s\to+\infty$，形成一个天然的**对数障碍**（log-barrier），网络输出不可能跑到 0 或负数；
- 当 $s$ 很大时，梯度被压缩，相当于在**相对尺度**上衡量误差。比值从 $10^{-3}$ 到 $10^3$ 的变化都能得到合理的学习信号。

这正好解决了第 2 节里 $\ell^2$ 损失的两个问题。

### 3.5 隐式 score entropy

$\mathcal L_{\text{SE}}$ 里出现了真实比值 $p(y)/p(x)$，而它正是我们不知道的东西。连续分数匹配遇到同样的问题时，有两条出路：隐式分数匹配（分部积分消掉未知的 score）和去噪分数匹配（用条件 score 代替）。score entropy 也有完全对应的两种形式。

**命题（隐式 score entropy）.** 在相差一个与 $\theta$ 无关的常数的意义下，

$$
\mathcal L_{\text{SE}} = \mathcal L_{\text{ISE}} + C,\qquad
\mathcal L_{\text{ISE}} = \mathbb E_{x\sim p}\left[\sum_{y\ne x}\Big(w_{xy}\,s_\theta(x)_y - w_{yx}\log s_\theta(y)_x\Big)\right]
$$

**证明。** $K$ 项与 $\theta$ 无关，归入常数；第一项 $w_{xy}s_\theta(x)_y$ 两边相同。只需处理含未知比值的那一项：

$$
\mathbb E_{x\sim p}\left[\sum_{y\ne x} w_{xy}\frac{p(y)}{p(x)}\log s_\theta(x)_y\right]
= \sum_x\sum_{y\ne x} p(y)\,w_{xy}\log s_\theta(x)_y
$$

$p(x)$ 被约掉了。交换求和变量的名字（把 $x$ 和 $y$ 互换）：

$$
= \sum_y\sum_{x\ne y} p(x)\,w_{yx}\log s_\theta(y)_x
= \mathbb E_{x\sim p}\left[\sum_{y\ne x}w_{yx}\log s_\theta(y)_x\right]
$$

不再出现未知比值。$\square$

这和隐式分数匹配里"分部积分把 $\nabla\log p$ 换成 $\nabla\cdot s_\theta$"是同一个思路：利用期望的结构，把未知的比值换成只依赖网络的量。

**问题在于代价。** 注意第二项是 $s_\theta(y)_x$：网络要在**所有邻居** $y$ 处各做一次前向传播。对文本序列，邻居数是 $d\times(N-1)$（每个位置换成任一其他 token），$d=1024$、$N\approx 5\times10^4$ 时完全不可行。即使用蒙特卡洛只采样部分邻居，方差也很大。这和隐式分数匹配需要计算 Jacobian 的迹、在高维时不可行是同一类问题。

### 3.6 去噪 score entropy

实际训练用的是去噪形式，它和去噪分数匹配（DSM）一一对应。

**定理（去噪 score entropy）.** 设 $p$ 是基分布 $p_0$ 经过扰动核 $p(\cdot\mid x_0)$ 得到的：

$$
p(x) = \sum_{x_0} p(x\mid x_0)\,p_0(x_0)
$$

则在相差一个常数的意义下，$\mathcal L_{\text{SE}}$ 等于

$$
\mathcal L_{\text{DSE}} = \mathbb E_{x_0\sim p_0}\,\mathbb E_{x\sim p(\cdot\mid x_0)}\left[\sum_{y\ne x}w_{xy}\left(s_\theta(x)_y - \frac{p(y\mid x_0)}{p(x\mid x_0)}\log s_\theta(x)_y\right)\right]
$$

**证明。** 同样只需处理含未知比值的项。把 $p(y)$ 展开成混合形式：

$$
\begin{aligned}
\mathbb E_{x\sim p}\left[\sum_{y\ne x}w_{xy}\frac{p(y)}{p(x)}\log s_\theta(x)_y\right]
&= \sum_x\sum_{y\ne x} w_{xy}\,p(y)\log s_\theta(x)_y \\
&= \sum_x\sum_{y\ne x} w_{xy}\sum_{x_0}p(y\mid x_0)\,p_0(x_0)\log s_\theta(x)_y \\
&= \sum_{x_0}p_0(x_0)\sum_x p(x\mid x_0)\sum_{y\ne x} w_{xy}\frac{p(y\mid x_0)}{p(x\mid x_0)}\log s_\theta(x)_y \\
&= \mathbb E_{x_0\sim p_0}\,\mathbb E_{x\sim p(\cdot\mid x_0)}\left[\sum_{y\ne x} w_{xy}\frac{p(y\mid x_0)}{p(x\mid x_0)}\log s_\theta(x)_y\right]
\end{aligned}
$$

第三步在分子分母同乘 $p(x\mid x_0)$。另一项 $\mathbb E_{x\sim p}[\sum w_{xy}s_\theta(x)_y]$ 本来就可以写成 $\mathbb E_{x_0}\mathbb E_{x\mid x_0}[\cdot]$。两式合并即得。$\square$

这里的关键是：**真实比值 $p(y)/p(x)$ 被换成了条件比值 $p(y\mid x_0)/p(x\mid x_0)$**，而后者由前向过程决定，有闭式解（第 1.2 节的矩阵指数）。

和连续情形对照：

| | 连续 | 离散 |
|---|---|---|
| 原始目标 | $\lVert s_\theta - \nabla\log p_t\rVert^2$ | $\mathcal L_{\text{SE}}$，含 $p_t(y)/p_t(x)$ |
| 隐式形式 | 含 $\mathrm{tr}(\nabla s_\theta)$，高维太贵 | 含 $s_\theta(y)_x$，要遍历邻居，太贵 |
| 去噪形式 | 用 $\nabla\log p(x_t\mid x_0)$ 代替 | 用 $p(y\mid x_0)/p(x\mid x_0)$ 代替 |

训练流程和 DDPM 一样：采一个干净样本，按前向过程加噪，让网络在加噪样本上回归一个可以算出来的目标。

## 4. 似然界：把损失变成困惑度

语言模型的标准指标是困惑度（perplexity），它需要计算或上界 $-\log p_\theta(x_0)$。SEDD 证明了 score entropy 可以直接给出这个上界。

### 4.1 参数化的逆向过程

把网络输出代入第 1.3 节的逆向速率公式，就得到**参数化的逆向速率矩阵**：

$$
\tilde Q^\theta_t(y,x) =
\begin{cases}
s_\theta(x,t)_y\,Q_t(x,y), & y\ne x \\[4pt]
-\sum_{z\ne x}\tilde Q^\theta_t(z,x), & y = x
\end{cases}
$$

它定义了一个从 $p_{\text{base}}$ 出发的逆向 CTMC，对应的分布族 $p^\theta_t$ 满足

$$
\frac{dp^\theta_{T-t}}{dt} = \tilde Q^\theta_{T-t}\,p^\theta_{T-t},\qquad p^\theta_T = p_{\text{base}}
$$

它在 $t=0$ 时的分布 $p^\theta_0$ 就是模型的生成分布。

### 4.2 定理

**定理（似然上界）.** 对任意数据点 $x_0$，

$$
-\log p^\theta_0(x_0) \le \mathcal L_{\text{DWDSE}}(x_0) + D_{\mathrm{KL}}\big(p_{T|0}(\cdot\mid x_0)\,\big\|\,p_{\text{base}}\big)
$$

其中**扩散加权的去噪 score entropy**（diffusion weighted denoising score entropy）为

$$
\mathcal L_{\text{DWDSE}}(x_0) = \int_0^T \mathbb E_{x_t\sim p_{t|0}(\cdot\mid x_0)}\sum_{y\ne x_t} Q_t(x_t,y)\left(s_\theta(x_t,t)_y - \frac{p_{t|0}(y\mid x_0)}{p_{t|0}(x_t\mid x_0)}\log s_\theta(x_t,t)_y + K\!\left(\frac{p_{t|0}(y\mid x_0)}{p_{t|0}(x_t\mid x_0)}\right)\right)dt
$$

和第 3.6 节的 $\mathcal L_{\text{DSE}}$ 比较，它就是：

- 权重取 $w_{xy} = Q_t(x,y)$，即前向过程从 $y$ 跳到 $x$ 的速率；
- 对所有时间 $t\in[0,T]$ 积分；
- 保留常数项 $K$，这样它才是似然的真正上界，而不只是差一个常数。

第二项 $D_{\mathrm{KL}}(p_{T|0}\,\|\,p_{\text{base}})$ 是先验项，和 $\theta$ 无关，$T$ 足够大时接近 0。它对应 DDPM ELBO 中的 $L_T$。

### 4.3 推导思路

这个定理的证明和连续扩散里用 Girsanov 定理推导似然界是同一个结构，这里给出关键步骤。

**第一步：路径 KL。** 考虑两条从 $T$ 到 $0$ 的逆向 CTMC：真实逆向过程（速率 $\tilde Q_t$）和模型逆向过程（速率 $\tilde Q^\theta_t$）。两条马尔可夫链在整条路径上的 KL 散度，有一个标准公式：

$$
D_{\mathrm{KL}}(\text{路径}\,\|\,\text{路径}^\theta) = D_{\mathrm{KL}}(p_T\,\|\,p_{\text{base}}) + \int_0^T\mathbb E_{x_t\sim p_t}\sum_{y\ne x_t}\left(\tilde Q^\theta_t(y,x_t) - \tilde Q_t(y,x_t) + \tilde Q_t(y,x_t)\log\frac{\tilde Q_t(y,x_t)}{\tilde Q^\theta_t(y,x_t)}\right)dt
$$

直观上，被积项是在每个瞬间"真实跳跃速率"与"模型跳跃速率"之间的广义 KL。

**第二步：代入比值。** 把 $\tilde Q_t(y,x) = r\,Q_t(x,y)$、$\tilde Q^\theta_t(y,x) = s\,Q_t(x,y)$ 代入（$r$ 为真实比值，$s$ 为网络输出），被积项变成

$$
Q_t(x,y)\big(s - r + r\log r - r\log s\big) = Q_t(x,y)\big(s - r\log s + K(r)\big)
$$

这正是以 $Q_t(x,y)$ 为权重的 score entropy。所以**路径 KL 等于 score entropy 对时间的积分**，这是 score entropy 形式的来源。

**第三步：数据处理不等式与条件化。** 路径 KL 大于等于两端边缘分布的 KL，于是 $D_{\mathrm{KL}}(p_0\,\|\,p_0^\theta)\le$ 上式。把它对单个数据点 $x_0$ 条件化（真实路径从 $x_0$ 出发），再用第 3.6 节的去噪变换把真实比值换成条件比值，就得到对单点的界 $-\log p_0^\theta(x_0)\le\mathcal L_{\text{DWDSE}}(x_0)+D_{\mathrm{KL}}(p_{T|0}(\cdot\mid x_0)\,\|\,p_{\text{base}})$。

**意义。** 训练时最小化的东西，就是对数似然上界的主体。降低 score entropy 损失，就是在降低困惑度的上界。这让 SEDD 可以和 GPT-2 在同一个指标下比较，也是它与先前很多扩散语言模型（只能报告生成样本质量）的重要区别。

## 5. 落到文本上：实现细节

### 5.1 序列状态空间与稀疏的 $Q_t$

文本是长度为 $d$ 的 token 序列，状态空间 $\mathcal X = \{1,\dots,N\}^d$，大小是 $N^d$。完整的 $N^d\times N^d$ 速率矩阵显然不可能存储，比值向量也有 $N^d$ 个分量。

SEDD 的做法和 D3PM 一样：**前向过程中每个 token 独立地加噪**。在连续时间里，这意味着同一个瞬间只有一个位置会跳变（两个独立的指数时钟同时响的概率为 0）。所以序列级的速率矩阵只在**汉明距离为 1** 的序列之间非零：

$$
Q_t\big(x^1\dots\hat x^i\dots x^d,\ x^1\dots x^i\dots x^d\big) = Q^{\text{tok}}_t(\hat x^i, x^i)
$$

读法：只把第 $i$ 个位置从 $x^i$ 换成 $\hat x^i$、其余位置不变，速率由单 token 的矩阵 $Q^{\text{tok}}_t$ 决定；改动两个及以上位置的速率为 0。

由于逆向速率是 $s\cdot Q_t$，$Q_t$ 为 0 的地方逆向速率也为 0。所以**只需要学习汉明距离为 1 的序列之间的比值**。

### 5.2 网络结构：序列到序列

需要的比值总数是 $d\times(N-1)$：每个位置 $i$、每个可能的替换 token $\hat x^i$。这正好是一个序列到序列网络（如 Transformer）的输出形状 $d\times N$：

$$
s_\theta(x,t)_{i,\hat x^i} \approx \frac{p_t(x^1\dots\hat x^i\dots x^d)}{p_t(x^1\dots x^i\dots x^d)}
$$

输入是加噪后的整个序列 $x$（以及时间 $t$），输出是"每个位置换成每个 token 后，整句概率变为原来的多少倍"。一次前向传播就得到所有需要的比值，这和 BERT 输出每个位置上的词表分布的形状完全一样。比值必须为正，所以网络在最后对输出取指数（输出的是对数比值）。

$\hat x^i = x^i$ 的那一项没有意义（比值恒为 1），实现时直接忽略或置为 1。

### 5.3 token 级转移概率的闭式解

因为各位置独立加噪，序列级转移概率分解为 token 级的乘积：

$$
p^{\text{seq}}_{t|0}(\hat x\mid x) = \prod_{i=1}^d p^{\text{tok}}_{t|0}(\hat x^i\mid x^i)
$$

取 $Q^{\text{tok}}_t = \sigma(t)\,Q^{\text{tok}}$，由第 1.2 节，token 级转移概率是矩阵指数 $\exp(\bar\sigma(t)Q^{\text{tok}})$ 的第 $x^i$ 列。

GPT-2 的词表大小 $N = 50257$，一个 $N\times N$ 的矩阵指数无法直接计算和存储。SEDD 用了两种有闭式解的结构。

**均匀（Uniform）.**

$$
Q^{\text{uniform}} = \mathbf 1\mathbf 1^\top - N I =
\begin{bmatrix}
1-N & 1 & \cdots & 1 \\
1 & 1-N & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 1-N
\end{bmatrix}
$$

每个 token 以相同的速率跳到任意其他 token。验证：每列有 $N-1$ 个 1 和一个 $1-N$，和为 0。

记 $U = \frac1N\mathbf 1\mathbf 1^\top$，它是幂等的（$U^2 = U$）。$Q^{\text{uniform}} = N(U - I)$，而 $I$ 与 $U$ 可交换，所以

$$
\exp\big(\bar\sigma Q^{\text{uniform}}\big) = e^{-N\bar\sigma}\exp(N\bar\sigma U)
= e^{-N\bar\sigma}\big(I + (e^{N\bar\sigma}-1)U\big)
= e^{-N\bar\sigma}\,I + \big(1 - e^{-N\bar\sigma}\big)\,U
$$

第二步用了幂等矩阵的指数：$\exp(cU) = I + (e^c - 1)U$（把级数展开，$U^k = U$）。读法：以概率 $e^{-N\bar\sigma}$ 保持原样，以概率 $1-e^{-N\bar\sigma}$ 重新从词表中均匀采样。这正是 D3PM 的均匀转移 $\bar\alpha I + (1-\bar\alpha)U$，其中 $\bar\alpha = e^{-N\bar\sigma}$。

**吸收态（Absorb）.** 在词表外加一个 [MASK] 状态 $m$，状态数变为 $N+1$：

$$
Q^{\text{absorb}} = e_m\mathbf 1^\top - I =
\begin{bmatrix}
-1 & 0 & \cdots & 0 & 0 \\
0 & -1 & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & -1 & 0 \\
1 & 1 & \cdots & 1 & 0
\end{bmatrix}
$$

最后一行、最后一列对应 [MASK]。每个普通 token 以速率 1 跳到 [MASK]；[MASK] 那一列全为 0，一旦变成 [MASK] 就不再离开。验证：普通 token 的列是 $-1$ 加 $1$，和为 0。

记 $A = e_m\mathbf 1^\top$，同样有 $A^2 = e_m(\mathbf 1^\top e_m)\mathbf 1^\top = A$，于是

$$
\exp\big(\bar\sigma Q^{\text{absorb}}\big) = e^{-\bar\sigma}\,I + \big(1 - e^{-\bar\sigma}\big)\,e_m\mathbf 1^\top
$$

读法：每个 token 以概率 $e^{-\bar\sigma}$ 保持原样，以概率 $1-e^{-\bar\sigma}$ 变成 [MASK]。这就是 D3PM 的吸收态扩散，$\bar\alpha = e^{-\bar\sigma}$。

两种结构都只需要 $O(N)$ 的计算，不用存储矩阵。论文发现**吸收态在困惑度上明显更好**，这和 D3PM 在文本上的结论一致，也和 BERT 式的掩码预训练相呼应。

**噪声调度。** 吸收态常用 log-linear 调度：

$$
\bar\sigma(t) = -\log\big(1 - (1-\epsilon)\,t\big),\qquad t\in[0,1]
$$

此时保留概率 $e^{-\bar\sigma(t)} = 1 - (1-\epsilon)t$ 随时间线性下降，$t=1$ 时几乎全部被 mask（$\epsilon$ 是一个很小的数，避免 $\bar\sigma$ 发散）。这和 D3PM 的互信息调度 $\bar\alpha_t = 1 - t/T$ 是同一件事的连续版本。

### 5.4 吸收态下的损失长什么样

把吸收态代入 $\mathcal L_{\text{DWDSE}}$，可以把损失化成非常具体的形式，这一步对理解后面的 MDLM 和 RADD 很重要。

固定加噪序列 $x_t$。对位置 $i$ 和候选 token $v$，记 $y$ 为把 $x_t$ 的第 $i$ 位换成 $v$ 的序列。需要权重 $Q_t(x_t, y) = \sigma(t)\,Q^{\text{absorb}}(x_t^i, v)$，即**从 $y$ 跳到 $x_t$ 的速率**：

- 若 $x_t^i$ 是普通 token：要让 $y$ 跳到 $x_t$，第 $i$ 位必须从 $v$ 变成一个普通 token。但普通 token 之间不互相跳，[MASK] 也不会跳回 token，所以速率为 0。**未被 mask 的位置对损失没有贡献。**
- 若 $x_t^i = m$（[MASK]）：$Q^{\text{absorb}}(m, v) = 1$ 对所有普通 token $v$ 成立。

再看条件比值。在被 mask 的位置 $i$，只有第 $i$ 位不同，其他位置的因子约掉：

$$
\frac{p_{t|0}(y\mid x_0)}{p_{t|0}(x_t\mid x_0)} = \frac{p^{\text{tok}}_{t|0}(v\mid x_0^i)}{p^{\text{tok}}_{t|0}(m\mid x_0^i)}
=
\begin{cases}
\dfrac{e^{-\bar\sigma(t)}}{1-e^{-\bar\sigma(t)}}, & v = x_0^i \\[8pt]
0, & v\ne x_0^i
\end{cases}
$$

所以吸收态的 $\mathcal L_{\text{DWDSE}}$（忽略常数项 $K$）是

$$
\int_0^1\sigma(t)\,\mathbb E_{x_t}\sum_{i:\,x_t^i = m}\left[\sum_{v\ne m}s_\theta(x_t,t)_{i,v} - \frac{e^{-\bar\sigma(t)}}{1-e^{-\bar\sigma(t)}}\log s_\theta(x_t,t)_{i,x_0^i}\right]dt
$$

只在被 mask 的位置上计算，且只有正确 token 那一项带 $\log$。这已经非常接近带权重的掩码语言模型交叉熵。

在 log-linear 调度下（取 $\epsilon\to 0$），$\bar\sigma(t) = -\log(1-t)$，$\sigma(t) = \frac{1}{1-t}$，于是 $\log$ 项前面的总权重是

$$
\sigma(t)\cdot\frac{e^{-\bar\sigma(t)}}{1-e^{-\bar\sigma(t)}} = \frac{1}{1-t}\cdot\frac{1-t}{t} = \frac1t
$$

和上一篇 D3PM 推出的"吸收态 D3PM 是按 $1/t$ 加权的 MLM"完全一致。

**预告：真实比值有更简单的结构。** 在吸收态下，真实的 concrete score 也可以直接写出来。设 $x_t$ 中被 mask 的位置集合是 $M$，未 mask 的部分记作 $x_t^{\text{UM}}$。$x_t$ 的概率等于"未 mask 的 token 恰好保留、其余恰好被 mask"的概率乘以干净数据在未 mask 位置上的边缘：

$$
p_t(x_t) = \big(e^{-\bar\sigma}\big)^{d-|M|}\big(1-e^{-\bar\sigma}\big)^{|M|}\,p_0\big(x_t^{\text{UM}}\big)
$$

把位置 $i\in M$ 从 [MASK] 换成 $v$ 后，未 mask 的位置多一个，mask 的位置少一个，于是

$$
\frac{p_t(y)}{p_t(x_t)} = \frac{e^{-\bar\sigma(t)}}{1-e^{-\bar\sigma(t)}}\cdot p_0\big(x^i = v\mid x_t^{\text{UM}}\big)
$$

真实比值 = 一个**只依赖时间的标量** × 一个**与时间无关的条件分布**（给定未 mask 的 token，被 mask 位置的干净 token 的分布）。这意味着网络其实不需要输入时间 $t$，也意味着 SEDD 吸收态和"直接预测 $x_0$"的 MDLM 本质上是同一个东西。后面 MDLM 和 RADD 两篇会分别从这两个角度展开。

### 5.5 训练流程

1. 从数据中采样一个序列 $x_0$；
2. 采样时间 $t\sim\mathcal U[0,1]$，算出 $\bar\sigma(t)$ 和 $\sigma(t)$；
3. 每个位置独立地按 token 级转移概率加噪，得到 $x_t$（吸收态：每个位置以概率 $1-e^{-\bar\sigma(t)}$ 变成 [MASK]）；
4. 网络前向传播，输出 $s_\theta(x_t,t)\in\mathbb R^{d\times N}$；
5. 按 $\mathcal L_{\text{DWDSE}}$ 的被积项计算损失（权重 $\sigma(t)Q^{\text{tok}}(x_t^i,v)$、条件比值用闭式解），反向传播。

评估困惑度时，对 $t$ 做多次蒙特卡洛采样估计 $\mathcal L_{\text{DWDSE}}(x_0)$，加上先验项，就得到 $-\log p_\theta(x_0)$ 的上界。

## 6. 采样：用 concrete score 模拟逆向过程

训练好 $s_\theta$ 之后，要从 $x_T\sim p_{\text{base}}$ 出发，模拟逆向 CTMC 得到 $x_0$。

### 6.1 直接用 Euler 法太慢

最直接的做法是对逆向 CTMC 做 Euler 离散化，在序列级别上采样：

$$
p(x_{t-\Delta t}=y\mid x_t = x) \approx \delta_{xy} + \Delta t\,\tilde Q^\theta_t(y,x)
$$

但序列级的 $\tilde Q^\theta_t$ 只在汉明距离为 1 的序列之间非零，**每一步最多只能改动一个位置**。生成 1024 个 token 至少要 1024 步，比自回归还慢，失去了扩散模型并行生成的意义。

### 6.2 $\tau$-leaping：所有位置并行更新

**$\tau$-leaping**（来自化学反应模拟，Campbell et al. 2022 引入离散扩散）的思路是：在一个时间步 $\Delta t$ 内，假设所有位置的跳变**互相独立、同时发生**。每个位置 $i$ 独立地按下式采样：

$$
p\big(x^i_{t-\Delta t}\mid x_t\big) = \delta_{x^i_t}\big(x^i_{t-\Delta t}\big) + \Delta t\,Q^{\text{tok}}_t\big(x^i_t, x^i_{t-\Delta t}\big)\,s_\theta(x_t,t)_{i,\,x^i_{t-\Delta t}}
$$

各项的含义：

- $\delta_{x^i_t}(\cdot)$ 是 Kronecker delta：大部分情况下 token 保持不变；
- $\Delta t\,Q^{\text{tok}}_t(x^i_t, \cdot)$：前向过程中从候选 token 跳到当前 token 的速率，决定了逆向时**哪些跳变是允许的**。例如吸收态下，只有 [MASK] 可以跳回普通 token；
- $s_\theta(x_t,t)_{i,\cdot}$：网络给出的比值，决定**往哪个 token 跳**。没有它，逆向过程就只是按前向速率乱跳；乘上它之后，跳变被引向让整句概率更高的方向。

两个实现细节：对角项（保持不变的概率）由"1 减去其他项之和"确定；$\Delta t$ 太大时某些概率可能为负或总和超过 1，需要截断后再归一化。

注意 $s_\theta$ 的输入是整个序列 $x_t$，所以每个位置的更新是"看过全句"的；独立性只是指在同一步内各位置**互相不知道对方要怎么变**。这是并行采样引入的近似，步数越少，近似误差越大。

### 6.3 离散 Tweedie 公式

$\tau$-leaping 只用了 $s_\theta$ 的一阶信息（"速率 × 比值 × $\Delta t$"）。SEDD 指出，concrete score 其实包含足够的信息来做**最优去噪**，这是连续扩散中 Tweedie 公式的离散版本。

**定理（离散 Tweedie）.** 设 $p_t$ 满足 $\frac{dp_t}{dt} = Q\,p_t$（$Q$ 不随时间变化，否则把下式中的 $t$ 换成 $\bar\sigma(t)$）。则真实的去噪分布为

$$
p_{0|t}(x_0\mid x_t) = \underbrace{\left(\exp(-tQ)\left[\frac{p_t(i)}{p_t(x_t)}\right]_{i=1}^N\right)_{x_0}}_{\text{(A)}}\times\underbrace{\exp(tQ)(x_t, x_0)}_{\text{(B)}}
$$

**证明。** 由 ODE 的解 $p_t = \exp(tQ)\,p_0$，有 $p_0 = \exp(-tQ)\,p_t$。两边同除以标量 $p_t(x_t)$：

$$
\frac{p_0}{p_t(x_t)} = \exp(-tQ)\left[\frac{p_t(i)}{p_t(x_t)}\right]_{i=1}^N
$$

取第 $x_0$ 个分量，左边是 $p_0(x_0)/p_t(x_t)$，右边就是 (A)。再由贝叶斯公式

$$
p_{0|t}(x_0\mid x_t) = \frac{p_{t|0}(x_t\mid x_0)\,p_0(x_0)}{p_t(x_t)}
$$

其中 $p_{t|0}(x_t\mid x_0) = \exp(tQ)(x_t, x_0)$ 就是 (B)。两者相乘即得。$\square$

两项的含义：

- **(B) 似然**：如果干净数据是 $x_0$，经过时间 $t$ 变成 $x_t$ 的概率。这由前向过程决定，已知；
- **(A) 先验比**：$p_0(x_0)/p_t(x_t)$。它用 $\exp(-tQ)$ 把 $t$ 时刻的比值"倒推"回 $0$ 时刻，得到干净数据的相对概率。

连续情形的 Tweedie 公式说的是"知道 score 就知道后验均值 $\mathbb E[x_0\mid x_t]$"；离散版本说的是"知道 concrete score 就知道整个后验分布 $p_{0|t}$"。

### 6.4 Tweedie $\tau$-leaping

把同样的思路用在从 $t$ 到 $t-\Delta t$ 的一小步上（而不是一步到 $0$），就得到 SEDD 实际使用的采样器。记这一步内的噪声增量

$$
\sigma^{\Delta t}_t = \bar\sigma(t) - \bar\sigma(t-\Delta t)
$$

每个位置 $i$ 独立地按下式采样：

$$
p\big(x^i_{t-\Delta t}\mid x_t\big) \propto \Big(\exp\big(-\sigma^{\Delta t}_t Q^{\text{tok}}\big)\,s_\theta(x_t,t)_i\Big)_{x^i_{t-\Delta t}}\times\exp\big(\sigma^{\Delta t}_t Q^{\text{tok}}\big)\big(x^i_t,\,x^i_{t-\Delta t}\big)
$$

这里 $s_\theta(x_t,t)_i$ 是位置 $i$ 上的 $N$ 维比值向量，当前 token 自己那一项取 1。

**为什么是这个形式。** 对单个 token，从 $t$ 到 $s = t-\Delta t$ 的**真实**逆向转移是

$$
p(x_s\mid x_t) = \frac{p_{t|s}(x_t\mid x_s)\,p_s(x_s)}{p_t(x_t)}
$$

$p_{t|s}(x_t\mid x_s) = \exp(\sigma^{\Delta t}_t Q)(x_t, x_s)$ 是前向在这一小段上的转移；$p_s = \exp(-\sigma^{\Delta t}_t Q)\,p_t$，所以 $p_s(x_s)/p_t(x_t)$ 就是 $\exp(-\sigma^{\Delta t}_t Q)$ 作用在比值向量上再取第 $x_s$ 个分量。这和 6.3 节的推导完全相同，只是把 "$0$ 到 $t$" 换成了 "$s$ 到 $t$"。所以 Tweedie $\tau$-leaping 在**单个 token 上是精确的**逆向转移，而不是 $\tau$-leaping 那样的一阶近似。

两项可以这样理解：

- 右项 $\exp(\sigma^{\Delta t}_t Q)(x^i_t,\cdot)$：前向过程在这一小段时间里能否、以多大概率把候选 token 变成当前 token，决定了**哪些跳变在物理上合理**；
- 左项 $\big(\exp(-\sigma^{\Delta t}_t Q)\,s_\theta\big)$：把网络对比值的估计倒推到 $t-\Delta t$ 时刻，决定**候选 token 有多好**。

**吸收态下的具体形式。** 设位置 $i$ 当前是 [MASK]，记 $\bar\alpha_t = e^{-\bar\sigma(t)}$。可以验证，若 $s_\theta$ 是真实比值，上式化为

$$
p\big(x^i_{t-\Delta t} = v\mid x_t\big) = \frac{\bar\alpha_{t-\Delta t} - \bar\alpha_t}{1-\bar\alpha_t}\,p_0\big(x^i = v\mid x^{\text{UM}}_t\big),
\qquad
p\big(x^i_{t-\Delta t} = m\mid x_t\big) = \frac{1-\bar\alpha_{t-\Delta t}}{1-\bar\alpha_t}
$$

即：以一定概率在这一步"揭开"这个 mask，揭开时从条件分布中采样 token；否则继续保持 mask。这正是 D3PM 吸收态后验 $q(x_{t-1}\mid x_t, x_0)$ 对 $x_0$ 取期望的结果。已经揭开的 token 不会再变。

**最优性。** 论文证明：如果所有位置必须在一步内**独立并行**地更新（这是所有 $\tau$-leaping 类方法共同的限制），那么在 $s_\theta$ 准确的前提下，Tweedie $\tau$-leaping 是使采样分布与真实逆向过程 KL 散度最小的选择。从上面的推导看这很自然：它在每个位置上都用了精确的单 token 逆向转移，剩下的误差只来自"位置之间独立"这个近似本身。

### 6.5 任意位置的提示与填空

自回归模型只能从左往右生成，给定后文生成前文、或者在中间填空都很别扭。SEDD 可以在**不重新训练**的情况下做任意位置的条件生成。

**问题.** 把位置分成两部分：已知位置 $\bar\Omega$（提示，取值固定为 $y$）和待生成位置 $\Omega$。目标是从条件分布 $p_t(x^\Omega\mid x^{\bar\Omega} = y)$ 采样。

**关键等式.** 条件分布的比值等于无条件分布的比值：

$$
\frac{p_t\big(x^\Omega = z'\mid x^{\bar\Omega} = y\big)}{p_t\big(x^\Omega = z\mid x^{\bar\Omega} = y\big)} = \frac{p_t\big(x = z'\oplus_\Omega y\big)}{p_t\big(x = z\oplus_\Omega y\big)}
$$

其中 $z\oplus_\Omega y$ 表示在 $\Omega$ 位置填 $z$、在 $\bar\Omega$ 位置填 $y$ 拼成的完整序列。

**证明.** 由条件概率的定义，

$$
p_t\big(x^\Omega = z\mid x^{\bar\Omega}=y\big) = \frac{p_t(z\oplus_\Omega y)}{p_t\big(x^{\bar\Omega} = y\big)}
$$

两个条件概率相除时，分母 $p_t(x^{\bar\Omega}=y)$ 相同，直接约掉。$\square$

**含义.** 左边是"给定提示，在空里填 $z'$ 比填 $z$ 好多少"，右边是"填好之后的整句 $z'\oplus y$ 比 $z\oplus y$ 好多少"。后者正是无条件模型 $s_\theta$ 输出的比值（在 $\Omega$ 内只改动一个位置时，$z$ 和 $z'$ 汉明距离为 1，正好是网络输出的那些比值）。这和连续扩散里"条件 score 与无条件 score 只差一个与 $x$ 无关的项"是同一个现象：归一化常数在比值里自动消失。

**算法.**

1. 初始化：$\bar\Omega$ 位置填入提示 $y$，$\Omega$ 位置从 $p_{\text{base}}$ 采样（吸收态下全为 [MASK]）；
2. 每一步按 Tweedie $\tau$-leaping 对整个序列计算更新；
3. 只保留 $\Omega$ 位置的更新，把 $\bar\Omega$ 位置重置为 $y$；
4. 重复直到 $t = 0$。

由于网络看到的是整句，而 $\bar\Omega$ 被固定住，网络只能调整 $\Omega$ 位置的 token 去配合提示。提示可以在开头、结尾、中间，也可以是分散的若干片段。

一个需要注意的细节：严格来说，条件化应当在 $t$ 时刻的加噪序列上进行。对吸收态这一点自然成立，因为未被 mask 的 token 就是干净 token，把提示位置一直保持为干净 token 等价于"这些位置恰好一直没被 mask"。

## 7. 实验与局限

**实验设置.** 论文在 OpenWebText 上训练与 GPT-2 small、medium 规模相当的 SEDD 模型（使用 GPT-2 的分词器），和 GPT-2 以及先前的扩散语言模型比较。

**困惑度.** 论文报告：

- 相比先前的扩散语言模型（D3PM、PLAID 等），在同等规模下困惑度降低 25% 到 75%；
- 在 LAMBADA、WikiText、PTB、One Billion Words 等零样本困惑度评测上，SEDD 吸收态与 GPT-2 处于同一水平，在部分数据集上更好；
- 吸收态一致好于均匀扩散。

这里 SEDD 的数字是似然的**上界**，所以真实困惑度只会更低。

**生成质量.** 用更大的语言模型（如 GPT-2 large）给生成样本打分（生成困惑度）：

- 不做温度退火等技巧时，SEDD 的生成困惑度比同规模的 GPT-2 采样好得多（论文报告约 6 到 8 倍）；
- 采样步数可以灵活调整，用远少于 token 数的步数（论文报告约 32 倍少的网络调用）就能达到与 GPT-2 相近的质量。

**可控生成.** 第 6.5 节的填空能力在实验中得到展示：可以给定开头、结尾或中间的片段，生成连贯的文本，质量与从左往右的提示相当。

**局限.**

- **似然是上界**：$\mathcal L_{\text{DWDSE}}$ 与真实负对数似然之间仍有差距，且需要蒙特卡洛估计；
- **规模仍然有限**：实验集中在 GPT-2 规模，和真正大规模的自回归模型还有距离；
- **采样仍然要多步**：虽然步数灵活，但质量随步数减少而下降，并行更新的独立性近似在少步时代价明显；
- **生成困惑度的局限**：这个指标偏好低熵、重复的文本，和多样性存在权衡，后续工作（Zheng et al., 2024）还指出低精度的类别采样会让这个指标偏低，比较时需要谨慎；
- **时间输入可能是多余的**：第 5.4 节的分解表明，吸收态下网络其实不需要 $t$。SEDD 仍然把 $t$ 作为输入，这一点被后续工作简化。

## 8. 小结

- **离散空间里的 score 是概率比值** $p_t(y)/p_t(x)$（concrete score）。连续时间离散扩散（CTMC）的逆向速率就是"前向速率 × 比值"，学会比值就能把过程倒过来。
- **score entropy** 是专门为正数比值设计的损失：它是 $u\log u$ 的 Bregman 散度（广义 KL），非负、凸，在真实比值处为 0；梯度相对 $\ell^2$ 损失自动乘以 $1/s$，形成对数障碍。
- 和分数匹配一样，score entropy 有**隐式**和**去噪**两种可计算形式；实际训练用去噪形式，目标比值来自前向过程的闭式转移概率。
- **似然上界**：按前向速率加权、对时间积分的去噪 score entropy，加上先验项，就是 $-\log p_\theta(x_0)$ 的上界。这来自两条 CTMC 之间的路径 KL。
- **实现**：每个 token 独立加噪，只需学汉明距离为 1 的比值，正好是序列到序列网络的输出形状；均匀和吸收态两种 $Q$ 的矩阵指数都有闭式解，吸收态效果最好。
- **采样**：Tweedie $\tau$-leaping 在每个位置上用精确的单 token 逆向转移，是并行更新类方法中最优的；条件比值等于无条件比值，所以同一个模型可以直接做任意位置的填空。

SEDD 留下了一条线索：在吸收态下，真实比值 = 时间的标量函数 × 与时间无关的条件分布。换句话说，学比值和学"被 mask 的位置是什么 token"是一回事。下一篇的 MDLM（以及同期的 MD4）就沿着这个方向，把掩码扩散写成一个极其简洁的、加权的掩码语言模型损失。

## 参考文献

- Hyvärinen. *Some Extensions of Score Matching.* Computational Statistics & Data Analysis, 2007.
- Song & Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS 2019.
- Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021.
- Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS 2021.
- Campbell et al. *A Continuous Time Framework for Discrete Denoising Models.* NeurIPS 2022.
- Meng et al. *Concrete Score Matching: Generalized Score Matching for Discrete Data.* NeurIPS 2022.
- Lou, Meng & Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* ICML 2024.
- Zheng et al. *Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling.* 2024.
