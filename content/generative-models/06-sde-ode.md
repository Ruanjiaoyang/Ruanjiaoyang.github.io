---
title: "SDE 与概率流 ODE：用连续时间统一扩散模型"
title_en: "SDEs and the Probability Flow ODE: A Continuous-Time View of Diffusion"
series: "从 GAN 到 Diffusion"
order: 6
date: 2026-10-16
tags: [生成模型, Diffusion, SDE, ODE, Fokker-Planck]
summary_en: >
  Letting the number of diffusion steps go to infinity turns the forward process into a stochastic
  differential equation. This post derives the reverse-time SDE from Bayes' rule and a Taylor expansion,
  the continuous-time denoising score matching objective, and the VE and VP SDEs that recover NCSN and
  DDPM, and proves they are equivalent up to rescaling. It then derives the Fokker–Planck equation with
  the Dirac delta trick, uses it to build a family of SDEs that share the same marginals, and arrives at
  the probability flow ODE, showing why DDIM is its Euler discretization and covering samplers
  (reverse diffusion, predictor–corrector, ODE solvers) and exact likelihoods.
---

前两篇的 DDPM 和 NCSN 都是**离散**的：噪声水平是 $T=1000$ 个或 $L=10$ 个固定的值。Song et al. 在 ICLR 2021 发表的 *Score-Based Generative Modeling through Stochastic Differential Equations*（下文简称 SDE 论文）把步数取到无穷，用一条**随机微分方程**（Stochastic Differential Equation，SDE）连续地描述加噪过程，从 SDE 的视角对 score-based 模型和 DDPM 做了统一的解读。

在这个视角下：

- DDPM 和 NCSN 只是同一个框架下两组不同的系数；
- 生成过程是这条 SDE 的**时间反演**，只需要知道每个时刻的 score；
- 存在一条**确定性的常微分方程**（ODE），和 SDE 具有完全相同的边际分布，DDIM 就是它的离散化。

这一篇比前几篇用到更多数学工具，但每一步都会尽量从离散形式出发、用最朴素的方式推导。

## 1. 前向过程：从离散到连续

### 1.1 为什么用 SDE

扩散模型中 $x_0\to x_T$ 的加噪过程和 $x_T\to x_0$ 的去噪过程，本质上都是**随机过程**，而刻画随机过程的自然工具就是随机微分方程。

之前我们都是离散地看扩散模型：$x_0\to x_1\to\cdots\to x_T$。SDE 则是**连续**的描述，可以把前向过程改写成连续形式（和第 5 篇朗之万动力学的连续化思路一样）：

$$
dx = \underbrace{f(x,t)\,dt}_{\text{确定性部分}} + \underbrace{g(t)\,dw}_{\text{随机部分}}
$$

其中：

- $dx = \lim_{\Delta t\to0}(x_{t+\Delta t} - x_t)$，表示 $x$ 在一个无穷小时间段内的变化；
- $f(x,t)$：**漂移系数**（drift coefficient），一个向量值函数，决定数据确定性的演化趋势；
- $g(t)$：**扩散系数**（diffusion coefficient），一个标量函数，控制注入噪声的强度；
- $w$：标准**布朗运动**（Brownian motion，也叫维纳过程），一种具有独立增量和连续轨迹的随机过程。

时间也连续化了：之前的离散步数 $1\to2\to\cdots\to T$，现在改写为 $0\to\Delta t\to2\Delta t\to\cdots\to1$，即 $t\in[0,1]$。$t=0$ 对应干净数据，$t=1$ 对应（近似）纯噪声。

### 1.2 布朗运动与 $\sqrt{\Delta t}$

布朗运动的关键性质是：在任意一段长为 $\Delta t$ 的时间内，增量 $w(t+\Delta t)-w(t)$ 服从 $\mathcal{N}(0,\Delta t\,I)$，并且不同时间段的增量相互独立。所以

$$
dw \;\approx\; \sqrt{\Delta t}\;\epsilon, \qquad \epsilon\sim\mathcal{N}(0,I)
$$

注意是 $\sqrt{\Delta t}$ 而不是 $\Delta t$。这一点非常重要：它意味着随机部分的量级（$\sqrt{\Delta t}$）比确定部分（$\Delta t$）**大得多**，所以在做泰勒展开时，随机项要展开到二阶，因为 $(\sqrt{\Delta t}\,\epsilon)^2$ 和 $\Delta t$ 同阶。这是随机微积分（伊藤微积分）和普通微积分最大的区别，第 5 节推导 Fokker–Planck 方程时会用到。

### 1.3 离散形式

于是前向 SDE 的离散形式（Euler–Maruyama 格式）是

$$
x_{t+\Delta t} = x_t + f(x_t,t)\,\Delta t + g(t)\sqrt{\Delta t}\;\epsilon, \qquad \epsilon\sim\mathcal{N}(0,I)
$$

- $f(x_t,t)\,\Delta t$ 是**确定部分**，决定数据的演化趋势；
- $g(t)\sqrt{\Delta t}\,\epsilon$ 是**随机部分**，引入扩散噪声。

之前的离散前向过程是 $x_t\to x_{t+1}$，现在的连续形式则是 $x_t\to x_{t+\Delta t}$。等价地，一步转移是一个高斯分布：

$$
p(x_{t+\Delta t}\mid x_t) = \mathcal{N}\big(x_{t+\Delta t};\; x_t + f(x_t,t)\,\Delta t,\; g^2(t)\,\Delta t\,I\big) \propto \exp\Big(-\frac{\|x_{t+\Delta t} - x_t - f(x_t,t)\,\Delta t\|^2}{2g^2(t)\,\Delta t}\Big)
$$

## 2. 反向 SDE

### 2.1 目标

和 DDPM 的分析一样，有了一个连续的前向过程，我们希望有一个**反向的、还原真实分布的过程**：从 $t=1$ 的噪声出发，走回 $t=0$。为此需要反向一步的转移 $p(x_t\mid x_{t+\Delta t})$。

（为了简洁，下面把时刻 $t$ 的边际分布 $p_t(x_t)$ 简写为 $p(x_t)$。注意它既是 $x_t$ 的函数，也是 $t$ 的函数。）

### 2.2 推导

**第一步：贝叶斯公式。**

$$
p(x_t\mid x_{t+\Delta t}) = \frac{p(x_{t+\Delta t}\mid x_t)\;p(x_t)}{p(x_{t+\Delta t})} = p(x_{t+\Delta t}\mid x_t)\,\exp\big(\log p(x_t) - \log p(x_{t+\Delta t})\big)
$$

代入前向转移：

$$
p(x_t\mid x_{t+\Delta t}) \propto \exp\Big(-\frac{\|x_{t+\Delta t} - x_t - f(x_t,t)\,\Delta t\|^2}{2g^2(t)\,\Delta t} + \log p(x_t) - \log p(x_{t+\Delta t})\Big)
$$

**第二步：泰勒展开。** 对 $\log p(x_{t+\Delta t})$ 在 $(x_t,t)$ 处做一阶泰勒展开。注意 $p(x_t)$ 其实是 $(x_t,t)$ 两个变量的函数，所以展开时有两项：

$$
\log p(x_{t+\Delta t}) \approx \log p(x_t) + (x_{t+\Delta t} - x_t)\cdot\nabla_{x_t}\log p(x_t) + \Delta t\,\frac{\partial}{\partial t}\log p(x_t)
$$

代入后，$\log p(x_t)$ 抵消：

$$
p(x_t\mid x_{t+\Delta t}) \propto \exp\Big(-\frac{\|x_{t+\Delta t} - x_t - f\,\Delta t\|^2}{2g^2\,\Delta t} - (x_{t+\Delta t}-x_t)\cdot\nabla_{x_t}\log p(x_t) - \Delta t\,\frac{\partial}{\partial t}\log p(x_t)\Big)
$$

（$f$、$g$ 是 $f(x_t,t)$、$g(t)$ 的简写。）

**第三步：配方。** 记 $\Delta x = x_{t+\Delta t}-x_t$，$s = \nabla_{x_t}\log p(x_t)$。把前两项合并成一个完全平方：

$$
\|\Delta x - f\Delta t\|^2 + 2g^2\Delta t\;\Delta x\cdot s = \big\|\Delta x - \big(f - g^2 s\big)\Delta t\big\|^2 + \underbrace{2g^2\Delta t^2\,f\cdot s - g^4\Delta t^2\,\|s\|^2}_{\mathcal{O}(\Delta t^2)}
$$

（直接展开右边即可验证。）除以 $2g^2\Delta t$ 之后，多出来的项是 $\mathcal{O}(\Delta t)$；最后那个 $\Delta t\,\partial_t\log p$ 也是 $\mathcal{O}(\Delta t)$。于是

$$
p(x_t\mid x_{t+\Delta t}) \propto \exp\Big(-\frac{\big\|x_{t+\Delta t} - x_t - \big[f(x_t,t) - g^2(t)\,\nabla_{x_t}\log p(x_t)\big]\Delta t\big\|^2}{2g^2(t)\,\Delta t} + \mathcal{O}(\Delta t)\Big)
$$

$\Delta t\to0$ 时，$\mathcal{O}(\Delta t)\to0$，可以忽略。

**第四步：换成已知时刻的量。** 上式中的系数都在 $t$ 时刻求值，但在反向过程中，我们已知的是 $x_{t+\Delta t}$，未知的是 $x_t$。所以要把系数中的 $t$ 换成 $t+\Delta t$。$\Delta t\to0$ 时，这种替换只带来高阶小量的误差，可以近似认为 $t\approx t+\Delta t$：

$$
p(x_t\mid x_{t+\Delta t}) \approx \mathcal{N}\Big(x_t;\; x_{t+\Delta t} - \big[f(x_{t+\Delta t},t+\Delta t) - g^2(t+\Delta t)\,\nabla\log p(x_{t+\Delta t})\big]\Delta t,\;\; g^2(t+\Delta t)\,\Delta t\,I\Big)
$$

也就是说，反向一步也是高斯：从 $x_{t+\Delta t}$ 出发，**沿着 $-[f - g^2\nabla\log p]$ 的方向退一步**，再加上方差为 $g^2\Delta t$ 的噪声。

**第五步：取极限。** 令 $\Delta t\to0$，得到**反向 SDE**：

$$
dx = \big[f(x,t) - g^2(t)\,\nabla_x\log p_t(x)\big]\,dt + g(t)\,d\bar w
$$

这里 $dt$ 是一个**负的**无穷小时间增量（时间从 1 流向 0），$\bar w$ 是反向时间的布朗运动。这个结果最早由 Anderson (1982) 严格证明，上面是一个启发式的推导，但结论是对的。

### 2.3 结论：只需要 score

整理一下：

$$
\text{前向：}\quad dx = f(x,t)\,dt + g(t)\,dw
$$

$$
\text{反向：}\quad dx = \big[f(x,t) - g^2(t)\,\nabla_x\log p_t(x)\big]\,dt + g(t)\,d\bar w
$$

```
          前向 SDE：dx = f dt + g dw
  x(0) ~ p_data  ─────────────────────────▶  x(1) ~ 先验（近似高斯）
  x(0) ~ p_data  ◀─────────────────────────  x(1) ~ 先验
          反向 SDE：dx = [f − g²·∇log p_t] dt + g dw̄
```

前向过程是我们自己定义的，所以 $f$ 和 $g$ 都是已知的。**反向过程中唯一不知道的，就是每个时刻边际分布的 score $\nabla_x\log p_t(x)$。** 只要能估计出它，就能把前向 SDE 反过来跑，从噪声生成数据。而怎样估计 score，第 5 篇已经讲过了。

## 3. 训练：连续时间的去噪分数匹配

### 3.1 为什么不直接算 score

原则上，边际分布的 score 可以写成条件分布的期望：

$$
\nabla_{x_t}\log p(x_t) = \frac{\nabla_{x_t}p(x_t)}{p(x_t)} = \frac{\mathbb{E}_{x_0}\big[\nabla_{x_t}p(x_t\mid x_0)\big]}{\mathbb{E}_{x_0}\big[p(x_t\mid x_0)\big]} = \frac{\mathbb{E}_{x_0}\big[p(x_t\mid x_0)\,\nabla_{x_t}\log p(x_t\mid x_0)\big]}{\mathbb{E}_{x_0}\big[p(x_t\mid x_0)\big]}
$$

其中 $\mathbb{E}_{x_0}$ 是对数据分布取期望。这个式子在理论上很有用（它说明边际 score 是条件 score 的加权平均，权重是 $x_0$ 的后验），但实际计算需要遍历整个数据集，计算量过大，而且对没见过的 $x_t$ 泛化不好。

### 3.2 训练目标

所以，和之前一样，用一个神经网络 $s_\theta(x,t)$ 去学习 score。第 5 篇的去噪分数匹配告诉我们，可以用**条件分布的 score** 作为回归目标，学到的就是边际分布的 score。推广到连续时间：

$$
\mathcal{L}(\theta) = \mathbb{E}_{t\sim\mathcal{U}[0,1]}\Big\{\lambda(t)\;\mathbb{E}_{x_0\sim p_\text{data}}\,\mathbb{E}_{x_t\sim p(x_t|x_0)}\Big[\big\|s_\theta(x_t,t) - \nabla_{x_t}\log p(x_t\mid x_0)\big\|_2^2\Big]\Big\}
$$

内层的期望写成积分是

$$
\int\mathbb{E}_{x_0}\Big[p(x_t\mid x_0)\,\big\|s_\theta(x_t,t) - \nabla_{x_t}\log p(x_t\mid x_0)\big\|^2\Big]dx_t = \mathbb{E}_{x_0,\,x_t\sim p(x_t|x_0)}\Big[\big\|s_\theta(x_t,t) - \nabla_{x_t}\log p(x_t\mid x_0)\big\|^2\Big]
$$

$\lambda(t)>0$ 是一个权重函数，和 NCSN 一样，通常取 $\lambda(t)\propto1/\mathbb{E}\big[\|\nabla_{x_t}\log p(x_t\mid x_0)\|^2\big]$，也就是条件分布的方差，让各个时刻的损失量级相当。

### 3.3 什么时候条件分布有闭式解

训练需要能从 $p(x_t\mid x_0)$ 采样、并算出它的 score。这个条件分布（论文中称为**扰动核**，perturbation kernel）一般没有闭式解，但当 **$f(x,t)$ 是 $x$ 的线性函数**时，SDE 是线性的，$p(x_t\mid x_0)$ 一定是高斯。

对 $f(x,t)=f(t)\,x$，记 $x_t$ 的条件均值为 $m(t)$、每一维的条件方差为 $v(t)$，对离散形式取期望和方差再令 $\Delta t\to0$，可以得到它们满足的常微分方程：

$$
\frac{dm}{dt} = f(t)\,m, \qquad \frac{dv}{dt} = 2f(t)\,v + g^2(t)
$$

（方差方程来自 $\mathrm{Var}\big[(1+f\Delta t)\,x + g\sqrt{\Delta t}\,\epsilon\big] = (1+f\Delta t)^2 v + g^2\Delta t$。）初值为 $m(0)=x_0$，$v(0)=0$。解出 $m(t)$ 和 $v(t)$ 后，$p(x_t\mid x_0) = \mathcal{N}(m(t),v(t)I)$，它的 score 就是 $-(x_t-m(t))/v(t)$，训练和 DDPM 一样简单。下面的两个例子都属于这种情况。

## 4. VE-SDE 与 VP-SDE：NCSN 和 DDPM 的统一

现在我们有了一个一般的前向过程和反向过程的框架。前向过程是我们自己定义的，$f(x,t)$ 和 $g(t)$ 都已知；唯一未知的 $\nabla_x\log p_t(x)$ 就是 score，第 5 篇已经介绍过如何估计。

能不能把 DDPM 和 NCSN 都归入这个框架？只需要看它们对应的 $f$、$g$ 应该取什么形式。我们从前向 SDE 的离散形式出发：

$$
x_{t+\Delta t} = x_t + f(x_t,t)\,\Delta t + g(t)\sqrt{\Delta t}\;\epsilon
$$

### 4.1 NCSN → VE-SDE

NCSN 的扩散公式是 $x_t = x_0 + \sigma_t\,\epsilon$，数据本身不缩放，只叠加噪声。噪声标准差 $\sigma_t$ 随 $t$ 增大，到最后 $\sigma_1$ 非常大，$x_1$ 主要由噪声主导，所以称为 **Variance Exploding（VE，方差爆炸）SDE**。

写出相邻两个时刻之间的关系。$x_{t+\Delta t}$ 和 $x_t$ 的噪声方差分别是 $\sigma_{t+\Delta t}^2$ 和 $\sigma_t^2$，由于独立高斯的方差可以相加，从 $x_t$ 到 $x_{t+\Delta t}$ 需要补上方差为 $\sigma_{t+\Delta t}^2-\sigma_t^2$ 的噪声：

$$
\begin{aligned}
x_{t+\Delta t} &= x_t + \sqrt{\sigma_{t+\Delta t}^2 - \sigma_t^2}\;\epsilon \\
&= x_t + \sqrt{\frac{\sigma_{t+\Delta t}^2 - \sigma_t^2}{\Delta t}\cdot\Delta t}\;\epsilon \\
&= x_t + \sqrt{\frac{\Delta\sigma_t^2}{\Delta t}}\,\sqrt{\Delta t}\;\epsilon
\end{aligned}
$$

和离散形式对比，得到 NCSN 的 SDE 形式：

$$
f(x,t) = 0, \qquad g(t) = \sqrt{\frac{d\,\sigma^2(t)}{dt}}
$$

注意 $g(t)$ 是 $\sigma^2(t)$ 导数的**平方根**。扰动核为 $p(x_t\mid x_0) = \mathcal{N}\big(x_0,\;(\sigma^2(t)-\sigma^2(0))I\big)$。

NCSN 的噪声级别是几何级数，连续化后 $\sigma(t) = \sigma_\text{min}\big(\frac{\sigma_\text{max}}{\sigma_\text{min}}\big)^t$。

### 4.2 DDPM → VP-SDE

DDPM 的扩散公式是 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$。$\sqrt{\bar\alpha_t}$ 对信号进行了压制，而总方差始终保持在 1 附近，并不发散，所以称为 **Variance Preserving（VP，方差保持）SDE**。

DDPM 的一步为

$$
x_{i+1} = \sqrt{1-\beta_{i+1}}\;x_i + \sqrt{\beta_{i+1}}\;\epsilon, \qquad i = 0,1,\dots,T-1
$$

为了把离散公式转换为连续形式，定义缩放后的序列 $\bar\beta_i = T\beta_i$（$i=1,\dots,T$），改写为

$$
x_{i+1} = \sqrt{1-\frac{\bar\beta_{i+1}}{T}}\;x_i + \sqrt{\frac{\bar\beta_{i+1}}{T}}\;\epsilon
$$

当步数 $T\to\infty$ 时，序列 $\{\bar\beta_i\}_{i=1}^T$ 变成一个连续函数 $\beta(t)$，$t\in[0,1]$，满足 $\beta\big(\frac iT\big) = \bar\beta_i$，步长 $\Delta t = \frac1T$。于是 $\beta_i = \beta(t)\Delta t$，上式变成

$$
x_{t+\Delta t} = \sqrt{1-\beta(t+\Delta t)\,\Delta t}\;x_t + \sqrt{\beta(t+\Delta t)\,\Delta t}\;\epsilon
$$

利用近似 $(1-u)^a\approx1-au$（$u\to0$），以及 $\beta(t+\Delta t)\approx\beta(t)$：

$$
\begin{aligned}
x_{t+\Delta t} &\approx \Big(1 - \frac12\beta(t+\Delta t)\,\Delta t\Big)x_t + \sqrt{\beta(t+\Delta t)\,\Delta t}\;\epsilon \\
&\approx x_t - \frac12\beta(t)\,x_t\,\Delta t + \sqrt{\beta(t)}\,\sqrt{\Delta t}\;\epsilon
\end{aligned}
$$

和离散形式对比，得到 DDPM 的 SDE 形式：

$$
f(x,t) = -\frac12\beta(t)\,x, \qquad g(t) = \sqrt{\beta(t)}
$$

漂移项 $-\frac12\beta(t)\,x$ 在加噪的同时把信号往原点拉。

**为什么叫"方差保持"？** 用 3.3 节的方差方程，把 $x_0$ 也看成随机的，设 $x_t$ 的总方差为 $V(t)$，则

$$
\frac{dV}{dt} = 2\cdot\Big(-\frac12\beta(t)\Big)V + \beta(t) = \beta(t)\,\big(1 - V\big)
$$

如果数据的方差 $V(0)=1$，那么 $V(t)\equiv1$；即使不为 1，$V(t)$ 也会指数收敛到 1。所以总方差始终被"保持"在 1 附近。对比 VE-SDE，$dV/dt = g^2 >0$，方差单调增长，最终"爆炸"。

VP-SDE 的扰动核为

$$
p(x_t\mid x_0) = \mathcal{N}\Big(x_0\,e^{-\frac12\int_0^t\beta(s)ds},\;\big(1-e^{-\int_0^t\beta(s)ds}\big)I\Big)
$$

对照 DDPM 的 $\mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I)$，连续时间下 $\bar\alpha(t) = e^{-\int_0^t\beta(s)ds}$。这和离散时 $\bar\alpha_t=\prod(1-\beta_i)$、$\log\bar\alpha_t = \sum\log(1-\beta_i)\approx-\sum\beta_i$ 是一致的。

SDE 论文还提出了第三种 **sub-VP SDE**：$f$ 与 VP 相同，$g(t) = \sqrt{\beta(t)\big(1-e^{-2\int_0^t\beta(s)ds}\big)}$。它的方差始终被 VP-SDE 的方差上界住，在似然指标上表现更好。

**结论：NCSN 和 DDPM 只是 SDE 框架下不同的 $f$、$g$。**

| | VE-SDE | VP-SDE |
|---|---|---|
| 对应的离散模型 | NCSN（SMLD） | DDPM |
| 漂移 $f(x,t)$ | $0$ | $-\frac12\beta(t)\,x$ |
| 扩散 $g(t)$ | $\sqrt{d\sigma^2(t)/dt}$ | $\sqrt{\beta(t)}$ |
| 扰动核均值 | $x_0$ | $\sqrt{\bar\alpha(t)}\,x_0$ |
| 扰动核方差 | $\sigma^2(t)-\sigma^2(0)$ | $1-\bar\alpha(t)$ |
| 终点先验 | $\mathcal{N}(0,\sigma_\text{max}^2I)$ | $\mathcal{N}(0,I)$ |

### 4.3 NCSN 和 DDPM 完全等价

VE 和 VP 看起来不同，但其实只差一个随时间变化的缩放。对 NCSN 的扩散公式 $x_t = x_0 + \sigma_t\,\epsilon$，两边同除以 $\sqrt{1+\sigma_t^2}$：

$$
\frac{x_t}{\sqrt{1+\sigma_t^2}} = \frac{1}{\sqrt{1+\sigma_t^2}}\,x_0 + \frac{\sigma_t}{\sqrt{1+\sigma_t^2}}\,\epsilon
$$

令

$$
y_t = \frac{x_t}{\sqrt{1+\sigma_t^2}}, \qquad \sqrt{\bar\alpha_t} = \frac{1}{\sqrt{1+\sigma_t^2}}
$$

则 $\sqrt{1-\bar\alpha_t} = \frac{\sigma_t}{\sqrt{1+\sigma_t^2}}$，于是

$$
y_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\;\epsilon
$$

正是 DDPM 的形式。**VE 过程的 $x_t$ 除以 $\sqrt{1+\sigma_t^2}$，就是 VP 过程的 $x_t$。** 两者描述的是同一族加噪分布，只是坐标的缩放不同；在一个框架下训练出的网络，经过简单的换算可以在另一个框架下使用。

### 4.4 DDPM 从没估计过 score，怎么放进 SDE 框架？

最后解释一个问题：DDPM 训练的是噪声预测网络 $\epsilon_\theta(x_t,t)$，从来没有显式估计过 score，那它怎么用到 SDE 框架里？

这个问题第 5 篇已经回答了。对 DDPM 的扰动核，

$$
\nabla_{x_t}\log p(x_t\mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}, \qquad \epsilon = \frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}
$$

所以 DDPM 的回归目标 $\epsilon$ 和条件 score 之间只差一个系数，于是有一个重要的换算公式：

$$
s_\theta(x_t,t) = -\frac{1}{\sqrt{1-\bar\alpha_t}}\;\epsilon_\theta(x_t,t)
$$

**噪声的方向就是 score 的反方向。** 所以 score-based 模型的 score 网络和 DDPM 的噪声网络完全可以互相使用。

## 5. Fokker–Planck 方程：从轨迹到边际分布

### 5.1 动机：SDE 版的 DDIM

SDE 给出了扩散模型训练和生成的一般框架，可以说是 DDPM 的一个非常一般化的推广。一个很自然的问题是：SDE 能不能也像 DDIM 那样推广？DDIM 的关键在于不考虑前向过程 $p(x_t\mid x_{t-1})$，而只考虑边际分布 $p(x_t\mid x_0)$。对应到 SDE 上，我们需要找到**描述边际分布的方程**。

这个方程就是 **Fokker–Planck 方程**（FP 方程）。它是一个偏微分方程，描述了一条 SDE 对应的边际分布 $p_t(x)$ 如何随时间演化：对前向 SDE，给定初始分布 $p_0(x)$，就能确定任意时刻的 $p_t(x)$。

### 5.2 工具：Dirac 函数

前向 SDE 的离散形式

$$
x_{t+\Delta t} = x_t + f_t(x_t)\,\Delta t + g_t\sqrt{\Delta t}\;\epsilon, \qquad \epsilon\sim\mathcal{N}(0,I)
$$

描述的是**随机变量** $x_{t+\Delta t}$、$x_t$、$\epsilon$ 之间的关系（这里把 $f(x,t)$、$g(t)$ 写成 $f_t(x)$、$g_t$）。我们可以方便地对两边求期望，但我们想要的不是期望，而是**分布** $p_{t+\Delta t}(x)$ 所满足的关系。怎样把分布写成期望的形式？答案是 **Dirac 函数** $\delta(\cdot)$：

$$
p(x) = \int\delta(x-y)\,p(y)\,dy = \mathbb{E}_y\big[\delta(x-y)\big]
$$

Dirac 函数的严格定义属于泛函分析的内容，但我们通常把它当作普通函数来处理，一般都能得到正确的结果。由上式还可以得到，对任意函数 $h(x)$：

$$
p(x)\,h(x) = \int\delta(x-y)\,p(y)\,h(y)\,dy = \mathbb{E}_y\big[\delta(x-y)\,h(y)\big]
$$

直接对两边关于 $x$ 求导：

$$
\nabla_x\big[p(x)\,h(x)\big] = \mathbb{E}_y\big[h(y)\,\nabla_x\delta(x-y)\big]
$$

这是后面要用到的关键性质：**对 Dirac 函数求的导数，在取期望后可以转移到它所乘的函数和密度的乘积上。**

### 5.3 推导 Fokker–Planck 方程

由离散形式，

$$
\delta(x - x_{t+\Delta t}) = \delta\big(x - x_t - f_t(x_t)\,\Delta t - g_t\sqrt{\Delta t}\;\epsilon\big)
$$

把 $\delta(\cdot)$ 当作普通函数，在 $x - x_t$ 处做泰勒展开。记扰动 $u = f_t(x_t)\Delta t + g_t\sqrt{\Delta t}\,\epsilon$，展开到二阶，只保留不超过 $\mathcal{O}(\Delta t)$ 的项：

$$
\delta(x - x_{t+\Delta t}) \approx \delta(x-x_t) - \big(f_t(x_t)\,\Delta t + g_t\sqrt{\Delta t}\,\epsilon\big)\cdot\nabla_x\delta(x-x_t) + \frac12\big(g_t\sqrt{\Delta t}\;\epsilon\cdot\nabla_x\big)^2\delta(x-x_t)
$$

二阶项里只保留了 $\epsilon$ 的部分，因为 $f\Delta t$ 的平方和交叉项都是 $\mathcal{O}(\Delta t^{3/2})$ 或更高阶；而 $(g_t\sqrt{\Delta t}\,\epsilon)^2$ 是 $\mathcal{O}(\Delta t)$，必须保留。这正是 1.2 节说的"随机项要展开到二阶"。

两边对 $x_t$ 和 $\epsilon$ 求期望。$\epsilon$ 与 $x_t$ 独立，$\mathbb{E}[\epsilon]=0$ 消掉一阶的 $\epsilon$ 项，$\mathbb{E}[\epsilon\epsilon^\top]=I$ 把二阶项变成拉普拉斯算子：

$$
\begin{aligned}
p_{t+\Delta t}(x) &= \mathbb{E}_{x_{t+\Delta t}}\big[\delta(x - x_{t+\Delta t})\big] \\
&\approx \mathbb{E}_{x_t,\epsilon}\Big[\delta(x-x_t) - \big(f_t(x_t)\,\Delta t + g_t\sqrt{\Delta t}\,\epsilon\big)\cdot\nabla_x\delta(x-x_t) + \frac12\big(g_t\sqrt{\Delta t}\,\epsilon\cdot\nabla_x\big)^2\delta(x-x_t)\Big] \\
&= \mathbb{E}_{x_t}\Big[\delta(x-x_t) - f_t(x_t)\,\Delta t\cdot\nabla_x\delta(x-x_t) + \frac12 g_t^2\,\Delta t\;\nabla_x\cdot\nabla_x\delta(x-x_t)\Big] \\
&= p_t(x) - \Delta t\;\nabla_x\cdot\big[f_t(x)\,p_t(x)\big] + \frac12 g_t^2\,\Delta t\;\nabla_x\cdot\nabla_x\,p_t(x)
\end{aligned}
$$

最后一步用了 5.2 节的性质（$\delta$ 的导数转移到 $f_t\,p_t$ 和 $p_t$ 上）。两边减去 $p_t(x)$、除以 $\Delta t$，取 $\Delta t\to0$ 的极限：

$$
\frac{\partial}{\partial t}p_t(x) = -\nabla_x\cdot\big[f_t(x)\,p_t(x)\big] + \frac12 g_t^2\;\nabla_x\cdot\nabla_x\,p_t(x)
$$

这就是前向 SDE 所对应的 **Fokker–Planck 方程**，描述了边际分布的演化。右边第一项（"漂移项"）描述概率密度被 $f$ 搬运；第二项（"扩散项"，$\nabla_x\cdot\nabla_x$ 即拉普拉斯算子 $\Delta$）描述概率密度因噪声而向四周扩散、被抹平。

### 5.4 一族共享边际分布的 SDE

这里并不打算研究怎么求解这个偏微分方程，只是借助它来引出一个**等价变换**。

利用 $\nabla_x p_t = p_t\,\nabla_x\log p_t$，对任意满足 $\sigma_t^2\le g_t^2$ 的函数 $\sigma_t$，把扩散项拆成两部分：$\frac12 g_t^2 = \frac12(g_t^2-\sigma_t^2) + \frac12\sigma_t^2$，其中前一部分写成散度形式并入漂移项：

$$
\begin{aligned}
\frac{\partial}{\partial t}p_t(x) &= -\nabla_x\cdot\Big[f_t(x)\,p_t(x) - \frac12\big(g_t^2-\sigma_t^2\big)\nabla_x p_t(x)\Big] + \frac12\sigma_t^2\;\nabla_x\cdot\nabla_x\,p_t(x) \\
&= -\nabla_x\cdot\Big[\Big(f_t(x) - \frac12\big(g_t^2-\sigma_t^2\big)\nabla_x\log p_t(x)\Big)p_t(x)\Big] + \frac12\sigma_t^2\;\nabla_x\cdot\nabla_x\,p_t(x)
\end{aligned}
$$

这个方程和原来的 FP 方程**完全等价**（只是把同一个式子换了个写法）。但从形式上看，它又相当于把原来 FP 方程中的 $f_t(x)$ 换成了 $f_t(x) - \frac12(g_t^2-\sigma_t^2)\nabla_x\log p_t(x)$，$g_t$ 换成了 $\sigma_t$。由于 FP 方程和 SDE 是一一对应的，它就对应于另一条 SDE：

$$
dx = \Big(f_t(x) - \frac12\big(g_t^2 - \sigma_t^2\big)\nabla_x\log p_t(x)\Big)dt + \sigma_t\,dw
$$

但两个 FP 方程是完全等价的，这意味着**原来的前向 SDE 和这一条 SDE，对应的边际分布 $p_t(x)$ 完全相同！**

这个结果告诉我们：**不同噪声强度的前向过程，可以产生一模一样的边际分布。** 两条 SDE 的单条轨迹不同，但在每一个时刻，粒子群的整体分布完全一样。这正是 DDIM 中"保持边际、换一种过程"的连续版本，可以看作 DDIM 的升级版。（第 6.4 节会证明，当 $f_t(x)$ 是 $x$ 的线性函数时，它和 DDIM 完全等价。）

由第 2 节的结果，这条 SDE 对应的反向 SDE 为

$$
dx = \Big(f_t(x) - \frac12\big(g_t^2 + \sigma_t^2\big)\nabla_x\log p_t(x)\Big)dt + \sigma_t\,d\bar w
$$

（把新的漂移 $\tilde f = f - \frac12(g^2-\sigma^2)\nabla\log p$ 和新的扩散 $\sigma$ 代入 $d x = [\tilde f - \sigma^2\nabla\log p]dt + \sigma d\bar w$ 即得。）$\sigma_t = g_t$ 时就是第 2 节的反向 SDE。

## 6. 概率流 ODE

### 6.1 $\sigma_t=0$：SDE 退化为 ODE

上面说明我们可以自由改变采样过程的噪声强度。特别考虑 $\sigma_t = 0$ 的极端情形，此时随机项消失，**SDE 退化为常微分方程（ODE）**：

$$
\frac{dx}{dt} = f_t(x) - \frac12 g_t^2\,\nabla_x\log p_t(x)
$$

这个 ODE 称为**概率流 ODE**（probability flow ODE）。实践中 $\nabla_x\log p_t(x)$ 要用神经网络 $s_\theta(x,t)$ 来近似，所以它也是一个**神经 ODE**（Neural ODE）。

注意 $\sigma_t=0$ 时，5.4 节的"前向"和"反向"方程是**同一个方程**。也就是说，这条 ODE 既描述了从数据到噪声的确定性变换，也描述了从噪声到数据的逆变换，只是积分方向相反。

对 VP-SDE，代入 $f=-\frac12\beta(t)x$，$g^2=\beta(t)$：

$$
\frac{dx}{dt} = -\frac12\beta(t)\Big[x + \nabla_x\log p_t(x)\Big] = -\frac12\beta(t)\Big[x - \frac{\epsilon_\theta(x,t)}{\sqrt{1-\bar\alpha(t)}}\Big]
$$

对 VE-SDE，代入 $f=0$，$g^2 = d\sigma^2/dt$：

$$
\frac{dx}{dt} = -\frac12\,\frac{d\sigma^2(t)}{dt}\,\nabla_x\log p_t(x) = -\sigma(t)\,\dot\sigma(t)\,\nabla_x\log p_t(x)
$$

### 6.2 为什么要特别研究 $\sigma_t=0$

**1. 确定且可逆，和 Flow 模型一致。** 此时变换过程不带噪声，从 $x_0$ 到 $x_1$ 是一个确定性变换，直接反向求解 ODE 就能得到从 $x_1$ 到 $x_0$ 的逆变换，也是确定性的。这和 **Flow 模型**完全一致：通过一个可逆变换把噪声变成样本。所以概率流 ODE 让我们可以把扩散模型和 Flow 模型的结果对应起来，享受 Flow 模型的好处：

- **隐变量表征**：每个数据点 $x_0$ 都对应唯一的噪声 $x_1$，可以把真实图像编码到噪声空间；
- **图像编辑与插值**：在噪声空间里做插值、编辑，再解码回图像空间；
- **唯一可辨识的编码**：在数据分布和网络都足够好的情况下，编码由数据分布决定，与具体的网络结构关系不大。

**2. 精确的似然计算。** Flow 模型的一个核心优势是可以精确计算对数似然。对于 ODE $\frac{dx}{dt}=\tilde f(x,t)$，有"瞬时变量替换公式"（Neural ODE）：

$$
\log p_0(x_0) = \log p_1(x_1) + \int_0^1 \nabla_x\cdot\tilde f(x_t,t)\,dt
$$

其中 $\tilde f$ 是概率流 ODE 的右侧，$x_t$ 是从 $x_0$ 出发沿 ODE 积分的轨迹。散度 $\nabla_x\cdot\tilde f$ 在高维时很难直接算，可以用 Skilling–Hutchinson 迹估计 $\mathbb{E}_v[v^\top\nabla_x\tilde f\,v]$（和第 5 篇切片分数匹配是同一个技巧）无偏估计。这样就得到了扩散模型**精确的**对数似然，而不是 DDPM 那样的一个下界。SDE 论文用这个方法在 CIFAR-10 上得到了约 3 bits/dim 的负对数似然，当时是同类模型中最好的。

**3. 可以用成熟的 ODE 求解器加速。** 从 $x_1$ 到 $x_0$ 的变换由一个 ODE 描述，这意味着可以用各种高阶的 ODE 数值算法来加速。原则上 SDE 的求解也有一些加速方法，但 SDE 的数值方法远不如 ODE 成熟。SDE 论文直接用通用的自适应步长求解器（RK45），只需要几十到一两百次网络前向，就能得到和上千步 SDE 采样相当的质量；还可以通过调节误差容限，在速度和质量之间自由取舍。

总的来说，相比 SDE，ODE 在理论分析和实际求解中都更简单直接。

### 6.3 SDE 与 ODE 的对比

| | 反向 SDE | 概率流 ODE |
|---|---|---|
| 方程 | $dx = [f - g^2\nabla\log p_t]dt + g\,d\bar w$ | $dx = [f - \frac12 g^2\nabla\log p_t]dt$ |
| score 前的系数 | $g^2$ | $\frac12 g^2$ |
| 随机性 | 每步注入噪声 | 完全确定 |
| 边际分布 | $p_t(x)$ | 相同的 $p_t(x)$ |
| 可逆 | 否 | 是 |
| 似然 | 只能算下界 | 可以精确计算 |
| 数值求解 | 需要较多步 | 可用高阶求解器，步数少 |
| 对误差的鲁棒性 | 随机噪声能"冲刷"掉部分累积误差 | 误差会沿轨迹累积 |

最后一行值得一提：实践中，SDE 采样在步数足够多时通常略好于 ODE 采样，因为随机项能起到一定的纠错作用；ODE 采样的优势在于步数少时质量更高。这和第 4 篇 DDIM 的实验观察是一致的。

### 6.4 DDIM 就是概率流 ODE 的欧拉离散化

第 4 篇最后提到，DDIM（$\eta=0$）其实是在沿一条 ODE 的轨迹走。现在可以严格地验证这一点。

对 VP-SDE 的概率流 ODE，做变量替换：

$$
\bar x = \frac{x}{\sqrt{\bar\alpha(t)}}, \qquad \bar\sigma = \sqrt{\frac{1-\bar\alpha(t)}{\bar\alpha(t)}}
$$

由 $\bar\alpha(t) = e^{-\int_0^t\beta(s)ds}$ 得 $\frac{d\bar\alpha}{dt} = -\beta(t)\,\bar\alpha$。分别计算：

$$
\frac{d\bar x}{dt} = \frac{1}{\sqrt{\bar\alpha}}\frac{dx}{dt} + \frac{\beta}{2\sqrt{\bar\alpha}}\,x = \frac{1}{\sqrt{\bar\alpha}}\cdot\frac{\beta}{2}\,\frac{\epsilon_\theta}{\sqrt{1-\bar\alpha}} = \frac{\beta\,\epsilon_\theta}{2\sqrt{\bar\alpha}\sqrt{1-\bar\alpha}}
$$

（ODE 中的 $-\frac12\beta x$ 正好和缩放产生的 $+\frac12\beta x$ 抵消。）而由 $\bar\sigma^2 = 1/\bar\alpha - 1$，

$$
2\bar\sigma\,\frac{d\bar\sigma}{dt} = \frac{\beta}{\bar\alpha} \quad\Longrightarrow\quad \frac{d\bar\sigma}{dt} = \frac{\beta}{2\bar\alpha\,\bar\sigma} = \frac{\beta}{2\sqrt{\bar\alpha}\sqrt{1-\bar\alpha}}
$$

两式相除，得到一个极其简洁的 ODE：

$$
\frac{d\bar x}{d\bar\sigma} = \epsilon_\theta(x, t)
$$

对它用**欧拉法**从 $\bar\sigma_t$ 走到 $\bar\sigma_{t-1}$：

$$
\bar x_{t-1} = \bar x_t + \big(\bar\sigma_{t-1} - \bar\sigma_t\big)\,\epsilon_\theta(x_t,t)
$$

换回原变量：

$$
\frac{x_{t-1}}{\sqrt{\bar\alpha_{t-1}}} = \frac{x_t}{\sqrt{\bar\alpha_t}} + \Big(\sqrt{\frac{1-\bar\alpha_{t-1}}{\bar\alpha_{t-1}}} - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\Big)\epsilon_\theta(x_t,t)
$$

两边乘以 $\sqrt{\bar\alpha_{t-1}}$，整理得

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\Big(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}\Big)}_{\hat x_0} + \sqrt{1-\bar\alpha_{t-1}}\;\epsilon_\theta(x_t,t)
$$

**这正是 $\eta=0$ 的 DDIM 更新公式。** 所以 DDIM 就是概率流 ODE 在 $(\bar x,\bar\sigma)$ 坐标下的一阶欧拉离散化。这也解释了 DDIM 的所有性质：确定性、可逆、步长大了误差变大。

更进一步，这个 ODE 在 $(\bar x,\bar\sigma)$ 坐标下的右边只有 $\epsilon_\theta$，这种"半线性"结构可以被充分利用：**DPM-Solver** 把 ODE 的解写成关于 $\log$ 信噪比的指数积分，对 $\epsilon_\theta$ 做泰勒展开，得到二阶、三阶的求解器，10 到 20 步就能得到很好的样本。

## 7. 采样器

有了 SDE 和 ODE 两种视角，就有了多种采样方法。

### 7.1 反向扩散采样器

直接对反向 SDE 做 Euler–Maruyama 离散化：

$$
x_{t-\Delta t} = x_t - \big[f(x_t,t) - g^2(t)\,s_\theta(x_t,t)\big]\Delta t + g(t)\sqrt{\Delta t}\;z, \qquad z\sim\mathcal{N}(0,I)
$$

对 VP-SDE，这一步是

$$
x_{t-\Delta t} = x_t + \frac12\beta(t)\,x_t\,\Delta t + \beta(t)\,s_\theta(x_t,t)\,\Delta t + \sqrt{\beta(t)\,\Delta t}\;z
$$

可以验证，DDPM 的祖先采样公式 $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(x_t + \beta_t\,s_\theta\big) + \sqrt{\beta_t}\,z$（把 $\epsilon_\theta$ 换成 $s_\theta$）在 $\beta_t$ 很小时，一阶展开 $\frac{1}{\sqrt{1-\beta_t}}\approx1+\frac12\beta_t$ 后与上式一致。**DDPM 的采样就是反向 VP-SDE 的一种离散化。**

### 7.2 预测-校正采样器（Predictor–Corrector）

SDE 论文提出的 PC 采样器结合了两类方法：

- **预测器**（Predictor）：用某种数值格式沿反向 SDE（或 ODE）走一步，从 $t$ 到 $t-\Delta t$；
- **校正器**（Corrector）：在新的噪声水平 $t-\Delta t$ 上，用第 5 篇的**朗之万动力学**跑若干步。由于此时已经知道这一时刻分布的 score，朗之万动力学能把样本"拉回"正确的边际分布 $p_{t-\Delta t}$，修正预测器累积的误差。

算法：

1. 采样 $x_N\sim p_1$（先验）
2. **for** $i = N-1,\dots,0$ **do**
3. 　**预测**：$x_i\leftarrow$ 反向 SDE 从 $x_{i+1}$ 走一步
4. 　**for** $j = 1,\dots,M$ **do**
5. 　　**校正**：$z\sim\mathcal{N}(0,I)$，$x_i\leftarrow x_i + \delta_i\,s_\theta(x_i,t_i) + \sqrt{2\delta_i}\;z$
6. 　**end for**
7. **end for**
8. **return** $x_0$

校正步的步长 $\delta_i$ 通常根据一个目标"信噪比" $r$ 自适应地选取：$\delta_i = 2\big(r\,\|z\|/\|s_\theta(x_i,t_i)\|\big)^2$，让漂移项和噪声项保持固定的比例。

这个框架把前面的方法都统一了起来：

- **NCSN 的退火朗之万动力学** = 只有校正器（预测器为恒等映射）；
- **DDPM 的祖先采样** = 只有预测器（不做校正）；
- 两者结合，在相同计算量下往往效果更好，尤其是对 VE-SDE。

### 7.3 ODE 求解器

对概率流 ODE，可以使用任何 ODE 求解器：

- **欧拉法**：即 DDIM；
- **通用的自适应求解器**（如 RK45）：SDE 论文使用的方法，自动选择步长；
- **Heun 法**（二阶）：EDM（Karras et al., 2022）把 ODE 写成 $\frac{dx}{d\sigma} = \frac{x - D(x;\sigma)}{\sigma}$（$D$ 为去噪器，$x = x_0+\sigma\epsilon$ 的 VE 形式），用 Heun 法配合精心设计的时间步，35 次网络前向就在 CIFAR-10 上达到了当时最好的 FID；
- **专用求解器**：DPM-Solver、DPM-Solver++、UniPC 等利用扩散 ODE 的半线性结构，10 到 20 步即可。

## 8. 一个延伸：条件生成

SDE 框架还给出了条件生成的一般方法。要从 $p(x\mid y)$ 采样，只需把反向 SDE 中的 score 换成条件 score：

$$
dx = \big[f(x,t) - g^2(t)\,\nabla_x\log p_t(x\mid y)\big]dt + g(t)\,d\bar w
$$

由贝叶斯公式，$\nabla_x\log p_t(x\mid y) = \nabla_x\log p_t(x) + \nabla_x\log p_t(y\mid x)$。第一项就是无条件 score 网络，第二项可以用一个在加噪数据上训练的分类器（或已知的观测模型）得到。SDE 论文用这个方法做了类别条件生成、图像修补和上色，这就是下一篇要讲的 **classifier guidance** 的雏形。

## 9. 小结

```
               前向 SDE：dx = f dt + g dw
                         │
          ┌──────────────┴──────────────┐
       VE-SDE                         VP-SDE
   f = 0, g = √(dσ²/dt)        f = −½βx, g = √β
       NCSN                           DDPM
          └──── 只差一个随时间变化的缩放 ────┘
                         │
     反向 SDE：漂移项减去 g²·score，唯一需要学的是 score
                         │       （DDPM 的 ε 与 score 只差系数）
     Fokker–Planck：存在一族共享边际分布的 SDE
                         │
     σ = 0：概率流 ODE
     确定、可逆、可算精确似然、可用高阶求解器
     DDIM = 它的欧拉离散化；DPM-Solver、EDM = 更好的求解器
```

SDE 视角把前面几篇串成了一个整体：

- **加噪方式**由 $f$ 和 $g$ 决定，NCSN 和 DDPM 只是两种选择；
- **训练**只需要学 score，用的是连续时间的去噪分数匹配；
- **生成**可以是随机的（反向 SDE），也可以是确定的（概率流 ODE），两者边际分布相同；
- **DDIM、DDPM、NCSN** 分别对应 ODE 的欧拉法、反向 SDE 的离散化和朗之万校正。

后来的很多工作都是在这个框架上继续简化和改进：EDM 系统地梳理了噪声调度、网络预处理、损失加权和采样器的设计空间；Flow Matching 和 Rectified Flow 则直接设计从噪声到数据的路径（比如直线 $x_t = (1-t)x_0 + t\epsilon$）和对应的速度场，跳过了 SDE，直接学习一个 ODE。

## 参考文献

- Anderson. *Reverse-Time Diffusion Equation Models.* Stochastic Processes and their Applications, 1982.
- Chen et al. *Neural Ordinary Differential Equations.* NeurIPS 2018.
- Grathwohl et al. *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models.* ICLR 2019.
- Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021.
- Song, Durkan, Murray & Ermon. *Maximum Likelihood Training of Score-Based Diffusion Models.* NeurIPS 2021.
- Lu et al. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.* NeurIPS 2022.
- Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022.
- Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023.
- Liu, Gong & Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023.
