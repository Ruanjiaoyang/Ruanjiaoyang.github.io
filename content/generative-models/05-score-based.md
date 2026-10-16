---
title: "Score-based 模型：不学密度，学密度的梯度"
title_en: "Score-Based Generative Models: Learn the Gradient, Not the Density"
series: "从 GAN 到 Diffusion"
order: 5
date: 2026-10-16
tags: [生成模型, Score Matching, Langevin Dynamics, NCSN]
summary_en: >
  The score of a distribution, the gradient of its log-density, does not depend on the normalizing
  constant. This post derives implicit, sliced and denoising score matching, explains Langevin
  dynamics and why its stationary distribution is the target, analyzes the three failure modes of the
  naive approach (manifold hypothesis, inaccurate low-density scores, wrong mode weights), and walks
  through NCSN: multi-scale noise, the noise-conditional objective and annealed Langevin sampling.
  It closes by showing that predicting the score is the same as predicting DDPM's noise, plus inpainting.
---

前面几篇讲的 DDPM 和 DDIM，是从**变分推断**的角度出发的：写出一个隐变量模型，推导 ELBO，化简成噪声回归。这一篇从一个完全不同的角度重新出发：**不去建模概率密度本身，而是去学习它的梯度。** 最后我们会看到，两条路走到了同一个地方。

这一篇主要基于 Yang Song 和 Stefano Ermon 2019 年的论文 *Generative Modeling by Estimating Gradients of the Data Distribution*。它的核心贡献是基于**分数匹配**（Score Matching）和**朗之万动力学**（Langevin Dynamics）提出了一种新的生成建模方法：不需要显式地对数据概率密度建模，而是学习它的梯度（score function），再用梯度来采样，从而实现数据生成。

## 1. 什么是 Score

### 1.1 定义

生成模型的目标是估计数据的概率分布 $p(x)$。分布的 **score function**（分数函数）定义为**对数密度关于数据 $x$ 的梯度**：

$$
s(x) = \nabla_x \log p(x)
$$

注意两点：

- 求导对象是数据 $x$，不是模型参数 $\theta$。（统计学里也有"score"这个词，指对数似然关于参数的梯度，那是另一个概念。）
- score 和数据同维：$x\in\mathbb{R}^d$，$s(x)\in\mathbb{R}^d$。所以 score 是空间中的一个**向量场**，在每个点上指向对数密度上升最快的方向。

想象一个由两个高斯混合而成的二维分布：它的密度像两座小山，而 score 场就是遍布整个平面的一组箭头，每个箭头都指向离它最近的那座山顶，离山顶越远箭头越长，在山顶处箭头长度为零。只要知道这组箭头，从任何一点出发顺着箭头走，就能走到高密度区域。

### 1.2 几个例子

**一维高斯。** $p(x) = \mathcal{N}(x;\mu,\sigma^2)$，$\log p(x) = -\frac{(x-\mu)^2}{2\sigma^2} + C$，所以

$$
\nabla_x\log p(x) = -\frac{x-\mu}{\sigma^2}
$$

score 指向均值，大小和离均值的距离成正比，和方差成反比。

**多维高斯。** $p(x) = \mathcal{N}(x;\mu,\Sigma)$，

$$
\nabla_x\log p(x) = -\Sigma^{-1}(x-\mu)
$$

**混合分布。** $p(x) = \sum_k \pi_k p_k(x)$，

$$
\nabla_x\log p(x) = \sum_k \underbrace{\frac{\pi_k p_k(x)}{p(x)}}_{w_k(x)}\,\nabla_x\log p_k(x)
$$

是各成分 score 的加权平均，权重 $w_k(x)$ 是"$x$ 属于第 $k$ 个成分"的后验概率。这个式子在第 5 节分析模式权重问题时会用到。

## 2. 为什么要学 Score：绕开归一化常数

### 2.1 能量模型与归一化常数

任意一个概率分布都可以写成更一般的形式：

$$
p_\theta(x) = \frac{e^{-f_\theta(x)}}{Z(\theta)}, \qquad Z(\theta) = \int e^{-f_\theta(x)}\,dx
$$

这来自**能量模型**（Energy-Based Models，EBM）。$f_\theta(x)$ 称为能量函数，可以是任意神经网络，能量越低的地方概率越高；$Z(\theta)$ 是归一化常数（也叫配分函数），保证 $p_\theta(x)$ 积分为 1。

能量模型的表达力很强，因为 $f_\theta$ 不受任何结构限制。问题在于训练：标准做法是最大似然估计，

$$
\log p_\theta(x) = -f_\theta(x) - \log Z(\theta)
$$

而 $Z(\theta)$ 是一个高维积分，算不出来，它对 $\theta$ 的梯度也算不出来（只能用 MCMC 近似，代价很大）。

### 2.2 各类生成模型怎样对付 $Z$

回头看整个系列，不同的生成模型其实就是不同的"对付 $Z$"的方式：

- **自回归模型**：用链式法则把联合分布分解成一维条件分布的乘积 $p(x)=\prod_i p(x_i\mid x_{<i})$，每个一维分布（比如 softmax）的归一化很容易。代价是要规定一个生成顺序，且采样必须逐个进行。
- **Flow 模型**：限制网络为可逆变换，由变量替换公式，密度可以用雅可比行列式精确算出，相当于从结构上保证 $Z=1$。代价是网络结构受到很大限制。
- **VAE**：不直接算似然，而是优化它的一个下界（ELBO）。
- **GAN**：完全绕开似然，用判别器网络去学一个分布之间的距离度量。
- **Score-based 模型**：对 $\log p$ 关于 $x$ 求梯度，**直接让 $Z$ 消失**。

最后一点的推导只有一行。对 $p_\theta(x) = e^{-f_\theta(x)}/Z(\theta)$ 求对数梯度：

$$
\nabla_x\log p_\theta(x) = -\nabla_x f_\theta(x) - \underbrace{\nabla_x\log Z(\theta)}_{=\,0} = -\nabla_x f_\theta(x)
$$

$Z(\theta)$ 只依赖参数、和 $x$ 无关，所以它关于 $x$ 的梯度为零。**用 score 函数代替密度函数，就避开了难以处理的归一化常数。**

### 2.3 直接用网络拟合 score

更进一步，既然我们最终只需要 score，那连能量函数 $f_\theta$ 都不必定义，直接用一个神经网络 $s_\theta(x)$ 去拟合 score：

$$
s_\theta(x)\approx\nabla_x\log p(x)
$$

这就是 **score-based model**：它不学习概率分布本身，而是学习 score。这样做的好处是对网络几乎没有任何结构要求，**唯一的要求是输入和输出的维度相同**。所以可以直接使用 U-Net 等图像到图像的网络。

（严格来说，一个任意的向量场不一定是某个标量函数的梯度，所以 $s_\theta$ 不一定对应一个合法的密度。实践中这并不影响采样。）

## 3. Score Matching：怎样学一个看不见的梯度

### 3.1 Fisher 散度

最自然的目标，是让网络输出的 score 和数据真实的 score 尽量接近。用平方误差在数据分布下取期望，得到 **Fisher 散度**：

$$
\mathcal{J}(\theta) = \frac12\,\mathbb{E}_{p_\text{data}(x)}\Big[\big\|s_\theta(x) - \nabla_x\log p_\text{data}(x)\big\|_2^2\Big] = \frac12\int p_\text{data}(x)\,\big\|s_\theta(x) - \nabla_x\log p_\text{data}(x)\big\|_2^2\,dx
$$

在适当的条件下（比如两个分布的密度都处处为正且足够光滑），Fisher 散度为零当且仅当两个分布相等，所以它是一个合法的训练目标。

问题是，直接计算 Fisher 散度是不可行的：**数据的真实 score $\nabla_x\log p_\text{data}(x)$ 是未知的**，我们手里只有样本，没有密度。所以需要换个思路，用一类叫做**分数匹配**（Score Matching）的方法：它们把 Fisher 散度改写成不含真实 score 的形式，可以直接在数据集上估计，并用随机梯度下降优化。下面介绍三种。

### 3.2 隐式分数匹配

Hyvärinen (2005) 证明，通过分部积分可以把真实 score 消掉。以一维为例，展开平方：

$$
\mathcal{J}(\theta) = \frac12\int p(x)\,s_\theta(x)^2\,dx - \int p(x)\,s_\theta(x)\,\frac{\partial\log p(x)}{\partial x}\,dx + \underbrace{\frac12\int p(x)\Big(\frac{\partial\log p(x)}{\partial x}\Big)^2dx}_{\text{与 }\theta\text{ 无关}}
$$

麻烦的是中间那项。注意 $p(x)\,\frac{\partial\log p(x)}{\partial x} = \frac{\partial p(x)}{\partial x}$，然后分部积分：

$$
\int s_\theta(x)\,\frac{\partial p(x)}{\partial x}\,dx = \Big[s_\theta(x)\,p(x)\Big]_{-\infty}^{\infty} - \int p(x)\,\frac{\partial s_\theta(x)}{\partial x}\,dx = -\,\mathbb{E}_{p(x)}\Big[\frac{\partial s_\theta(x)}{\partial x}\Big]
$$

（假设 $p(x)\to0$ 足够快，边界项为零。）代回去，真实 score 不见了。推广到 $d$ 维：

$$
\mathcal{J}(\theta) = \mathbb{E}_{p_\text{data}(x)}\Big[\mathrm{tr}\big(\nabla_x s_\theta(x)\big) + \frac12\big\|s_\theta(x)\big\|_2^2\Big] + C
$$

其中 $\nabla_x s_\theta(x)$ 是 $s_\theta$ 的雅可比矩阵。这个目标只依赖网络本身和数据样本，可以直接优化。

但它有一个计算上的问题：$\mathrm{tr}(\nabla_x s_\theta(x)) = \sum_{i=1}^d \frac{\partial s_{\theta,i}(x)}{\partial x_i}$ 需要雅可比矩阵的对角线，每一项都要一次反向传播，总共 $d$ 次。对一张 $256\times256\times3$ 的图像，$d\approx 2\times10^5$，完全不可行。

### 3.3 切片分数匹配

切片分数匹配（Sliced Score Matching，Song et al., 2019）的思路是：既然高维向量场难以比较，那就把它**投影到随机方向上**再比较。取随机向量 $v\sim p(v)$（比如标准高斯），目标变为

$$
\mathcal{J}_\text{SSM}(\theta) = \mathbb{E}_{p(v)}\,\mathbb{E}_{p_\text{data}(x)}\Big[v^\top\nabla_x s_\theta(x)\,v + \frac12\big(v^\top s_\theta(x)\big)^2\Big]
$$

$v^\top\nabla_x s_\theta(x)\,v$ 可以先算 $v^\top s_\theta(x)$（一个标量），再对 $x$ 求一次梯度、和 $v$ 做内积得到，只需要**一次**反向传播。这样就把计算量从 $d$ 次降到了常数次。它本质上是用随机投影来无偏地估计雅可比矩阵的迹（Hutchinson 迹估计）。

### 3.4 去噪分数匹配

去噪分数匹配（Denoising Score Matching，Vincent, 2011）完全绕开了雅可比矩阵，后来的扩散模型实际采用的都是它。

**思路：先给数据加噪。** 选一个已知的加噪分布 $q_\sigma(\tilde x\mid x)$，比如 $\tilde x = x + \sigma\epsilon$，$\epsilon\sim\mathcal{N}(0,I)$。加噪后的数据分布为

$$
q_\sigma(\tilde x) = \int q_\sigma(\tilde x\mid x)\,p_\text{data}(x)\,dx
$$

我们改为去估计**加噪后分布** $q_\sigma(\tilde x)$ 的 score。它的 Fisher 散度仍然算不出来，因为 $q_\sigma(\tilde x)$ 本身也不知道。但关键的结论是：

$$
\mathbb{E}_{q_\sigma(\tilde x)}\Big[\big\|s_\theta(\tilde x) - \nabla_{\tilde x}\log q_\sigma(\tilde x)\big\|^2\Big] = \mathbb{E}_{p_\text{data}(x)}\,\mathbb{E}_{q_\sigma(\tilde x|x)}\Big[\big\|s_\theta(\tilde x) - \nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\big\|^2\Big] + C
$$

**左边需要边际分布的 score（未知），右边只需要条件分布的 score（已知），两者只差一个与 $\theta$ 无关的常数。**

**证明。** 两边展开平方。$\|s_\theta\|^2$ 项在两边相同（都是对 $\tilde x$ 的边际分布取期望）；不含 $s_\theta$ 的项都是常数。只需验证交叉项相等。左边的交叉项：

$$
\begin{aligned}
\mathbb{E}_{q_\sigma(\tilde x)}\big[s_\theta(\tilde x)^\top\nabla_{\tilde x}\log q_\sigma(\tilde x)\big]
&= \int s_\theta(\tilde x)^\top\nabla_{\tilde x}q_\sigma(\tilde x)\,d\tilde x \\
&= \int s_\theta(\tilde x)^\top\nabla_{\tilde x}\int p_\text{data}(x)\,q_\sigma(\tilde x\mid x)\,dx\,d\tilde x \\
&= \iint p_\text{data}(x)\,q_\sigma(\tilde x\mid x)\;s_\theta(\tilde x)^\top\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\,dx\,d\tilde x \\
&= \mathbb{E}_{p_\text{data}(x)\,q_\sigma(\tilde x|x)}\big[s_\theta(\tilde x)^\top\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\big]
\end{aligned}
$$

第一行用了 $q\nabla\log q = \nabla q$，第三行把梯度移进积分并再次使用同一个恒等式。这正是右边的交叉项。证毕。

**高斯噪声下的闭式解。** 对 $q_\sigma(\tilde x\mid x) = \mathcal{N}(\tilde x;x,\sigma^2 I)$，

$$
\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x) = -\frac{\tilde x - x}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

于是去噪分数匹配的目标就是

$$
\mathcal{J}_\text{DSM}(\theta) = \frac12\,\mathbb{E}_{x\sim p_\text{data},\,\epsilon\sim\mathcal{N}(0,I)}\Big[\Big\|s_\theta(x+\sigma\epsilon) + \frac{\epsilon}{\sigma}\Big\|_2^2\Big]
$$

意思是：**看到加噪后的样本 $\tilde x$，预测加进去的噪声方向（乘以 $-1/\sigma$）。** 它不需要任何雅可比矩阵，就是一个普通的回归问题。

**代价：学到的是加噪分布的 score。** 最优解满足 $s_{\theta^*}(\tilde x) = \nabla_{\tilde x}\log q_\sigma(\tilde x)$，而不是 $\nabla_x\log p_\text{data}(x)$。只有 $\sigma$ 足够小时，$q_\sigma\approx p_\text{data}$，两者才近似相等。但 $\sigma$ 太小又会带来别的问题（见第 5 节）。

**Tweedie 公式：score 就是去噪器。** 加噪分布的 score 和"最优去噪"之间有一个漂亮的关系：

$$
\mathbb{E}[x\mid\tilde x] = \tilde x + \sigma^2\,\nabla_{\tilde x}\log q_\sigma(\tilde x)
$$

也就是说，给定一个加噪样本，**沿着 score 方向走 $\sigma^2$ 那么远，就得到了对干净样本的最优（均方误差意义下）估计**。估计 score 和训练一个去噪器是同一件事，这也是"去噪分数匹配"这个名字的含义。

### 3.5 套用 DDPM 的加噪方式

去噪分数匹配可以用任何已知的加噪分布。套用 DDPM 的加噪方式：

$$
q(x_t\mid x_0) = \mathcal{N}\big(x_t;\;\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big)
$$

其概率密度为

$$
q(x_t\mid x_0) \propto \exp\Big\{-\frac{(x_t - \sqrt{\bar\alpha_t}\,x_0)^\top(x_t - \sqrt{\bar\alpha_t}\,x_0)}{2(1-\bar\alpha_t)}\Big\}
$$

对 $x_t$ 求对数梯度，并代入 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$：

$$
\begin{aligned}
\nabla_{x_t}\log q(x_t\mid x_0) &= -\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t} \\
&= -\frac{\sqrt{1-\bar\alpha_t}\;\epsilon}{1-\bar\alpha_t} \\
&= -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
\end{aligned}
$$

也就是说，**条件分布的 score 和加在原图上的噪声只差一个系数**。由上一小节的结论，用它做回归目标学到的就是边际分布 $q(x_t)$ 的 score。所以可以用一个**噪声估计网络**来估计 score：

$$
s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}
$$

**估计噪声就是估计 score，也就是估计（加噪后的）数据分布的对数梯度。** 第 8 节会更完整地展开这条联系。

## 4. 朗之万动力学：只用 score 来采样

### 4.1 从布朗运动说起

朗之万动力学（Langevin dynamics）原本指朗之万方程，是物理学中描述布朗运动的微分方程。布朗运动是悬浮在液体或气体中的微小颗粒所做的无规则运动：颗粒一方面受到外力（比如势能的梯度）的牵引，另一方面不断受到周围分子随机撞击的扰动。

把这个思想借过来作为采样方法：**把 score 当作"外力"，把高斯噪声当作"分子撞击"。** 朗之万动力学可以通过一组迭代的算法从一个分布中采样，而这个算法只依赖分布的 score：

1. 从某个先验分布中**随机采一个初始样本**；
2. 利用 score 把样本逐渐推向数据分布的**高概率密度区域**；
3. 为了保证生成结果的多样性，采样过程需要带有**随机性**。布朗运动本身就是随机的，朗之万方程因此也自带一个随机项。

### 4.2 迭代公式

建立好 score 模型 $s_\theta(x)\approx\nabla_x\log p(x)$ 之后，就可以用朗之万动力学从任意初始分布出发，一步步逼近目标分布。它提供了一种马尔可夫链蒙特卡罗（MCMC）过程，**只用 $\nabla_x\log p(x)$ 就能从 $p(x)$ 中采样**。

给定初始样本 $\tilde x_0\sim\pi(x)$（$\pi$ 是任意一个先验分布，比如均匀分布或高斯），迭代：

$$
\tilde x_{k+1} = \tilde x_k + \frac{\delta}{2}\,\nabla_x\log p(\tilde x_k) + \sqrt{\delta}\,z_k, \qquad z_k\sim\mathcal{N}(0,I), \quad k = 0,1,\dots,K-1
$$

$\delta>0$ 是步长。实践中把真实 score 换成学到的 $s_\theta$：

$$
\tilde x_{k+1} = \tilde x_k + \frac{\delta}{2}\,s_\theta(\tilde x_k) + \sqrt{\delta}\,z_k
$$

（为了不和 DDPM 的时间步 $t$、噪声 $\epsilon$ 混淆，这里用 $k$ 表示迭代次数，$\delta$ 表示步长。）

当 $\delta\to0$、$K\to\infty$ 时，在一定的正则条件下，$\tilde x_K$ 的分布收敛到 $p(x)$。$\delta$ 足够小、$K$ 足够大时，误差可以忽略。（$\delta$ 不够小时，可以用 Metropolis-Hastings 步骤来修正离散化误差，这就是 MALA 算法；但在实践中通常直接忽略。）

### 4.3 两项各自的作用

- **漂移项** $\frac{\delta}{2}\nabla_x\log p$：沿着 score 做梯度上升，把样本推向高密度区域。如果只有这一项，就是普通的梯度上升，所有样本最终都会收敛到 $\log p$ 的某个局部极大值（众数），失去多样性。
- **扩散项** $\sqrt\delta\,z$：注入随机噪声，防止样本塌缩到峰值上，让它们在高密度区域附近按概率"散开"。

两项的尺度是精心配平的：漂移项和 $\delta$ 成正比，扩散项和 $\sqrt\delta$ 成正比。正是这个比例保证了采样分布是 $p(x)$，而不是更尖或更平的分布。

### 4.4 为什么平稳分布是 $p(x)$

把离散迭代看成连续时间过程，步长 $\delta$ 对应时间增量 $d\tau$，得到朗之万随机微分方程

$$
dx = \nabla_x\log p(x)\,d\tau + \sqrt2\,dW
$$

$W$ 是布朗运动。它描述的粒子群的密度 $\rho(x,\tau)$ 满足 Fokker–Planck 方程

$$
\frac{\partial\rho}{\partial\tau} = -\nabla\cdot\big(\rho\,\nabla\log p\big) + \Delta\rho
$$

把 $\rho = p$ 代入：$p\,\nabla\log p = \nabla p$，所以右边是 $-\nabla\cdot\nabla p + \Delta p = -\Delta p + \Delta p = 0$。也就是说，**$p$ 是这个过程的平稳分布**：粒子一旦按 $p$ 分布，就会一直保持按 $p$ 分布。在一定条件下，从任何初始分布出发，最终都会收敛到它。第 6 篇会更系统地使用 Fokker–Planck 方程。

## 5. 朴素方法的三个问题

把"去噪分数匹配学 score"和"朗之万动力学采样"直接组合起来，在真实图像上效果很差。原因有三个。

### 5.1 流形假设：score 在数据流形外没有定义

**流形假设**认为，真实世界的数据（比如图像）往往集中在高维空间中的一个**低维流形**上。一张 $256\times256$ 的 RGB 图像有约 20 万个维度，但"看起来像自然图像"的点只占其中极小的一部分，可以用少得多的自由度来描述。

这对 score-based 模型带来两个问题：

- score 是在整个空间中定义的梯度，但在低维流形之外，$p_\text{data}=0$，$\log p_\text{data}$ 没有定义，score 也就没有定义；
- 只有当数据分布的支撑集是整个空间时，score matching 的目标才是一个一致的估计。数据集中在低维流形上时，这个条件不成立。

论文的实验也验证了这一点：在 CIFAR-10 上直接用切片分数匹配训练，损失曲线剧烈震荡，无法收敛；而给数据加上很小的高斯噪声（比如 $\sigma=0.01$）后，损失立刻变得平稳。

### 5.2 低密度区域的 score 估计不准

score matching 的目标是在数据分布下取期望的：

$$
\mathbb{E}_{p_\text{data}(x)}\big[\|s_\theta(x) - \nabla_x\log p_\text{data}(x)\|^2\big] = \int p_\text{data}(x)\,\|\cdots\|^2\,dx
$$

误差被 $p_\text{data}(x)$ 加权。在数据密度很低的区域，几乎没有训练样本，这部分误差对损失几乎没有贡献，网络在那里学到什么都可以。所以**在数据密度较低的位置，score 的估计往往是不准确的**。

而朗之万采样恰恰是从一个随机初始化出发的，初始样本几乎必然落在低密度区域（高维空间中，随机一个点几乎不可能恰好落在数据流形附近）。于是在采样的早期阶段，模型很容易根据错误的梯度走偏，"脱轨"之后就很难回到正确的区域，生成结果不好。

### 5.3 朗之万动力学分不清各模式的权重

即使 score 估计得很准，朗之万动力学在**多模态分布**上也会出问题。

考虑两个支撑集不相交的分布的混合：$p(x) = \pi\,p_1(x) + (1-\pi)\,p_2(x)$。在 $p_1$ 的支撑集上，$p_2(x)=0$，所以

$$
\nabla_x\log p(x) = \nabla_x\log\big(\pi\,p_1(x)\big) = \nabla_x\log p_1(x)
$$

同理，在 $p_2$ 的支撑集上 score 等于 $\nabla_x\log p_2(x)$。**混合权重 $\pi$ 完全从 score 里消失了。** 不论 $\pi$ 是 0.1 还是 0.9，score 场一模一样。

这意味着，单凭 score 无法知道两个模式应该各占多少比例。即使两个模式之间不是完全不相交、而是由一片低密度区域隔开，朗之万动力学也需要极长的时间才能穿过这片区域在两个模式之间"混合"，在有限步内，样本落在哪个模式主要取决于初始化，而不是真实的权重。论文的实验显示，在一个权重为 1:4 的二维高斯混合上，朗之万采样得到的两个模式的样本数大致相等，比例完全不对。

## 6. 解决办法：给数据加噪声

### 6.1 一个办法同时解决三个问题

三个问题有同一个解：**给数据加高斯噪声**。

- **解决流形问题**：高斯噪声的支撑集是整个空间。加噪之后的数据不再局限于低维流形，score 处处有定义，score matching 的目标也变得良定。
- **解决低密度区域的问题**：较大的高斯噪声会把原本数据分布中的低密度区域"填满"，扩大了数据密度的覆盖范围。这样能比较准确地估计 score 的区域就扩大了很多，score matching 在这些区域也能得到训练信号。
- **解决模式权重的问题**：加噪后各个模式的支撑集重叠，混合权重重新出现在 score 中（由 1.2 节的式子，权重 $w_k(x)$ 里含有 $\pi_k$），朗之万动力学也更容易在模式之间来回移动。

### 6.2 噪声大小的两难

但给原始分布加入噪声，会改变原始分布。所以噪声的大小需要权衡：

- 加**较强**的噪声：更多的区域可以准确估计 score，模式之间更容易混合；但会**严重破坏原本的数据分布**，学到的是一个被高度模糊过的分布。
- 加**较弱**的噪声：尽可能保留原始数据分布；但**无法在大多数区域准确估计 score**，三个问题依然存在。

论文给出了一个巧妙的解决思路：**不只用一种噪声，而是用一系列不同强度的噪声，在推理的不同阶段使用不同强度的噪声。推理刚开始时噪声较大，推理快结束时噪声较小。** 这样大噪声的好处（覆盖广、易混合）和小噪声的好处（接近真实分布）可以兼得。

## 7. NCSN：噪声条件分数网络

### 7.1 多个噪声级别

基于上面的分析，NCSN（Noise Conditional Score Network）从两个方面改进 score-based 生成：

1. 用**多个级别的噪声**扰动数据；
2. **训练单个条件分数网络，同时估计所有噪声级别对应的 score。**

具体地，取一组几何递减的噪声标准差：

$$
\sigma_i>0, \qquad \frac{\sigma_1}{\sigma_2} = \frac{\sigma_2}{\sigma_3} = \cdots = \frac{\sigma_{L-1}}{\sigma_L} > 1
$$

- $\sigma_1$ 足够大，大到能"抹平"数据分布，缓解上面三个问题；
- $\sigma_L$ 足够小，小到加噪后的数据几乎和原始数据无法区分。

对每个噪声级别，加噪后的分布为 $q_{\sigma_i}(\tilde x) = \int p_\text{data}(x)\,\mathcal{N}(\tilde x;x,\sigma_i^2 I)\,dx$。这一系列分布从"高度模糊"逐渐过渡到"几乎就是真实数据"，最终收敛于真实数据分布。

### 7.2 训练目标

训练一个以噪声级别为条件的网络 $s_\theta(\tilde x,\sigma)$，要求它对所有 $\sigma_i$ 都满足 $s_\theta(\tilde x,\sigma_i)\approx\nabla_{\tilde x}\log q_{\sigma_i}(\tilde x)$。对单个噪声级别，用 3.4 节的去噪分数匹配：

$$
\ell(\theta;\sigma) = \frac12\,\mathbb{E}_{p_\text{data}(x)}\,\mathbb{E}_{\tilde x\sim\mathcal{N}(x,\sigma^2I)}\Big[\Big\|s_\theta(\tilde x,\sigma) + \frac{\tilde x - x}{\sigma^2}\Big\|_2^2\Big]
$$

再把所有噪声级别加权求和：

$$
\mathcal{L}(\theta;\{\sigma_i\}_{i=1}^L) = \frac1L\sum_{i=1}^L\lambda(\sigma_i)\,\ell(\theta;\sigma_i)
$$

**权重怎么选？** 我们希望各个噪声级别的损失大致在同一个量级上。经验上，训练好的网络满足 $\|s_\theta(\tilde x,\sigma)\|_2\propto1/\sigma$（噪声越小，分布越尖，score 越大）。取 $\lambda(\sigma)=\sigma^2$，则

$$
\lambda(\sigma)\,\ell(\theta;\sigma) = \frac12\,\mathbb{E}\Big[\Big\|\sigma\,s_\theta(\tilde x,\sigma) + \epsilon\Big\|_2^2\Big], \qquad \epsilon = \frac{\tilde x - x}{\sigma}\sim\mathcal{N}(0,I)
$$

其中 $\sigma s_\theta$ 和 $\epsilon$ 都是 $O(1)$ 的量，损失的量级与 $\sigma$ 无关。这个形式和 DDPM 的 $\|\epsilon-\epsilon_\theta\|^2$ 已经非常接近了（只差一个符号和系数）。

**网络结构。** 原始 NCSN 用了一个基于 RefineNet 的 U 形网络，并用"条件实例归一化"把噪声级别 $\sigma_i$ 注入网络。后续的 NCSNv2 发现，直接把无条件网络的输出除以 $\sigma$，即 $s_\theta(\tilde x,\sigma) = s_\theta(\tilde x)/\sigma$，效果同样好，而且可以推广到训练时没见过的噪声级别。

### 7.3 采样：退火朗之万动力学

训练完成后，用朗之万动力学生成样本时，**先使用大噪声对应的 score，再逐渐"退火"降低噪声水平**。这有助于把大噪声级别的好处平稳地转移到小噪声级别，而在最小的噪声级别上，加噪数据和原始数据几乎无法区分。

**退火朗之万动力学**（Annealed Langevin Dynamics）算法：

1. 初始化 $\tilde x_0$（比如从均匀分布或高斯中采样）
2. **for** $i = 1,\dots,L$ **do**　　（从大噪声到小噪声）
3. 　设置步长 $\alpha_i = \delta\cdot\sigma_i^2/\sigma_L^2$
4. 　**for** $k = 1,\dots,K$ **do**
5. 　　采样 $z_k\sim\mathcal{N}(0,I)$
6. 　　$\tilde x_k = \tilde x_{k-1} + \frac{\alpha_i}{2}\,s_\theta(\tilde x_{k-1},\sigma_i) + \sqrt{\alpha_i}\,z_k$
7. 　**end for**
8. 　$\tilde x_0\leftarrow\tilde x_K$　　（作为下一个噪声级别的初始化）
9. **end for**
10. **return** $\tilde x_K$

```python
@torch.no_grad()
def annealed_langevin(score_net, shape, sigmas, K=100, delta=2e-5):
    x = torch.rand(shape)                                  # 均匀初始化
    for sigma in sigmas:                                   # sigmas 从大到小
        alpha = delta * (sigma / sigmas[-1]) ** 2
        for _ in range(K):
            x = x + 0.5 * alpha * score_net(x, sigma) + alpha ** 0.5 * torch.randn_like(x)
    return x
```

原论文的设置是 $L=10$，$\sigma_1=1$，$\sigma_{10}=0.01$，每个级别 $K=100$ 步，$\delta=2\times10^{-5}$，总共 1000 次网络前向。

**为什么步长按 $\sigma_i^2$ 缩放？** 看一步更新中"漂移"和"噪声"的比值。漂移项大小约为 $\frac{\alpha_i}{2}\|s_\theta\|\propto\alpha_i/\sigma_i$，噪声项大小约为 $\sqrt{\alpha_i}$，比值约为 $\sqrt{\alpha_i}/\sigma_i$。取 $\alpha_i\propto\sigma_i^2$，这个比值与 $i$ 无关，各个噪声级别上的朗之万采样就有相同的"信噪比"。

**直观理解**：这和模拟退火的思想一致。先在大噪声下跑，此时 score 在全空间都可靠，分布很平滑，样本能先找到正确的大致区域，并且按正确的比例分配到各个模式；然后逐步降低噪声，每一级都以上一级的结果为初始化，此时样本已经在高密度区域附近，score 估计是准确的，可以放心地精修细节。**先粗后细，先找到对的区域，再精修细节。**

NCSN 在 CIFAR-10 上达到了 8.87 的 Inception Score，是当时无条件生成模型中最好的结果，第一次证明了不依赖对抗训练、只靠 score 估计就能生成高质量图像。

### 7.4 NCSNv2 的改进

Song & Ermon (2020) 在 *Improved Techniques for Training Score-Based Generative Models* 中，从理论上分析了噪声级别和采样参数应该怎么选，让 NCSN 可以稳定地扩展到 $256\times256$ 的高分辨率图像：

- **最大噪声 $\sigma_1$**：应当和训练集中两两样本之间的最大欧氏距离相当，这样才能保证朗之万动力学能在任意两个模式之间移动；
- **噪声级别的间隔**：相邻两级之间的比值要选得让相邻的加噪分布有足够的重叠，从而能平稳过渡；
- **噪声条件的方式**：如上所述，用 $s_\theta(\tilde x)/\sigma$ 代替复杂的条件归一化；
- **步长与步数**：给出了根据 $\sigma$ 序列计算最优步长的方法；
- **参数的指数移动平均**（EMA）：显著提升了样本的稳定性。

此外，采样结束后常常再做一步**去噪**：$\tilde x\leftarrow\tilde x + \sigma_L^2\,s_\theta(\tilde x,\sigma_L)$。由 3.4 节的 Tweedie 公式，这正是对干净样本的最优估计，可以去掉最后残留的一点噪声。

## 8. 和 DDPM 的联系：殊途同归

### 8.1 预测噪声就是预测 score

3.5 节已经推出，在 DDPM 的加噪方式下

$$
\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

所以 DDPM 的噪声预测网络和 score 网络之间是一个简单的换算：

$$
s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}
$$

把它代入去噪分数匹配的目标：

$$
\Big\|s_\theta(x_t,t) - \nabla_{x_t}\log q(x_t\mid x_0)\Big\|^2 = \frac{1}{1-\bar\alpha_t}\,\big\|\epsilon_\theta(x_t,t) - \epsilon\big\|^2
$$

如果取权重 $\lambda(t) = 1-\bar\alpha_t$（对应 NCSN 的 $\lambda(\sigma)=\sigma^2$），就**恰好**得到 DDPM 的 $L_\text{simple}$。也就是说，**DDPM 的 $L_\text{simple}$ 就是一个特定加权的多噪声级别去噪分数匹配**。DDPM 预测的噪声，就是 score 的反方向，只差一个只依赖 $t$ 的系数。

### 8.2 两条路线的对照

| | NCSN（Score-based） | DDPM |
|---|---|---|
| 出发点 | 估计数据分布的梯度 | 变分推断，优化 ELBO |
| 加噪方式 | $\tilde x = x + \sigma_i\epsilon$，$x$ 不缩放 | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，$x_0$ 被缩小 |
| 噪声级别 | $L\approx10$ 个，几何递减 | $T=1000$ 个，由 $\beta_t$ 决定 |
| 网络输出 | score $s_\theta(\tilde x,\sigma)$ | 噪声 $\epsilon_\theta(x_t,t)$ |
| 训练目标 | $\sum_i\sigma_i^2\,\ell(\theta;\sigma_i)$ | $L_\text{simple}$ |
| 采样 | 退火朗之万：每个噪声级别跑多步 | 祖先采样：每个噪声级别走一步 |

两者在训练上本质相同：都是在一系列噪声水平上学习 score（或者等价地，学习去噪）；采样也都是由粗到细地从大噪声走向小噪声。差别主要在两处：

- **加噪方式**：NCSN 直接叠加越来越大的噪声，数据本身不缩放，总方差随噪声增长而**爆炸**；DDPM 在加噪的同时把数据按 $\sqrt{\bar\alpha_t}$ 缩小，总方差**保持**不变。
- **采样方式**：NCSN 在每个噪声级别上用朗之万动力学"原地"修正多步；DDPM 每个级别只走一步，直接跳到下一个级别。

下一篇会看到，这两种加噪方式恰好对应 SDE 框架下的 **VE-SDE**（Variance Exploding）和 **VP-SDE**（Variance Preserving），而两种采样方式可以组合成**预测-校正采样器**（Predictor-Corrector）：用 DDPM 式的一步作为"预测"，再用几步朗之万作为"校正"。

## 9. 一个应用：图像修补（Inpainting）

score-based 模型天然适合做图像修补这类任务，而且**不需要为修补重新训练**。

设一张图像 $y$ 中有一部分像素已知、一部分缺失，用二值掩码 $m$ 表示（$m=1$ 为已知像素，$m=0$ 为待补全像素）。我们想从 $p(x\mid x\odot m = y\odot m)$ 中采样，即在保持已知像素不变的前提下，补全缺失部分。

做法是在退火朗之万动力学中，**每一步都把已知区域替换为对应噪声水平下的加噪真实值**，只让未知区域跟随 score 演化：

1. 初始化 $\tilde x_0$
2. **for** $i = 1,\dots,L$ **do**
3. 　$\alpha_i = \delta\cdot\sigma_i^2/\sigma_L^2$
4. 　采样 $\tilde z\sim\mathcal{N}(0,I)$，令 $\tilde y = y + \sigma_i\tilde z$　　（已知区域在当前噪声水平下的样子）
5. 　**for** $k = 1,\dots,K$ **do**
6. 　　采样 $z_k\sim\mathcal{N}(0,I)$
7. 　　$\tilde x_k = \tilde x_{k-1} + \frac{\alpha_i}{2}\,s_\theta(\tilde x_{k-1},\sigma_i) + \sqrt{\alpha_i}\,z_k$
8. 　　$\tilde x_k \leftarrow \tilde x_k\odot(1-m) + \tilde y\odot m$　　（已知区域用加噪真实值覆盖）
9. 　**end for**
10. 　$\tilde x_0\leftarrow\tilde x_K$
11. **end for**
12. **return** $\tilde x_K$

为什么已知区域要替换为"加噪后的"真实值，而不是干净的真实值？因为在噪声级别 $\sigma_i$ 下，score 网络只认识 $q_{\sigma_i}$ 的样本；如果已知区域是干净的、未知区域是带噪声的，整张图就不像任何一个噪声级别下的样本，score 估计会出错。把已知区域也加上同样水平的噪声，整张图在每个噪声级别上都"自洽"。

随着噪声逐渐降低，未知区域在 score 的引导下，逐渐长出与已知区域协调的内容。同样的思路可以推广到上色、超分辨率、去模糊等**逆问题**：只要知道观测与原图之间的关系，就可以在采样中加入约束，用一个无条件训练的 score 模型来解决各种条件生成任务。第 7 篇的 guidance 是这一思路更系统的版本。

## 10. 小结

- **score 与归一化常数无关**：$\nabla_x\log p(x) = -\nabla_x f(x)$，所以可以用任意输入输出同维的网络去拟合，不需要关心 $Z$。
- **分数匹配**把"拟合一个未知的梯度"变成可以优化的目标：隐式分数匹配需要雅可比矩阵的迹，切片分数匹配用随机投影降低代价，**去噪分数匹配**把它变成"预测加入的噪声"，最简单也最常用。
- **朗之万动力学**只用 score 就能采样：漂移项沿 score 爬坡，扩散项注入随机性，两者配平后平稳分布恰好是目标分布。
- **朴素方法有三个问题**：流形外 score 无定义、低密度区估计不准、模式权重丢失。**加噪声**同时解决了这三个问题。
- **NCSN** 用一组几何递减的噪声，训练一个噪声条件的 score 网络，用退火朗之万动力学由粗到细地采样。
- **预测 score 与预测 DDPM 的噪声只差一个系数**：$s_\theta = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$，DDPM 的 $L_\text{simple}$ 就是特定加权的去噪分数匹配。两条路线殊途同归。

到这里，我们手上有了两个看起来很像的东西：DDPM 用一千个离散的噪声级别，NCSN 用十个离散的噪声级别。很自然的问题是：如果让噪声级别变成**连续的**，会发生什么？下一篇会回答：两者都变成了随机微分方程（SDE），而且只是同一个框架下的两个特例。

## 参考文献

- Hyvärinen. *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR 2005.
- Vincent. *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation 2011.
- Welling & Teh. *Bayesian Learning via Stochastic Gradient Langevin Dynamics.* ICML 2011.
- Efron. *Tweedie's Formula and Selection Bias.* JASA 2011.
- Song, Garg, Shi & Ermon. *Sliced Score Matching: A Scalable Approach to Density and Score Estimation.* UAI 2019.
- Song & Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS 2019.
- Song & Ermon. *Improved Techniques for Training Score-Based Generative Models.* NeurIPS 2020.
- Ho, Jain & Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021.
