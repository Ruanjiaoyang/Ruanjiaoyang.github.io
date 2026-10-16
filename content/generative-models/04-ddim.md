---
title: "DDIM：不重新训练，把 1000 步采样压到几十步"
title_en: "DDIM: Fewer Sampling Steps Without Retraining"
series: "从 GAN 到 Diffusion"
order: 4
date: 2026-10-16
tags: [生成模型, Diffusion, DDIM, 采样加速]
summary_en: >
  The DDPM training loss only depends on the marginals q(x_t|x_0), not on the Markov forward chain.
  DDIM exploits this freedom to define a family of non-Markovian inference processes with the same
  marginals. This post derives the family in two ways (by induction on the paper's definition and
  by solving for undetermined coefficients), shows why one network trained with L_simple serves every
  member of the family, and covers the eta parameter, deterministic sampling, step skipping, inversion,
  interpolation, and the interpretation of DDIM as an Euler step of an ODE.
---

## 0. 问题：DDPM 太慢

上一篇的 DDPM 有一个很大的缺点：**需要很多扩散步数才能得到好的效果，所以生成很慢。** 扩散步数取 1000，生成一个样本就要做 1000 次网络推理。相比之下，GAN 和 VAE 生成一个样本只需要一次前向。

能不能跳着走，比如每次走 20 步？在 DDPM 的框架下不行：反向过程是对马尔可夫前向链逐步求逆，$p_\theta(x_{t-1}\mid x_t)$ 被训练来逼近 $q(x_{t-1}\mid x_t,x_0)$，而这个后验是从**相邻两步**之间的转移 $q(x_t\mid x_{t-1})$ 推出来的。一旦跳步，单个高斯就很难描述一大步的反向转移（见上一篇 2.1 节）。

DDIM（Denoising Diffusion Implicit Models，Song, Meng & Ermon, 2021）给出了一个漂亮的答案：

- **DDIM 和 DDPM 的训练目标完全相同**，可以直接复用训练好的 DDPM 模型，不需要重新训练；
- 但它**不再要求扩散过程是一条马尔可夫链**，从而可以用少得多的采样步数加速生成；
- DDIM 的另一个特点是，从一个随机噪声生成样本的过程可以是**确定的**，中间不加入任何随机噪声。

一句话概括：**DDIM 不是一个新模型，而是同一个模型的另一种采样方法。**

## 1. 记号说明

DDIM 原论文的记号和 DDPM 不同，读论文时很容易混淆：

| | DDPM 论文 | DDIM 论文 |
|---|---|---|
| 累积系数 | $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$ | $\alpha_t$ |
| 前向闭式解 | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ | $x_t = \sqrt{\alpha_t}\,x_0 + \sqrt{1-\alpha_t}\,\epsilon$ |

也就是说，**DDIM 论文里的 $\alpha_t$，就是 DDPM 论文里的 $\bar\alpha_t$**。换成 DDIM 的记号，DDPM 前向过程每一步的方差为

$$
\beta_t = 1 - \frac{\alpha_t}{\alpha_{t-1}} \qquad (\text{DDIM 记号})
$$

为了和上一篇衔接，**本文统一使用 DDPM 的记号** $\bar\alpha_t$，$\alpha_t = 1-\beta_t$ 表示单步系数。读原论文时，把下面所有的 $\bar\alpha$ 换成 $\alpha$ 即可。

## 2. 关键观察：训练只用到了边际分布

### 2.1 DDPM 训练目标里有什么

回顾 DDPM：前向过程是一条马尔可夫链，反向过程也是一条马尔可夫链，最终的优化目标是

$$
L_\text{simple} = \mathbb{E}_{t,\,x_0\sim q(x_0),\,\epsilon\sim\mathcal{N}(0,I)}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big)\big\|^2\Big]
$$

仔细看这个式子：训练时，我们从 $x_0$ **一步**加噪得到 $x_t$，这里只用到了**边际分布** $q(x_t\mid x_0)$。至于前向过程的**联合分布** $q(x_{1:T}\mid x_0)$ 长什么样、中间各步之间怎么转移，训练根本没有用到。

这带来一个重要启示：**DDPM 这个隐变量模型可以有很多种推理分布（前向过程）可选，只要它们满足同样的边际分布**

$$
q(x_t\mid x_0) = \mathcal{N}\big(x_t;\;\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big)
$$

**就行，而且这些推理过程不一定是马尔可夫链。**

### 2.2 还缺一样东西：后验

但有一点需要注意。要推出 DDPM 的优化目标（或者说，要知道怎么采样），我们还需要后验 $q(x_{t-1}\mid x_t, x_0)$，因为 ELBO 中的 $L_{t-1}$ 就是网络与这个后验之间的 KL。

上一篇用贝叶斯公式推这个后验时，**依赖了前向转移 $q(x_t\mid x_{t-1})$ 和马尔可夫性**：

$$
q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t\mid x_{t-1})\;q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

如果要摆脱对马尔可夫前向过程的依赖，就得换个思路：**不定义前向转移，直接定义后验 $q(x_{t-1}\mid x_t, x_0)$**，只要求它与边际分布相容。下面给出两种推导：第 3 节按论文的写法，先给出定义再验证；第 4 节用待定系数法，从约束条件直接解出来。

## 3. 推导一：按论文的定义

### 3.1 定义一族推理分布

DDIM 论文把推理分布定义为

$$
q_\sigma(x_{1:T}\mid x_0) = q_\sigma(x_T\mid x_0)\prod_{t=2}^{T} q_\sigma(x_{t-1}\mid x_t, x_0)
$$

注意这里是**从 $x_T$ 往回**写的乘积，并没有定义"从 $x_{t-1}$ 到 $x_t$"的前向转移。其中要求

$$
q_\sigma(x_T\mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_T}\,x_0,\;(1-\bar\alpha_T)I\big)
$$

并且对所有 $t\ge 2$，

$$
q_\sigma(x_{t-1}\mid x_t, x_0) = \mathcal{N}\Big(x_{t-1};\;\sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}},\;\sigma_t^2 I\Big)
$$

这里的 $\sigma_t^2$ 是一个可以自由选择的实数（要求 $\sigma_t^2\le 1-\bar\alpha_{t-1}$）。**不同的 $\sigma=(\sigma_1,\dots,\sigma_T)$ 对应不同的分布**，所以 $q_\sigma(x_{1:T}\mid x_0)$ 实际上是**一族**推理分布。

后验的均值同样是 $x_0$ 和 $x_t$ 的线性组合，但写成了一个很有意义的形式：注意

$$
\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}} = \epsilon
$$

正是 $x_t$ 中所含的噪声。所以这个均值就是"$x_0$ 按 $t-1$ 时刻的比例缩放，再加上按 $t-1$ 时刻的比例缩放的**同一份噪声**"，剩下的方差 $\sigma_t^2$ 用新的随机噪声补上。

### 3.2 验证边际分布：数学归纳法

为什么后验要定义成这个形式？因为它保证了对**所有** $t$，边际分布都满足

$$
q_\sigma(x_t\mid x_0) = \mathcal{N}\big(x_t;\;\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big)
$$

用数学归纳法从 $t=T$ 往回证明。

- **起点**：$t=T$ 时由定义成立。
- **归纳步**：假设对某个 $t$ 成立，即 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，$\epsilon\sim\mathcal{N}(0,I)$。按后验的定义，用重参数化采样 $x_{t-1}$：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\underbrace{\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}}_{=\,\epsilon} + \sigma_t z, \qquad z\sim\mathcal{N}(0,I)
$$

$\epsilon$ 和 $z$ 是两个独立的标准高斯，所以 $x_{t-1}$ 是高斯，均值为 $\sqrt{\bar\alpha_{t-1}}\,x_0$，方差为

$$
\big(1-\bar\alpha_{t-1}-\sigma_t^2\big) + \sigma_t^2 = 1-\bar\alpha_{t-1}
$$

于是 $q_\sigma(x_{t-1}\mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_{t-1}}\,x_0,\,(1-\bar\alpha_{t-1})I\big)$，归纳完成。

这个证明很短，但它揭示了后验形式背后的设计思路：**两份独立的噪声，方差加起来恰好凑够 $t-1$ 时刻应有的噪声水平。** $\sigma_t$ 只是决定了"沿用旧噪声"和"引入新噪声"的比例。

### 3.3 这族推理分布满足两个条件

可以看到，这里定义的推理分布 $q_\sigma(x_{1:T}\mid x_0)$ 并没有直接定义前向过程，但它满足了我们需要的两个条件：

1. **边际分布与 DDPM 相同**：$q_\sigma(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I)$，所以训练目标里用到的东西不变；
2. **后验分布已知**：$q_\sigma(x_{t-1}\mid x_t,x_0)$ 是一个显式的高斯，可以用来推导 ELBO，也可以用来采样。

如果非要问"前向过程"是什么，可以用贝叶斯公式反推：

$$
q_\sigma(x_t\mid x_{t-1}, x_0) = \frac{q_\sigma(x_{t-1}\mid x_t, x_0)\;q_\sigma(x_t\mid x_0)}{q_\sigma(x_{t-1}\mid x_0)}
$$

它也是一个高斯。但关键在于：**生成 $x_t$ 时不仅依赖 $x_{t-1}$，还依赖 $x_0$**，所以这个前向过程一般**不是马尔可夫链**。这是 DDIM 和 DDPM 的根本区别。

### 3.4 $\eta=1$ 时退回 DDPM

如果取

$$
\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

即 DDPM 的后验方差，那么

$$
1-\bar\alpha_{t-1}-\tilde\beta_t = (1-\bar\alpha_{t-1})\Big(1 - \frac{\beta_t}{1-\bar\alpha_t}\Big) = (1-\bar\alpha_{t-1})\cdot\frac{\alpha_t - \bar\alpha_t}{1-\bar\alpha_t} = \frac{\alpha_t\,(1-\bar\alpha_{t-1})^2}{1-\bar\alpha_t}
$$

（用了 $1-\bar\alpha_t-\beta_t = \alpha_t-\bar\alpha_t = \alpha_t(1-\bar\alpha_{t-1})$。）开方得 $\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{\sqrt{1-\bar\alpha_t}}$。代入后验均值，$x_t$ 的系数是 $\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$，$x_0$ 的系数是 $\sqrt{\bar\alpha_{t-1}} - \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\sqrt{\bar\alpha_t}}{1-\bar\alpha_t} = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}$，和上一篇 2.3 节的 $\tilde\mu_t$ **完全一样**。

所以 **DDPM 的后验只是这一族分布中的一个特例**。虽然两篇论文对参数的定义不太相同，但在这个取值下，DDIM 的后验和 DDPM 的后验是完全等价的，此时的推理过程也恰好是马尔可夫的。

## 4. 推导二：从边际约束出发，用待定系数解出来

上一节是"先给答案，再验证"。这一节换个角度，看这个答案是怎么"找"出来的。

### 4.1 马尔可夫性从哪里来

DDPM 的后验由贝叶斯公式推出：

$$
q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t\mid x_{t-1})\;q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

它之所以和马尔可夫性绑在一起，主要源于分子中的 $q(x_t\mid x_{t-1})$ 这一项。怎样"消除"这一项？

注意我们真正需要的约束只有一个：**后验要和边际分布相容**。用全概率公式写出来就是

$$
\int q(x_{t-1}\mid x_t, x_0)\;q(x_t\mid x_0)\;dx_t = q(x_{t-1}\mid x_0)
$$

意思是：先按边际分布采一个 $x_t$，再按后验采 $x_{t-1}$，得到的 $x_{t-1}$ 必须服从 $t-1$ 时刻的边际分布。DDPM 的后验满足这个方程，但它**只是这个方程的一个特解**。只要找出更一般的解，就摆脱了马尔可夫性。

### 4.2 为什么可以不管 $q(x_t\mid x_{t-1})$

更一般地，考虑任意两个时刻 $s<k$（不必相邻），想要一个从 $x_k$ 直接跳到 $x_s$ 的采样公式。由贝叶斯公式

$$
q(x_s\mid x_k, x_0) = \frac{q(x_k\mid x_s, x_0)\;q(x_s\mid x_0)}{q(x_k\mid x_0)}
$$

在 DDPM 中，由于马尔可夫性，右边三项都是已知的。但如果我们选择的采样方式不满足马尔可夫性，右边就有一部分是未知的。怎么办？

回想 DDPM 的训练：我们是从 $x_0$ **一步**生成 $x_t$，即拟合的是 $q(x_t\mid x_0)$。所以右边的 $q(x_s\mid x_0)$ 和 $q(x_k\mid x_0)$ 是训练中确定下来的，**必须保持**；而 $q(x_k\mid x_s,x_0)$（在 DDPM 里对应前向转移 $q(x_t\mid x_{t-1})$）**在训练中根本没用到**。既然没用到，说明它不重要，我们可以不管它，直接去设定左边的 $q(x_s\mid x_k,x_0)$，只要它和两个边际相容。

这也回答了为什么说**DDIM 其实是一种采样方法，训练还是 DDPM 的训练**。

### 4.3 待定系数

既然不关心前向转移，那就直接假设后验是一个高斯，均值是 $x_k$ 和 $x_0$ 的线性组合：

$$
q(x_s\mid x_k, x_0) = \mathcal{N}\big(\kappa\,x_k + \lambda\,x_0,\;\sigma^2 I\big)
$$

这里有三个未知数 $\kappa,\lambda,\sigma$。用重参数化写成

$$
x_s = \kappa\,x_k + \lambda\,x_0 + \sigma\,z, \qquad z\sim\mathcal{N}(0,I)
$$

要求它满足两个边际：$x_k = \sqrt{\bar\alpha_k}\,x_0 + \sqrt{1-\bar\alpha_k}\,\epsilon$，并且 $x_s$ 的分布应为 $\mathcal{N}(\sqrt{\bar\alpha_s}\,x_0,(1-\bar\alpha_s)I)$。把 $x_k$ 代入：

$$
x_s = \big(\kappa\sqrt{\bar\alpha_k} + \lambda\big)\,x_0 + \kappa\sqrt{1-\bar\alpha_k}\;\epsilon + \sigma\,z
$$

$\epsilon$ 与 $z$ 独立，对照目标分布的均值和方差，得到两个方程：

$$
\kappa\sqrt{\bar\alpha_k} + \lambda = \sqrt{\bar\alpha_s}, \qquad \kappa^2\,(1-\bar\alpha_k) + \sigma^2 = 1-\bar\alpha_s
$$

三个未知数、两个方程，**有一个自由度**。把 $\sigma$ 当作自由参数，解出

$$
\kappa = \frac{\sqrt{1-\bar\alpha_s-\sigma^2}}{\sqrt{1-\bar\alpha_k}}, \qquad
\lambda = \sqrt{\bar\alpha_s} - \frac{\sqrt{\bar\alpha_k}\,\sqrt{1-\bar\alpha_s-\sigma^2}}{\sqrt{1-\bar\alpha_k}}
$$

代回去，得到新的后验：

$$
q(x_s\mid x_k, x_0) = \mathcal{N}\Big(\sqrt{\bar\alpha_s}\,x_0 + \sqrt{1-\bar\alpha_s-\sigma^2}\cdot\frac{x_k - \sqrt{\bar\alpha_k}\,x_0}{\sqrt{1-\bar\alpha_k}},\;\sigma^2 I\Big)
$$

取 $s=t-1$、$k=t$，就是第 3 节论文给出的定义。两种推导殊途同归，而这一种还顺便说明了：**$s$ 和 $k$ 不必相邻**，这正是第 7 节跳步采样的依据。

## 5. 为什么同一个网络对整族分布都适用

### 5.1 生成过程

有了后验，生成过程就照着它来写：从 $x_t$ 采样 $x_{t-1}$ 时，需要 $x_0$，但生成时 $x_0$ 是未知的。和 DDPM 一样，用网络预测的噪声来估计它：

$$
\hat x_0 = f_\theta(x_t, t) = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

然后把 $\hat x_0$ 代入后验，定义生成过程：

$$
p_\theta(x_{t-1}\mid x_t) = q_\sigma\big(x_{t-1}\mid x_t,\, \hat x_0\big), \qquad t\ge 2
$$

最后一步 $t=1$ 时取 $p_\theta(x_0\mid x_1) = \mathcal{N}\big(\hat x_0(x_1),\,\sigma_1^2 I\big)$。

### 5.2 不同 $\sigma$ 的 ELBO 只差权重

对每一个 $\sigma$，都可以像 DDPM 那样写出 ELBO。其中第 $t$ 项是两个方差相同的高斯之间的 KL：

$$
\mathrm{KL}\big(q_\sigma(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big) = \frac{1}{2\sigma_t^2}\big\|\text{均值}(x_0) - \text{均值}(\hat x_0)\big\|^2
$$

两个均值只在 $x_0$ 和 $\hat x_0$ 处不同，$x_0$ 的系数是 4.3 节的 $\lambda_t$，所以

$$
\mathrm{KL} = \frac{\lambda_t^2}{2\sigma_t^2}\,\big\|x_0 - \hat x_0\big\|^2 = \frac{\lambda_t^2\,(1-\bar\alpha_t)}{2\sigma_t^2\,\bar\alpha_t}\,\big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2
$$

最后一步用了 $x_0 - \hat x_0 = \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}(\epsilon_\theta - \epsilon)$。

也就是说，**对任何 $\sigma$，ELBO 都是各个时间步上噪声预测误差 $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$ 的加权和**，加上一个常数，只是权重 $\gamma_t$ 随 $\sigma$ 变化。这就是 DDIM 论文的定理 1：对任意 $\sigma$，存在权重 $\gamma$ 和常数 $C$，使得

$$
J_\sigma(\epsilon_\theta) = L_\gamma(\epsilon_\theta) + C, \qquad L_\gamma = \sum_{t=1}^T \gamma_t\,\mathbb{E}\big[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\big]
$$

### 5.3 权重不影响最优解

关键的一步：如果网络在不同时间步之间不共享参数（或者说容量足够大，每个 $t$ 都能独立学到最优），那么 $L_\gamma$ 中每一项可以分别最小化，**最优解与权重 $\gamma_t$ 无关**，都是"在每个 $t$ 上把噪声预测到最好"。

于是：**用 $L_\text{simple}$（$\gamma_t\equiv1$）训练出来的 $\epsilon_\theta$，同时也是整族 $J_\sigma$ 的最优解。** 训练一次，就可以用任意一个 $\sigma$ 来采样。这就是为什么 DDIM 可以直接拿 DDPM 训练好的模型来用。

（严格地说，$\sigma_t\to0$ 时上面的 KL 会发散，ELBO 不再有意义，这时的模型可以看作 $\sigma\to0$ 的极限。这也是"implicit"这个名字的由来：$\sigma=0$ 时，生成过程是一个从 $x_T$ 到 $x_0$ 的确定性映射，和 GAN 的生成器一样，是一个**隐式**定义的分布，能采样，但写不出密度。）

## 6. 采样公式与 $\eta$

### 6.1 三项分解

把 $\hat x_0$ 代入后验并用重参数化，注意 $\frac{x_t - \sqrt{\bar\alpha_t}\,\hat x_0}{\sqrt{1-\bar\alpha_t}} = \epsilon_\theta(x_t,t)$，得到从 $x_t$ 生成 $x_{t-1}$ 的公式：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\underbrace{\Big(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}\Big)}_{\text{predicted } x_0}
+ \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\epsilon_\theta(x_t,t)}_{\text{direction pointing to } x_t}
+ \underbrace{\sigma_t\,z}_{\text{random noise}}
$$

三项各有含义：

- **predicted $x_0$**：用网络预测的噪声，从 $x_t$ 反推出的干净数据估计 $\hat x_0$，再按 $t-1$ 时刻的信号比例 $\sqrt{\bar\alpha_{t-1}}$ 缩放；
- **direction pointing to $x_t$**：沿着预测噪声的方向，补回 $t-1$ 时刻应有的一部分噪声，相当于"指向 $x_t$"的偏移量；
- **random noise**：额外加入的新随机噪声 $z\sim\mathcal{N}(0,I)$，与 $x_t$ 独立。

整体的直观理解是：**先猜一个干净图像 $\hat x_0$，再按 $t-1$ 时刻应有的噪声水平，把它"重新加噪"回去。** 其中一部分噪声沿用网络在 $x_t$ 中识别出的那份噪声 $\epsilon_\theta$，另一部分用新的随机噪声。

### 6.2 用 $\eta$ 控制随机性

论文把 $\sigma_t$ 参数化为

$$
\sigma_t = \eta\,\sqrt{\tilde\beta_t} = \eta\,\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\;\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}
$$

（第二个等号用了 $1-\bar\alpha_t/\bar\alpha_{t-1} = 1-\alpha_t = \beta_t$。）$\eta\ge0$ 控制每一步注入多少新的随机性：

- **$\eta=1$**：$\sigma_t^2=\tilde\beta_t$，由 3.4 节，生成过程**等同于 DDPM**。
- **$\eta=0$**：$\sigma_t=0$，生成过程中**没有随机噪声**，是一个**确定性的去噪过程**，这就是通常所说的 DDIM。一旦最初的随机噪声 $x_T$ 确定，生成的样本就唯一确定。
- **$0<\eta<1$**：介于两者之间。

$\eta=0$ 时，采样公式简化为

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\;\epsilon_\theta(x_t,t)
$$

这个形式和前向闭式解 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ 一模一样，只是把真实的 $x_0$ 和 $\epsilon$ 换成了网络的估计。也就是说，**DDIM 每一步都在假装"估计出的 $\hat x_0$ 和 $\epsilon_\theta$ 就是真的"，然后直接算出 $t-1$ 时刻应该是什么样子。**

## 7. 跳步：为什么可以加速

### 7.1 从 $x_k$ 直接跳到 $x_s$

$\eta=0$ 时整个过程是确定的，没有随机性，那就不必从 $x_T$ 一步步走到 $x_0$。更重要的是，第 4 节的推导中**从未用到"$s$ 必须紧挨着 $k$"**。由于我们不再要求马尔可夫性，可以把采样公式改写为任意 $s<k$ 之间的跳跃：

$$
x_s = \sqrt{\bar\alpha_s}\,\underbrace{\Big(\frac{x_k - \sqrt{1-\bar\alpha_k}\,\epsilon_\theta(x_k,k)}{\sqrt{\bar\alpha_k}}\Big)}_{\text{predicted } x_0}
+ \underbrace{\sqrt{1-\bar\alpha_s-\sigma_k^2}\;\epsilon_\theta(x_k,k)}_{\text{direction pointing to } x_k}
+ \underbrace{\sigma_k\,z}_{\text{random noise}}
$$

### 7.2 在子序列上采样

具体做法：取 $[1,\dots,T]$ 的一个递增子序列 $\tau_1<\tau_2<\cdots<\tau_S$，长度 $S\ll T$。在这个子序列上定义推理过程，它在 $\tau_i$ 处的边际分布仍然是 $q(x_{\tau_i}\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_{\tau_i}}\,x_0,(1-\bar\alpha_{\tau_i})I)$，与训练时一致。所以**用同一个网络，只在这 $S$ 个时刻上采样，不需要重新训练**。

以 $T=1000$、$S=50$ 为例，可以取 $\tau = (1, 21, 41, \dots, 981)$（等间隔），也可以让步长在 $t$ 小的一端更密（论文在 CIFAR-10 上用了二次方间隔）。

```
DDPM 采样：  x_1000 → x_999 → x_998 → ··· → x_1 → x_0      （1000 次网络前向，每步加随机噪声）
DDIM 采样：  x_981 ──▶ x_961 ──▶ x_941 ──▶ ··· ──▶ x_21 ──▶ x_1 ──▶ x_0   （50 次网络前向，η=0 时无随机噪声）
            每一步：先估计 x̂_0，再直接"重新加噪"到下一个时刻
```

```python
@torch.no_grad()
def ddim_sample(model, shape, taus, eta=0.0):
    x = torch.randn(shape)
    for i in reversed(range(len(taus))):
        t = taus[i]
        ab_t = alphas_bar[t]
        ab_s = alphas_bar[taus[i - 1]] if i > 0 else torch.tensor(1.0)
        eps = model(x, torch.full((shape[0],), t))
        x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        sigma = eta * ((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)).sqrt()
        x = ab_s.sqrt() * x0_hat + (1 - ab_s - sigma**2).sqrt() * eps + sigma * torch.randn_like(x)
    return x
```

### 7.3 效果与限制

论文在 CIFAR-10 和 CelebA 上的实验结论：

- **步数少时，DDIM（$\eta=0$）远好于 DDPM（$\eta=1$）**。例如在 CIFAR-10 上只走 10 步，DDIM 的 FID 约 13，而 $\eta=1$ 的采样超过 40；如果用 DDPM 原文那个更大的方差 $\sigma_t^2=\beta_t$，少步数时几乎完全失效。
- **步数多时，两者差距缩小**。走满 1000 步时，DDPM 略好一些。
- 总体上，DDIM 用 **20 到 100 步**就能达到接近 1000 步 DDPM 的质量，带来 10 到 50 倍的加速。

为什么少步数时确定性采样更好？一种直观解释是：加入随机噪声相当于每一步都引入新的扰动，需要后面的步骤去"修正"；步数很少时，没有足够的步骤来修正这些扰动。确定性采样则没有这个问题。

但也不能跳得太狠。网络对 $\epsilon$ 的预测并不完美，每一步对 $\hat x_0$ 的估计都有误差，步子越大，误差的累积越严重。实践中一次跳 10 到 20 步是比较安全的；如果只走 5 步甚至更少，质量会明显下降。这正是后来 DPM-Solver 等高阶求解器、以及渐进蒸馏、Consistency Models 等蒸馏方法要解决的问题。

## 8. 确定性带来的额外能力

$\eta=0$ 时，$x_T\to x_0$ 是一个确定性映射，而且（步长足够小时）近似可逆。这带来了几个 DDPM 没有的能力。

### 8.1 一致性

用同一个初始噪声 $x_T$，不论用 20 步、50 步还是 1000 步采样，生成的图像在高层语义上基本一致，只是细节上有差别。这说明 $x_T$ 本身就像 GAN 的隐变量 $z$ 一样，编码了图像的主要信息。DDPM 由于每一步都注入随机噪声，同一个 $x_T$ 会生成完全不同的图像。

### 8.2 隐空间插值

既然 $x_T$ 编码了图像，就可以在两个初始噪声 $x_T^{(0)}$、$x_T^{(1)}$ 之间插值，得到语义上平滑过渡的图像序列。由于 $x_T$ 来自高维高斯分布，它们的范数集中在 $\sqrt d$ 附近，直线插值的中点范数会变小、偏离分布，所以通常用**球面插值**（slerp）：

$$
x_T^{(\lambda)} = \frac{\sin\big((1-\lambda)\theta\big)}{\sin\theta}\,x_T^{(0)} + \frac{\sin(\lambda\theta)}{\sin\theta}\,x_T^{(1)}, \qquad \theta = \arccos\Big(\frac{\langle x_T^{(0)},x_T^{(1)}\rangle}{\|x_T^{(0)}\|\,\|x_T^{(1)}\|}\Big)
$$

### 8.3 反演（Inversion）

既然 $x_T\to x_0$ 是确定的、近似可逆的，那就可以反过来：给一张**真实图像** $x_0$，按 DDIM 的公式**正向**走到 $x_T$，得到它的"噪声编码"。正向一步为

$$
x_{t+1} = \sqrt{\bar\alpha_{t+1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t+1}}\;\epsilon_\theta(x_t,t), \qquad \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

这里有一个近似：严格来说应该用 $\epsilon_\theta(x_{t+1},t+1)$，但 $x_{t+1}$ 还不知道，所以用 $\epsilon_\theta(x_t,t)$ 代替。步长较小时，这个近似误差很小，从得到的 $x_T$ 出发再做 DDIM 采样，可以近似重建原图。

DDIM inversion 是很多基于扩散模型的**图像编辑**方法的基础：先把真实图像反演到噪声，再在采样时修改条件（比如把文本提示里的"猫"换成"狗"），就能在保留原图结构的同时完成编辑。在使用 classifier-free guidance（第 7 篇）时，反演误差会被放大，Null-text Inversion 等方法专门处理这个问题。

### 8.4 确定性采样背后是一个 ODE

为什么 DDIM 有这些性质？把 $\eta=0$ 的更新公式两边同除以 $\sqrt{\bar\alpha_{t-1}}$：

$$
\frac{x_{t-1}}{\sqrt{\bar\alpha_{t-1}}} = \frac{x_t}{\sqrt{\bar\alpha_t}} + \Big(\sqrt{\frac{1-\bar\alpha_{t-1}}{\bar\alpha_{t-1}}} - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\Big)\,\epsilon_\theta(x_t,t)
$$

令 $\bar x = x/\sqrt{\bar\alpha}$，$\bar\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$（它是信噪比平方根的倒数），上式就是

$$
\bar x_{t-1} - \bar x_t = \big(\bar\sigma_{t-1} - \bar\sigma_t\big)\,\epsilon_\theta(x_t,t)
$$

这正是常微分方程

$$
\frac{d\bar x}{d\bar\sigma} = \epsilon_\theta\Big(\frac{\bar x}{\sqrt{1+\bar\sigma^2}},\, t\Big)
$$

的**欧拉法**离散化（$\bar\alpha = 1/(1+\bar\sigma^2)$，所以 $x = \bar x/\sqrt{1+\bar\sigma^2}$）。也就是说，**DDIM 的确定性采样，就是在沿着一条 ODE 的轨迹走**：

- ODE 的解是确定的，所以同一个起点总是到达同一个终点（一致性）；
- ODE 可以正着解也可以反着解，所以可以反演；
- 欧拉法是最简单的一阶求解器，步长大了误差就大，所以跳步太狠会变差；换成高阶求解器（DPM-Solver、DPM-Solver++ 等），10 到 20 步就能得到很好的结果。

第 6 篇会看到，这条 ODE 就是**概率流 ODE**（probability flow ODE）。DDPM 对应一个随机微分方程（SDE），DDIM 对应和它具有相同边际分布的 ODE。

## 9. 小结

| | DDPM | DDIM（$\eta=0$） |
|---|---|---|
| 训练 | $L_\text{simple}$ | **完全相同**，直接复用 DDPM 权重 |
| 推理分布（前向过程） | 马尔可夫链 | 一族非马尔可夫过程，只要求边际分布一致 |
| 后验 | 由前向转移经贝叶斯公式推出 | 直接定义，含自由参数 $\sigma_t$ |
| 采样 | 随机，每步加噪声 | 确定，不加噪声 |
| 能否跳步 | 不能，必须走满 $T$ 步 | 能，在任意子序列上采样 |
| 典型步数 | 1000 | 20 到 100 |
| 额外能力 | 无 | 一致性、插值、反演、图像编辑 |
| 连续时间视角 | 反向 SDE | 概率流 ODE 的欧拉离散化 |

核心思想可以归结为一句话：**训练只依赖边际分布 $q(x_t\mid x_0)$，所以"训练用什么过程"和"采样用什么过程"可以解耦。** DDIM 利用了这个自由度，在不改变训练的前提下，换了一个对采样更友好的过程。

这个"边际分布相同、但过程不同"的思想，在第 6 篇会以更一般的形式再次出现：同一组边际分布，可以对应一整族随机微分方程，其中噪声为零的那一个，就是概率流 ODE。不过在那之前，下一篇先从一个完全不同的角度重新出发：**不学密度，学密度的梯度。**

## 参考文献

- Song, Meng & Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
- Ho, Jain & Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021.
- Salimans & Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022.
- Lu et al. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.* NeurIPS 2022.
- Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022.
- Song et al. *Consistency Models.* ICML 2023.
- Mokady et al. *Null-text Inversion for Editing Real Images using Guided Diffusion Models.* CVPR 2023.
