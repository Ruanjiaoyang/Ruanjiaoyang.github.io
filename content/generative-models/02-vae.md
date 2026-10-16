---
title: "从 AE 到 VAE 再到 VQ-VAE：让隐空间可以采样"
title_en: "From AE to VAE to VQ-VAE: Making the Latent Space Samplable"
series: "从 GAN 到 Diffusion"
order: 2
date: 2026-10-16
tags: [生成模型, VAE, VQ-VAE, 变分推断]
summary_en: >
  An autoencoder compresses data but cannot generate, because nothing constrains its latent space.
  VAEs fix this with a probabilistic encoder and a KL regularizer. This post derives the evidence
  lower bound in two ways, the closed-form Gaussian KL and the reparameterization trick, explains
  blurriness and posterior collapse, and then covers VQ-VAE in detail: codebook, the three loss terms,
  the straight-through estimator, EMA updates, the learned prior, codebook collapse, VQ-VAE-2 and
  VQGAN, which together became the tokenizer behind latent diffusion and autoregressive image models.
---

上一篇的 GAN 走的是"不写密度，只学采样器"的路线。这一篇换一条路：**写出一个带隐变量的概率模型，用变分推断去优化它的似然下界。** 我们从最朴素的自编码器出发，看它为什么不能生成；再看 VAE 怎样用一个 KL 正则项修好这个问题；最后看 VQ-VAE 怎样把隐空间离散化，成为后来几乎所有视觉 tokenizer 的起点。

这一篇的 ELBO 推导在下一篇 DDPM 里会原样再走一遍，所以值得仔细看。

## 1. AE：自编码器

### 1.1 基本架构

自编码器（Auto-Encoder，AE）是一种无监督学习模型，用神经网络实现数据的压缩和重构。它由两个部分组成：

- **编码器（Encoder）**：把输入数据 $x$ 映射到一个低维的潜在空间（latent space），得到隐表示 $z = f_\phi(x)$，$\phi$ 是编码器参数。
- **解码器（Decoder）**：从隐变量 $z$ 中重构数据，得到 $\hat x = g_\theta(z)$，$\theta$ 是解码器参数。

$z$ 的维度远小于 $x$，中间形成一个"瓶颈"。网络被迫只保留重构所必需的信息，于是学到的是数据的压缩表示。

```
  x ──▶ [ 编码器 f_φ ] ──▶ z（低维瓶颈） ──▶ [ 解码器 g_θ ] ──▶ x̂
  例：784 维 MNIST 图像           2 到 32 维                    784 维重建图像
```

### 1.2 损失函数

AE 的目标是最小化输入和输出之间的重构误差，常用均方误差：

$$
\mathcal{L}_\text{AE}(\phi,\theta) = \mathbb{E}_{x}\Big[\big\| x - g_\theta\big(f_\phi(x)\big)\big\|^2\Big]
$$

如果数据是二值的（如二值化的 MNIST），也可以用逐像素的二元交叉熵。这个损失衡量的是：编码器得到的隐表示 $z$ 经过解码器还原后，和原始输入差了多少。

### 1.3 AE 与 PCA

线性自编码器（编码器和解码器都是线性映射、没有激活函数）在 MSE 损失下的全局最优解，恰好张成数据的前 $k$ 个主成分所在的子空间（Baldi & Hornik, 1989）。也就是说，**线性 AE 学到的就是 PCA**，只是基向量不一定正交、不一定按方差排序。

既然如此，为什么还要用 AE？因为 AE 的强大之处在于**非线性**。加入非线性激活函数和更多隐藏层之后，AE 可以学到弯曲的低维流形，而不只是一个线性子空间，因此能在更低的维度上保留更多信息。PCA 只能把一个瑞士卷"压扁"，而非线性 AE 有能力把它"展开"。

### 1.4 AE 的局限：会压缩，不会生成

AE 在数据压缩、降维和特征提取上应用广泛，但有一个根本问题：**它不能用来生成新样本。**

**潜在空间的结构不受约束。** AE 只要求"编码后能解回来"，对隐空间里的点如何分布没有任何约束。在 MNIST 上训练一个 2 维隐空间的 AE，再把所有训练样本的编码画出来，会看到：

- 相同数字的编码倾向于聚在一起，形成若干个簇；
- 簇的形状不规则，大小差别很大，有的细长，有的分散；
- **簇与簇之间有大片空白**，这些区域不对应任何训练样本。

如果在隐空间中随便采一个点送进解码器，它很可能落在空白区域。解码器从没见过这样的输入，输出往往不像任何一个数字。只有落在某个簇内部的点才能解码出合理的结果，而一旦训练数据被移除，我们甚至无法知道哪些区域是"安全"的。这就是所谓的**潜在空间没有正则化**。

**过拟合会让情况更糟。** 理论上，一个容量足够大的编码器可以把每个训练样本映射到实数轴上的一个独立点，解码器再把每个点精确还原，重构误差为零，但隐空间完全没有结构，相邻的点之间没有任何语义联系。

所以 AE 主要用于压缩：对有效的输入，它能把数据压到更少的维度、去掉冗余；但由于隐空间没有正则化，解码器不能可靠地从随便采样的隐向量生成有效数据。要生成，我们需要一个**知道该从哪里采样**的隐空间：它最好是连续的（相近的点解码出相近的结果），并且是完整的（任何采样到的点都能解码出有意义的结果）。

## 2. VAE：变分自编码器

VAE（Variational Auto-Encoder，Kingma & Welling, 2014）的核心思想是在自编码器的基础上引入**概率图模型**，用**变分推断**来近似难以计算的后验分布。从结构上看，它和 AE 很像，但每一个部件都换成了概率版本。

### 2.1 从生成过程出发

VAE 首先写下一个数据的生成过程：

$$
z \sim p(z) = \mathcal{N}(0, I), \qquad x \sim p_\theta(x \mid z)
$$

- $p(z)$ 是**先验分布**，通常取标准正态分布。它就是我们生成时要采样的地方，形状已知、处处连续。
- $p_\theta(x\mid z)$ 是**似然**，由解码器参数化，表示从隐变量 $z$ 生成数据 $x$ 的条件分布。

我们的目标是最大化数据的**边际似然**（也叫证据，evidence）：

$$
p_\theta(x) = \int p_\theta(x\mid z)\, p(z)\, dz
$$

### 2.2 为什么不能直接算

这个积分在高维隐空间上没有闭式解。能不能用蒙特卡洛估计？

$$
p_\theta(x) \approx \frac1N\sum_{i=1}^N p_\theta(x\mid z_i), \qquad z_i\sim p(z)
$$

原则上可以，但效率极低：对于一个具体的 $x$（比如某张数字"3"的图），绝大多数从先验里随便采的 $z_i$ 会解码成别的东西，$p_\theta(x\mid z_i)$ 几乎为零。只有隐空间中极小一块区域对 $x$ 有贡献，要采到它们需要天文数字的样本。

能对 $x$ 有贡献的 $z$，恰好是**后验分布** $p_\theta(z\mid x)$ 集中的地方。但由贝叶斯公式，

$$
p_\theta(z\mid x) = \frac{p_\theta(x\mid z)\,p(z)}{p_\theta(x)}
$$

分母正是我们算不出的 $p_\theta(x)$。于是陷入循环：要算似然需要后验，要算后验需要似然。

### 2.3 变分推断：用编码器近似后验

VAE 的办法是：**引入一个可学习的分布 $q_\phi(z\mid x)$ 来近似真实后验 $p_\theta(z\mid x)$。** 这个 $q_\phi$ 就是编码器。

注意它是一个以 $x$ 为输入的神经网络，对所有数据点共享同一组参数 $\phi$。传统的变分推断会为每个数据点单独优化一组变分参数，而 VAE 用一个网络一次性"预测"出每个数据点的近似后验，这叫**摊销推断**（amortized inference）。

### 2.4 ELBO 推导

**推导一：从对数似然出发，拆出 KL。** 由于 $\log p_\theta(x)$ 与 $z$ 无关，而 $q_\phi(z\mid x)$ 积分为 1：

$$
\begin{aligned}
\log p_\theta(x)
&= \int q_\phi(z\mid x)\,\log p_\theta(x)\, dz \\
&= \int q_\phi(z\mid x)\,\log \frac{p_\theta(x,z)}{p_\theta(z\mid x)}\, dz \\
&= \int q_\phi(z\mid x)\,\log \Big[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\cdot\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\Big]\, dz \\
&= \underbrace{\mathbb{E}_{q_\phi(z|x)}\Big[\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\Big]}_{\text{ELBO}} + \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)
\end{aligned}
$$

第二行用了贝叶斯公式 $p_\theta(x) = p_\theta(x,z)/p_\theta(z\mid x)$；第三行在分式里同乘同除 $q_\phi(z\mid x)$。

- **第一项**称为证据下界（Evidence Lower BOund，ELBO），记作 $\mathcal{L}(\theta,\phi;x)$。它只包含 $p_\theta(x,z) = p_\theta(x\mid z)p(z)$ 和 $q_\phi(z\mid x)$，都能算。
- **第二项**是近似后验与真实后验之间的 KL 散度，非负，但算不出来（因为含有 $p_\theta(z\mid x)$）。

由于 KL 散度非负，

$$
\log p_\theta(x) \;\ge\; \mathcal{L}(\theta,\phi;x) = \mathbb{E}_{q_\phi(z|x)}\Big[\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\Big]
$$

**推导二：Jensen 不等式。** 把 $q_\phi$ 当作重要性采样的提议分布：

$$
\log p_\theta(x) = \log \int p_\theta(x,z)\,dz = \log \mathbb{E}_{q_\phi(z|x)}\Big[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\Big] \;\ge\; \mathbb{E}_{q_\phi(z|x)}\Big[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\Big]
$$

不等号来自 Jensen 不等式：$\log$ 是**凹函数**，所以 $\log\mathbb{E}[\cdot]\ge\mathbb{E}[\log(\cdot)]$。

两种推导得到同一个下界。推导一的价值在于它精确地告诉我们**下界和真实值之间差了多少**：

$$
\log p_\theta(x) - \mathcal{L}(\theta,\phi;x) = \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)
$$

因此最大化 ELBO 一举两得：

- 对 $\theta$ 最大化，推高数据的对数似然；
- 对 $\phi$ 最大化，由于左边的 $\log p_\theta(x)$ 与 $\phi$ 无关，等价于最小化 $\mathrm{KL}(q_\phi\|p_\theta(z\mid x))$，即让编码器逼近真实后验。

当 $q_\phi$ 恰好等于真实后验时，下界是紧的。

### 2.5 ELBO 分解：重构项与正则项

把 $p_\theta(x,z) = p_\theta(x\mid z)\,p(z)$ 代入 ELBO：

$$
\begin{aligned}
\mathcal{L}(\theta,\phi;x)
&= \mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x\mid z)\big] + \mathbb{E}_{q_\phi(z|x)}\Big[\log \frac{p(z)}{q_\phi(z\mid x)}\Big] \\
&= \underbrace{\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x\mid z)\big]}_{\text{重构项}} - \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{正则项}}
\end{aligned}
$$

VAE 的训练损失就是负的 ELBO：

$$
\mathcal{L}_\text{VAE} = \underbrace{-\,\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x\mid z)\big]}_{\text{重构损失}} + \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{KL 损失}}
$$

两项各有分工：

- **重构损失**衡量解码器生成数据的质量：从编码器给出的分布里采一个 $z$，它要能被解码回原来的 $x$。具体形式取决于 $p_\theta(x\mid z)$ 的选择（见 2.8 节），常见的是均方误差或交叉熵。
- **KL 损失**衡量编码器输出的分布 $q_\phi(z\mid x)$ 与先验 $p(z)$ 之间的差异。通过最小化它，模型被迫让每个样本的后验都接近标准正态分布，从而实现隐空间的**正则化**。

注意这里的 KL 是对**先验** $p(z)$ 的，可以算；2.4 节里那个算不出的 KL 是对**真实后验** $p_\theta(z\mid x)$ 的。两者不要混淆。

### 2.6 编码器：输出一个分布

和 AE 不同，VAE 的编码器不输出一个确定的隐向量，而是输出一个**分布的参数**。通常取对角协方差的高斯：

$$
q_\phi(z\mid x) = \mathcal{N}\big(z;\; \mu_\phi(x),\; \mathrm{diag}\big(\sigma^2_\phi(x)\big)\big)
$$

编码器网络有两个输出头，一个输出均值 $\mu_\phi(x)$，一个输出方差。实践中通常输出 $\log\sigma^2$ 而不是 $\sigma$，以保证方差为正、数值稳定。

这样每个输入不再对应隐空间里的一个点，而是一小团"云"。相近的输入的云会彼此重叠，解码器必须对一整片区域都给出合理的输出，隐空间因此变得**连续**。

在高斯假设下，KL 损失有闭式解。对单个维度：

$$
\begin{aligned}
\mathrm{KL}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\big)
&= \mathbb{E}_{z\sim\mathcal{N}(\mu,\sigma^2)}\Big[\log\mathcal{N}(z;\mu,\sigma^2) - \log\mathcal{N}(z;0,1)\Big] \\
&= \mathbb{E}\Big[-\tfrac12\log\sigma^2 - \tfrac{(z-\mu)^2}{2\sigma^2} + \tfrac{z^2}{2}\Big] \\
&= -\tfrac12\log\sigma^2 - \tfrac12 + \tfrac12\big(\mu^2+\sigma^2\big)
\end{aligned}
$$

最后一步用了 $\mathbb{E}[(z-\mu)^2]=\sigma^2$ 和 $\mathbb{E}[z^2]=\mu^2+\sigma^2$（两个 $\log\sqrt{2\pi}$ 抵消）。对 $d$ 维对角高斯，各维独立相加：

$$
\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,\mathcal{N}(0,I)\big) = \frac12\sum_{j=1}^d \Big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\Big)
$$

这个式子很好理解：$\mu_j^2$ 惩罚均值偏离原点，$\sigma_j^2 - \log\sigma_j^2 - 1$ 在 $\sigma_j=1$ 时取最小值 0，惩罚方差偏离 1。两者合起来把每朵"云"都往标准高斯的位置和大小上拉。

### 2.7 重参数化技巧

重构项 $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x\mid z)]$ 需要从 $q_\phi$ 中采样 $z$，然后对 $\phi$ 求梯度。问题是"采样"这个操作本身不可导：梯度无法穿过一个随机数生成器传回 $\mu_\phi$ 和 $\sigma_\phi$。

**一种办法是得分函数估计**（score function estimator，也叫 REINFORCE）：

$$
\nabla_\phi\,\mathbb{E}_{q_\phi(z)}\big[f(z)\big] = \mathbb{E}_{q_\phi(z)}\big[f(z)\,\nabla_\phi\log q_\phi(z)\big]
$$

它对任何分布都适用，但方差很大，训练很不稳定。（强化学习里的策略梯度用的就是它，所以 RL 才需要基线、优势函数等一系列降方差的手段。）

**VAE 用的是重参数化技巧**：把随机性挪到一个与参数无关的变量上，

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

$\odot$ 是逐元素乘法。这样采样过程被分解为**确定性部分**（$\mu$ 和 $\sigma$，依赖参数）和**随机部分**（$\epsilon$，不依赖参数）。期望可以改写为对 $\epsilon$ 的期望：

$$
\nabla_\phi\,\mathbb{E}_{q_\phi(z|x)}\big[f(z)\big] = \mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\Big[\nabla_\phi f\big(\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon\big)\Big]
$$

梯度可以沿着 $z\to\mu,\sigma\to\phi$ 正常反向传播，而且方差比得分函数估计小得多。实践中每个样本只采一个 $\epsilon$ 就足够了。

```
            确定性路径（可导）                 随机路径（不需要求导）
x ──▶ [编码器] ──▶ μ, log σ² ──┐
                               ├──▶ z = μ + σ ⊙ ε ──▶ [解码器] ──▶ x̂
               ε ~ N(0, I) ────┘
```

这个技巧会在扩散模型里再次出现：DDPM 的前向加噪 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ 就是同一个写法。

### 2.8 解码器：似然的选择决定重构损失

解码器接收采样得到的 $z$，输出 $p_\theta(x\mid z)$ 的参数。重构损失的具体形式由这个分布的选择决定：

**高斯似然。** 取 $p_\theta(x\mid z) = \mathcal{N}\big(x;\,\mu_\theta(z),\,\sigma_x^2 I\big)$，$\sigma_x$ 固定，则

$$
-\log p_\theta(x\mid z) = \frac{1}{2\sigma_x^2}\big\|x - \mu_\theta(z)\big\|^2 + \text{常数}
$$

即**加权的均方误差**。这里有一个容易忽略的细节：$\sigma_x$ 的取值决定了重构项和 KL 项的相对权重。$\sigma_x$ 越小，重构项权重越大。很多实现直接用 MSE 而不写 $\frac{1}{2\sigma_x^2}$，相当于隐式地选了某个 $\sigma_x$，这和下面 $\beta$-VAE 调 KL 权重是同一件事。

**伯努利似然。** 对于像素值在 $[0,1]$ 的图像，取每个像素独立的伯努利分布，解码器输出概率 $\hat x_i$，则

$$
-\log p_\theta(x\mid z) = -\sum_i\Big[x_i\log\hat x_i + (1-x_i)\log(1-\hat x_i)\Big]
$$

即**二元交叉熵**。

### 2.9 完整的训练与生成流程

**训练**（每个 batch）：

1. 编码器读入 $x$，输出 $\mu_\phi(x)$ 和 $\log\sigma^2_\phi(x)$；
2. 采样 $\epsilon\sim\mathcal{N}(0,I)$，计算 $z = \mu + \sigma\odot\epsilon$；
3. 解码器读入 $z$，输出重建 $\hat x$；
4. 计算重构损失与 KL 损失之和，反向传播，同时更新 $\phi$ 和 $\theta$。

```python
def vae_loss(x, encoder, decoder):
    mu, logvar = encoder(x)
    eps = torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * eps            # 重参数化
    x_hat = decoder(z)
    recon = F.mse_loss(x_hat, x, reduction="sum")      # 或 BCE
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return (recon + kl) / x.size(0)
```

**生成**：扔掉编码器，直接从先验采样 $z\sim\mathcal{N}(0,I)$，送进解码器即可。这正是 AE 做不到的事：因为 KL 项把所有样本的后验都推向了标准高斯，所以先验中的任何一个点附近都"有人负责"。

### 2.10 两项损失的拉扯：VAE 的隐空间长什么样

两项损失的目标是相互矛盾的：

- 只有重构损失时，VAE 会退化成 AE：编码器把方差压到接近 0，每个样本占据一个孤立的点，隐空间重新出现空洞。
- 只有 KL 损失时，所有样本的后验都变成同一个标准高斯，隐空间被完美地"铺满"了，但 $z$ 不携带任何关于 $x$ 的信息，解码器只能输出一个平均图像。

训练过程在两者之间找平衡。在 MNIST 上训练一个 2 维隐空间的 VAE，把编码画出来，会看到和 AE 很不一样的图景：

- 整体分布接近一个以原点为中心的单位圆形高斯；
- 相同数字仍然聚在一起，但簇与簇之间**没有明显的空隙**，相似的数字（比如 4 和 9）的簇在边界上有重叠；
- 在隐空间中沿直线移动，解码结果会从一个数字平滑地过渡到另一个数字。

这正是我们想要的：一个**连续**且**完整**的隐空间。

### 2.11 VAE 的问题

**生成结果偏模糊。** 这是 VAE 最常被诟病的一点。一种解释正是上面的拉扯：KL 项让相近样本的后验重叠，同一个 $z$ 附近混入了多个不同样本的信息。在高斯似然（MSE）下，解码器对这些样本的最优输出是它们的**平均**，而多张图像的平均就是一张模糊的图。另一种解释是对角高斯后验的表达能力有限，无法刻画真实后验的复杂形状，ELBO 与真实似然之间的差距较大。

**后验坍缩（posterior collapse）。** 当解码器足够强大时（例如用自回归的 PixelCNN 或 LSTM 作解码器），它不依赖 $z$ 也能把 $x$ 建模得很好。此时最省事的解法是让 $q_\phi(z\mid x) = p(z)$，KL 项直接降到 0，$z$ 被完全忽略。常见的缓解办法：

- **KL 退火**（KL annealing）：训练初期把 KL 项的权重从 0 逐渐增加到 1，先让编码器学会传递信息；
- **Free bits**：对每个隐变量维度，KL 低于某个阈值 $\lambda$ 时不再惩罚，即用 $\max(\lambda,\mathrm{KL}_j)$ 代替 $\mathrm{KL}_j$；
- 削弱解码器，迫使它依赖 $z$。

**$\beta$-VAE。** 给 KL 项加一个权重：

$$
\mathcal{L}_\beta = -\,\mathbb{E}_{q_\phi(z|x)}\big[\log p_\theta(x\mid z)\big] + \beta\,\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)
$$

$\beta>1$ 时，隐空间更规整，各个维度倾向于编码相互独立的因素（比如一维控制旋转、一维控制大小），称为**解耦表示**，代价是重构更模糊；$\beta<1$ 时重构更清晰，但隐空间更不规整。后面会看到，Stable Diffusion 的自编码器用的就是一个 $\beta$ 极小的 VAE。

**其他扩展**（简要）：条件 VAE（CVAE）给编码器和解码器都输入条件 $y$；分层 VAE（如 NVAE）使用多层隐变量，大幅提升了生成质量；IWAE 用多个样本做重要性加权，得到更紧的下界。分层 VAE 的思路值得记住，下一篇的 DDPM 就是一个特殊的分层 VAE。

## 3. VQ-VAE：向量量化变分自编码器

VQ-VAE（Vector Quantized VAE）由 DeepMind 在 2017 年提出（van den Oord et al., 2017），是一种结合离散表示学习和自编码器结构的生成模型，用于高质量图像、语音和视频的生成。它的核心创新是用一个**离散码本（codebook）**来表示隐变量。

### 3.1 为什么要离散化

和 VAE 相比，VQ-VAE 有两个关键区别：

- VAE 用连续的高斯分布建模隐变量；VQ-VAE 通过**向量量化**把隐变量映射到一个有限的离散码本上，类似 K-means 聚类中把每个点分配给最近的中心。
- VAE 的先验是固定的标准高斯；VQ-VAE 的先验是**另外学出来的**，可以很复杂。

离散化的动机有几层：

1. **很多数据本身就是离散的**：语言由词组成，语音由音素组成，图像也可以用"物体、部件、纹理"这样的离散概念来描述。
2. **绕开后验坍缩**：VQ-VAE 的 KL 项是常数（见 3.6 节），不存在"KL 降到 0、编码器被忽略"的问题，因此可以和强大的自回归解码器或先验配合使用。
3. **离散 token 可以直接用序列模型建模**：图像被表示成一串 token 之后，可以用 PixelCNN、Transformer 这类在离散序列上非常成熟的模型去学它的分布。这一点后来被证明影响深远。

### 3.2 结构

VQ-VAE 包含**编码器、向量量化模块、解码器**三部分。

**(1) 编码器。** 输入 $x$ 经过编码器 $E$ 得到一个连续的隐表示：

$$
z_e(x) = E(x)
$$

对图像来说，$z_e(x)$ 是一张 $h\times w\times d$ 的特征图，例如把 $256\times256$ 的图像下采样 8 倍，得到 $32\times32$ 个位置，每个位置一个 $d$ 维向量。此时 $z_e(x)$ 仍然是连续的。

**(2) 向量量化。** 引入一个离散码本

$$
\mathcal{E} = \{e_1, e_2, \dots, e_K\}, \qquad e_j\in\mathbb{R}^d
$$

$K$ 是码本大小（比如 512 或 8192），每个码字 $e_j$ 是一个可学习的向量。对特征图上的每个位置，找到欧氏距离最近的码字，用它替换原来的向量：

$$
z_q(x) = e_k, \qquad k = \arg\min_j \big\| z_e(x) - e_j \big\|_2
$$

这一步和 K-means 中把样本分配给最近的聚类中心完全一样。量化之后，一张图像就被表示成一张 $h\times w$ 的**整数索引图**，每个位置是 $\{1,\dots,K\}$ 中的一个 token。

**(3) 解码器。** 量化后的 $z_q(x)$ 送入解码器，重建输入数据：

$$
\hat x = D\big(z_q(x)\big)
$$

整体流程：

```
x ──▶ [编码器 E] ──▶ z_e(x) ∈ R^{h×w×d}
                        │  逐位置查找最近码字
                        ▼
                  索引图 k ∈ {1..K}^{h×w}  ◀── 这就是"离散 token"
                        │  查表
                        ▼
                  z_q(x) ∈ R^{h×w×d} ──▶ [解码器 D] ──▶ x̂
```

### 3.3 训练目标与损失函数

VQ-VAE 的训练损失由三部分组成：

$$
\mathcal{L} = \underbrace{\big\|x - D\big(z_q(x)\big)\big\|^2}_{\mathcal{L}_\text{recon}} + \underbrace{\big\|\mathrm{sg}\big[z_e(x)\big] - e_k\big\|^2}_{\mathcal{L}_\text{codebook}} + \beta\,\underbrace{\big\|z_e(x) - \mathrm{sg}\big[e_k\big]\big\|^2}_{\mathcal{L}_\text{commit}}
$$

其中 $\mathrm{sg}[\cdot]$ 表示**停止梯度**（stop-gradient）：前向传播时它是恒等映射，反向传播时它的梯度为零。

**(1) 重构损失 $\mathcal{L}_\text{recon}$。** 确保解码器能从量化后的隐变量恢复原始数据。它更新解码器，并通过下面的直通估计更新编码器；但它**不会更新码本**，因为 $\arg\min$ 那一步把码本和梯度隔开了。

**(2) 码本损失 $\mathcal{L}_\text{codebook}$。** 码本需要单独的训练信号。这一项把选中的码字 $e_k$ 拉向编码器的输出 $z_e(x)$。由于对 $z_e$ 加了停止梯度，它**只更新码本**，不影响编码器。这和 K-means 中"把聚类中心移到分配给它的点的中心"是一回事。

**(3) 承诺损失 $\mathcal{L}_\text{commit}$。** 反过来，把编码器的输出 $z_e(x)$ 拉向它选中的码字。由于对 $e_k$ 加了停止梯度，它**只更新编码器**。为什么需要这一项？隐空间的维度是无界的，如果编码器的输出可以随意增长，而码本的学习速度跟不上，编码器输出就会越漂越远，在不同码字之间来回跳，训练不稳定。承诺损失让编码器"承诺"使用码本，不要跑远。$\beta$ 是控制其强度的超参数，原文发现结果对它不敏感，取 0.25。

为什么要把"码本靠近编码器"和"编码器靠近码本"拆成两项，而不是直接写一个 $\|z_e(x)-e_k\|^2$？因为这样可以给两个方向不同的权重：码本应当积极地追随编码器，编码器只需轻微地约束自己。

### 3.4 直通估计（Straight-Through Estimator）

量化过程包含一个最近邻查找（$\arg\min$），它是不可导的：$z_q$ 关于 $z_e$ 的梯度几乎处处为零。重构损失的梯度到了 $z_q$ 就传不下去了，编码器得不到训练。

VQ-VAE 用**直通估计**来近似：

- **前向传播**时，编码器输出 $z_e(x)$ 被最近的码字 $e_k$ 替换，解码器看到的是 $z_q(x)$；
- **反向传播**时，假装量化这一步是恒等映射，把 $z_q$ 处的梯度原封不动地复制给 $z_e$：

$$
\frac{\partial \mathcal{L}}{\partial z_e(x)} \approx \frac{\partial \mathcal{L}}{\partial z_q(x)}
$$

代码里通常一行实现：

```python
z_q = z_e + (z_q - z_e).detach()
```

前向时括号内外的 $z_e$ 抵消，结果就是 $z_q$；反向时 `.detach()` 的部分没有梯度，梯度全部流向 $z_e$。

为什么这种近似是合理的？因为承诺损失保证了 $z_e$ 和 $z_q$ 离得很近，在 $z_q$ 处计算的梯度对 $z_e$ 来说也是一个不错的方向。直观上，解码器的梯度告诉编码器"往哪个方向改能让重建更好"，下一次前向时，编码器的输出移动后可能就被分配到另一个更合适的码字上。

把三项损失和直通估计合起来，各个参数的梯度来源是：

| 参数 | 从哪些损失获得梯度 |
|---|---|
| 解码器 | 重构损失 |
| 编码器 | 重构损失（经直通估计）+ 承诺损失 |
| 码本 | 码本损失 |

### 3.5 用 EMA 更新码本

原文还提出了码本损失的一个替代方案：用**指数移动平均**（EMA）直接更新码本，不再通过梯度下降。对第 $i$ 个码字，记当前 batch 中分配给它的编码器输出为 $\{z_{i,1},\dots,z_{i,n_i}\}$，则

$$
N_i \leftarrow \gamma N_i + (1-\gamma)\,n_i, \qquad
m_i \leftarrow \gamma m_i + (1-\gamma)\sum_{j=1}^{n_i} z_{i,j}, \qquad
e_i \leftarrow \frac{m_i}{N_i}
$$

衰减系数 $\gamma$ 通常取 0.99。这就是在线版本的 K-means：码字等于分配给它的所有向量的（滑动）平均。EMA 更新收敛更快、更稳定，是后来多数实现的默认选择。

### 3.6 先验：训练时均匀，训练后另学

VQ-VAE 的名字里有"VAE"，那它的 ELBO 在哪里？

把编码器看成一个确定性的 one-hot 后验：$q(z=k\mid x) = 1$ 当 $k$ 是最近码字，否则为 0。训练时假设先验是 $K$ 个码字上的**均匀分布** $p(z=k)=1/K$，则

$$
\mathrm{KL}\big(q(z\mid x)\,\|\,p(z)\big) = \sum_k q(z=k\mid x)\log\frac{q(z=k\mid x)}{1/K} = 1\cdot\log\frac{1}{1/K} = \log K
$$

是一个常数，对训练没有影响，可以从损失中省略。所以 VQ-VAE 的训练目标只剩下重构项（加上为量化服务的两个辅助项）。

但均匀先验显然不是真实的 token 分布：不同 token 的出现频率不同，相邻位置的 token 之间也有很强的依赖。所以生成分两个阶段：

1. **第一阶段**：训练 VQ-VAE（编码器、码本、解码器），得到把图像变成 token 图、再变回图像的能力。
2. **第二阶段**：冻结 VQ-VAE，把训练集的所有图像都编码成 token 图，然后**单独训练一个自回归模型**去拟合 token 的联合分布 $p(k_1,k_2,\dots,k_{hw})$。原文用的是 PixelCNN，后来普遍换成 Transformer。

生成时：用自回归先验逐个采样出一张 token 图，查码本得到 $z_q$，再用解码器还原成图像。

这个两阶段范式的好处在于分工：VQ-VAE 负责把高维、冗余的像素压缩成短得多的 token 序列，并处理局部细节；先验模型只需要在这个紧凑的离散空间里建模全局结构，计算量大幅降低。

### 3.7 码本坍缩

VQ-VAE 训练中一个常见的问题是**码本利用率低**：$K$ 个码字里只有一小部分被用到，其余的码字从未被选中过，也就从未得到更新，成为"死码"（dead codes）。这和 K-means 初始化不好时有的中心永远分配不到点是一回事。

常见的解决办法：

- 用 K-means 在第一批编码器输出上初始化码本；
- 定期把长期未使用的码字重新初始化为当前 batch 里的随机编码器输出；
- 在低维空间里做查找：先把 $z_e$ 投影到很低的维度（如 8 维或 32 维）再量化，并对向量做 L2 归一化（ViT-VQGAN），用余弦相似度代替欧氏距离；
- 干脆不学码本：FSQ（Finite Scalar Quantization）把每一维直接四舍五入到几个固定取值，码本是这些取值的笛卡尔积；LFQ（Lookup-Free Quantization，MAGVIT-v2）把每一维量化为 $\pm1$。这两种方法都不需要码本损失和承诺损失，码本利用率接近 100%。

### 3.8 VQ-VAE-2

DeepMind 在 2019 年提出 VQ-VAE-2（Razavi et al., 2019），用于生成高分辨率图像（如 ImageNet $256\times256$、FFHQ $1024\times1024$）。

- **层级结构**：在 VQ-VAE 基础上使用多层级的隐变量，像金字塔一样。以 $256\times256$ 图像为例，顶层是一张 $32\times32$ 的 token 图，捕捉全局信息（如物体形状、场景布局）；底层是一张 $64\times64$ 的 token 图，以顶层为条件，捕捉细节信息（如纹理）。
- **层级先验**：顶层先验用带自注意力的 PixelCNN 建模全局依赖，底层先验以顶层 token 为条件。
- **拒绝采样**：用一个预训练分类器对生成结果打分，丢弃得分低的样本，以多样性换质量。

VQ-VAE-2 生成的样本在质量上接近当时的 BigGAN，同时多样性更好，没有 GAN 常见的模式崩溃。

### 3.9 VQGAN 与后续发展

VQ-VAE 的重建仍然偏模糊，因为重构损失是 MSE。VQGAN（Esser et al., 2021）在此基础上做了三处改动：

- 把 MSE 换成 L1 损失加**感知损失**（LPIPS：在预训练网络的特征空间里比较）；
- 加入一个 **PatchGAN 判别器**，用对抗损失让重建图像具有真实的高频纹理（这就是第 1 篇最后提到的"对抗损失作为组件"）；
- 自适应地平衡重构损失和对抗损失的权重：

$$
\lambda = \frac{\big\|\nabla_{G_L}\mathcal{L}_\text{rec}\big\|}{\big\|\nabla_{G_L}\mathcal{L}_\text{GAN}\big\| + \delta}
$$

其中 $\nabla_{G_L}$ 表示对解码器最后一层参数的梯度，$\delta$ 是一个很小的数。这样两种损失对解码器的"拉力"大致相当。

在此基础上，VQGAN 把第二阶段的先验换成了 Transformer。由于 token 序列很短（如 $16\times16=256$ 个），Transformer 可以在其上建模全局结构，生成百万像素级别的图像。

"**先学一个 tokenizer，再在 token 上建模**"这个范式后来成为了主流：

- **自回归文生图**：DALL·E（第一代）用 dVAE（用 Gumbel-Softmax 松弛代替直通估计的离散 VAE）得到图像 token，再和文本 token 拼接，交给一个 Transformer 自回归建模；Parti 用 ViT-VQGAN 做 tokenizer。
- **掩码生成**：MaskGIT、MUSE 在 VQ token 上做类似 BERT 的掩码预测，并行生成多个 token。
- **Latent Diffusion（Stable Diffusion）**：在自编码器的隐空间里做扩散，而不是在像素空间。论文同时试了两种正则化：一种是 VQ 正则，一种是 KL 正则（即权重极小的 VAE，KL 权重约 $10^{-6}$，只是防止隐空间方差无限增大）。Stable Diffusion 实际用的是后者，下采样 8 倍、4 个通道。这个自编码器同样用了感知损失和 PatchGAN 对抗损失。
- **多模态大模型**：统一理解与生成的多模态模型中，视觉 tokenizer 也基本沿用 VQ 系列的思路，图像 token 与文本 token 共享同一个词表和同一个 Transformer。

## 4. 小结

| 模型 | 隐变量 | 先验 | 损失 | 能否生成 | 主要问题 |
|---|---|---|---|---|---|
| AE | 连续、确定的点 | 无 | 重构误差 | 否 | 隐空间不规则、有空洞 |
| VAE | 连续、高斯分布 | 固定的 $\mathcal{N}(0,I)$ | 重构 + KL（负 ELBO） | 能，一步采样 | 生成模糊、后验坍缩 |
| VQ-VAE | 离散码本索引 | 另训的自回归模型 | 重构 + 码本 + 承诺 | 能，两阶段 | 码本利用率、重建偏糊（VQGAN 改进） |

几条贯穿后续文章的线索：

- **ELBO**：$\log p(x)\ge\mathbb{E}_q[\log p(x,z)/q(z\mid x)]$，下界与真实值的差距是近似后验与真实后验的 KL。下一篇的 DDPM 会对一条长长的隐变量链 $x_1,\dots,x_T$ 写出完全相同的下界。
- **重参数化**：$z = \mu + \sigma\epsilon$。DDPM 的前向加噪、训练目标都建立在这个写法上。
- **在隐空间里做生成**：VQ-VAE 与 VQGAN 开创的"tokenizer + 生成模型"两阶段范式，被 Latent Diffusion 继承，成为今天文生图的标准架构。

下一篇的 DDPM 可以看成一个**层数很多、每层隐变量都和数据同维、编码器固定不学**的分层 VAE。我们会看到，固定编码器这个看似"限制"的选择，恰恰让 ELBO 化简成了一个极其简单的噪声回归问题。

## 参考文献

- Baldi & Hornik. *Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima.* Neural Networks 1989.
- Kingma & Welling. *Auto-Encoding Variational Bayes.* ICLR 2014.
- Rezende, Mohamed & Wierstra. *Stochastic Backpropagation and Approximate Inference in Deep Generative Models.* ICML 2014.
- Bowman et al. *Generating Sentences from a Continuous Space.* CoNLL 2016.（KL 退火与后验坍缩）
- Burda, Grosse & Salakhutdinov. *Importance Weighted Autoencoders.* ICLR 2016.
- Kingma et al. *Improved Variational Inference with Inverse Autoregressive Flow.* NeurIPS 2016.（free bits）
- Higgins et al. *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR 2017.
- van den Oord, Vinyals & Kavukcuoglu. *Neural Discrete Representation Learning.* NeurIPS 2017.
- Razavi, van den Oord & Vinyals. *Generating Diverse High-Fidelity Images with VQ-VAE-2.* NeurIPS 2019.
- Vahdat & Kautz. *NVAE: A Deep Hierarchical Variational Autoencoder.* NeurIPS 2020.
- Esser, Rombach & Ommer. *Taming Transformers for High-Resolution Image Synthesis.* CVPR 2021.（VQGAN）
- Ramesh et al. *Zero-Shot Text-to-Image Generation.* ICML 2021.（DALL·E）
- Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
- Yu et al. *Vector-quantized Image Modeling with Improved VQGAN.* ICLR 2022.（ViT-VQGAN）
- Chang et al. *MaskGIT: Masked Generative Image Transformer.* CVPR 2022.
- Mentzer et al. *Finite Scalar Quantization: VQ-VAE Made Simple.* ICLR 2024.
- Yu et al. *Language Model Beats Diffusion: Tokenizer is Key to Visual Generation.* ICLR 2024.（MAGVIT-v2，LFQ）
