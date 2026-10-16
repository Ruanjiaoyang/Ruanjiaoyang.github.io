---
title: "从 GAN 到 Diffusion：系列导读"
title_en: "From GANs to Diffusion: Series Overview"
series: "从 GAN 到 Diffusion"
order: 0
date: 2026-10-16
tags: [生成模型, 系列导读]
summary_en: >
  A reading guide to seven posts on deep generative models, organized around one question:
  how do we learn a distribution we can sample from when its normalizing constant is intractable?
---

## 这个系列在讲什么

所有生成模型都在回答同一个问题：**给定一批样本，怎样学到一个能采样的分布？**

难点在于高维空间里的归一化常数。一个灵活的密度函数很容易写，但让它积分为 1 几乎不可能精确计算。这个系列的七篇文章，可以看成七种和归一化常数打交道的方式，以及它们如何一步步汇合到扩散模型。

## 路线图

```
GAN ──────────────── 不写密度，只学采样器
 │
VAE / VQ-VAE ─────── 写出概率模型，优化似然下界（ELBO）
 │
DDPM ─────────────── 很多层的 VAE：编码器固定，ELBO 化为"预测噪声"
 │        ╲
DDIM       Score-based ── 学 log 密度的梯度，归一化常数自动消失
 │        ╱
SDE / ODE ────────── 连续时间统一：DDPM、NCSN、DDIM 都是特例
 │
Guidance ─────────── 按条件生成，文生图的核心机制
```

## 篇目

| # | 文章 | 一句话 |
|---|---|---|
| 1 | GAN：用一场博弈学习数据分布 | 最优判别器下，GAN 在最小化 JS 散度；这也是它难训的原因 |
| 2 | 从 AE 到 VAE 再到 VQ-VAE | KL 正则让隐空间可以采样；离散码本是后来视觉 tokenizer 的起点 |
| 3 | DDPM：一步步加噪，再一步步去噪 | 前向闭式解 + 可解后验，ELBO 化简为噪声回归 |
| 4 | DDIM：不重新训练，把 1000 步压到几十步 | 训练只依赖边际分布，于是可以换一个确定、可跳步的采样过程 |
| 5 | Score-based 模型 | score 与归一化常数无关；预测 score 等于预测 DDPM 的噪声 |
| 6 | SDE 与概率流 ODE | VE/VP 两种 SDE 统一 NCSN 与 DDPM；ODE 带来可逆、似然和加速 |
| 7 | Guidance：让扩散模型听指令 | 条件 score 的贝叶斯拆分；CFG 用一个网络完成引导 |

## 怎么读

- **只想理解扩散模型**：从第 3 篇开始，读 3 → 4 → 7。
- **想搞清楚推导**：按顺序读。第 2 篇的 ELBO 推导会在第 3 篇原样复用，第 5 篇的 score 会在第 6、7 篇反复出现。
- **前置知识**：概率论基础（贝叶斯公式、高斯分布、KL 散度）和一点微积分。第 6 篇用到随机微分方程，但只需要直观理解。

## 贯穿全系列的三条线索

1. **似然下界**：VAE 的 ELBO 和 DDPM 的变分下界是同一件事。
2. **噪声与 score**：$s_\theta(x_t,t) = -\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$，这条式子把 DDPM、NCSN、SDE 和 guidance 全部连在一起。
3. **质量、多样性和速度的取舍**：GAN 快但漏模式，DDPM 质量高但慢，DDIM 和 ODE 求解器用确定性换速度，guidance 用多样性换贴合度。

## 之后

这个系列只覆盖连续数据（图像）上的扩散。下一个系列"扩散语言模型"会讨论 D3PM、SEDD、MDLM、RADD 等工作：当数据变成离散 token，加噪和去噪要怎么重新定义。
