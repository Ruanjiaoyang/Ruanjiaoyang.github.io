---
title: "扩散语言模型：系列导读"
title_en: "Diffusion Language Models: Series Overview"
series: "扩散语言模型"
order: 0
date: 2026-10-16
tags: [扩散语言模型, 系列导读]
summary_en: >
  A reading guide to four posts on discrete diffusion for language, organized around one question:
  what has to change for diffusion to move from continuous images to discrete text? The answer runs from
  transition matrices (D3PM) to probability ratios (SEDD), to a weighted masked-language-modeling loss
  (MDLM / MD4), to time-independent conditionals equivalent to any-order autoregression (RADD).
---

## 这个系列在讲什么

上一个系列"从 GAN 到 Diffusion"里的扩散模型都建立在一个前提上：**数据可以和高斯噪声相加**。图像满足这个前提，文本不满足。"猫"加上 0.1 的噪声没有意义，"score" $\nabla_x\log p(x)$ 在离散的词表上也没有定义。

这个系列的四篇文章回答一个问题：**要把扩散模型从连续的图像搬到离散的文本，需要改掉什么、保留什么？** 答案分成四步：

1. 把"加高斯噪声"换成"按转移矩阵跳到别的 token"；
2. 把"对数密度的梯度"换成"概率的比值"；
3. 发现在最好用的掩码噪声下，整个训练目标化简成一个加权的 BERT 式掩码语言模型损失；
4. 证明这个模型学的是与时间无关的条件分布，而且训练目标等价于对所有生成顺序取平均的自回归模型。

## 路线图

```
连续扩散（上一系列）
 │  加高斯噪声、学 score
 ▼
D3PM ─────────── 噪声换成转移矩阵 Q_t；均匀 / 吸收态 / 离散高斯 / 嵌入近邻
 │                 BERT、自回归、MLM 都是离散扩散的特例
 ▼
SEDD ─────────── 连续时间（CTMC）；离散的 score 是比值 p(y)/p(x)
 │                 score entropy 损失，给出困惑度上界
 ▼
MDLM / MD4 ───── 吸收态 + SUBS 参数化：损失 = 按 1/t 加权的 MLM
 │                 与 SEDD 吸收态损失逐项相等；缓存采样、半自回归生成
 ▼
RADD ─────────── 比值 = 时间标量 × 与时间无关的条件分布
                   训练目标 = 任意顺序自回归模型的期望负对数似然
```

## 篇目

| # | 文章 | 一句话 |
|---|---|---|
| 1 | D3PM：把扩散模型搬到离散状态空间 | 用转移矩阵代替高斯噪声，整套 ELBO 推导与 DDPM 一一对应；吸收态（[MASK]）在文本上最好 |
| 2 | SEDD：离散空间里的 score 是概率比值 | concrete score 取代 $\nabla\log p$；score entropy 是正数上的 Bregman 散度，给出似然上界 |
| 3 | MDLM 与 MD4：掩码扩散就是加权的掩码语言模型 | SUBS 参数化让 KL 解析化简；连续时间目标对调度不变，线性调度下是 $1/t$ 加权的 MLM |
| 4 | RADD：吸收态扩散在学干净数据的条件分布 | 网络不需要时间输入，可以缓存采样；损失等价于任意顺序自回归 |

## 怎么读

- **只想知道今天的扩散语言模型在训练什么**：读第 3 篇。它的结论（掩码扩散 = 加权 MLM）是 LLaDA 等大规模扩散语言模型的直接基础。
- **想搞清楚来龙去脉**：按顺序读。第 1 篇的后验公式在第 3 篇复用，第 2 篇 5.4 节的比值分解在第 3、4 篇被反复用到。
- **前置知识**：上一个系列的 DDPM（ELBO 推导）和 Score-based（score 与分数匹配）两篇。第 2 篇用到连续时间马尔可夫链，文中从头介绍，只需要矩阵指数的基本概念。

## 贯穿全系列的三条线索

1. **同一个权重 $1/t$。** D3PM 在互信息调度下、SEDD 在 log-linear 调度下、MDLM 在线性调度下，吸收态的损失都化成"被 mask 位置上的交叉熵 × $1/t$"。RADD 用 $\lambda$ 代换说明这个权重和调度无关，并把它对应到任意顺序自回归中的 $\frac{1}{k}$。
2. **吸收态的特殊结构。** [MASK] 一旦出现就不变、没被 mask 的 token 就是干净 token。这一结构让后验极其简单（第 1、3 篇），让 concrete score 可以分解（第 2、4 篇），让网络可以不看时间、采样可以缓存（第 3、4 篇）。
3. **扩散与自回归的关系。** 第 1 篇说自回归是确定性的、从右往左 mask 的离散扩散；第 4 篇说吸收态扩散是对所有顺序取平均的自回归。两种模型的差别不在"扩散"还是"自回归"，而在固定顺序还是任意顺序、一次揭开一个还是并行揭开多个。

## 记号提醒

四篇论文各自用了不同的记号，文中保留了论文的写法，并在每篇开头做了对照。最需要注意的两处：

- **行约定与列约定**：D3PM 和 RADD 用"行和为 1（或 0）"的矩阵，SEDD 用"列和为 0"的矩阵，两者差一个转置；
- **保留概率**：D3PM 的 $\bar\alpha_t$、SEDD 的 $e^{-\bar\sigma(t)}$、MDLM 的 $\alpha_t$、RADD 的 $1-\lambda$ 是同一个量，都是一个 token 在 $t$ 时刻还没被 mask 的概率。

## 之后

在这些理论基础之上，扩散语言模型在 2025 年进入了大规模阶段：LLaDA、Dream 等工作把掩码扩散扩展到数十亿参数，块扩散（Block Diffusion）把扩散与自回归在块的粒度上结合起来，也有不少工作开始研究扩散语言模型的推理加速和强化学习后训练。这些会在之后的文章里继续展开。
