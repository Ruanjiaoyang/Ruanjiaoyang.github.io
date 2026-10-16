---
title: "On-Policy Distillation（六）：教哪些 token，用什么目标"
title_en: "On-Policy Distillation (6): Which Tokens to Teach, and With What Objective"
series: "On-Policy Distillation"
order: 6
date: 2026-10-16
tags: [On-Policy Distillation, Variance Reduction, Token Selection, Forward KL, Reverse KL, Entropy]
summary_en: >
  The sixth post of the OPD series covers the 2026 papers that change the OPD loss itself. It starts from
  a 2x2 map (prefix source by KL direction) and then groups the work by what it modifies: the estimator
  (vOPD subtracts the per-token reverse KL as a closed-form, critic-free baseline), the target (TOP-D mixes
  teacher and student in probability space to bound the reward, Veto builds a logit-space intermediate
  target, EOPD adds forward KL on high-entropy teacher positions to keep diversity), and which tokens or
  trajectories count (TA-OPD keeps only tokens that are both informative and reachable, FiRe-OPD filters
  then reweights, while SCOPE and Uni-OPD use a verifier and are really OPD-plus-outcome-reward hybrids).
  NumPy code checks the vOPD variance reduction and the TOP-D reward bound. Each paper is labelled with
  how strictly on-policy it is and how strong its evidence is.
---

## 0. 这一篇讲什么

第 4 篇讲了 OPD 的几类失败。这一篇讲直接修改 OPD 损失的工作。它们改动的地方可以分成三类：

| 改什么 | 代表工作 | 针对的问题 |
|---|---|---|
| **估计量** | vOPD | 采样 token 的梯度方差大 |
| **目标分布** | TOP-D、Veto、EOPD | 奖励无界、梯度不稳定；reverse KL 压缩多样性 |
| **教哪些 token / 轨迹** | TA-OPD、FiRe-OPD；SCOPE、Uni-OPD（需要验证器） | 不是所有不一致都值得教 |

开始之前，先用一篇论文给出的框架把设计空间摆清楚。

## 1. 一张 2×2 的地图

*Decoupling KL and Trajectories*（Zhao et al., 2026）把第 1 篇 4.4 节的链式分解写成了一个统一框架：

$$
\mathrm{KL}(p_T\|q_\theta)=\sum_t\mathbb{E}_{s_t\sim d_T^t}\,\mathrm{KL}(p_T\|q_\theta)(s_t), \qquad
\mathrm{KL}(q_\theta\|p_T)=\sum_t\mathbb{E}_{s_t\sim d_S^t}\,\mathrm{KL}(q_\theta\|p_T)(s_t)
$$

$d_T^t$、$d_S^t$ 分别是 teacher 和 student 在第 $t$ 步的前缀分布。序列级 KL 把"前缀来源"和"KL 方向"绑在一起，但在逐 token 的实现里两者可以独立选择，于是得到四种组合：

| 前缀来源 | 逐 token forward KL | 逐 token reverse KL |
|---|---|---|
| teacher 的前缀 | off-policy SFT / 词级蒸馏 | 类似离线 RL |
| student 的前缀 | DAgger 式（GKD 的 forward KL 版本） | 标准 OPD |

这就是第 1 篇 4.5 节说的"on-policy 和 reverse KL 是两个独立的选择"。论文在 Qwen3-0.6B-Base 上的小规模实验给出了几个与本系列其他结论一致的观察：KL 方向带来准确率和熵之间的权衡（reverse KL 的平均准确率更高，pass@k 相近）；reverse KL 的回答长度会膨胀（4096 token 上限下平均 8189 对 forward KL 的 2031，说明大量回答被截断）；student 前缀确实有帮助，但每步的计算量约是使用缓存 teacher logits 的 teacher 前缀方法的 3.6 倍。实验只有一个 0.6B 的 student，AIME 分数都在个位数，数字只作参考，框架本身是这篇论文的主要价值。

本篇后面的方法都可以放进这张图：多数在右下角（student 前缀 + reverse KL）上修改，EOPD 和 Veto 往左下角挪了一部分。

## 2. 改估计量：vOPD

### 2.1 一个"免费"的基线

vOPD（Oh et al., *KL for a KL: On-Policy Distillation with Control Variate Baseline*, 2026）的出发点是第 1 篇第 5 节的观察：采样 token 的估计量无偏但方差很大，全词表计算方差小但贵（Qwen3 的词表约 15 万），top-$k$ 截断则有偏。

把 OPD 当成策略梯度，每个 token 的奖励是 $r_t=\log\pi_T(y_t\mid c_t)-\log\pi_\theta(y_t\mid c_t)$。RL 里降方差的标准做法是减去一个基线，最好的基线是价值函数。在 OPD 里，**价值函数有闭式解**：

$$
V^{\pi_\theta}(c_t)=\mathbb{E}_{y_t\sim\pi_\theta}[r_t]=-D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid c_t)\,\|\,\pi_T(\cdot\mid c_t)\big)
$$

也就是当前位置的 reverse KL 本身，所以论文标题叫"KL for a KL"。vOPD 的梯度是

$$
\nabla_\theta J_{\mathrm{vOPD}}=\mathbb{E}\Big[\sum_t\big(r_t+D_{\mathrm{KL}}(\pi_\theta(\cdot\mid c_t)\,\|\,\pi_T(\cdot\mid c_t))\big)\nabla_\theta\log\pi_\theta(y_t\mid c_t)\Big]
$$

基线不依赖采样到的 $y_t$，所以估计仍然无偏；它不需要训练 Critic，只用前向计算就能得到（可以用 student top-$k$ 上的 KL 近似）。对比 RL 系列：PPO 需要训练一个 Critic 来估计价值函数，GRPO 用组内平均来近似它；OPD 因为有 teacher 的分布，价值函数可以直接算出来。

下面在一个词表大小为 1000 的位置上比较两种估计：

```python
import numpy as np
rng = np.random.default_rng(0)
V, N = 1000, 2000
def softmax(z): e = np.exp(z - z.max()); return e/e.sum()
zS = rng.normal(size=V)*2.0; pS = softmax(zS)
pT = softmax(zS + rng.normal(size=V)*1.5)          # teacher: correlated with the student, but different
kl = np.sum(pS*np.log(pS/pT))
exact = pS*(np.log(pS/pT) - kl)                    # d KL(S||T) / d logits
a = rng.choice(V, size=N, p=pS)                    # N sampled tokens at one position
r = np.log(pT[a]) - np.log(pS[a])                  # sampled-token OPD reward
score = np.eye(V)[a] - pS                          # d log pS(a) / d logits
for name, adv in [("plain OPD  (adv = r)", r), ("vOPD (adv = r + KL)", r + kl)]:
    g = -adv[:, None]*score                        # per-sample descent direction on KL
    print(f"{name}: mean-gradient error {np.linalg.norm(g.mean(0) - exact):.4f}, "
          f"per-sample variance {g.var(0).sum():.4f}")
print(f"KL = {kl:.3f};  reward r: mean {r.mean():.3f}, std {r.std():.3f}, min {r.min():.2f}")
# TOP-D: probability-space mixture bounds the reward from below by log(1 - alpha)
for alpha in [0.1, 0.2]:
    rt = np.log(alpha*np.exp(r) + 1 - alpha)
    print(f"TOP-D alpha={alpha}: reward min {rt.min():.3f} (bound {np.log(1-alpha):.3f}), std {rt.std():.3f}")
```

输出：

```
plain OPD  (adv = r): mean-gradient error 0.0469, per-sample variance 4.4604
vOPD (adv = r + KL): mean-gradient error 0.0306, per-sample variance 2.5235
KL = 1.431;  reward r: mean -1.408, std 1.609, min -6.37
TOP-D alpha=0.1: reward min -0.105 (bound -0.105), std 0.186
TOP-D alpha=0.2: reward min -0.223 (bound -0.223), std 0.309
```

减去 KL 基线后单样本方差从 4.46 降到 2.52，2000 个样本的平均梯度离真值也更近。原因很直观：奖励 $r$ 的均值是 $-\mathrm{KL} = -1.43$，不减基线时每个样本的梯度都被这个共同的负偏移放大了。最后两行留到 3.1 节。

### 2.2 实验

vOPD 在四组 teacher–student 上比较（Qwen3-1.7B → 1.7B-Base、Qwen3-4B → 4B-Base、Qwen3-1.7B → 0.6B-Base、Olmo-3-7B-Think → Olmo-3-7B-Base），用 LoRA 训练，回答长度上限 2048。数学平均分：

| 组合 | OPD（采样） | OPD top-$k$ | OPD 全词表 | vOPD top-$k$ | vOPD 全词表 |
|---|---|---|---|---|---|
| Qwen3-1.7B | 29.8 | 30.2 | 33.0 | 33.0 | 33.1 |
| Qwen3-4B | 42.9 | 42.8 | 45.1 | 45.3 | 45.4 |
| Olmo-3-7B | 29.9 | 29.5 | 31.8 | 33.1 | 32.9 |

两个结论：用了基线的采样估计（vOPD top-$k$ 基线）能达到全词表计算的水平；单纯截断到 top-$k$ 几乎没有帮助（29.8 到 30.2），这和第 4 篇"截断有偏"的结论一致。限制是模型不超过 7B、回答很短、用 LoRA。这是本篇证据最干净的一篇：推导一句话说得清，四组对照设置一致。

## 3. 改目标分布

### 3.1 TOP-D：在概率空间里混合，让奖励有界

TOP-D（Xie et al., *Trust Region Policy Distillation*, 2026）认为 OPD 不稳定的根源是奖励 $\log(\pi^*/\pi_\theta)$ 无界：student 在 teacher 认为极不可能的 token 上采样时，奖励可以是很大的负数。它用一个"近端 teacher"代替 teacher：

$$
\tilde\pi^*=\alpha\pi^*+(1-\alpha)\pi_\theta, \qquad \tilde r=\log\frac{\tilde\pi^*}{\pi_\theta}=\log(\alpha\rho+1-\alpha), \quad \rho=\frac{\pi^*}{\pi_\theta}
$$

这个奖励有下界 $\log(1-\alpha)$。上面代码的最后两行验证了这一点：原始奖励最小到 −6.37，$\alpha=0.1$ 时最小只有 −0.105，恰好等于下界。论文指出，如果在对数空间里混合，只是把奖励整体缩放成 $\alpha r$，并不能截掉尾部，所以必须在概率空间里混合。这和第 2 篇 DistiLLM 的 skew KL 是同一个思路（把对方分布混进来让散度有界），区别在于 TOP-D 把它用在 RL 形式的奖励上，并配合 PPO 式的裁剪和信赖域理论分析。

TOP-D 在 Qwen3-30B-A3B → Qwen3-8B-Base 上报告 AIME24 从 OPD 的 24.58 提高到 50.42。这个数字需要谨慎看待：

- **不是严格的 on-policy。** 一批 rollout 被分成 16 个小批次重复使用，论文自己说它"打破了严格的 on-policy 数据复用限制"。所以提升混合了"奖励有界"和"样本复用"两个因素。
- **OPD 基线偏弱。** 同一张表里 OPD 在 AIME24 上的 24.58 低于 GRPO 的 30.10 和 DAPO 的 32.92，这和第 3 篇工业报告里 OPD 明显优于 RL 的结果相反，说明这里的 OPD 基线可能没调好。
- 训练只有 200 到 400 步，单次运行。

所以 TOP-D 的思路值得记住，但它的数字不能作为"OPD 可以提高 25 个点"的证据。

### 3.2 Veto：在 logit 空间里构造中间目标

Veto（Jang et al., 2026）在 logit 空间里构造一个介于 teacher 和 student 之间的目标：

$$
Q(y\mid x)\propto\exp(z_T+\beta z_S)=\tfrac{1}{Z}P_T\,P_S^{\beta}
$$

然后最小化 $D_{\mathrm{KL}}(Q\,\|\,P_\theta)$（默认是 forward 方向）。$P_S^\beta$ 这个因子会压低 student 自己都不相信的 token 上的目标概率，论文称之为梯度"否决"，避免 forward KL 在 student 忽略的 token 上产生爆炸的梯度；$\beta$ 在训练中线性衰减到 0，最后回到原始 teacher。

注意它的**默认是 forward KL**，属于 GKD 那一类（2×2 表的左下角）。实验规模很小（0.5B student、每个任务 1000 个训练提示），在 GSM8K 上从 on-policy KD 的 35.1 提高到 39.9。作为一个思路（"不要一步到位地追 teacher，而是追一个逐渐靠近 teacher 的中间目标"），它和 TOP-D、DistiLLM 的 skew KL 属于同一家族；作为证据，规模不够。

### 3.3 EOPD：在高熵位置补上 forward KL

Entropy-Aware OPD（EOPD；Jin et al., *Entropy-Aware On-Policy Distillation of Language Models*, ICML 2026）处理的是 reverse KL 的 mode-seeking（第 1 篇第 4 节）。论文测量到：OPD 之后 student 只有 6.8% 的 token 熵大于等于 1.0，而 teacher 有 18.5%。也就是说，teacher 在很多位置是"有几种都可以"的，student 学成了只选一种。

修补办法是在 teacher 熵高的位置额外加一个 forward KL：

$$
\mathcal{L}_t^{\mathrm{EOPD}}=\mathcal{L}_t^{\mathrm{OPD}}+\alpha\,\mathbb{I}\big[H_t^{\text{teacher}}>\tau\big]\,\mathcal{L}_t^{\mathrm{FKL}}, \qquad
\mathcal{L}_t^{\mathrm{FKL}}\approx\sum_{v\in S_t^k}\tilde\pi_T(v)\log\frac{\tilde\pi_T(v)}{\pi_\theta(v)}
$$

forward KL 在 teacher 的 top-16 上计算（重新归一化），默认 $\tau = 0.8$、$\alpha = 1$。额外开销约为每步 4.5% 的时间。

Qwen3-8B 教 Qwen3-0.6B / 1.7B / 4B-Base，六个数学评测平均：Pass@8 相对 OPD 分别提高 1.37、2.39、5.05；Avg@8 只提高约 1 个点。提升主要体现在 Pass@8，正是多样性指标，与论文的动机一致。它在 2×2 表里是"student 前缀 + 两种 KL 按位置混合"，并且给出了一个可操作的混合规则：**teacher 不确定的地方要覆盖，teacher 确定的地方要聚焦。**

## 4. 教哪些 token

### 4.1 TA-OPD：显著不等于可学

TA-OPD（Wang et al., *Not All Disagreement Is Learnable: Token Teachability in On-Policy Distillation*, 2026）的观点是：按 KL 大小或熵挑 token，挑出的是**显著**的 token，不一定是**可学**的 token。第 4 篇讲过，teacher 在 student 前缀上可能已经不可靠，或者 teacher 想要的 token 在 student 的分布里根本够不着。

它为每个位置定义两个量：

- **局部分歧**：在 student 和 teacher 的 top-$K$ 并集 $U_t$ 上重新归一化后的 KL，$D_t=D_{\mathrm{KL}}(\bar p_T^{U_t}\,\|\,\bar p_\theta^{U_t})$。
- **可达性**：teacher 的概率有多少落在 student 的 top-$K$ 里，$C_t=\sum_{v\in S_t^S(K)}p_T(v\mid c_t)$。

可教性分数是两者归一化后的乘积 $s_t=\tilde D_t\,\tilde C_t$，只在分数最高的一部分 token 上施加 OPD 损失。论文的回归分析显示，"可学"那部分分歧对后续改进的贡献系数约是"不可达"部分的两倍（0.086 对 0.043）。

在只监督 10% token 的预算下，六个评测的平均分：

| 组合 | 全部 token 的 OPD | 按另一种重要性选 10% | TA-OPD 选 10% |
|---|---|---|---|
| Qwen3-4B → 1.7B | 42.37 | 43.05 | 44.89 |
| Qwen3-8B-GRPO → 4B | 53.71 | 56.81 | 56.87 |
| Qwen3-14B → 4B | 54.64 | 53.62 | 54.65 |
| R1-Distill-14B → Qwen2.5-3B（跨家族） | 28.76 | 28.11 | 30.62 |

只教 10% 的 token 就能达到或超过教全部 token 的效果。注意这个"10%"是**监督预算，不是加速**：所有 token 仍然要生成、要让 teacher 打分，只是只有一部分进入损失。14B 那一组的差距几乎为零，评测用了 5 个种子，但训练只有一次。这篇论文和第 4 篇 THUNLP 的"只在重叠 token 上优化就够了"相互印证：**有效信号集中在 student 够得着、teacher 又确实有不同意见的少数位置上。**

### 4.2 FiRe-OPD：先过滤轨迹，再给 token 加权

FiRe-OPD（Li et al., *Filter, Then Reweight*, 2026）在两个粒度上操作：先按 teacher 对整条回答的平均 log-likelihood 去掉每批中最差的 20%（teacher 都觉得离谱的回答，前缀上的 teacher 信号也不可靠）；再按"teacher 越确定、student 越困惑"给 token 加权，$w_t=(1+\alpha c_t^T)(1+\beta c_t^S)$，其中 $c_t^T=1-H(\pi^*)/\max H$，$c_t^S=H(\pi_\theta)/\max H$。

在 Qwen3-30B-A3B → Qwen3-4B 的强到弱设定下，八个数学评测平均从 OPD 的 58.70 提高到 60.83，同表中 ExOPD 是 60.16；单教师设定下只从 61.21 到 61.74。论文的正文和消融表在哪个组件更重要上说法不一致，单次运行。它是"轨迹 + token 两级筛选"的一个例子，证据中等。

### 4.3 用验证器分流：SCOPE 与 Uni-OPD

还有两篇工作也在决定"在哪些样本上用 teacher 信号"，但它们依赖答案验证器，严格说属于 OPD 与 outcome reward 的混合（第 5 篇第 3 节的那条路线）：

- **SCOPE**（Zheng et al., 2026）：验证器把一组回答分成对的和错的。对的回答做最大似然，按 student 困惑度加权（对了但 student 自己没把握的多学）；错的回答做 OPD，按 teacher 困惑度加权（teacher 觉得还算合理的错误回答，teacher 的纠正信号更可信）。这和 KDRL"只在答错时蒸馏"的思路一致。DS-1.5B 学生上数学 Avg@32 从 52.3 到 55.2。但它每个提示采样 8 条回答而 OPD 基线只采 1 条，每步耗时约是 OPD 的 2.8 倍，比较不是计算量对齐的。
- **Uni-OPD**（Hou et al., 2026）：一方面按难度和正确率平衡数据，让 student 多访问"有信息量"的状态；另一方面做"结果引导的边距校准"，检查 teacher 的序列级 OPD 回报能否把同一提示下对的回答排在错的前面，不能时屏蔽或修正这组样本。Qwen3-4B 单教师上数学平均从 44.6 到 48.7、代码从 59.0 到 63.2，并扩展到多教师和多模态。

这两篇和第 4 篇 Demystifying OPD 的"信息量"指标其实在检查同一件事：**teacher 的 token 级偏好与最终对错是否一致**。不一致的地方，用 outcome reward 来纠正或屏蔽 teacher 信号。

## 5. 小结

| 方法 | 2×2 位置 | 改了什么 | 严格 on-policy | 证据 |
|---|---|---|---|---|
| vOPD | student 前缀 + RKL | 减去闭式 KL 基线，无偏降方差 | 是 | 较强：4 组对照，推导干净 |
| TOP-D | student 前缀 + RKL | 概率空间混合，奖励有下界 | 否（小批次复用） | 中等：OPD 基线偏弱 |
| Veto | student 前缀 + FKL（默认） | logit 空间中间目标，$\beta$ 退火 | 是 | 弱：0.5B，1000 个提示 |
| EOPD | student 前缀 + RKL，高熵处加 FKL | 按 teacher 熵混合两个方向 | 是 | 中等偏强：ICML 2026 |
| TA-OPD | student 前缀 + RKL | 只教"分歧大且够得着"的 token | 是 | 中等偏强 |
| FiRe-OPD | student 前缀 + RKL | 过滤轨迹 + token 加权 | 是 | 中等，有内部不一致 |
| SCOPE / Uni-OPD | student 前缀 + RKL + 验证器 | 按对错分流或校准 teacher 信号 | 是，但需验证器 | 中等，比较未对齐计算量 |

三条可以带走的经验：

1. **估计量方面**，采样 token 加上闭式 KL 基线就能接近全词表的效果，截断 top-$k$ 而不加基线几乎没用，不归一化的 top-$k$ 是错的（第 4 篇）。
2. **目标方面**，reverse KL 不是唯一选择：在 teacher 不确定的位置补 forward KL 能保住多样性，把 teacher 和 student 混合能让奖励有界。
3. **token 选择方面**，信号集中在少数既有分歧又够得着的位置；当有验证器时，用最终对错来决定在哪里相信 teacher 是有效的。

## 参考文献

- Zhao et al. *Decoupling KL and Trajectories: A Unified Perspective for SFT, DAgger, Offline RL, and OPD in LLM Distillation.* 2026. arXiv:2605.16826.
- Oh et al. *KL for a KL: On-Policy Distillation with Control Variate Baseline.* 2026. arXiv:2605.07865.
- Xie et al. *Trust Region Policy Distillation.* 2026. arXiv:2607.04751.
- Jang et al. *Veto: Stable On-Policy Distillation through Adaptive Target Reformulation.* 2026. arXiv:2601.07155.
- Jin et al. *Entropy-Aware On-Policy Distillation of Language Models.* ICML 2026. arXiv:2603.07079.
- Wang et al. *Not All Disagreement Is Learnable: Token Teachability in On-Policy Distillation.* 2026. arXiv:2605.26844.
- Li et al. *Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation.* 2026. arXiv:2606.02684.
- Zheng et al. *SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting.* 2026. arXiv:2604.10688.
- Hou et al. *Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe.* 2026. arXiv:2605.03677.
- Ko et al. *DistiLLM: Towards Streamlined Distillation for Large Language Models.* ICML 2024.
