---
title: "On-Policy Distillation（七）：拿不到 logits 或词表不同时怎么办"
title_en: "On-Policy Distillation (7): Black-Box Teachers and Mismatched Tokenizers"
series: "On-Policy Distillation"
order: 7
date: 2026-10-16
tags: [On-Policy Distillation, Black-Box Distillation, GAD, Rubrics, Cross-Tokenizer, Knowledge Distillation]
summary_en: >
  The seventh post of the OPD series covers the two access problems that standard OPD assumes away: the
  teacher may only be reachable through an API that returns text, and the teacher and student may use
  different tokenizers. For black-box teachers, GAD trains a discriminator alongside the student and uses
  it as an on-policy reward, OVD asks the teacher for verbal scores, and ROPD has the teacher write
  contrastive rubrics that a verifier checks; all three turn OPD back into sequence-level RL with a
  teacher-derived reward. For mismatched vocabularies, SimCT and BPM project the teacher's next-token
  distribution onto units both models share, and a recent counterpoint argues that coverage matters less
  than supervision reliability. A NumPy-free toy shows byte-prefix marginalization on one position.
---

## 0. 这一篇讲什么

前面几篇的 OPD 都默认两个条件：

1. **白盒 teacher**：能拿到 teacher 在任意前缀上的 log-prob，至少是采样 token 的 log-prob。
2. **同一个词表**：teacher 和 student 的 token 一一对应，逐 token 的 KL 才有意义。

现实中两个条件经常不满足。最强的 teacher 往往是只能通过 API 访问的闭源模型，API 一般只返回文本（有的返回 top-k log-prob，但不能对任意前缀打分）。开源模型之间的 tokenizer 也各不相同。Joanna 笔记里提到 OVD 和 ROPD 是"把 OPD 扩展到黑盒 teacher"的方向，这一篇把这两类问题分开讲：

| 问题 | 代表工作 | 思路 |
|---|---|---|
| teacher 只返回文本 | GAD、OVD、ROPD | 用 teacher 构造一个序列级奖励，让 student 用 RL 在自己的样本上优化 |
| 词表不同 | SimCT、BPM；以及一篇反方观点 | 把 teacher 的分布投影到两边共享的单位上，继续做逐 token 的 OPD |

两类方法要分开看：跨词表方法**仍然需要 teacher 的 logits**，解决的是对齐问题，不是访问权限问题。

## 1. 黑盒 teacher：从逐 token 信号退回到序列级奖励

没有 log-prob，就没法计算第 1 篇定义的逐 token 散度。三篇代表工作的共同做法是：**student 仍然在自己的样本上训练（保留 on-policy 的好处），但信号变成序列级的奖励，由 teacher 以某种方式提供。** 这实际上又回到了 RL，只是奖励来自 teacher 而不是验证器。

### 1.1 GAD：用一个判别器当奖励模型

GAD（Ye et al., *Black-Box On-Policy Distillation of Large Language Models*, 2025）是这个方向的奠基工作。它把 student 看成生成器 $G$，训练一个判别器 $D$ 区分 teacher 的回答和 student 的回答，目标是 Bradley–Terry 形式的极小极大问题：

$$
\max_G\min_D\ \mathbb{E}_{(x,\,y_T)}\Big[-\log\sigma\big(D(y_T)-D(G(x))\big)\Big]
$$

student 用 GRPO 最大化 $D(G(x))$。关键在于判别器**和 student 一起更新**，论文称之为 on-policy 的奖励模型：它总是在当前 student 的样本上判断"像不像 teacher"。论文的对照实验显示，如果把判别器冻结（相当于一个普通的离线奖励模型），大约 300 步后就会被钻空子，回答长到 1300 token 左右；一起更新的判别器则保持稳定。这和 RL 系列第 1 篇讲的奖励过优化是同一个问题，GAD 的解法是让奖励模型持续跟着策略走。

实验中 teacher 是 GPT-5-Chat，提示来自 LMSYS-Chat 的 20 万条，student 是 Qwen2.5 3B/7B/14B 和 Llama 3B/8B。用 GPT-4o 打分，Qwen2.5-14B 在 LMSYS 上 GAD 52.1、SeqKD（在 teacher 回答上做 SFT）50.6、teacher 51.7。Qwen2.5-3B 在 Dolly 上 SeqKD 甚至低于蒸馏前（44.8 对 45.1），GAD 是 46.7。

需要注意的问题：评测只用一个 GPT-4o 评委；论文按 GPT-4o 在测试集上的分数挑选检查点，这会高估结果；单次运行。GAD 的价值在于它确立了黑盒 OPD 的基本形式：**teacher 的作用从"逐 token 打分"变成了"提供正样本"，判别器把这些正样本转化成 student 样本上的奖励。**

### 1.2 OVD：让 teacher 口头打分

OVD（Xiong et al., *On-policy Verbal Distillation*, 2026）让 teacher 直接给 student 的轨迹（或每一步）打一个 0 到 9 的分数，低于阈值的步骤由 teacher 重新生成，再在混合后的数据上用 GRPO / PPO 优化，同时使用环境奖励。论文强调这样不需要传输词表大小的 logits，显存开销小。

在搜索问答上平均 43.96（对比 ZeroSearch 31.07、Search-R1 29.69）。但这篇论文的问题较多：训练数据混入了 teacher 生成的样本，不是纯 on-policy；还用了环境奖励，很难说提升来自 teacher 的打分；正文和表格的数字对不上（正文写 43.6 对 32.8，表格是 43.96 对 29.69），评分范围在不同地方写成 0–9 和 1–10。可以把它看作设计空间里的一个点（"teacher 给轨迹级口头反馈"），不要依赖它的数字。

### 1.3 ROPD：让 teacher 写评分细则

ROPD（Fang et al., *Rubric-based On-policy Distillation*, 2026）的做法更结构化。对每个提示：

1. teacher 生成几个回答（默认 4 个），student 生成几个回答（默认 8 个）。
2. teacher 作为"细则生成器"，**对比**自己的回答和 student 的回答，写出 4 到 12 条带权重的评分细则 $\{(\rho_k, w_k)\}$，内容是"好回答应该具备而 student 回答缺少的东西"。
3. 一个验证器逐条检查 student 的每个回答是否满足，得到加权通过率 $s_i=\sum_k w_k v_{i,k}/\sum_k w_k$ 作为奖励，用 GRPO 优化。

这里"对比"是关键：细则不是通用的评分标准，而是针对当前 student 的弱点生成的，所以随着 student 变化，细则也在变化。这和 GAD 的"判别器跟着 student 走"是同一个原则。

结果（teacher 是 GPT-5.2-chat-latest，student 是 Qwen3-4B）：AIME25（思考模式）上 ROPD 68.75，teacher 评委打分（T-Judge）65.48，OVD 65.83，GAD 61.28，teacher 自己 67.08。更出人意料的是白盒对比：用 Qwen3-30B-A3B 当 teacher 时，ROPD 的四个数学评测平均 45.87，基于 logits 的普通 OPD 只有 32.82，ExOPD 35.25（teacher 56.46）。论文还称 ROPD 的样本效率最高可达 10 倍。

这些数字需要谨慎看待：

- "10 倍"只是在 AIME24 非思考模式下对比 logits OPD 得到的（1.6k 对 15.4k 个样本），不是普遍结论。
- 用评分细则打败逐 token logits 蒸馏 13 个点，和第 3 篇工业报告以及本系列其他论文的结论差距很大，logits OPD 的基线可能没调好。
- API 调用（细则生成和验证）的成本没有报告。
- ROPD 和下面的 SimCT 以及第 8 篇的 EffOPD 有共同作者，ROPD 中对 OVD、GAD 的比较是同一作者群体做的复现。

ROPD 是目前黑盒设定下结果最好的方法，它的思路（让 teacher 用自然语言指出 student 缺什么，再用验证器把它变成奖励）也和下一篇要讲的自蒸馏、RL with rich feedback 相通。但它的白盒结论在被独立复现之前不宜引用。

### 1.4 黑盒 OPD 的代价

三种方法都用序列级奖励代替了 teacher 的逐 token 分布，于是也失去了 OPD 相对 RL 最大的优势：稠密的 credit assignment。第 1 篇说 OPD 每个 episode 带来 $O(N)$ 比特信息、RL 只有 $O(1)$ 比特，黑盒方法介于两者之间：细则或判别器给出的奖励比"对/错"信息量大，但仍然是整条回答一个数。它们保留的是 on-policy 的另一半好处：在 student 自己的分布上训练，遗忘少、没有 exposure bias。

## 2. 词表不同：把 teacher 的分布投影到共享单位上

### 2.1 问题

两个 tokenizer 不同的模型，同一段文本会被切成不同的 token 序列。在某个位置上，teacher 给出的是"它的下一个 token"的分布，student 需要的是"student 的下一个 token"的分布，两者的支撑集不同，逐 token 的 KL 无法直接计算。早期的离线方法如 ULD（Boizard et al., TMLR 2025）用排序后分布之间的 Wasserstein 距离绕开对齐，但它在 teacher 生成的文本上训练，不是 on-policy 的。

### 2.2 BPM：按字节前缀合并概率

BPM（Wang et al., *Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization*, 2026）的做法很直接：teacher 的每个 token $w$，找 student 词表里字节是 $w$ 的前缀、并且最长的那个 token $u$，把 $w$ 的概率加到 $u$ 上；找不到的归入一个残差项：

$$
\phi(w)=\arg\max_{u:\ \mathrm{bytes}(u)\preceq\mathrm{bytes}(w)}|\mathrm{bytes}(u)|
$$

直觉是：如果 teacher 下一步要写 " then"，那么 student 写同一段文本时，下一步写出的就是 " the"（后面再接 "n"）。所以 teacher 在 " then" 上的概率正是 student 下一步选 " the" 的概率的一部分。下面是一个位置上的玩具例子：

```python
# Byte-prefix marginalization (BPM-style), toy version.
# Teacher and student tokenize differently. At one position the teacher gives a distribution
# over ITS next tokens; map each teacher token w to the longest student token u whose text is a
# prefix of w, and sum the mass. Mass with no matching student token goes to a residual bucket.
teacher_next = {" the": 0.45, " then": 0.20, " there": 0.15, " thus": 0.12, "\n\n": 0.05, " Θ": 0.03}
student_vocab = [" the", " th", " t", " thus", "\n"]
def phi(w):
    cands = [u for u in student_vocab if w.startswith(u)]
    return max(cands, key=len) if cands else None
mapped = {}
for w, p in teacher_next.items():
    u = phi(w) or "<residual>"; mapped[u] = mapped.get(u, 0) + p
    print(f"teacher {w!r:9} p={p:.2f} -> student {u!r}")
print("\nteacher mass projected onto the student vocabulary:")
for u, p in sorted(mapped.items(), key=lambda kv: -kv[1]): print(f"  {u!r:8} {p:.2f}")
```

输出：

```
teacher ' the'    p=0.45 -> student ' the'
teacher ' then'   p=0.20 -> student ' the'
teacher ' there'  p=0.15 -> student ' the'
teacher ' thus'   p=0.12 -> student ' thus'
teacher '\n\n'    p=0.05 -> student '\n'
teacher ' Θ'      p=0.03 -> student '<residual>'

teacher mass projected onto the student vocabulary:
  ' the'   0.80
  ' thus'  0.12
  '\n'     0.05
  '<residual>' 0.03
```

投影后得到一个定义在 student 词表（加一个残差项）上的分布，就可以和 student 的分布计算散度了。这个玩具版本忽略了一个细节：student 的 tokenizer 在切 " then" 时未必先切出 " the"（可能切成 " t" + "hen"），真实方法需要按 student tokenizer 的实际切分来判断；论文报告在 99% 以上的位置投影是精确的，其余位置用一个链式下界处理。损失用广义 JSD（$\beta = 1/2$）。

以 Qwen3.5-2B 为 student、三个不同家族的 teacher（Qwen3-32B、GLM-Z1-9B、MiniMax-M2.7），六个数学和代码评测平均，BPM 比最强基线高 3.7 到 6.6 个点（如 GLM-Z1-9B teacher 时 41.5 对 SeqKD 的 34.9）。

### 2.3 SimCT：在最小对齐单位上计算

SimCT（Sun et al., *Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation*, 2026）从另一个角度出发：只在两个词表的交集上做 OPD 会丢掉大量监督。它把两个 tokenizer 的切分边界合并成一个片段图，取其连通分量作为"最小对齐单位"，在共享词表加这些单位组成的空间上计算 reverse KL，多 token 的单位用长度归一化的 log-likelihood 打分。论文证明把单位切得更粗只会丢失 KL 信息。

Qwen2.5-7B → Phi-4-mini 上平均 55.06，只用共享词表的 OPD 是 52.03，其他跨词表方法中最好的 GOLD 是 53.90。

### 2.4 一个反方观点

2026 年 10 月的一篇论文（Hou et al., *Rethinking Cross-Tokenizer OPD: From Alignment Coverage to Supervision Reliability*，我只读到了摘要）提出了不同看法：严格一一对齐的位置已经覆盖了绝大多数 token；在共享词表的 top-16 上做 reverse KL 就能达到完整共享词表 OPD 的效果；为了提高覆盖率额外加入片段级的匹配损失反而有害。这与 SimCT"找回丢失的监督"的出发点相反，和第 6 篇 TA-OPD、第 4 篇 THUNLP 的结论（有效信号集中在少数可靠的位置）倒是一致的：**重要的不是覆盖多少位置，而是覆盖到的位置上信号是否可靠。** 这个争论目前没有定论。

## 3. 小结

| 方法 | 需要 teacher 的什么 | 信号粒度 | 是否 on-policy | 证据 |
|---|---|---|---|---|
| GAD | 只要文本 | 序列级（判别器分数） | student 样本 + 跟随更新的判别器 | 中等；按测试集挑选检查点 |
| OVD | 文本打分 + teacher 重生成 | 轨迹 / 步骤级 | 否（混入 teacher 样本） | 弱；内部数字不一致 |
| ROPD | 文本（细则生成与验证） | 序列级（细则通过率） | student 样本 | 中等；白盒对比待独立复现 |
| BPM | logits（不同词表） | 逐 token | 是 | 中等 |
| SimCT | logits（不同词表） | 逐 token / 对齐单位 | 是 | 中等 |

两条经验：

1. **黑盒 OPD 本质上是"用 teacher 构造奖励的 RL"。** 它保留了 on-policy 的好处，但失去了逐 token 的稠密信号。有效的黑盒方法都让奖励随 student 一起变化（判别器一起训练、细则针对当前 student 生成），否则会被钻空子。
2. **跨词表是对齐问题，不是访问问题。** 按字节前缀合并概率在绝大多数位置是精确的，现有证据倾向于"保证对齐位置的信号可靠"比"尽量覆盖所有位置"更重要。

## 参考文献

- Ye et al. *Black-Box On-Policy Distillation of Large Language Models.* 2025. arXiv:2511.10643.
- Xiong et al. *OVD: On-policy Verbal Distillation.* 2026. arXiv:2601.21968.
- Fang et al. *Rubric-based On-policy Distillation.* 2026. arXiv:2605.07396.
- Wang et al. *Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization.* 2026. arXiv:2607.22334.
- Sun et al. *Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation.* 2026. arXiv:2605.07711.
- Hou et al. *Rethinking Cross-Tokenizer OPD: From Alignment Coverage to Supervision Reliability.* 2026. arXiv:2610.08448.（仅读摘要）
- Boizard et al. *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs.* TMLR 2025. arXiv:2402.12030.
