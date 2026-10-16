---
title: "分词器：word、char、subword 与 BPE"
title_en: "Tokenization: Words, Characters, Subwords and BPE"
series: "Transformer 架构"
order: 1
date: 2026-10-16
tags: [Transformer, 分词器, BPE, Tokenizer]
summary_en: >
  How text becomes token ids. The post compares word, character and subword granularity, works through
  BPE training and encoding by hand, then covers byte-level BPE and pre-tokenization (GPT-2), WordPiece
  (BERT), the Unigram language model and SentencePiece, and the trade-offs of vocabulary size. It closes
  with the visible consequences of tokenization: why LLMs struggle to spell or count letters, how numbers
  are split, why some languages cost more tokens, and where glitch tokens come from.
---

## 0. 分词器在哪里

Transformer 处理的是向量序列，而我们手里的是字符串。**分词器**（tokenizer）站在两者之间，做两件事：

1. **切分**：把文本切成一个个片段，称为 **token**；
2. **映射**：把每个 token 映射成词表中的一个整数 id。

```
"unbelievable results"  →  ["un", "believ", "able", " results"]  →  [403, 22327, 540, 2482]
```

之后嵌入层把 id 查表变成向量（第 2 篇）。生成时方向相反：模型输出 id，分词器再把 id 拼回文本。

分词器不是神经网络的一部分，它在训练语言模型**之前**就单独训练好、固定下来。但它对模型的影响非常深：它决定了序列有多长、词表有多大、模型"看见"的最小单位是什么。很多看起来奇怪的大模型行为，根源都在分词上。

## 1. 三种粒度

### 1.1 按词（word）

按空格和标点把文本切成词，每个词一个 token。

- **优点**：词的边界和含义完整保留，序列短。
- **缺点**：
  - **词表外（OOV）问题**：训练时没见过的词（新词、拼写错误、专有名词）只能映射成一个 `[UNK]`，信息完全丢失；
  - **形态变化无法共享**：`walk`、`walks`、`walked`、`walking` 是四个互不相关的 token，模型要分别学习，词缀的规律无法泛化；
  - **长尾稀疏**：语言中的词频服从齐夫定律，大量词只出现几次，它们的嵌入几乎学不好。要覆盖足够多的词，词表会非常大（几十万），嵌入层和输出层的参数随之膨胀；
  - **不适用于没有空格的语言**：中文、日文需要额外的分词工具。

### 1.2 按字符（char）

每个字符一个 token。

- **优点**：词表极小（英文几十个，加上 Unicode 也有限），不存在 OOV。
- **缺点**：
  - **序列变长**：一个英文单词平均五六个字符，序列长度变成按词切分的数倍。注意力的计算量随长度平方增长，代价很高；
  - **语义被打散**：单个字符几乎没有含义，模型要用很多层去把字符重新组装成有意义的单元。

### 1.3 子词（subword）

折中方案：**高频词保持完整，低频词拆成更小的、可复用的片段。**

```
"the"            →  ["the"]
"tokenization"   →  ["token", "ization"]
"Transformerify" →  ["Trans", "former", "ify"]
```

- 常见词是一个 token，序列不会太长；
- 罕见词被拆成见过的片段，不会出现 OOV（配合字节级回退，见 3.1 节）；
- 词缀成为独立 token（`ization`、`ing`、`un`），形态规律可以在不同词之间共享。

子词分词在词表大小和表达能力之间取得了平衡，是今天所有主流大语言模型的选择。它的几种算法（BPE、WordPiece、Unigram）区别在于**如何决定哪些片段进入词表**。

## 2. BPE：从字符开始，反复合并最高频的相邻对

**BPE**（Byte Pair Encoding）最初是一种数据压缩算法，Sennrich et al.（2016）把它用于神经机器翻译的分词。GPT 系列、Llama、Qwen 等都使用 BPE 或它的变体。

### 2.1 训练算法

1. 准备训练语料，设定目标词表大小 $V$；
2. 把语料切成词，统计每个词的频次；每个词拆成字符序列，词尾加一个结束标记（如 `</w>`，用来区分词中的 `st` 和词尾的 `st`）；
3. 初始词表 = 所有出现过的字符；
4. 统计所有**相邻符号对**在语料中出现的总次数，选出次数最多的一对，合并成一个新符号，加入词表，并记录这条**合并规则**；
5. 在语料中把所有这一对替换成新符号；
6. 重复第 4、5 步，直到词表达到 $V$，或者没有出现次数大于 1 的相邻对。

最终产出两样东西：**词表**和**有序的合并规则列表**。

### 2.2 手算一个例子

语料中有四个词及其频次（经典例子）：

| 词 | 频次 | 初始切分 |
|---|---|---|
| low | 5 | `l o w </w>` |
| lower | 2 | `l o w e r </w>` |
| newest | 6 | `n e w e s t </w>` |
| widest | 3 | `w i d e s t </w>` |

**统计相邻对**（每一对的次数是它所在词的频次之和）：

| 对 | 次数 | 来源 |
|---|---|---|
| `e s` | 9 | newest 6 + widest 3 |
| `s t` | 9 | newest 6 + widest 3 |
| `t </w>` | 9 | newest 6 + widest 3 |
| `w e` | 8 | lower 2 + newest 6 |
| `l o` | 7 | low 5 + lower 2 |
| `o w` | 7 | low 5 + lower 2 |
| `n e`、`e w` | 6 | newest |
| `w </w>` | 5 | low |
| 其他 | ≤ 3 | |

最大次数是 9，有三对并列。实现中通常按某个确定的规则打破平局（比如取先出现的）。这里取 `e s`。

**第 1 次合并**：`e s → es`。newest 变成 `n e w es t </w>`，widest 变成 `w i d es t </w>`。重新统计后 `es t` 出现 9 次，是最多的。

**第 2 次合并**：`es t → est`。

**第 3 次合并**：`est </w> → est</w>`（9 次）。现在 newest 是 `n e w est</w>`，widest 是 `w i d est</w>`。

**第 4 次合并**：剩下最多的是 `l o`（7 次）和 `o w`（7 次），取 `l o → lo`。

**第 5 次合并**：`lo w → low`（7 次）。low 变成 `low </w>`，lower 变成 `low e r </w>`。

此时词表 = 初始字符 + `es`、`est`、`est</w>`、`lo`、`low`。可以看到算法自动发现了有意义的单元：最高级后缀 `est` 和词根 `low`。继续合并下去，`newest`、`low</w>` 等常见词会变成完整的 token。

### 2.3 编码：按顺序应用合并规则

训练好之后，对一个新词编码：

1. 把词拆成字符（加 `</w>`）；
2. 按**训练时学到的顺序**，依次检查每条合并规则，能合并就合并；
3. 剩下的符号序列就是这个词的 token。

例如对没见过的词 `lowest`：`l o w e s t </w>` → 应用 `e s` → `l o w es t </w>` → 应用 `es t` → `l o w est </w>` → 应用 `est </w>` → `l o w est</w>` → 应用 `l o` → `lo w est</w>` → 应用 `lo w` → `low est</w>`。

结果是 `["low", "est</w>"]`：虽然 `lowest` 不在训练语料里，它被拆成了两个有意义的片段。这就是子词分词解决 OOV 问题的方式。

（工程实现中通常不是逐条扫描规则，而是反复在当前序列中找"在规则列表中排名最靠前的相邻对"进行合并，结果相同但更快。）

### 2.4 一个最小实现

```python
from collections import Counter

def get_pair_counts(vocab):
    """vocab: {('l','o','w','</w>'): 5, ...}  ->  {('l','o'): 7, ...}"""
    pairs = Counter()
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(pair, vocab):
    """把 vocab 中每个词里的相邻对 pair 合并成一个符号"""
    a, b = pair
    new_vocab = {}
    for word, freq in vocab.items():
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(word[i])
                i += 1
        new_vocab[tuple(out)] = freq
    return new_vocab

def train_bpe(word_freqs, num_merges):
    vocab = {tuple(w) + ('</w>',): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best, count = pairs.most_common(1)[0]
        if count < 2:
            break
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return merges

def encode_word(word, merges):
    symbols = list(word) + ['</w>']
    for pair in merges:                 # 按学到的顺序应用
        symbols = list(merge_pair(pair, {tuple(symbols): 1}).keys())[0]
    return list(symbols)

merges = train_bpe({'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}, num_merges=10)
print(merges[:5])                     # [('e','s'), ('es','t'), ('est','</w>'), ('l','o'), ('lo','w')]
print(encode_word('lowest', merges))  # ['low', 'est</w>']
```

`Counter.most_common` 在次数相同时保留先插入的项，所以这里的平局打破方式与手算一致。

## 3. 现代 BPE 的两个关键改进

### 3.1 字节级 BPE

上面的 BPE 以**字符**为初始词表。但 Unicode 有十几万个字符，如果语料中出现一个没见过的字符（某种罕见文字、emoji），仍然会出现 OOV。

GPT-2 提出了**字节级 BPE**（byte-level BPE）：先把文本按 UTF-8 编码成字节序列，以 **256 个字节**作为初始词表，在字节上做 BPE 合并。

- **永远不会 OOV**：任何字符串都是字节序列，最坏情况下退化成逐字节编码；
- **初始词表固定为 256**：与语言无关。

GPT-2 的词表大小是 50257 = 256 个字节 + 50000 次合并 + 1 个特殊 token `<|endoftext|>`。

代价是：一个汉字的 UTF-8 编码是 3 个字节，如果训练语料中中文不多、对应的合并不够，一个汉字可能要用 2 到 3 个 token 表示，中文文本的序列会明显变长（见 6.3 节）。

另一种做法是 SentencePiece 的 **byte fallback**：以字符为单位做分词，遇到词表外的字符时才退回到字节表示。Llama 1、Llama 2 使用这种方式。

### 3.2 预分词

如果直接在整段文本上做 BPE，可能会学到跨越词边界的合并，比如 `dog.`、`dog!`、`dog?` 各自成为 token，浪费词表。

所以在 BPE 之前先做**预分词**（pre-tokenization）：用正则表达式把文本切成粗粒度的块（单词、数字、标点、空白分开），BPE 合并**只在块内部**进行。GPT-2 的正则大致把文本切成：

- 英文缩写（`'s`、`'t`、`'re` 等）；
- 可选的前导空格 + 一串字母（` hello`）；
- 可选的前导空格 + 一串数字；
- 可选的前导空格 + 一串标点；
- 空白。

注意空格被并入了**后面**的词：` hello` 和 `hello` 是不同的 token。所以句首的 `Hello` 和句中的 ` Hello` 在模型看来是两个不同的东西。后来的分词器（如 GPT-4 的 cl100k、Llama 3 的 tiktoken 词表）调整了正则，例如把数字限制为最多 3 位一组、更好地处理换行和缩进。

## 4. 其他子词算法

### 4.1 WordPiece（BERT）

WordPiece 和 BPE 一样自底向上合并，区别在**选择合并哪一对**的标准。BPE 选**出现次数最多**的对；WordPiece 选**使训练语料似然增加最多**的对，等价于按下面的分数排序：

$$
\mathrm{score}(a,b) = \frac{\mathrm{count}(ab)}{\mathrm{count}(a)\cdot\mathrm{count}(b)}
$$

直觉：如果 $a$ 和 $b$ 本身都很常见，它们碰巧相邻的次数自然也多，这不说明它们"应该"在一起；这个分数相当于衡量 $a$ 和 $b$ 的**关联程度**（类似点互信息），优先合并那些"几乎总是一起出现"的片段。

WordPiece 用 `##` 标记**非词首**的片段：`playing` → `["play", "##ing"]`。BERT 的英文词表约 3 万个 token。编码时 WordPiece 通常用**最长匹配**（从词首开始，每次取词表中能匹配的最长前缀），而不是像 BPE 那样回放合并规则。

### 4.2 Unigram 语言模型与 SentencePiece

**Unigram**（Kudo, 2018）走相反的方向：**自顶向下删减**。

1. 从一个很大的初始词表开始（例如语料中所有高频子串）；
2. 假设每个 token 独立出现，概率为 $p(x)$。一个词的一种切分 $\mathbf x = (x_1,\dots,x_m)$ 的概率是 $\prod_i p(x_i)$，词的概率是所有切分概率之和；
3. 用 EM 算法估计每个 token 的概率 $p(x)$，使语料似然最大；
4. 对每个 token，计算"把它从词表中删掉，语料似然会下降多少"，删掉下降最少的一部分（比如 20%）；
5. 重复，直到词表缩小到目标大小。单个字符始终保留，保证任何文本都能切分。

编码时，用 Viterbi 算法找概率最大的切分。Unigram 的一个特点是**同一个词可以有多种切分**，各有概率。训练语言模型时可以按概率随机采样不同的切分（**subword regularization**），让模型对切分方式更鲁棒。

**SentencePiece**（Kudo & Richardson, 2018）是一个分词工具库，支持 BPE 和 Unigram。它的关键设计是**把输入当作原始字符流，不依赖空格做预分词**：空格被替换成一个可见符号 `▁`（U+2581），当作普通字符参与分词：

```
"Hello world"  →  ["▁Hello", "▁world"]
```

这样分词完全可逆（把 `▁` 换回空格就是原文），也天然适用于中文、日文等不用空格分词的语言。T5 使用 SentencePiece + Unigram（32K 词表），Llama 1、Llama 2 使用 SentencePiece + BPE（32K 词表）。

### 4.3 三种算法对比

| | BPE | WordPiece | Unigram |
|---|---|---|---|
| 方向 | 自底向上合并 | 自底向上合并 | 自顶向下删减 |
| 选择标准 | 相邻对的频次 | 合并带来的似然增益 | 删除带来的似然损失 |
| 编码方式 | 按顺序回放合并规则 | 最长前缀匹配 | Viterbi 求最优切分 |
| 切分是否唯一 | 是 | 是 | 可以有多种（可采样） |
| 代表模型 | GPT 系列、Llama、Qwen、DeepSeek | BERT | T5、ALBERT、XLNet |

## 5. 词表大小怎么选

词表大小 $V$ 是一个关键的超参数，它在几个方向上互相牵制：

**$V$ 越大：**

- 序列越短（同样的文本用更少的 token 表示），注意力计算更省，同样的上下文窗口能装下更多内容；
- 嵌入层和输出层的参数（各 $V\times d$）越多，输出层算 softmax 的成本越高（第 2 篇）；
- 每个 token 平均出现的次数越少，罕见 token 的嵌入越难学好。

**$V$ 越小：** 反过来。

早期模型多用 3 万到 5 万的词表（BERT 约 3 万，GPT-2 约 5 万，Llama 1/2 为 3.2 万）。随着模型变大、训练数据变多、对多语言和代码的需求增加，词表普遍扩大：Llama 3 扩到约 12.8 万，Qwen 系列约 15 万，一些更新的模型达到 25 万左右。模型越大，嵌入层占总参数的比例越小，大词表带来的序列缩短收益就越划算。

用**压缩率**衡量分词器的效率：平均每个 token 对应多少个字符（或字节）。英文在主流分词器下大约是每个 token 4 个字符左右。

## 6. 分词的可见后果

### 6.1 为什么大模型数不清单词里的字母

经典问题："strawberry 里有几个 r？"很多模型会答错。原因不在推理能力，而在分词：模型看到的不是字母序列 `s-t-r-a-w-b-e-r-r-y`，而是几个 token id，比如可能被切成 `["str", "aw", "berry"]`（具体切分取决于分词器）。

每个 token 对模型来说是一个不可分的整体，它的嵌入向量里并没有直接写着"这个 token 由哪几个字母组成"。模型只能从训练数据中间接学到这些拼写知识，而这类知识在自然文本中很少被显式表达。同理，拼写单词、倒序输出字母、按字母数截断等任务都会受影响。

如果让模型先把单词逐个字母写出来（每个字母成为一个 token），再去数，正确率就会大幅提高。这也从侧面说明问题出在输入表示上。

### 6.2 数字的切分

数字如何分词直接影响算术能力。如果 `12345` 被切成 `["123", "45"]`，而 `12346` 被切成 `["12", "346"]`，同一位上的数字在不同数里落在 token 的不同位置，模型很难学到按位运算的规律。

不同分词器的处理方式：

- **逐位切分**：每个数字一个 token（Llama 1/2 等采用）。规律最清晰，但长数字的序列变长；
- **最多 3 位一组**：GPT-4 的 cl100k 等分词器在预分词时限制数字块最多 3 位，在效率和规律性之间折中。

### 6.3 不同语言的代价不同

分词器的合并规则是在训练语料上统计出来的。如果语料以英文为主，英文常用词大多是单个 token，而其他语言的词会被拆得更碎。同样意思的一句话，在某些语言中可能需要英文数倍的 token。

这意味着这些语言的用户：同样的上下文窗口能放的内容更少，同样的内容推理更慢、API 费用更高。扩大词表、在多语言语料上训练分词器，是缓解这个问题的直接办法，这也是 Qwen 等面向中文和多语言的模型使用大词表的原因之一。

### 6.4 Glitch token

如果某个 token 在**分词器的训练语料**中频繁出现（因而进入了词表），但在**语言模型的训练语料**中几乎不出现，它的嵌入几乎没有被训练过，仍接近初始化的随机值。模型遇到它时会表现异常，比如无法复述、输出无关内容。这类 token 被称为 glitch token，一个著名的例子是早期 GPT 分词器中的 ` SolidGoldMagikarp`（某个网络用户名）。根源是分词器和模型使用了分布不同的语料。

### 6.5 特殊 token

词表里还有一些不来自文本的**特殊 token**：

- 序列边界：`<s>`、`</s>`、`<|endoftext|>`；
- BERT 的 `[CLS]`、`[SEP]`、`[MASK]`、`[PAD]`；
- 对话模型的角色标记，例如 `<|im_start|>`、`<|im_end|>`，以及思考模式的 `<think>`、`</think>`。

它们在预分词阶段被整体识别，不参与 BPE 切分。对话模板（chat template）就是用这些特殊 token 把多轮对话拼成一条序列，这部分会在后训练相关的文章中展开。

## 7. 小结

- **分词器**把文本切成 token 并映射成 id，独立于模型训练，但决定了序列长度、词表大小和模型看到的最小单位。
- **子词**粒度在 word 和 char 之间取得平衡：常见词完整，罕见词拆成可复用的片段，没有 OOV 问题。
- **BPE** 从字符（或字节）开始，反复合并出现次数最多的相邻对；编码时按学到的顺序回放合并规则。
- **字节级 BPE** 以 256 个字节为初始词表，彻底消除 OOV；**预分词**防止合并跨越词边界。
- **WordPiece** 按似然增益合并；**Unigram** 从大词表出发按似然损失删减，并支持多种切分；**SentencePiece** 把空格当作普通字符，适用于任何语言。
- **词表大小**在序列长度、参数量和罕见 token 的学习质量之间折中，大模型倾向于更大的词表。
- 数不清字母、算术困难、多语言代价不均、glitch token，这些现象的根源都在分词。

下一篇讲 token id 进入模型后的第一步和输出前的最后一步：嵌入层与 LM Head。

## 参考文献

- Sennrich, Haddow & Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016.
- Schuster & Nakajima. *Japanese and Korean Voice Search.* ICASSP 2012.（WordPiece）
- Wu et al. *Google's Neural Machine Translation System.* 2016.
- Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL 2018.
- Kudo & Richardson. *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer.* EMNLP 2018.
- Radford et al. *Language Models are Unsupervised Multitask Learners.* 2019.（GPT-2，字节级 BPE）
- Rumbelow & Watkins. *SolidGoldMagikarp (plus, prompt generation).* LessWrong, 2023.
