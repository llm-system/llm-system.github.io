---
title: 深入大模型系统札记：第4章 预训练语言模型基础
description: "归纳偏置决定标记化粒度，自监督学习通过借口任务从无标注数据学习；编码器-解码器建立语义空间，Transformer架构实现记忆规模化与工作记忆外化。"
draft: false
---

> **本章概要**：本章的核心是两层递进的洞察。**第一层：如何用全网数据自监督训练**，归纳偏置决定标记化粒度的选择，自监督学习通过借口任务从数据本身创造监督信号，使模型能够从互联网上近乎无限的无标注文本中学习语言规律和世界知识。**第二层：如何实现记忆的规模化**，编码器-解码器架构将表示学习从视觉延伸到语言，建立语义空间；Transformer 架构实现记忆规模化，并通过工作记忆的外化突破信息瓶颈。

---

这一章要用一章的篇幅讲清楚标记化、自监督学习、编码器-解码器、Transformer，每一个都可以单独写一本书。如何串联？用一个统一的视角：**记忆**。

模型如何记住训练数据中的知识（参数记忆）？模型如何在生成时"记住"之前的内容（工作记忆）？抓住这两个问题，你就抓住了本章的主线。

第三章建立了"预训练→微调"的方法论框架，本章讨论如何将这一框架应用于语言领域。

**第一层：如何用全网数据自监督训练**。互联网上有近乎无限的文本，但几乎都没有标签。自监督学习的核心突破是从数据本身创造监督信号，使模型能够从无标注数据中学习语言规律和世界知识。

**第二层：如何实现记忆的规模化**。自监督学习让模型能从海量数据中学习，但学到的知识存储在哪里？如何让模型在生成时高效检索这些知识？编码器-解码器架构将表示学习从视觉延伸到语言，建立语义空间；Transformer 架构则让记忆真正规模化，使模型能存储更多知识、处理更长的上下文。

---

## 第一部分：如何用全网数据自监督训练

自监督训练需要解决两个问题：如何表示文本，如何从无标注数据中学习。

### 一、归纳偏置与标记化

第三章讨论了归纳偏置如何提高参数利用效率并增强结构化迁移能力：CNN 利用物理世界实体的空间连续性，推荐系统利用协同过滤，语言模型利用时序性状态机。在语言领域，归纳偏置的第一个体现就是**标记化粒度的选择**：字符级、词级还是子词级，每种选择都隐含了对语言结构的不同先验假设。

自然语言是连续的、非结构化的文本流，而神经网络只能处理数值。**标记化**（Tokenization）将文本分割成离散的基本单元（记号），再通过词汇表把每个记号映射为唯一的整数编号。

但词汇表是有限的，而语言是开放的。罕见词、新词、拼写错误都可能不在词汇表中，这就是**未登录词**问题。真正的突破来自**子词级标记化**：常见词保持完整，罕见词拆分为更小的有意义单元。字节对编码（BPE）、WordPiece、SentencePiece 等算法实现了这一思想，成为现代大语言模型的标准实践。

标记化之后，每个整数编号还需要转换为稠密向量（嵌入）才能被神经网络处理。这个嵌入层是模型学习"文字如何表示"的第一站，语义相近的词会被映射到向量空间中相邻的位置。

### 二、自监督学习与借口任务

传统监督学习需要大量人工标注数据，成本高且难以规模化。能否让模型从无标注数据中自动学习？

**自监督学习**的核心思想是**从数据本身创造监督信号**。方法是设计一个"借口任务"，人为地对输入进行部分遮盖或破坏，然后让模型预测被遮盖的内容。通过在海量文本上反复解决这类"自设谜题"，模型被迫学习语言的深层结构性规律。

两种主流范式：**掩码语言建模**（MLM）随机遮盖部分词汇，让模型根据双向上下文预测被遮盖的词，训练出的模型（如 BERT）擅长语义理解任务。**因果语言建模**（CLM）给定前文预测下一个词，严格单向，模拟人类写作过程，训练出的模型（如 GPT）擅长文本生成任务。

自监督学习带来了三重革命：**数据规模的解放**，可以利用互联网上近乎无限的文本；**强大的泛化能力**，学到的是语言的普遍规律；**知识的涌现**，为了更好地预测下一个词，模型自发学习了语法、语义、事实性知识、常识推理。模型的初始目标只是"完形填空"，但当处理了数万亿词元之后，它发现持续降低预测损失的最优策略是**理解世界**。这正是大语言模型取得成功的根本原因。

至此，第一部分的结论是清晰的：**自监督学习让模型能够从全网无标注数据中学习**。但学到的知识存储在哪里？如何让模型在生成时高效检索这些知识？

---

## 第二部分：如何实现记忆的规模化

模型的记忆能力决定了它能存储多少知识、处理多长的上下文、多高效地检索。

### 三、编码器-解码器与表示学习

文字只是语言的表层符号。一个句子的"意思"不等于其中每个词的意思之和，"我爱打篮球"和"篮球爱打我"用词相同，语义却截然不同。如何将变长的符号序列映射到一个能捕捉整体语义的统一空间？

**编码器-解码器架构**给出了答案。编码器负责"理解与压缩"，读取整个输入序列，将其语义精华压缩为一个固定维度的**上下文向量**。解码器负责"生成与重构"，基于上下文向量，以自回归方式逐个生成输出序列。

上下文向量存在于一个经过学习的**隐层语义空间**中。在这个空间里，语义相似的输入会被映射到相邻位置。这把"理解语义"的问题转化为了"在向量空间中计算距离"的问题。

从表示学习的视角看，编码器-解码器架构是第三章「自动表示」思想在语言领域的延伸：CNN 学习的是视觉层次化表示，编码器-解码器学习的则是**语言的语义表示**。模型权重中压缩存储了从训练数据中学到的语言规律和世界知识（**参数记忆**）。上下文向量既是语义桥梁，也是信息瓶颈，它迫使模型学会舍弃冗余、保留精华。

### 四、Transformer 架构与记忆规模化

编码器-解码器架构把序列压缩为单一的上下文向量，这是个严重的信息瓶颈。模型需要在处理序列时"记住"之前的内容，并在需要时检索相关信息，这就是**工作记忆**。语言模型架构的演进，本质上是一系列将记忆规模化的努力：如何让模型存储更多知识、处理更长的上下文、更高效地检索。

**第一阶段：用循环结构引入工作记忆。**循环神经网络（RNN）通过状态机为模型引入了工作记忆，每读一个词，就结合当前输入和之前的隐藏状态更新理解。长短期记忆网络（LSTM）通过门控机制缓解了信息衰减问题。但本质上，这仍是一个**被动衰减的记忆系统**，信息只能沿时间轴单向流动。

**第二阶段：用注意力机制扩展工作记忆。**注意力机制的核心是"查询-键-值"框架：当前位置生成查询向量，与所有历史位置的键向量计算相关性，再按相关性对值向量加权求和。这相当于给模型配备了一个**可微分的检索系统**，大幅扩展了工作记忆的有效容量。

**第三阶段：用 Transformer 实现全局并行的向量数据库。**Transformer 的核心洞察是**完全抛弃循环结构，把整个序列当作一个可并行查询的向量数据库**。在自注意力中，每个位置同时独立计算，生成自己的查询、键、值向量，然后向所有其他位置发起查询。这本质上就是一个**可微分的向量数据库**：键是索引，值是存储的内容，查询是检索条件。

**位置编码**解决了自注意力不感知顺序的问题。**旋转位置编码**（RoPE）将相对位置信息直接融入注意力计算。**多头注意力**让模型能从多个视角同时查询，相当于同时运行多个独立的向量数据库，然后融合检索结果。

GPT 系列选择了**仅解码器架构**，只保留带因果掩码的自注意力，每个位置只能查询自己和之前的位置。这种设计与因果语言建模完美匹配，并且推理高效（键值缓存可以复用之前的计算）。

基于因果语言建模的生成式语言模型带来了一个更深层的解放：**工作记忆的外化**。在编码器-解码器架构中，编码器必须把整个输入序列压缩到一个上下文向量中，这是严重的信息瓶颈。而仅解码器架构将大部分工作记忆从模型内部转移到了**显式的生成内容**，模型可以把中间推理步骤、关键信息、待处理的子任务"写"到输出序列中，再通过自注意力机制随时"读"回来。

记忆规模化的趋势仍在加速：Gemini 1.5 已经将上下文窗口推至百万 token 级别 (Gemini Team, 2024)，进一步验证了 Transformer 自注意力机制在超长序列上的可扩展性。至此，第二部分的结论是清晰的：**Transformer 架构实现了记忆的规模化，大幅扩展了工作记忆，并通过工作记忆的外化突破了信息瓶颈**。

"大语言模型到底是什么"？如果你需要一个直觉，把它理解为**超大规模的向量数据库**。训练阶段把知识压缩存储到参数记忆中，推理阶段通过工作记忆高效检索并生成连贯的文本输出。这个比喻不完美，但足以帮你理清大部分困惑。

---

## 小结

本章的核心是两层递进的洞察。

**第一层：如何用全网数据自监督训练**。归纳偏置决定标记化粒度的选择，自监督学习通过借口任务从数据本身创造监督信号。这使模型能够从互联网上近乎无限的无标注文本中学习语言规律和世界知识。

**第二层：如何实现记忆的规模化**。编码器-解码器架构将表示学习从视觉延伸到语言，建立语义空间，知识压缩存储在模型权重中。Transformer 架构实现记忆规模化，自注意力机制让模型能高效检索序列中的任意位置。仅解码器架构实现了工作记忆的外化，模型可以把中间状态"写"到输出序列中再"读"回来。

这条主线为后续的规模化预训练和后训练技术奠定了坚实的基础。特别值得注意的是，GPT 选择的"预测下一个词元"（Next Token Prediction）目标函数看似简单，却有一个深远的工程含义：它让训练数据近乎无限（整个互联网都是语料），且目标函数统一（无需为不同任务设计不同的损失函数）。正是这两个特性，使得缩放定律得以生效——这正是下一章的主题。

在工程实践层面，Hugging Face Transformers 库将本章讨论的几乎所有模型架构（BERT、GPT、LLaMA 等）统一到一个开源平台上 (Wolf et al., 2020)，使预训练语言模型从论文走向可复现、可部署的工程实践，是推动整个领域民主化的关键基础设施。

## 引用本章

```bibtex
@book{baillmsystem,
  title     = {深入大模型系统：提示工程、符号推理与智能体实践},
  author    = {Bai, Yu},
  publisher = {人民邮电出版社},
  year      = {2025},
  isbn      = {978-7-115-68707-4}
}
```

## 文献列表

- **Transformer: 注意力机制就是全部**
  Attention Is All You Need. Vaswani, Ashish et al.
  [原文](https://arxiv.org/abs/1706.03762)
- **BERT: 用于语言理解的深度双向 Transformer 预训练**
  Pre-training of Deep Bidirectional Transformers for Language Understanding. Devlin, Jacob et al.
  [原文](https://arxiv.org/abs/1810.04805)
- **GPT-1: 通过生成式预训练改进语言理解**
  Improving Language Understanding by Generative Pre-Training. Radford, Alec et al.
  [原文](https://openai.com/research/language-unsupervised)
- **GPT-2: 语言模型是无监督多任务学习者**
  Language Models are Unsupervised Multitask Learners. Radford, Alec et al.
  [原文](https://openai.com/research/better-language-models)
- **LSTM: 长短期记忆网络**
  Long Short-Term Memory. Hochreiter, Sepp and Schmidhuber, Jürgen.
  [原文](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **Seq2Seq: 使用神经网络的序列到序列学习**
  Sequence to Sequence Learning with Neural Networks. Sutskever, Ilya et al.
  [原文](https://arxiv.org/abs/1409.3215)
- **GRU / Encoder-Decoder: 使用 RNN 编码器-解码器学习短语表示**
  Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. Cho, Kyunghyun et al.
  [原文](https://arxiv.org/abs/1406.1078)
- **Attention: 通过联合学习对齐和翻译的神经机器翻译**
  Neural Machine Translation by Jointly Learning to Align and Translate. Bahdanau, Dzmitry et al.
  [原文](https://arxiv.org/abs/1409.0473)
- **BPE: 基于子词单元的稀有词神经机器翻译**
  Neural Machine Translation of Rare Words with Subword Units. Sennrich, Rico et al.
  [原文](https://arxiv.org/abs/1508.07909)
- **WordPiece: Google 的神经机器翻译系统**
  Google's Neural Machine Translation System. Wu, Yonghui et al.
  [原文](https://arxiv.org/abs/1609.08144)
- **SentencePiece: 简单且独立于语言的子词分词器**
  SentencePiece: A simple and language independent subword tokenizer. Kudo, Taku and Richardson, John.
  [原文](https://arxiv.org/abs/1808.06226)
- **RoPE: 旋转位置编码增强 Transformer 语言模型**
  RoFormer: Enhanced Transformer with Rotary Position Embedding. Su, Jianlin et al.
  [原文](https://arxiv.org/abs/2104.09864)
- **Neural LM: 神经概率语言模型**
  A Neural Probabilistic Language Model. Bengio, Yoshua et al.
  [原文](https://www.jmlr.org/papers/v3/bengio03a.html)
- **Word2Vec: 向量空间中词表示的高效估计**
  Efficient Estimation of Word Representations in Vector Space. Mikolov, Tomas et al.
  [原文](https://arxiv.org/abs/1301.3781)
- **GloVe: 词表示的全局向量**
  GloVe: Global Vectors for Word Representation. Pennington, Jeffrey, Socher, Richard and Manning, Christopher D.
  [原文](https://nlp.stanford.edu/pubs/glove.pdf)
- **ELMo: 深度上下文化词表示**
  Deep contextualized word representations. Peters, Matthew E. et al.
  [原文](https://arxiv.org/abs/1802.05365)
- **LLaMA: 开放且高效的基础语言模型**
  LLaMA: Open and Efficient Foundation Language Models. Touvron, Hugo et al.
  [原文](https://arxiv.org/abs/2302.13971)
- **Gemini 1.5: 跨越百万 token 上下文的多模态理解**
  Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. Gemini Team, Google.
  [原文](https://arxiv.org/abs/2403.05530)
- **Hugging Face Transformers: 自然语言处理的开源平台**
  Transformers: State-of-the-Art Natural Language Processing. Wolf, Thomas et al. EMNLP 2020 Demo.
  [原文](https://arxiv.org/abs/1910.03771)