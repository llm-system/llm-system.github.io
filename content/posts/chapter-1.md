---
title: 深入大模型系统札记：第1章 大模型技术概述
description: "辨析基础模型、GenAI、LLM等概念，大规模记忆能力、通用知识获取、指令跟随能力共同定义能力边界；系统化连接外部世界，MaaS以token为核心抽象成为大模型时代最重要的云基础设施。"
draft: false
---

> **本章概要**：本章的核心是两层递进的洞察。**第一层：大模型是什么**，辨析基础模型、GenAI 模型、大语言模型、大模型等概念；Transformer 架构提供大规模记忆能力，预训练范式解决通用知识获取，后训练方法赋予指令跟随能力，三者共同定义了大模型的能力边界。**第二层：大模型系统解决什么问题**，模型本身是"封闭系统"，无法获取实时信息、无法执行实际操作；系统化解决与真实世界交互的问题，MaaS 以 token 为核心抽象，成为大模型时代最重要的云基础设施。

---

如果你只能用两句话向一个技术背景但不熟悉大模型的朋友解释"大模型是什么"，你会怎么说？

这一章给你答案：大模型技术的演进，并不是"参数越大越好"的单线叙事，而是在两个层次上逐步解决核心问题。

**第一层：大模型是什么**。大模型的能力边界由三个技术突破共同定义：Transformer 架构提供大规模记忆能力，预训练范式解决通用知识获取，后训练方法赋予指令跟随能力。

**第二层：大模型系统解决什么问题**。模型本身是"封闭系统"，无法获取实时信息、无法执行实际操作、无法在真实任务中形成闭环。大模型系统要解决的是：系统化让模型连接外部世界，MaaS 以 token 为核心抽象，成为大模型时代最重要的云基础设施。

---

## 第一部分：大模型是什么

在讨论技术细节之前，需要先辨析几个容易混淆的概念。

**基础模型（Foundation Models）**在这组概念中范围最广。Bommasani 等学者在斯坦福研究报告中给出了定义：在广泛数据上训练（通常通过大规模自监督学习），并能适配到各种下游任务的模型 (Bommasani et al., 2021)。基础模型不限于语言，视觉基础模型（如 CLIP、SAM）、多模态基础模型（如 GPT-4V）都属于这一范畴。"基础"一词强调的是这类模型在技术生态中的**枢纽地位**：众多下游应用建立在同一个基础模型之上，其能力与缺陷都会被继承。从构建范式看，基础模型属于预训练模型，但并非所有预训练模型都是基础模型——只有训练数据足够广泛、能适配多种下游任务的预训练模型，才具备"基础"的地位。

**GenAI 模型（Generative AI Model）**是基础模型的一种应用形态：能够理解用户输入的提示（Prompt），并据此生成文本、图像、视频等内容的模型。GenAI 模型的核心是**提示-生成**范式：用户通过自然语言描述需求，模型生成相应的内容。当用户在 ChatGPT 对话框中输入"中国举办过几次奥运会？"时，这段文本就是提示，ChatGPT 生成的回答就是输出。基础模型是 GenAI 模型的能力基础，提示是激活这些能力的钥匙。

**大语言模型（Large Language Models，LLM）**特指以自然语言为核心模态的大规模基础模型。"大"的含义有两层：参数量大（通常数十亿到数万亿参数）和训练数据量大（通常数万亿 token）。GPT 系列、Claude、Llama、Qwen、DeepSeek 都是典型的大语言模型。LLM 的核心能力是**自然语言理解与生成**，但随着技术演进，其边界不断扩展：通过多模态扩展可以处理图像、音频，通过工具调用可以执行代码、访问 API。

理解这些概念还可以从三个分类维度切入：**按模态分类**，有语言模型、图像模型、语音模型、视频模型等；**按参数量分类**，有小参数模型（百万级，适用于轻量化场景）和大参数模型（十亿到万亿级，具备更强的泛化能力）；**按构建范式分类**，有预训练模型（先通用预训练再任务适配）和非预训练模型（针对具体任务从零训练）。在这一框架下，LLM 的精准定位是：从模态看属于语言模型，从参数量看属于大参数模型，从构建范式看属于预训练模型，其规模和通用性使之成为语言领域的基础模型。

与上述技术术语不同，**大模型**更多是一个传播概念而非严格的技术术语。它表面上描述的是模型参数量和训练数据量的庞大，本质上强调的是规模带来的能力提升，包括更强的推理能力、更好的泛化能力、更广的知识覆盖。本书为照顾大众语境，沿用了这一表述。

在此语境下，大模型的能力边界由三个技术突破共同定义。

### 一、Transformer：大规模记忆能力

传统的循环神经网络按时间步逐个处理词元，无法并行化，在长序列上训练效率极低，记忆容量受限。2017 年**Transformer**的出现彻底改变了这一局面，解决了大规模并行化训练的问题。从功能机制上看，自注意力机制可以被视为**可微分、可并行的向量数据库**：每个位置都能向所有其他位置发起查询，键是索引，值是存储的内容，查询是检索条件。这一架构突破带来了两个关键提升：**参数记忆**大幅扩展（更大的模型能存储更多知识），**工作记忆**大幅扩展（更长的上下文窗口能处理更复杂的任务）。Transformer 使"大规模预训练"成为可行且可扩展的主流范式，为后续一切发展奠定了基础。

### 二、预训练：通用知识获取

传统监督学习需要为每个任务单独收集标注数据，成本高昂且难以扩展。预训练范式给出了答案：模型在海量数据上进行自监督学习，获得可迁移的通用表征与世界知识。BERT 和 GPT 代表了两条不同的预训练路径：前者采用掩码语言建模，侧重双向理解；后者采用因果语言建模，侧重自回归生成。

当模型规模进一步扩大后，GPT-3 展示出惊人的少样本学习能力，揭示了"能力随规模涌现"这一现象：模型不仅学到了语言模式，还压缩了人类社会的大量先验知识。

### 三、后训练：指令跟随能力

"会生成"并不等于"可用"。预训练模型可能生成有害、不真实或不符合用户需求的内容。研究与工业界逐步形成了一整套**后训练方法**：从指令微调（如 FLAN）到基于人类偏好的强化学习（RLHF），再到近端策略优化（PPO）等策略优化算法，最终汇聚成 InstructGPT 所代表的技术路线。

ChatGPT 是这些方法的集大成者，它的交互体验让大模型真正从研究产物走向公众可用的平台能力。GPT-4 则普遍认为是引入 MoE 架构的一个关键节点，进一步展示了通用模型在能力边界、可靠性和工程体系上的新高度。

至此，第一部分的结论是清晰的：**大规模记忆能力（Transformer）+ 通用知识获取（预训练）+ 指令跟随能力（后训练）**共同定义了大模型的能力边界。

很多技术文章止步于此，讲完模型架构和训练方法就结束了。但如果你止步于此，你就只理解了大模型的一半。真正的问题在下一层：模型本身仍是"封闭系统"，它无法获取实时信息，无法执行实际操作，无法在真实任务中形成闭环。

---

## 第二部分：大模型系统解决什么问题

大模型系统要解决的是模型与真实世界之间的鸿沟。

### 四、产品演进：从能力到可用

从系统视角看，纯语言模型存在三个根本性局限：**知识静态**（训练数据有截止日期，无法获取实时信息）、**输出非结构化**（只能生成自然语言文本，无法与程序对接）、**无法行动**（不能调用外部工具和服务）。你可以从 OpenAI 的产品演进中清晰地看到如何一步步突破这些局限。

**GPT-3**展示了大规模语言模型的少样本学习能力，但上述三个局限全部存在。GPT-3 只能进行文本续写，输出是非结构化的自然语言，无法被程序直接解析和执行。

**Codex 和 GitHub Copilot**的意义常被低估。GPT 的第一个规模化商业应用不是 ChatGPT，而是 Copilot。这在第一天就揭示了大模型真正的 killer app：**AI4SE（AI for Software Engineering）**。直到今天蓦然回首，类 Cursor 应用已成为 token 消耗最高的大模型应用类别。技术上，Codex 证明了模型可以输出**可执行的结构化内容**，为后续的工具调用奠定了基础；商业上，软件工程场景具备"痛点明确、价值可量化、用户有付费能力"的特征，在大模型商业化的第一天就指明了方向。

**ChatGPT**的意义是**从技术到产品的跨越**。通过 RLHF 对齐人类偏好，模型变成了"符合用户预期的对话助手"，证明了大模型可以直接面向普通用户。但 ChatGPT 初期仍是封闭系统：知识静态、无法联网、无法执行代码。

### 五、系统化：连接外部世界

ChatGPT 证明了模型的可用性，但它仍是封闭系统。接下来的演进聚焦于打通模型与外部世界的接口。

**联网和插件**是**上下文工程的起点**。两者的本质相同：在推理时动态注入外部信息或能力，扩展模型的有效上下文。这一思路后来发展为 RAG、Agentic RAG 等系统化方法，成为大模型系统的核心技术之一。

**函数调用和代码解释器**标志着**智能体系统的技术基础成型**。函数调用是"调用已有工具"，代码解释器是"临时生成工具"，两者互补构成完整的行动能力。这也是 AI4SE 从辅助工具演进到自主智能体的关键跨越，Cursor、Devin 等 AI 编程智能体正是建立在这一技术基础之上。

**GPT-4o**为**具身智能**奠定了系统基础。多模态端到端原生整合带来两个关键特性：低延迟和跨模态理解。这正是具身智能的前提条件，机器人需要实时感知、实时决策、实时行动，任何环节的高延迟都会导致系统不可用。GPT-4o 证明了大模型可以在实时性要求下工作，这是从"数字世界的 AI 助手"迈向"物理世界的具身智能"的关键跨越。

回顾这条演进路径，你需要看清一个常被忽视的事实：GPT-3 到 GPT-4o 的跨越，与其说是"模型更强了"，不如说是"系统更完整了"。每一步都在扩展模型与外部世界的接口，竞争焦点因此从"模型本身"转向"系统能力"。

这正是这本书命名为"大模型**系统**"而非"大模型"的原因：模型只是系统的一部分，而非全部。

### 六、MaaS：模型即服务

**MaaS（Model as a Service）**是大模型时代最重要的云基础设施。传统云计算以计算、存储、网络为基本资源单元；MaaS 的核心抽象是**token**——输入 token 承载查询，输出 token 承载响应，定价按 token 计费，优化目标是单位成本的 token 吞吐量。这是一种**以 token 为中心的归纳偏置**：整个技术栈从架构到算子到硬件，都围绕 token 的生产效率来组织。

围绕这一归纳偏置，一场多层面的效率竞赛正在展开。架构层有稀疏激活（MoE），让模型"容量大但计算少"；算子层有内存优化（FlashAttention），让硬件利用率逼近极限；硬件层有专用芯片（TPU、推理加速卡），提供持续扩展的算力底座。每一层的优化都在压缩每 token 的边际成本。

DeepSeek 系列模型是这场竞赛的一个标志性节点：通过架构创新和工程优化，在保持顶尖性能的同时，将训练和推理成本降低一个数量级。这种"性能、成本、开放"的新平衡，正在重新定义 MaaS 的竞争规则。

---

## 小结

本章的核心是两层递进的洞察。

**第一层：大模型是什么**。Transformer 架构提供大规模记忆能力，预训练范式解决通用知识获取，后训练方法赋予指令跟随能力，三者共同定义了大模型的能力边界。

**第二层：大模型系统解决什么问题**。模型本身是"封闭系统"，无法获取实时信息、无法执行实际操作。系统化解决与真实世界交互的问题，MaaS 以 token 为核心抽象，成为大模型时代最重要的云基础设施。

理解"大模型系统"需要把视角从单一模型扩展到完整链路。

如果你读完这一章只记住一件事，记住这个：**不要只盯着模型看，要看整个系统**。这条主线为后续章节提供了统一的坐标系。

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
  [原文](https://arxiv.org/abs/1706.03762) [代码](https://github.com/tensorflow/tensor2tensor)
- **GPT-3: 语言模型是少样本学习者**
  Language Models are Few-Shot Learners. Brown, Tom B. et al.
  [原文](https://arxiv.org/abs/2005.14165)
- **InstructGPT: 使用人类反馈训练语言模型遵循指令**
  Training language models to follow instructions with human feedback. Ouyang, Long et al.
  [原文](https://arxiv.org/abs/2203.02155)
- **ChatGPT: 介绍**
  Introducing ChatGPT. OpenAI.
  [原文](https://openai.com/index/chatgpt/)
- **GPT-4: 技术报告**
  GPT-4 Technical Report. OpenAI.
  [原文](https://arxiv.org/abs/2303.08774) [代码](https://github.com/openai/evals)
- **基础模型的机遇与风险**
  On the Opportunities and Risks of Foundation Models. Bommasani, Rishi et al. Stanford CRFM, 2021.
  [原文](https://arxiv.org/abs/2108.07258)
- **AlexNet: 使用深度卷积神经网络进行 ImageNet 分类**
  ImageNet Classification with Deep Convolutional Neural Networks. Krizhevsky, Alex et al.
  [原文](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- **AlphaGo: 使用深度神经网络和树搜索掌握围棋**
  Mastering the game of Go with deep neural networks and tree search. Silver, David et al.
  [原文](https://www.nature.com/articles/nature16961) [代码](https://github.com/maxpumperla/deep_learning_and_the_game_of_go)
- **AlphaZero: 通过自我对弈掌握国际象棋、将棋和围棋的通用强化学习算法**
  Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. Silver, David et al.
  [原文](https://arxiv.org/abs/1712.01815) [代码](https://github.com/suragnair/alpha-zero-general)
- **BERT: 用于语言理解的深度双向 Transformer 预训练**
  Pre-training of Deep Bidirectional Transformers for Language Understanding. Devlin, Jacob et al.
  [原文](https://arxiv.org/abs/1810.04805) [代码](https://github.com/google-research/bert)
- **GPT-1: 通过生成式预训练改进语言理解**
  Improving Language Understanding by Generative Pre-Training. Radford, Alec et al.
  [原文](https://openai.com/research/language-unsupervised)
- **GPT-2: 语言模型是无监督多任务学习者**
  Language Models are Unsupervised Multitask Learners. Radford, Alec et al.
  [原文](https://openai.com/research/better-language-models) [代码](https://github.com/openai/gpt-2)
- **Codex: 评估基于代码训练的大语言模型**
  Evaluating Large Language Models Trained on Code. Chen, Mark et al.
  [原文](https://arxiv.org/abs/2107.03374) [代码](https://github.com/openai/human-eval)
- **GitHub Copilot: 你的 AI 编程伙伴**
  Introducing GitHub Copilot: your AI pair programmer. GitHub.
  [原文](https://github.blog/2021-06-29-introducing-github-copilot-ai-pair-programmer/)
- **FLAN: 微调的语言模型是零样本学习者**
  Finetuned Language Models Are Zero-Shot Learners. Wei, Jason et al.
  [原文](https://arxiv.org/abs/2109.01652) [代码](https://github.com/google-research/flan)
- **RLHF: 基于人类偏好的深度强化学习**
  Deep Reinforcement Learning from Human Preferences. Christiano, Paul F. et al.
  [原文](https://arxiv.org/abs/1706.03741) [代码](https://github.com/openai/lm-human-preferences)
- **PPO: 近端策略优化算法**
  Proximal Policy Optimization Algorithms. Schulman, John et al.
  [原文](https://arxiv.org/abs/1707.06347) [代码](https://github.com/openai/baselines)
- **GPT-4V: 系统卡片**
  GPT-4V(ision) System Card. OpenAI.
  [原文](https://openai.com/index/gpt-4v-system-card/)
- **GPT-4o: 介绍**
  Hello GPT-4o. OpenAI.
  [原文](https://openai.com/index/hello-gpt-4o/)
- **Sora: 作为世界模拟器的视频生成模型**
  Video generation models as world simulators. OpenAI.
  [原文](https://openai.com/index/video-generation-models-as-world-simulators/)
- **CLIP: 从自然语言监督中学习可迁移的视觉模型**
  Learning Transferable Visual Models From Natural Language Supervision. Radford, Alec et al.
  [原文](https://arxiv.org/abs/2103.00020)
- **SAM: 分割一切模型**
  Segment Anything. Kirillov, Alexander et al.
  [原文](https://arxiv.org/abs/2304.02643)
- **MoE: 超大神经网络：稀疏门控专家混合层**
  Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. Shazeer, Noam et al.
  [原文](https://arxiv.org/abs/1701.06538)
- **FlashAttention: 具有 IO 感知的快速且内存高效的精确注意力**
  Fast and Memory-Efficient Exact Attention with IO-Awareness. Dao, Tri et al.
  [原文](https://arxiv.org/abs/2205.14135)
- **Mamba: 使用选择性状态空间的线性时间序列建模**
  Linear-Time Sequence Modeling with Selective State Spaces. Gu, Albert et al.
  [原文](https://arxiv.org/abs/2312.00752)
- **DDPM: 去噪扩散概率模型**
  Denoising Diffusion Probabilistic Models. Ho, Jonathan et al.
  [原文](https://arxiv.org/abs/2006.11239)
- **DeepSeek-V3: 技术报告**
  DeepSeek-V3 Technical Report. DeepSeek-AI.
  [原文](https://arxiv.org/abs/2412.19437)
- **DeepSeek-R1: 通过强化学习激励大语言模型的推理能力**
  Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. DeepSeek-AI.
  [原文](https://arxiv.org/abs/2501.12948) [代码](https://github.com/deepseek-ai/DeepSeek-R1)
- **TPU: 数据中心张量处理单元性能分析**
  In-Datacenter Performance Analysis of a Tensor Processing Unit. Jouppi, Norman P. et al.
  [原文](https://arxiv.org/abs/1704.04760)
- **ChatGPT 插件**
  ChatGPT plugins. OpenAI.
  [原文](https://openai.com/index/chatgpt-plugins/) [代码](https://github.com/openai/chatgpt-retrieval-plugin)
- **函数调用和其他 API 更新**
  Function calling and other API updates. OpenAI.
  [原文](https://openai.com/index/function-calling-and-other-api-updates/) [代码](https://github.com/openai/openai-python)


