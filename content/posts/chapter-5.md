---
title: 深入大模型系统札记：第5章 大语言模型基础
description: "缩放定律揭示性能与资源的幂律关系，规模化工程突破数据、算力、参数瓶颈，输出续写模型；后训练通过结构化输出、指令跟随、RLHF对齐，输出指令模型。"
draft: false
---

> **本章概要**：本章的核心是两层递进的洞察。**第一层：预训练缩放定律**，缩放定律揭示了性能与资源的幂律关系，规模化工程突破数据、算力、参数三个维度的瓶颈，输出的是续写模型（completion model）。**第二层：后训练缩放定律**，三种对齐模式（结构化输出、指令跟随、RLHF）将续写能力转化为指令跟随能力，输出的是指令模型（instruct model）。

---

如果前四章是"大模型为什么能工作"，这一章就是"大模型如何被造出来"。

你需要认识到一个事实：**缩放定律改变了 AI 研究的游戏规则**。以前是"有好 idea 就能发论文"，现在是"没有算力就没有发言权"。这对学术界是挑战，但对工程师是机遇，当"方向明确、拼执行"成为主旋律时，工程能力的价值被前所未有地放大了。

第四章建立了预训练语言模型的技术基础。GPT 选择的"预测下一个词元"目标函数看似简单，却让训练数据近乎无限、目标函数高度统一，这正是缩放定律得以生效的前提。本章讨论大语言模型构建的两层缩放定律。

**第一层：预训练缩放定律**。预训练阶段的核心问题是如何高效地将计算资源转化为模型能力。缩放定律揭示了性能与资源的幂律关系，规模化工程突破数据、算力、参数的物理瓶颈。预训练输出的是**续写模型**（completion model），给定前文，预测后续文本。

**第二层：后训练缩放定律**。后训练阶段的核心问题是如何将续写能力转化为指令跟随能力。三种对齐模式服务于三类需求：结构化输出适配封闭式任务，指令跟随建立开放式自然语言接口，RLHF 通过强化学习适应动态环境。后训练输出的是**指令模型**（instruct model），理解用户意图，生成有用回复。

---

## 第一部分：续写模型与预训练

预训练阶段的核心问题是如何高效地将计算资源转化为模型能力。

### 一、缩放定律

单纯增加数据、算力或参数，性能增益并非线性。在投入数亿美元训练之前，能否预测不同资源配置下的模型性能？

**缩放定律**（Scaling Laws）给出了答案。通过大规模实验，研究者发现模型性能与参数规模、数据量、算力之间存在稳定的幂律关系。缩放定律的工程价值在于性能预测、资源优化、风险控制。Chinchilla 的研究进一步揭示了数据与参数的最优配比，指出早期模型普遍"参数过大、数据不足"。缩放定律使"更大即更强"从信念转变为可预测的工程学科。

### 二、规模化工程

缩放定律指明了方向，但具体执行时数据、算力、参数三个维度各有瓶颈。

**数据维度**：模型学习存在两种模式，**表面记忆**（存储并复现具体文本）和**结构化压缩**（将语法、语义、知识编码进参数）。前者无法泛化，后者才是真正的能力。数据工程的目标是抑制表面记忆、促进结构化压缩，方法包括多源采集、去噪去重、多样性平衡。

**算力维度**：大规模训练面临计算吞吐量和内存带宽的双重瓶颈。当数据搬运速度跟不上计算速度时，GPU 大量时钟周期浪费在等待，这就是**内存墙**。**混合并行**是基础架构：数据并行、模型并行、流水线并行。系统级优化最大化计算与通信重叠，如 DeepSeek 的双管线技术 DualPipe、以及优化注意力计算的 FlashAttention。推理服务同样面临内存瓶颈，vLLM 的 PagedAttention 借鉴操作系统虚拟内存的分页机制管理键值缓存 (Kwon et al., 2023)，已成为大规模 LLM 推理服务的事实标准。

**参数维度**：稠密架构的资源消耗与参数规模呈线性或超线性增长。**混合专家架构**（MoE）的核心思想是"拥有海量参数，但单次前向计算仅激活子集"，通过专家网络和路由器实现"总容量"与"单次计算量"的解耦。**模型压缩与蒸馏**解决能力落地问题：多头潜在注意力（MLA）减少键值缓存开销，知识蒸馏让小模型学习大模型的输出和内部表征。

至此，第一部分的结论是清晰的：**预训练输出续写模型**。续写模型具备强大的语言建模能力，但它只会"续写"，给定前文预测后续文本。续写能力不等于指令跟随能力，模型可能生成有害、不真实或不符合用户需求的内容。

---

## 第二部分：指令模型与后训练

后训练阶段的核心问题是如何将续写能力转化为指令跟随能力。

### 三、结构化输出对齐

许多应用要求模型输出预定义的封闭结构（类别标签、数值、序列标注）。如何将模型通用的、连续的内部表示映射到特定的、离散的输出空间？

**分类头微调**在预训练模型顶部附加任务专用的输出层，将通用表示映射到特定输出空间。特点是硬对齐（输出被严格限制在预定义集合内）、离线学习、高可控性。其思想延续到**参数高效微调**（PEFT），低秩适配（LoRA）、适配器（Adapter）在保持主干不变的前提下注入少量可训练参数，以低成本适配多种结构化任务。

### 四、指令跟随对齐

结构化输出要求"一任务一结构"，无法处理开放式需求。能否构建统一接口，使单一模型通过自然语言指令执行任意任务？

**指令微调**将所有任务转化为条件式语言生成，监督信号被完全"语言化"。模型学习的不是"选择正确标签"，而是"生成能够代表该标签的文本"。特点是软对齐（以语言指令替代固定标签）、统一架构（单一模型支持海量任务）、零样本涌现。但指令微调无法对齐**隐性偏好**，人类关于安全、有用、诚实等价值标准难以通过离线的显式指令完全描述。

### 五、在线环境对齐

现实部署环境是动态的，用户偏好会变化，安全合规要求会更新，业务指标会调整。这些外部信号往往是隐性的、情境依赖的、非二元的。传统离线监督学习无法适应这类动态环境。

**基于人类反馈的强化学习**（RLHF）是在线环境对齐的代表性方法，通过三阶段流程实现。**监督微调**（SFT）建立行为基线，使模型具备基本的指令遵循能力。**奖励模型训练**（RM）学习人类偏好，监督信号从"构造标准答案"转变为"评价候选答案"，让监督微调后的模型生成多个候选回答，人类标注者进行偏好排序，奖励模型成为人类偏好的可扩展代理。**强化学习**（RL）在线优化策略，监督微调模型作为初始策略，奖励模型提供奖励，通过近端策略优化（PPO）的剪切机制约束策略变化避免奖励作弊，KL 散度惩罚防止遗忘基础能力。

RLHF 的特点是在线学习（模型行为在部署后持续优化）、隐性对齐（从相对信号而非显式标签中学习）、闭环反馈（用户反馈、内容审核、业务指标等信号可持续迭代）。在方法论层面，RLHF 也在持续演进：Anthropic 的 Constitutional AI 提出用 AI 反馈替代人类反馈（RLAIF），通过一组预设原则让模型自我评估和修正，大幅降低了人类标注成本 (Bai et al., 2022)；DPO（直接偏好优化）则完全绕开了奖励模型和强化学习，将偏好对齐简化为一个分类损失函数，已成为业界广泛采用的轻量替代方案 (Rafailov et al., 2023)。

值得注意的是，**对齐的信号来源不仅限于人类反馈**。在智能体系统中，情节记忆积累的高质量交互数据（尤其是人在回路介入后人类的决策模式）同样可以作为对齐信号，将验证有效的经验固化到模型参数中。这意味着对齐不是一次性的训练过程，而是"运行→积累→训练→改进"的终身学习闭环。第九章将详细讨论这一机制。

至此，第二部分的结论是清晰的：**后训练输出指令模型**。指令模型能够理解用户意图、生成有用回复，是大语言模型产品化的基础。

---

## 小结

本章的核心是两层递进的洞察。

**第一层：预训练缩放定律**。缩放定律揭示了性能与资源的幂律关系，规模化工程突破数据、算力、参数的物理瓶颈。预训练输出**续写模型**，具备强大的语言建模能力，但只会给定前文预测后续文本。

**第二层：后训练缩放定律**。三种对齐模式，结构化输出、指令跟随、在线环境对齐，分别服务于封闭式任务、开放式接口、动态偏好学习三类不同需求。后训练输出**指令模型**，能够理解用户意图、生成有用回复。

对齐模式的选择不是递进关系，而是根据应用场景灵活选择或组合。

这里有一个更深层的认识：**对齐本质上是适应环境的问题**。面向开放世界的智能体，需要在专业任务、文化、历史背景等各种环境下做出恰当的响应。"有用、诚实、无害"不是固定的价值标准，而是随环境而变的对齐目标：医疗场景下的"有用"与法律场景下的"有用"截然不同，不同文化背景下的"恰当"也各有边界。这正是在线对齐不可或缺的原因，离线训练无法穷尽所有环境，模型必须在部署后持续适应。

从续写模型到指令模型的转变，是大语言模型从技术能力走向产品能力的关键一步。

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

- **Scaling Laws: 神经语言模型的缩放定律**
  Scaling Laws for Neural Language Models. Kaplan, Jared et al.
  [原文](https://arxiv.org/abs/2001.08361)
- **Chinchilla: 训练计算最优的大语言模型**
  Training Compute-Optimal Large Language Models. Hoffmann, Jordan et al.
  [原文](https://arxiv.org/abs/2203.15556)
- **C4: 探索迁移学习的极限**
  Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Raffel, Colin et al.
  [原文](https://arxiv.org/abs/1910.10683)
- **The Pile: 用于语言建模的 800GB 多样化文本数据集**
  The Pile: An 800GB Dataset of Diverse Text for Language Modeling. Gao, Leo et al.
  [原文](https://arxiv.org/abs/2101.00027)
- **MoE: 超大神经网络：稀疏门控专家混合层**
  Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. Shazeer, Noam et al.
  [原文](https://arxiv.org/abs/1701.06538)
- **DeepSeek-V3: 技术报告**
  DeepSeek-V3 Technical Report. DeepSeek-AI.
  [原文](https://arxiv.org/abs/2412.19437)
- **FlashAttention: 快速且内存高效的精确注意力**
  Fast and Memory-Efficient Exact Attention with IO-Awareness. Dao, Tri et al.
  [原文](https://arxiv.org/abs/2205.14135)
- **LoRA: 大语言模型的低秩适配**
  LoRA: Low-Rank Adaptation of Large Language Models. Hu, Edward J. et al.
  [原文](https://arxiv.org/abs/2106.09685)
- **FLAN: 微调的语言模型是零样本学习者**
  Finetuned Language Models Are Zero-Shot Learners. Wei, Jason et al.
  [原文](https://arxiv.org/abs/2109.01652)
- **InstructGPT: 使用人类反馈训练语言模型遵循指令**
  Training language models to follow instructions with human feedback. Ouyang, Long et al.
  [原文](https://arxiv.org/abs/2203.02155)
- **RLHF: 基于人类偏好的深度强化学习**
  Deep Reinforcement Learning from Human Preferences. Christiano, Paul F. et al.
  [原文](https://arxiv.org/abs/1706.03741)
- **Constitutional AI: 来自 AI 反馈的无害性对齐**
  Constitutional AI: Harmlessness from AI Feedback. Bai, Yuntao et al.
  [原文](https://arxiv.org/abs/2212.08073)
- **DPO: 直接偏好优化**
  Direct Preference Optimization: Your Language Model is Secretly a Reward Model. Rafailov, Rafael et al.
  [原文](https://arxiv.org/abs/2305.18290)
- **PPO: 近端策略优化算法**
  Proximal Policy Optimization Algorithms. Schulman, John et al.
  [原文](https://arxiv.org/abs/1707.06347)
- **Megatron-LM: 使用模型并行训练数十亿参数的语言模型**
  Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. Shoeybi, Mohammad et al.
  [原文](https://arxiv.org/abs/1909.08053)
- **ZeRO: 训练万亿参数模型的内存优化**
  ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. Rajbhandari, Samyam et al.
  [原文](https://arxiv.org/abs/1910.02054)
- **GPT-3: 语言模型是少样本学习者**
  Language Models are Few-Shot Learners. Brown, Tom et al.
  [原文](https://arxiv.org/abs/2005.14165)
- **涌现能力: 大语言模型的涌现能力**
  Emergent Abilities of Large Language Models. Wei, Jason et al.
  [原文](https://arxiv.org/abs/2206.07682)
- **知识蒸馏: 提炼神经网络中的知识**
  Distilling the Knowledge in a Neural Network. Hinton, Geoffrey et al.
  [原文](https://arxiv.org/abs/1503.02531)
- **Adapter: 面向 NLP 的参数高效迁移学习**
  Parameter-Efficient Transfer Learning for NLP. Houlsby, Neil et al.
  [原文](https://arxiv.org/abs/1902.00751)
- **vLLM: 基于 PagedAttention 的高效 LLM 推理服务**
  Efficient Memory Management for Large Language Model Serving with PagedAttention. Kwon, Woosuk et al.
  [原文](https://arxiv.org/abs/2309.06180)