---
title: 深入大模型系统札记：第3章 模型预训练技术基础
description: "监督学习通过结构压缩解决IID泛化问题，深度学习自动学习可泛化表示，迁移学习通过知识复用突破分布外问题，形成预训练-微调的现代范式。"
draft: false
---

> **本章概要**：本章的核心是两层递进的洞察。**第一层：如何解决独立同分布（IID）问题**，监督学习通过结构压缩从有限样本泛化到同分布的无限样本，深度学习通过层次化神经网络自动学习可泛化的表示。**第二层：如何解决分布外（OOD）问题**，迁移学习将源任务上学到的通用表示复用到新任务，形成"预训练→微调"的现代范式。

---

这一章是全书中最"基础"的一章，讨论的是预训练范式的方法论起源。你可能觉得监督学习和迁移学习是"老生常谈"，但如果你真正理解了这些基础，你就不会问"为什么模型会幻觉"或"为什么微调有时有效有时无效"这类问题。

**很多对大模型的误解，源于对这些基础问题的忽视。**

第一章提到"预训练范式解决通用知识获取问题"，本章讨论这一范式的方法论基础。

**第一层：如何解决独立同分布（IID）问题**。我们只有有限的训练样本，但希望模型能在无限的未知样本上做出准确预测。监督学习和深度学习共同解决了这个问题：前者通过结构压缩实现泛化，后者通过自动学习表示提升泛化能力。

**第二层：如何解决分布外（OOD）问题**。监督学习依赖一个关键假设：训练样本与测试样本来自同一分布。但现实中分布漂移无处不在。迁移学习突破了这一限制，使模型能够将源任务上学到的知识复用到不同分布的新任务。

---

## 第一部分：如何解决 IID 问题

独立同分布假设下，如何从有限样本泛化到无限样本？

### 一、监督学习：结构压缩

我们只有有限的训练样本，但希望模型能在无限的未知样本上做出准确预测。这可能吗？

监督学习的核心洞察是：不要记住样本，而要**发现并压缩结构性规律**。牛顿没有记住所有天体的运动轨迹，而是用一个参数 G 总结了万有引力定律。监督学习做的是同样的事：**有限样本 → 结构压缩 → 全样本预测**。

但"压缩"不等于"泛化"。只有当模型学到的是跨样本重复出现的模式，而非训练集中的偶然特征，它才具备真正的泛化能力。为此需要在"拟合数据"和"简化结构"之间寻求平衡，通过正则化、Dropout 等策略控制模型复杂度。

监督学习依赖一个关键假设，即**独立同分布（IID）**。训练样本与测试样本必须来自同一分布，模型在训练集上学到的规律才能推广到测试集。这个假设定义了监督学习的能力边界。

### 二、深度学习：自动表示

传统机器学习依赖人工设计的特征，但人工特征难以捕捉复杂的高阶模式。能否让模型自己学习"什么是好的特征"？

深度学习通过**层次化的神经网络**自动构建特征表示。深度神经网络的核心思想是通过多层非线性映射，将原始特征逐步转化为越来越抽象、稳定、紧凑的中间表示：浅层学习边缘、纹理等基础特征，中间层组合成部件，深层聚合为语义概念。

**卷积神经网络（CNN）**是视觉领域中表示学习的典型范式。其核心在于引入了基于物理世界“局部连续性”和“平移不变性”的**归纳偏置**：这不仅提升了参数利用效率，也增强了模型的结构化迁移能力。此后，这一“以归纳偏置驱动建模”的方法逐渐成为通用范式——例如，推荐系统依赖协同过滤偏置，语言模型则利用序列建模偏置。在此基础上，随着大规模数据集（如 ImageNet）的构建，以及深层网络训练关键难题（如梯度消失与爆炸）通过 ResNet 等方法得到缓解，视觉领域进入了预训练模型时代。CNN 通过卷积层提取局部模式、池化层压缩空间信息，并以多层堆叠逐步形成“边缘 → 部件 → 物体”的层次化表示。

至此，第一部分的结论是清晰的：**监督学习 + 深度学习解决了 IID 问题**。但这一切都依赖同分布假设，训练集和测试集必须来自同一分布。现实中分布漂移无处不在，怎么办？

---

## 第二部分：如何解决 OOD 问题

分布漂移无处不在，在医学影像上训练的模型需适应自然图像，白天训练的模型需部署到夜间场景。为每个新任务从零训练成本极高。

### 三、迁移学习：知识复用

迁移学习的核心思想是**复用已有知识**。在大规模源任务上预训练的模型已经学到了通用的结构化表示（如视觉中的边缘、纹理、形状），这些表示对新任务同样有价值。通过复用这些表示，可以降低对标注数据的需求、加速训练收敛、增强泛化能力。

两种策略：一是**冻结主干网络**，将预训练模型作为固定的特征提取器，只训练新的任务头；二是**微调**，在目标任务上更新部分或全部参数，使模型"专业化"。选择哪种策略取决于目标任务的数据量和与源任务的相似度。

这种"**预训练→微调**"的双阶段框架，已成为现代深度学习的标准范式。它将"学习通用知识"与"解决具体问题"在结构上明确分离，模型不再是解决特定问题的孤立工具，而是能够积累、泛化并复用知识的智能体系。

至此，第二部分的结论是清晰的：**迁移学习解决了 OOD 问题**。预训练模型学到的通用表示可以跨分布复用，使模型能够适应与训练数据不同分布的新任务。

---

## 小结

本章的核心是两层递进的洞察。

**第一层：如何解决 IID 问题**。监督学习通过结构压缩从有限样本泛化到同分布的无限样本。深度学习通过层次化神经网络自动学习可泛化的表示。但这一切都依赖同分布假设。

**第二层：如何解决 OOD 问题**。迁移学习将源任务上学到的通用表示复用到新任务，突破了同分布的限制。"预训练→微调"的双阶段框架成为现代深度学习的标准范式。

这条主线为后续的预训练语言模型奠定了方法论基础。

如果你是 AI 应用开发工程师，这一章给你一个核心检查项：**永远要问"训练数据和部署环境是否同分布"**。很多线上事故的根源，都是分布漂移，不是模型不行，是你喂错了数据。

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

- **Backpropagation: 通过反向传播误差学习表示**
  Learning representations by back-propagating errors. Rumelhart, David E., Hinton, Geoffrey E. and Williams, Ronald J.
  [原文](https://www.nature.com/articles/323533a0)
- **Deep Learning: 深度学习综述**
  Deep Learning. LeCun, Yann, Bengio, Yoshua and Hinton, Geoffrey.
  [原文](https://www.nature.com/articles/nature14539)
- **AlexNet: 使用深度卷积神经网络进行 ImageNet 分类**
  ImageNet Classification with Deep Convolutional Neural Networks. Krizhevsky, Alex et al.
  [原文](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- **Dropout: 防止神经网络过拟合的简单方法**
  Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Srivastava, Nitish et al.
  [原文](https://jmlr.org/papers/v15/srivastava14a.html)
- **ResNet: 深度残差学习用于图像识别**
  Deep Residual Learning for Image Recognition. He, Kaiming et al.
  [原文](https://arxiv.org/abs/1512.03385)
- **Batch Normalization: 通过减少内部协变量偏移加速深度网络训练**
  Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. Ioffe, Sergey and Szegedy, Christian.
  [原文](https://arxiv.org/abs/1502.03167)
- **Adam: 随机优化方法**
  Adam: A Method for Stochastic Optimization. Kingma, Diederik P. and Ba, Jimmy.
  [原文](https://arxiv.org/abs/1412.6980)
- **Hubel & Wiesel: 猫视觉皮层的感受野、双眼交互和功能架构**
  Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. Hubel, David H. and Wiesel, Torsten N.
  [原文](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/)
- **LeNet-5: 基于梯度的学习应用于文档识别**
  Gradient-based learning applied to document recognition. LeCun, Yann et al.
  [原文](https://ieeexplore.ieee.org/document/726791)
- **ImageNet: 大规模分层图像数据库**
  ImageNet: A Large-Scale Hierarchical Image Database. Deng, Jia et al.
  [原文](https://ieeexplore.ieee.org/document/5206848)
- **特征可迁移性: 深度神经网络中的特征迁移能力研究**
  How transferable are features in deep neural networks? Yosinski, Jason et al.
  [原文](https://arxiv.org/abs/1411.1792)
- **彩票假设: 寻找稀疏、可训练的神经网络**
  The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. Frankle, Jonathan and Carbin, Michael.
  [原文](https://arxiv.org/abs/1803.03635)
