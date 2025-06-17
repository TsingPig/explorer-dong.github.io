---
title: 视频理解
---

本文介绍的视频分类任务主要用于以视频为单位 [^ref] 的动作识别任务，即给定一个视频，输出视频中包含的动作类别。

[^ref]: [Video Analysis相关领域解读之Action Recognition(行为识别) | 林天威 - (zhuanlan.zhihu.com)](https://zhuanlan.zhihu.com/p/26460437)

## 动作识别

### 基本概念

**动作识别定义**。

假设时序长度为 $T$ 的视频表示为 $X = x_1, x_2, \cdots , x_T$，其中 $x_i$ 表示第 $i$ 帧图像。定义动作标签集合 $c\in C$，其中每个 $c$ 表示一个特定的动作类别。

动作识别 (Action Recognition) 任务可以看作是一个分类问题，其目标是找到一个映射函数 $f$，该函数将 $X$ 映射到动作标签集合 $C$ 中的一个标签：$f: X \to C$。

**数据集**。

![动作识别经典数据集](https://cdn.dwj601.cn/images/20250603091413644.png)

**评价指标**。

Clip Hit@k：从视频中提取多个片段 (clip)，每个片段独立预测，统计所有 clip 的 Top-k 准确率。

Video Hit@k：将同一个视频的多个 clip 的预测结果进行聚合，统计所有 video 的 Top-k 准确率。

*注：Top-k 表示模型输出 k 次的结果中是否存在正确答案，因此我们平时常说的准确率其实是 Top-1 指标。

### 传统方法

一种做法是将图像的特征融合为视频的特征 [^early-cnn] ，直接的融合方式可分为以下三种：

[^early-cnn]: Karpathy A, Toderici G, Shetty S, et al. Large-scale video classification with convolutional neural networks[C]//Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2014: 1725-1732.

- 早期融合 (Early fusion)
- 晚期融合 (Late fusion)
- 缓慢融合 (Slow fusion)

![将图像的特征融合为视频的特征](https://cdn.dwj601.cn/images/20250527125604034.png)

![实验结果](https://cdn.dwj601.cn/images/20250527125838198.png)

缓慢融合方式在网络内部逐渐融合相邻帧的信息，取得了最好的效果。但融合了时序信息的效果并没有显著优于基于静态图像的效果，说明基于 2D 卷积的网络难以有效提取运动信息。

### 引入光流的网络

为了让模型捕捉到不同帧之间的关系，需要引入物体运动的信息，光流就是物体运动的一种量化表示方式。

**光流的定义**。TODO

**双流网络**。

为了让网络更好地利用运动信息，双流网络 (Two-stream convolutional network) [^two-stream] 引入了光流 (Optical flow) 信息。

[^two-stream]: Simonyan K, Zisserman A. Two-stream convolutional networks for action recognition in videos[J]. Advances in neural information processing systems, 2014, 27.

![双流网络结构示意图](https://cdn.dwj601.cn/images/20250527130316523.png)

![实验结果](https://cdn.dwj601.cn/images/20250527130526231.png)

实验结果表明：

- 基于手工特征的 IDT 在当时取得了最高的准确率；
- 缓慢融合方式距离 IDT 差距明显；
- 双流网络的效果虽然弱于 IDT，但是充分验证了深度学习用于视频动作识别的可行性。

**基于双流网络的改进**。

如何设计池化层以理解更长时序的动作。论文阅读 [^two-stream-approve1]。

如何融合双流信息以增强模型的性能。论文阅读 [^two-stream-approve2]。

如何设计网络使能够处理更长时序的视频：时序片段网络 (Temporal Segment Network, TSN) [^tsn] 将视频划分为不同的片段，然后综合各个片段的分类结果。

[^two-stream-approve1]: Yue-Hei Ng J, Hausknecht M, Vijayanarasimhan S, et al. Beyond short snippets: Deep networks for video classification//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 4694-4702.
[^two-stream-approve2]: Feichtenhofer C, Pinz A, Zisserman A. Convolutional two-stream network fusion for video action recognition//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 1933-1941.
[^tsn]: Wang L, Xiong Y, Wang Z, et al. Temporal segment networks: Towards good practices for deep action recognition[C]//European conference on computer vision. Springer, Cham, 2016: 20-36.

### 基于 3D 卷积的网络

光流信息的计算与存储开销极大，人们开始尝试使用 3D 卷积来完成动作识别任务。

![2D 卷积 vs. 3D 卷积](https://cdn.dwj601.cn/images/20250603081454178.png)

**C3D**。C3D [^c3d] 直接将 VGG 的卷积操作从 $3\times 3$ 拓展到了 $3\times 3\times 3$，在 Sport-1M 数据集上进行训练。

[^c3d]: Tran D, Bourdev L, Fergus R, et al. Learning spatiotemporal features with 3d convolutional networks[C]//Proceedings of the IEEE international conference on computer vision. 2015: 4489-4497.

![C3D 模型架构](https://cdn.dwj601.cn/images/20250603081959259.png)

特点：

- C3D 直接在 Sport-1M 数据集上训练，取得的效果要好于缓慢融合网络；
- C3D 在更大的数据集上预训练，再进行微调，则可以取得更好的效果；
- C3D 的训练时间非常长，难以继续拓展。

![实验结果](https://cdn.dwj601.cn/images/20250603081944380.png)

**I3D**。使用预训练模型初始化网络的参数可以增强模型的效果并降低训练难度，因此 I3D [^i3d] 提出了膨胀 3D 卷积 (Inflated 3D convolution)。

[^i3d]: Carreira J, Zisserman A. Quo vadis, action recognition? a new model and the kinetics dataset[C]//proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 6299-6308.

![I3D 模型架构](https://cdn.dwj601.cn/images/20250603081936376.png)

![实验结果](https://cdn.dwj601.cn/images/20250603081922787.png)

### 基于 Transformer 的网络

TimeSformer [^timesformer] 是第一个尝试将 ViT [^vit] 引入到动作识别任务的工作。如何将应用于 2D 图像的 ViT 迁移到视频领域的动作识别任务？

[^timesformer]: Bertasius G, Wang H, Torresani L. Is space-time attention all you need for video understanding?[C]//ICML. 2021, 2(3): 4.
[^vit]: [轻松理解 ViT(Vision Transformer) 原理及源码 | 000_error - (zhuanlan.zhihu.com)](https://zhuanlan.zhihu.com/p/640013974)

<video controls src="https://cdn.dwj601.cn/videos/ViT.mp4"></video>

类似于之前的工作，TimeSformer 对于如何将 ViT 拓展到视频数据进行了大量的探索。

![TimeSformer 的探索](https://cdn.dwj601.cn/images/20250603092020812.png)

![可视化结果](https://cdn.dwj601.cn/images/20250603092041738.png)

![实验指标](https://cdn.dwj601.cn/images/20250603092057978.png)

用 Transformer 的好处：

- Transformer 的全局建模能力使得相关模型能够更好地利用时序上下文信息，更易于应用到长时序的视频；
- 相较于 CNN「局部性、平移不变性」的归纳偏执 (Inductive Bias)，以及 RNN「时序依赖性」的归纳偏置，Transformer「注意力机制」的归纳偏置更弱，更易于拓展成多模态模型。

## 时序动作检测

时序动作检测可以理解为目标检测的视频版。根据监督信息的不同，时序动作检测任务可以细分为全监督预测和弱监督预测，前者提供视频中每一帧的动作类别信息，后者仅提供整个视频的动作类别信息。

### 基本概念

**任务定义**。假设数据集 $X$ 中包含 $N$ 个视频，表示为 $X=\{X_i\}_{i=1}^N$，其中每个视频对应一个监督信息 $y_i$。

- 训练阶段，利用视频和对应的监督信息使模型能够准确识别每个类别的动作模式；
- 测试阶段，模型根据输入的无修剪视频，给出时序动作检测结果。第 $i$ 个视频的预测结果表示为 $p_i=\{t_j^s,t_j^e,c_j,s_j\}_{j=1}^M$，即模型给当前视频预测出了 $M$ 个动作，其中 $t_j^s,t_j^e,c_j,s_j$ 分别表示第 $i$ 个视频中第 $j$ 个动作的开始时间、结束时间、动作类别和置信度。

**评价指标**。与目标检测类似，时序动作检测也可以采用平均精度均值 (mean Average Precision, mAP) 作为评价指标。与 [目标检测的 mAP](./object-detection.md/#性能度量) 不同的是，这里的 IoU 变成了 tIoU。各种 IoU 的计算方法如下图所示：

![各种 IoU 的计算方法](https://cdn.dwj601.cn/images/20250617104551490.png)

## 时序动作分割

### 基本概念

**任务定义**。

假设数据集 𝒳 中包含 𝑁 个视频，表示为 𝒳 = {𝑋𝑖 }𝑖 𝑁 =1，其中每个视频对应一个监督信息 𝑌𝑖。

测试阶段，模型根据输入的视频，给出无修剪视频的时序动作分割结果，表示为𝑃𝑖 = {𝑝𝑗 }𝑗 𝐿 =1 ，表示第 𝑖 帧的动作类别。

根据监督信息的不同，时序动作分割任务可以细分为：

➢ **全监督**：提供视频中每一帧的动作类别信息；

➢ 弱监督：仅提供整个视频的部分信息：

• 转录（Transcripts）：视频中动作的序列；

• 动作集（Action set）：视频中动作的集合；

• 单帧标签（Single-frame label）：视频中每个动作中随机一帧的动作类别信息；

• 活动标签（Activity label）：整个视频的任务类别信息。

➢ 无监督：不提供任何监督信息。严格来说，无监督设置也属于弱监督设置。

**评价指标**。

帧级准确率（Frame-wise accuracy）

𝑎𝑐𝑐 = # 𝑜𝑓 𝑐𝑜𝑟𝑟𝑒𝑐𝑡 𝑓𝑟𝑎𝑚𝑒𝑠/# 𝑜𝑓 𝑎𝑙𝑙 𝑓𝑟𝑎𝑚𝑒

当不同动作的帧数的分布不平衡时，准确率可能存在问题。

➢ 准确率类似的模型可能生成不同质量的分割结果。

➢ 准确率无法反映**过度分割问题**（Over-segmentation problem）

过度分割问题是时序动作分割任务中的关键问题，指的是模型将一个动作片段错误的划分为多个片段的情况。在实际应用中，过度分割问题会严重影响模型的实际使用效果。

![过度分割的典型情况](https://cdn.dwj601.cn/images/20250617090355312.png)

片段级编辑分数（Segmental edit score）

编辑距离又称作莱文斯坦距离，常在NLP领域衡量两个字符串之间的差异程度，表示将一个单词更改为另一个单词所需的最少编辑操作（替换/插入/删除）的次数。

在时序动作分割任务中，编辑分数越大则两个动作序列越相似。

片段级F1分数（Segmental F1-Score）

F1 score 是精确度（Precision）和召回率（Recall）的调和平均数，综合反映了模型的性能。F1@k 表示时序分割模型在 tIoU的阈值为 k 时的F1 score。

对于 F1@k，假设模型输出的片段为 {𝑝𝑖 }𝑖 𝑀 =1，真实片段表示为 {𝑔𝑗 }𝑗 𝑁 =1，对于 𝑝𝑖：

➢ 在时序上计算与 {𝑔𝑗 }𝑗 𝑁 =1 的交并比 tIoU；

➢ 找出对应最大 tIoU 的 𝑔𝑗；

➢ 如果 𝑝𝑖 与 𝑔𝑗 的 tIoU 不小于 k 且 𝑔𝑗 未与任何预测片段匹配，则 tp+=1 且 𝑔𝑗 与 𝑝𝑖

匹配成功；

➢ 否则 fp+=1；

◼ 循环结束后，未匹配成功的 𝑔𝑗 的数量表示为 fn。

◼ 时序动作分割任务中通常采用 F1@{10，25，50}三种设置。

