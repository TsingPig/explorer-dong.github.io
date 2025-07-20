---
title: 知识图谱导读
status: new
---

![知识图谱研究分类](https://cdn.dwj601.cn/images/20250529081747782.png)

本文记录知识图谱的学习笔记。章节安排参考 *A Survey on Knowledge Graphs: Representation, Acquisition, and Applications* [^arch-ref] 提出的结构：

1. [知识获取](./kg-acquisition.md)：从非结构化/半结构化数据中提取结构化数据，得到形如（实体，关系，属性）这样的三元组，并融合成最终的知识图谱 [^trans]；
2. [知识表示](./kg-representation.md)：根据某种规则将知识图谱嵌入到向量空间；
3. [知识推理](./kg-aware-application.md)：在构建出的知识图谱上进行推理任务；
4. [时序图谱](./temporal-kg.md)：结合了时许信息的知识图谱。

[^arch-ref]: Ji, Shaoxiong , et al. "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications." IEEE Transactions on Neural Networks and Learning Systems PP.99(2021).
[^trans]: [Translate 模型汇总：TransE, TransH, TransR, TransD - (mp.weixin.qq.com)](https://mp.weixin.qq.com/s/2YbfL_1_SyM4wNozyaj4lw)

部分论文/工具参考内容如下：

- [Knowledge-Graph-Tutorials-and-Papers | heathersherry - (github.com)](https://github.com/heathersherry/Knowledge-Graph-Tutorials-and-Papers)
- [A systematic course about knowledge graph | SEU - (github.com)](https://github.com/npubird/KnowledgeGraphCourse)
