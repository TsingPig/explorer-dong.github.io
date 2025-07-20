<div align="center">
    <a href="https://wiki.dwj601.cn/">
        <img src="./overrides/assets/cover-image.png" alt="Site Cover Image" />
    </a>
</div>

本网站以「AI/CS 学科基础笔记」与「工程实践技术文章」为基础，旨在构建一个 **开放知识社群 (Open Wiki Community)**。内容按照 Markdown 撰写，站点采用 MkDocs 编译，云端基于 Aliyun OSS 部署。如果您觉得内容不错，欢迎⭐！访问链接：<https://wiki.dwj601.cn/>。

## 站点预览 / Site Preview

<div align="center">

```mermaid
flowchart LR
    %% 实体定义
    面向对象程序设计(面向对象程序设计)
    数据结构与算法(数据结构与算法)
    数字逻辑电路(数字逻辑电路)
    计算机系统基础(计算机系统基础)
    数据库(数据库)
    操作系统(操作系统)
    计算机组成(计算机组成)
    计算机网络(计算机网络)

    %% 关系定义
    面向对象程序设计 --> 数据结构与算法 --> 计算机系统基础
    数字逻辑电路 --> 计算机系统基础 --> 数据库 & 操作系统 & 计算机组成
    计算机组成 & 操作系统 --> 计算机网络

    %% 实体定义
    数字图像处理(数字图像处理)
    机器学习(机器学习)
    深度学习(深度学习)
    数据挖掘(数据挖掘)
    自然语言处理(自然语言处理)
    计算机视觉(计算机视觉)
    语音信号处理(语音信号处理)
    知识图谱(知识图谱)

    %% 关系定义
    机器学习 --> 数据挖掘
    深度学习 --> 语音信号处理 & 自然语言处理 & 计算机视觉
    自然语言处理 --> 知识图谱
    数字图像处理 --> 计算机视觉

    %% 实体定义
    高等数学(高等数学)
    线性代数(线性代数)
    概率统计(概率统计)
    最优化方法(最优化方法)

    %% 关系定义
    高等数学 & 线性代数 & 概率统计 --> 最优化方法 --> 机器学习 & 深度学习
```

<caption> 拓扑图 1. AI/CS 学科基础笔记 </caption>

</br>
</br>

```mermaid
graph LR
    %% 实体定义
    data_science(数据科学专栏)
    ds_and_algo(数据结构与算法专栏)
    dev_tools(开发工具专栏)
    front_end(前端开发专栏)
    back_end(后端开发专栏)
    dev_ops(运维开发专栏)

    %% 关系定义
    ds_and_algo --> data_science & back_end
    dev_tools --> front_end & back_end & dev_ops & data_science
```

<caption> 拓扑图 2. 工程实践技术文章 </caption>

</div>  

## 贡献名单 / Contributors

<a href="https://github.com/Explorer-Dong/wiki/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Explorer-Dong/wiki" />
</a>

## 星标历史 / Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Explorer-Dong/wiki&type=Date)](https://www.star-history.com/#Explorer-Dong/wiki&Date)
