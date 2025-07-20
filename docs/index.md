---
title: 首页
hide:
  - navigation
  - toc
---

## 本站定位

欢迎访问！本网站以「AI/CS 学科基础笔记」与「工程实践技术文章」为基础，旨在构建一个 **开放知识社群 (Open Wiki Community)**。内容按照 Markdown 撰写，站点采用 MkDocs 编译，云端基于 Aliyun OSS 部署。如果您觉得内容不错，欢迎⭐！

*[Wiki]: 一种允许一群用户用简单的描述来创建和连接一组网页的社会计算系统。

## 共建社区

若您有 **意见或建议**，欢迎参与贡献！贡献者名单将会出现在对应页面的底部，也会给您的 GitHub 账户累计贡献积分。详细的贡献方式：

1. 点击网站右上角的小猫娘前往 GitHub 仓库；
2. 点击右上角 Fork 按钮；
3. 回到原来的网站页面点击右上角铅笔按钮；
4. 编辑内容后向我发起 Pull Request。

当然您也可以将仓库克隆至本地，按照 `.github/workflows/bot.yml` 文件指引安装构建文档所需的依赖环境，完成修改/新增后，推送至仓库并向我发起 Pull Request。

## 行文规范

本站点行文格式主要参考 OI Wiki 格式手册 [^oi-format]。其中：

- 标题：单页文章标题为 H2 至 H3，低于 H3 等级的标题不应再出现，可以采用段首加粗的形式；
- 链接：所有内链采用 `标题/章节` 的链接格式，所有外链采用 `标题/章节 | 作者/组织 - (顶级域名)` 的链接/引用格式。

[^oi-format]: [格式手册 | OI Wiki - (oi-wiki.org)](https://oi-wiki.org/intro/format/)

## 站点地图

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

    %% 跳转链接
    click 高等数学 "./base/math/advanced-math/"
    click 线性代数 "./base/math/linear-algebra/"
    click 概率统计 "./base/math/probability-and-statistics/"
    click 最优化方法 "./base/math/optimization-method/"

    click 面向对象程序设计 "./base/cs/object-oriented-programming/"
    click 数字逻辑电路 "./base/cs/digital-logic-circuit/"
    click 计算机系统基础 "./base/cs/computer-system-basic/"
    click 数据库 "./base/cs/database/"
    click 数据结构与算法 "./ds-and-algo/"
    click 操作系统 "./base/cs/operating-system/"
    click 计算机组成 "./base/cs/computer-organization/"
    click 计算机网络 "./base/cs/computer-network/"

    click 数字图像处理 "./base/ai/digital-image-processing/"
    click 语音信号处理 "./base/ai/speech-signal-processing/"
    click 机器学习 "./base/ai/machine-learning/"
    click 深度学习 "./base/ai/deep-learning/"
    click 数据挖掘 "./base/ai/data-mining/"
    click 计算机视觉 "./base/ai/computer-vision/"
    click 自然语言处理 "./base/ai/natural-language-processing/"
    click 知识图谱 "./base/ai/knowledge-graph/"
```

<caption> 拓扑图 1. AI/CS 学科基础笔记 </caption>

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
  
    %% 跳转链接
    click data_science "./data-science/"
    click ds_and_algo "./ds-and-algo/"
    click dev_tools "./dev-tools/"
    click front_end "./front-end/"
    click back_end "./back-end/"
    click dev_ops "./operation/"
```

<caption> 拓扑图 2. 工程实践技术文章 </caption>

</div>
