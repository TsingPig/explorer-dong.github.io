---
title: Git PR 解读
---

早闻 Git 的 PR 功能，趁着端午放假在家便来实战一波，做个小小的记录。

## 原理

首先聊聊 PR 名字的由来。PR 即 `pull request`，顾名思义是「拉取请求」的意思。PR 根本逻辑是将原项目 fork 到自己的仓库后进行修改，然后向原项目作者发起 pull request 请求，直到 merge 到原项目或被拒绝。这样的一个工作流程为什么被叫做「拉取请求」呢？依稀记得之前在某乎上看到过相关文章，程序员届大概有两个流派，主要从这个工作流的执行主体角度来理解。对于原项目作者，需要考虑修改者的请求是否适合合并，就是 pull 的过程；相对的，修改者发送修改合并的请求是一个 push 的过程。因此我们从原项目作者的角度来理解这个名称就有据可寻了。下附一张图解：

![Pull Request 工作逻辑](https://cdn.dwj601.cn/images/202406091607490.svg)

## 实操

### fork 目标仓库

进入目标仓库，点击右上角的 fork 按钮进行 fork

![fork 目标仓库](https://cdn.dwj601.cn/images/202406091618430.png)

### clone 至本地

进入自己的仓库，找到对应的项目并复制克隆链接

![clone 至本地](https://cdn.dwj601.cn/images/202406091620622.png)

### 编辑内容并版本管理

!!! warning
    如果在进行 clone 时，只克隆了默认的主分支，我们需要按照原项目规定的 PR 分支创建同名的分支。比如此处：原项目只允许在 develop 分支上进行 PR，因此我们在默认只拉取了 master 分支的前提下，需要自己在本地创建一个 develop 分支，并在此基础之上进行内容的修改与编写。

我们将需要修改的内容完善后，进行 add 和 commit 操作即可。

### push 至仓库

此处将已经版本管理好的内容 push 到自己刚才 fork 出来的仓库即可。上述示例中需要 push 到 develop 分支上。

### 发起 PR 请求

在选择合适的分支后，点击 Contribute 按钮即可看到 Open pull request 选项，点击即可发起 PR 请求：

![发起 PR 请求](https://cdn.dwj601.cn/images/202406091634960.png)

## 参考

[吵疯了，Pull Request到底是个啥 - 知乎](https://zhuanlan.zhihu.com/p/347918608)

[百大技术UP带你玩转 Pull Request - bilibili](https://www.bilibili.com/video/BV11d4y1H74N/)
