---
title: Git 常用命令
---

Git 是一款版本管理软件，适用目前绝大多数操作系统；GitHub 是一个代码托管平台，与 Git 没有任何关系。只不过 Git 可以基于 GitHub 进行分布式云存储与交互，因此往往需要结合二者从而达到相对良好的 Teamwork 状态。

本文记录 Git 的常用命令。完整命令见官方文档：[https://git-scm.com/book/zh/v2](https://git-scm.com/book/zh/v2)。

下面的图和表概述了 Git 的整个逻辑框架。

Git 的三个区域：

|     工作区     |     暂存区     |     仓库区     |
| :------------: | :------------: | :------------: |
| 内容的编辑区域 | 变动的保存区域 | 版本的生成区域 |

![Git 的基本命令](https://cdn.dwj601.cn/images/202402271037959.png)

下面将会从 5 个方面展开 Git 的命令。

## 查看

### 查看状态

```bash
# 查看当前文件状态
git status
```

### 查看日志

```bash
# 从当前版本开始查询 commit 日志
git log
```

```bash
# 查看所有 commit 日志
git reflog
```

### 查看差异

```bash
# 查看工作区与暂存区的差异
git diff <FileName>

# 查看两个区域所有文件的差异
git diff
```

```bash
# 查看暂存区与仓库区的差异
git diff --cached <FileName>

# 查看两个区域所有文件的差异
git diff --cached
```

## 配置

### 初始化

```bash
# 初始化仓库
git init
```

### 查看配置

```bash
# 查看 git 配置信息
git config --list
```

```bash
# 查看 git 用户名、密码、邮箱的配置
git config user.name
git config user.password
git config user.email
```

```bash
# 查看代理
git config --global --get http.proxy
git config --global --get https.proxy
```

### 编辑配置

1）邮箱、密码、用户名

```bash
# 配置（修改） Email & Pwd & Username （局部）
git config user.name "xxx"
git config user.password "xxx"
git config user.email "xxx@xxx.com"

# 配置（修改） Email & Pwd & Username （全局）
git config --global user.name xxx
git config --global user.password xxx
git config --global user.email "xxx@xxx.com"
```

2）VPN

```bash
# 配置代理
git config --global http.proxy 127.0.0.1:<VpnPort>
git config --global https.proxy 127.0.0.1:<VpnPort>

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

3）远程

```bash
# 连接远程服务器
git remote add <RemoteName> https://github.com/用户名/仓库名.git

# 查看所有连接的远程
git remote -v

# 修改远程别名
git remote rename <OldRemoteName> <NewRemoteName>

# 修改远程 URL
git remote set-url <RemoteName> <NewURL>

# 增加远程 push 的仓库
git remote set-url --add github https://gitee.com/idwj/idwj.git

# 删除远程
git remote rm <RemoteName>
```

## 迭代

### 工作区到暂存区

```bash
# 工作区到暂存区（单文件）
git add <FileName>

# 工作区到暂存区（全部变动文件）
git add .
```

### 暂存区到仓库区

```bash
# 暂存区到仓库区
git commit -m '<Comment>'
```

### 仓库区到服务器

```bash
# 仓库区到云服务器（常规方法）
git push <RemoteName> <BranchName>

# 仓库区到云服务器（初始配置仓库推送默认地址）
git push -u <RemoteName> <BranchName>

# 仓库区到云服务器（已配置默认推送地址后）
git push
```

```bash
# 强制覆盖推送
git push --force <RemoteName> <BranchName>
```

### 服务器到本地

一键克隆整个项目

```bash
# 从服务器克隆仓库
git clone https://github.com/<UserName>/<ProjectName>.git <ProjectName>
```

远程更新，本地未更新（方法一）

```bash
# 抓取复制远程代码
git fetch <RemoteName> <BranchName>

# 更新本地分支
git merge <BranchName>
```

远程更新，本地未更新（方法二）

```bash
# 直接使用 pull 命令，等价于上述方法一的两步。即先抓取，后合并分支
git pull
```

远程更新，本地也更新

```bash
# 在将远程代码与本地合并后，再将个人修改的代码推送到远程
git pull
git push
```

## 回溯

### 取消版本管理

注意：该操作只能取消当前版本的文件管理，如果需要删除某个文件在所有版本中的管理，需要用到 [进阶命令](./git-tips.md/#将文件从-git-系统中彻底删除)。

```bash
# 希望某些文件取消版本管理，但是依然保留在工作区
git rm --cached <FileName>
git commit -m 'remove xxx file'
git push
# 之后在 .gitignore 中增加上述 <FileName>
```

- `--cached` 参数表示只删除已经 add/commit 的内容，原始文件保留，如果不加则会将原始文件也一并删除；
- `-r` 参数表示针对文件夹进行递归删除。

### 工作区到上一个版本

```bash
# 取消新文件的管理 or 将修改文件回溯到上一个版本的初始状态
git checkout -- <FileName>
```

### 暂存区到工作区状态

```bash
# 取消 add（一个文件），默认为 --mixed 模式，即保存修改但是从暂存区到工作区
git reset <FileName>

# 取消 add（全部文件），默认为 --mixed 模式，即保存修改但是从暂存区到工作区
git reset .
```

### 仓库区到暂存区状态

```bash
# 移动 HEAD 指针到指定的版本
git reset '<commit_id>'        # 默认为 --mixed，将指定版本与暂存区全部合并到工作区，暂存区清空
git reset --soft '<commit_id>' # 【更推荐】工作区不变，只是将指定版本合并到暂存区
git reset --hard '<commit_id>' # 【不推荐】工作区与暂存区全部被指定版本覆盖

# 取消上一次 comment 并进入 vim 编辑模式
git commit --amend
```

### 取消云端更改

在本地强制推送即可：

```bash
git push --force <RemoteName> <BranchName>
```

彻底删除某些文件在

取消 **当前版本** 某文件（夹）的版本管理

```bash
# 希望某些文件取消版本管理，但是依然保留在工作区
git rm --cached <FileName>
git commit -m 'remove xxx file'
git push
在 .gitignore 中增加上述 <FileName>

# 希望某些文件取消版本管理，同时在不保留在工作区
git rm <FileName>
git commit -m 'delete xxx file(folder)'
git push

# 如果希望某个目录离开暂存区可以添加 -r 参数，从而可以递归的将该目录下所有的文件 & 子目录全部取消版本管理
git rm -r --cached <Folder>
git commit -m 'remove xxx floder'
git push
```

取消 **所有版本** 某文件。参考：[单独删除某个文件的所有历史记录 - CSDN](https://blog.csdn.net/q258523454/article/details/83899911)

```bash
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch <FilePath>' --prune-empty --tag-name-filter cat -- --all
```

希望某些文件加入版本管理

```bash
删除在 .gitignore 中的相应语句即可
```

## 分支

### 创建分支

```bash
# 创建分支
git branch <BranchName>

# 远程同步
git push <RemoteName> <BranchName>
```

### 删除分支

```bash
# 切换到另一个分支再进行删除操作
git switch <AnotherBranchName>
git branch -d <BranchName>

# 远程同步
git push <RemoteName> --delete <BranchName>
```

### 修改分支

如果修改的分支为远程保护分支，则在远程更新之前，需要在远程相应的服务商家那里对保护分支进行重新设定

```bash
# 修改名称
git branch -m <OldName> <NewName>

# 远程同步
git push <RemoteName> <NewName>
git push <RemoteName> --delete <OldName>
```

### 合并分支

```bash
# 首先将当前分支切换到需要被合并的分支 <NowBranch>，接着合并需要被合并的分支 <TodoBranch>
git merge <TodoBranch>
```

### 拉取分支

```bash
# 在 clone 后只会拉取默认分支，想要拉取其余的分支执行
git checkout -t <RemoteName>/<BranchName>
```
