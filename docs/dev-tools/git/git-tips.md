---
title: Git 使用贴士
---

## 取消 Git 对中文的自动转义

在使用 Git Bash 命令行工具时，其配套的 Git 程序会自动对中文进行转义，其他的程序则不会，如下图：

![Git 程序 VS 其他程序](https://cdn.dwj601.cn/images/202409010902437.png)

取消自动转义即可。在 Git Bash 终端输入下面的命令取消 Git 程序对中文的转义：

```bash
git config --global core.quotepath false
```

最终效果如下：

![不会对中文自动转义了](https://cdn.dwj601.cn/images/202502071715678.png)

## 将文件从 Git 系统中彻底删除

有时我们会不小心将敏感文件加入到 Git 的版本管理中，并且经过不断迭代，导致曾经很多版本都记录了该敏感文件，此时 [基础的删除操作](./git-commands.md#取消版本管理) 无法在 Git 系统中彻底删除这些文件，就需要用到更高级的命令 [^sensitive]。

记得先把该文件备份（例如加个 `.bak` 后缀），然后再执行下面的命令：

```bash
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch <FilePath>' --prune-empty --tag-name-filter cat -- --all
```

[^sensitive]: [Removing sensitive data from a repository | GitHub - (docs.github.com)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
