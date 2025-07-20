---
title: 配置
---

本文记录 Python 的一些配置。包括外部工具、使用技巧等。

## 查看库的安装位置

由于本地安装了多个 Python 解释器，想要打印某个版本的解释器下载的「包或模块」的路径，整理一下大约有两种方法。

如果模块内置了 `__file__` 方法，则可以直接打印出来。比如下面的程序：

```python
import numpy as np

print(np.__file__)
```

![直接打印的运行结果](https://cdn.dwj601.cn/images/202406052305698.png)

有些库没有内置上述 `__file__` 方法，可以使用 pip 工具进行打印。语法规则如下：

```bash
pip show <PackgeName or ModelName>
```

例如想要打印 `sortedcontainers` 包的安装路径：

```bash
pip show sortedcontainers
```

![基于 pip 工具的运行结果](https://cdn.dwj601.cn/images/202406052310960.png)

## 自动激活虚拟环境 (VSCode)

打开用户设置 json 文件。加一行下面的配置即可：

```json
"python.terminal.activateEnvInCurrentTerminal": true,
```

不过奇怪的是使用 VSCode 创建虚拟环境后虽然在终端显示了确实是对应的虚拟环境，但是使用诸如 `which pip` 或 `which python` 后，显示的都是默认的路径而非虚拟环境对应的路径。使用 Pycharm 在当前路径创建虚拟环境就可以正常识别出来。

## 自动生成 requirements.txt

如果当前环境仅为当前项目独有，使用下面的命令即可：

```bash
pip freeze > requirements.txt
```

如果当前环境并非独属于当前项目，使用上述命令会生成很多与当前项目无关的依赖包，推荐使用 [`pipreqs`](https://github.com/bndr/pipreqs) 包，可以自动搜索独属于当前项目依赖的包，命令如下：

```bash
# 安装
pip install pipreqs

# 生成
pipreqs ./ --encoding=utf8 --force
```

!!! tip
    当然最好还是用虚拟环境然后用上述第一种方法生成依赖。

基于 requirements.txt 文件安装依赖：

```bash
pip install -r requirements.txt
```

## 解决安装时出现 0x80070422 和 0x80070643 报错

本质原因是之前没有按照官方程序卸载 Python。需要到注册表里彻底删掉对应的文件，在下面的三个路径下一一尝试，删除对应的版本的 Python 文件即可：

- `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall`
- `HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall`
- `HKEY_CURRENT_USER\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall`

## 更换 pip 下载源

默认的 pip 会从国外 PyPI 拉取库，要么本地有代理加速，要么换镜像源。

镜像源列表：

- 清华大学：`https://pypi.tuna.tsinghua.edu.cn/simple/`
- 中国科技大学：`https://pypi.mirrors.ustc.edu.cn/simple/`
- 阿里云：`https://mirrors.aliyun.com/pypi/simple/`
- 腾讯云：`https://mirrors.cloud.tencent.com/pypi/simple/`

临时换源：

```bash
pip install <PackageName> -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

永久换源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
```

查看当前配置：

```bash
pip config list
```

恢复默认源：

```bash
pip config unset global.index-url
```

## 终止并行程序

如果想要中断一个并行或并发任务，在终端输入 Ctrl+C 往往是无效的，因为 Ctrl+C 一次只能结束一个线程。我们直接用 taskkill 终止对应程序的所有进程即可。以 Python 炼丹为例，直接终止所有的 python.exe 进程即可：

=== "Windows"

    ```bash
    taskkill -F -IM python.exe
    ```

=== "Linux/MacOS"

    ```bash
    pkill -9 -f python
    ```

## 安装 CPython 库

TODO

```bash
pip install pesq
```

报错：`error: Microsoft Visual C++ 14.0 or greater is required.`

原因：没有 MSCV 的编译环境，需要下载 CPython 的构建工具

- [Microsoft C++ 生成工具](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)
- [WindowsCompilers](https://wiki.python.org/moin/WindowsCompilers)
