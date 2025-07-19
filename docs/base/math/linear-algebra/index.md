---
title: 线性代数
status: new
---

本文初稿完成于 2024-01-06，即大二上学期期末。参考《工程数学 线性代数》同济第七版。配套讲义、历年真题、框框老师讲义见百度网盘 [^resource]。

由于初稿写作时笔者的水平有限并且偏向于应试，导致在初窥机器学习的奥秘时，被满屏的矩阵吓软了腿，我一度怀疑自己没学过线代 🤡，因此本文会持续更新。后续的更新会对内容进行整合与补充，同时偏向实际的应用，包括 AI 相关的矩阵计算和矩阵微积分。

为什么要学习线性代数？为什么会有线性代数？

- 线性代数是为了解决多元线性方程组而诞生的。（2024-01-06）；
- 真的是这样吗？看到一篇文章 [^blog] 是从程序语言角度进行理解的，还挺有意思。（2024.10.21）

[^resource]: [配套讲义、历年真题、框框老师讲义 | 百度网盘 - (pan.baidu.com)](https://pan.baidu.com/s/1IbW-R4a44JcMr1JFAP76sw?pwd=cwjf)
[^blog]: [8 分钟带你彻底弄懂线性代数 | 嵌入式逍遥 - (blog.csdn.net)](https://blog.csdn.net/Neutionwei/article/details/109698699)

*注：使用小写字母表示的为标量，例如 5 可以用 $x$ 来表示；加粗小写字母表示的为向量，例如 $(3,2,4)^T$ 可以用 $\mathbf a$ 来表示；加粗大写字母表示的为矩阵，例如 $\begin{bmatrix}b_{11} & \cdots & b_{1m} \\ \vdots & \ddots & \vdots \\b_{n1} & \cdots & b_{nm}\end{bmatrix}$ 可以用 $\mathbf A_{n\times m}$ 来表示。

## 行列式

国内的教材喜欢在一开始讲行列式 (Determinant)，虽然笔者认为这其实和线代没什么关系，直接跳到 [矩阵](#矩阵) 开始学也行。

所谓行列式，其实就是一种运算，与 $+$、$-$、$\times$、$\div$ 是一个东西，符号表示为 $\vert \mathbf A\vert$。其中 $\mathbf A$ 为 $n\times n$ 的数表，最终的运算结果是一个实数，即 $\vert \mathbf A\vert$ 其实是一个实数。

### 基本概念

**排列**。若一个序列含有 $n$ 个数并且序列中每一个位置只出现 $[1,n]$ 一次，则称该序列为全排列，简称排列；$[1,n]$ 的升序排列称为标准排列。

**逆序数**。一个排列中每一个元素之前比其大的元素数量之和。听起来有些拗口，可以用下面的代码来表示：

```python
cnt = 0
for i in range(n):
    for j in range(i):
        cnt += a[j] > a[i]
print(cnt)
```

**对换**。即交换排列中的两个元素。有以下两个结论：

1. 一个排列中两个元素对换，排列逆序数的奇偶性改变；
2. 奇排列对换成标准排列的对换次数为奇数，偶排列对换成标准排列的对换次数为偶数。

### 行列式的定义与性质

**行列式的定义**。以 $n$ 阶行列式为例，其值为 $n!$ 个项之和。每一项的数值与符号定义为：

- 数值：每行选一个元素，每列选一个元素，行列各不相同；
- 符号：记 $t(p_1p_2\cdots p_n)$ 为排列 $p_1p_2\cdots p_n$ 的逆序数，那么符号为 $(-1)^{t(\text{行号})+N(\text{列号})}$。

**行列式的性质**。行列式的性质可以用来简化求值，下面简单介绍一下 5 个常见的行列式性质及其推论（都可以用定义证出来，此处省略）：

1. 行列式与其转置行列式相等；

2. 对换行列式的两个行或者列，行列式的符号改变；
    - 推论：若行列式有两行或两列完全相同，则行列式的值为 $0$。

3. 若行列式的某一行/列 $\times k$，则行列式的值也 $\times k$；
    - 推论一：行列式的某一行/列中的公因子可以提到行列式之外；
    - 推论二：若行列式有两行/列成比例，则行列式的值为零。

4. 若行列式的某一行/列都是两数之和，则可以拆分成两个行列式之和；

5. 把行列式的某一行/列乘一个常数累加到另一行/列上，行列式的值不变。

合理利用上述五个性质对行列式进行变换，就可以快速求解一个行列式。一般地，我们都会尽可能保证变换后的行列式左上角是数字 $1$，从而配凑出「上三角行列式」，进而直接用主对角线之积求解。

### 行列式的按行/列展开

这里简单介绍一下行列式求值的另一个策略：按行/列展开。我们用 $\mathbf{D}$ 表示行列式。用 $\mathbf{M}_{ij}$ 表示余子式，即行列式去掉 $\mathbf{D}_{ij}$ 元素所在的行和列后剩余元素拼接起来的行列式。用 $\mathbf{A}_{ij}$ 表示代数余子式，其中 $\mathbf{A}_{ij}=(-1)^{i+j}\mathbf{M}_{ij}$。

**若行列式的某一行/列只有一个元素不为零**。则有：

$$
\mathbf{D}= a_{ij}\mathbf{A}_{ij}
$$

证明：将某行/列唯一不为零的元素 $a_{ij}$ 经过 $i+j-2$ 次对换后整到 $a_{11}$ 的位置后，剩余的 $\mathbf{M}_{ij}$ 变换为下三角即可，即：

$$
\mathbf{D} =(-1)^{i+j-2}a_{ij}\mathbf{M}_{ij}=(-1)^{i+j}a_{ij}\mathbf{M}_{ij}= a_{ij}\mathbf{A}_{ij}
$$

**若行列式的某一行/列有多个元素不为零**。则有：

$$
D =\sum_{i = 1}^n a_{xi}A_{xi}
$$

证明：将展开的那一行/列通过加法原理进行拆分，然后利用上述只有一个元素不为零时的一般情况进行证明即可。

**补充一个性质及其例题**。已知 $n$ 阶行列式 $\mathbf{D}$，按第 $x$ 行展开后有 $\mathbf{D}=\sum_{i=1}^n a_{xi}\mathbf{A}_{xi}$，现在将 $a_{xi}$ 替换为 $a_{yi}$ 且 $x\ne y$，则 $\sum_{i=1}^n a_{yi}\mathbf{A}_{xi}=0$。道理很简单，现在求解的值其实也是一个行列式，并且这个行列式有两行/列的元素完全相等，那么显然的行列式的值就是 $0$。

例如下面这道题：

<img src="https://cdn.dwj601.cn/images/202406140938343.png" alt="例题" style="zoom:50%;" />

显然 (1) 的结果为 $0$；(2) 转化为 $3\mathbf{A}_{31}-5\mathbf{A}_{32}+2\mathbf{A}_{33}-\mathbf{A}_{34}$ 后，与第一行做差然后通过余子式求解即可，结果为 $-40$。

### 特殊的行列式

下面补充几个特殊的行列式及其计算方法。

**分块行列式**。如下图所示：

<img src="https://cdn.dwj601.cn/images/202406140938721.png" alt="分块行列式" style="zoom:50%;" />

计算方法：$\mathbf{0}$ 在左下或右上就是左上角与右下角行列式之积 ($\mathbf{D}=\mathbf{D}_1\mathbf{D}_2$)，$\mathbf{0}$ 在左上或右下就是左下角与右上角行列式之积加上符号判定。

证明：将两个框选出来的区域转换为上三角即可。

**$2n$ 阶行列式**。如下图所示：

<img src="https://cdn.dwj601.cn/images/202406140939983.png" alt="2n 阶行列式" style="zoom:67%;" />

先行对换再列对换，通过分块行列式和数学归纳法，可得行列式的值是一个等比数列。

**范德蒙德行列式**。如下图所示：

<img src="https://cdn.dwj601.cn/images/202406140939979.png" alt="范德蒙德行列式" style="zoom:50%;" />

证明。首先从最后一行开始，依次减去前一行的 $x_1$ 倍，凑出第一列一个元素不为零的情况，最后通过数学归纳法即可求解。项数为 $C_n^2$。

## 矩阵

行列式是一种运算，运算结果是一个数，而矩阵是一种表示，表示一个数表。

### 基本概念

下面补充几个常见的名词：

- 方阵。若矩阵的行列数相等（假设为 $n$），则可以称该矩阵为：方阵、$n$ 阶矩阵、$n$ 阶方阵；
- 对角阵。即方阵的非主对角线元素均为 $0$，主对角线元素不限。符号表示为 $\mathbf{\Lambda}=diag(\lambda_1,\lambda_2,\cdots,\lambda_n)$；
- 单位阵。即方阵的非主对角线元素均为 $0$，主对角线元素均为 $1$。符号表示为 $\mathbf{E}=diag(1,1,\cdots,1)$；
- 纯量阵。即方阵的非主对角线元素均为 $0$，主对角线元素均为 $\lambda$。符号表示为 $\mathbf{S}=diag(\lambda,\lambda,\cdots,\lambda)$。

### 矩阵运算

**元素级运算**。两个形状相同的矩阵按元素逐个进行加、减、乘、除等运算。

**向量级运算**。向量有内积（点积）和外积（叉积）两种运算。前者计算出一个标量，后者计算出一个矩阵。以 $\mathbf{x},\mathbf{y},\mathbf{z}$ 三个 $n$ 维向量和实数 $\lambda$ 为例：

- 向量的内积，记 $[\cdot,\cdot]$ 为两个长度相等的向量作内积，有以下性质：

    1. $[\mathbf x, \mathbf y] = [\mathbf y,\mathbf x]$；
    2. $[\lambda \mathbf x,\mathbf y] = \lambda [\mathbf x,\mathbf y]$；
    3. $[\mathbf x + \mathbf y,\mathbf z] = [\mathbf x,\mathbf z] + [\mathbf y,\mathbf z]$；
    4. $[\mathbf x, \mathbf x] \geq 0$，且当 $x \ne 0$ 时有 $[\mathbf x,\mathbf x] > 0$。

- 向量的长度，有以下性质：

    1. 非负性：当 $\mathbf x \ne \mathbf 0$ 时，$\|\mathbf x\| > 0$；当 $\mathbf x =\mathbf 0$ 时，$\|\mathbf x\| = 0$；
    2. 齐次性：$\|\lambda\mathbf x\| = \lambda\|\mathbf x\|$；
    3. 三角不等式：$\|\mathbf x +\mathbf y\| \le \|\mathbf x\| + \|\mathbf y\|$。

- 向量的夹角：

    1. 当 $\|\mathbf x\| = 1$ 时，称 $x$ 为单位向量；
    2. 当 $\|\mathbf x\| \ne \mathbf 0\land \|\mathbf y\| \ne \mathbf 0$ 时，$\theta = \arccos \frac{[\mathbf x,\mathbf y]}{\|\mathbf x\|\|\mathbf y\|}$。

**矩阵级运算**。矩阵算律如下：

1. 结合律：$(\mathbf A\mathbf B)\mathbf C=\mathbf A(\mathbf B \mathbf C)$；
2. 分配率：$\mathbf A(\mathbf B+\mathbf C)=\mathbf A\mathbf B+\mathbf A\mathbf C,(\mathbf B+\mathbf C)\mathbf A=\mathbf B\mathbf A+\mathbf C\mathbf A$；
3. 常数因子可以随意交换顺序：$\lambda(\mathbf A\mathbf B)=(\lambda\mathbf A)\mathbf B=\mathbf A(\lambda\mathbf B)$；
4. 单位阵可以随意交换顺序或直接省略：$\mathbf A\mathbf E=\mathbf E\mathbf A=\mathbf A$；
5. 幂运算：若 $\mathbf A$ 是 $n$ 阶矩阵，则 $\mathbf A$ 的 $k$ 次幂为 $\mathbf A^k=\underbrace{\mathbf A\mathbf A\cdots \mathbf A}_{k\text{个}}$，且 $\mathbf  A^m \mathbf A^k=\mathbf A^{m+k},(\mathbf A^m)^k=\mathbf A^{mk}$，其中 $m,k$ 为正整数。

*注：矩阵乘法没有交换律。$\mathbf{AB}$ 称为 $\mathbf{A}$ 左乘 $\mathbf{B}$。交换成立的前提是 $\mathbf{A}$ 和 $\mathbf{B}$ 左乘和右乘合法相等才可以。

**矩阵转置**。矩阵转置算律有以下四点：

1. $(\mathbf{A}^T)^T=\mathbf{A}$；
2. $(\mathbf{A}+\mathbf{B})^T=\mathbf{A}^T+\mathbf{B}^T$；
3. $(\lambda \mathbf{A})^T=\lambda \mathbf{A}^T$；
4. $(\mathbf{AB})^T=\mathbf{B}^T\mathbf{A}^T$。

证明 4：左边的 $c_{ij}$ 其实应该是 $AB$ 的 $c_{ji}$ ，对应 $A$ 的第 $j$ 行与 $B$ 的第 $i$ 列，那么反过来对于 $ij$ 就是 $B$ 转置的第 $i$ 行与 $A$ 转置的第 $j$ 列。

**对称矩阵**。对于一个方阵 $\mathbf{A}$，若有 $\mathbf{A} = \mathbf{A}^T$ 则称 $\mathbf{A}$ 为对称矩阵，简称对称阵。给一个对阵矩阵的例题：

<img src="https://cdn.dwj601.cn/images/202406140940661.png" alt="对称矩阵例题" style="zoom:50%;" />

**方阵的行列式**。行列式算律有以下三点：

1. $\vert \mathbf{A}^T\vert=\vert \mathbf{A}\vert$；
2. $\vert \lambda \mathbf{A}\vert=\lambda^n\vert \mathbf{A}\vert$；
3. $\vert \mathbf{AB}\vert=\vert \mathbf{A}\vert\vert \mathbf{B}\vert$。

对于上述第 3 点，显然有：$\vert \mathbf{AB}\vert=\vert \mathbf{A}\vert\vert \mathbf{B}\vert=\vert \mathbf{B}\vert\vert \mathbf{A}\vert=\vert \mathbf{BA}\vert$，即 $\vert \mathbf{AB}\vert=\vert \mathbf{BA}\vert$。

**伴随矩阵**。用 $\mathbf{A}^*$ 表示 $\mathbf{A}$ 的伴随矩阵，则有：

$$
\mathbf{AA}^* = \mathbf{A}^* \mathbf{A} = \vert \mathbf{A} \vert \mathbf{E}
$$

其中 $\mathbf{A}^*$ 表示为：

$$
\mathbf{A}^*=
\begin{pmatrix}
\mathbf{A}_{11} & \mathbf{A}_{21} & \cdots & \mathbf{A}_{n1}\\
\mathbf{A}_{12} & \mathbf{A}_{22} & \cdots & \mathbf{A}_{n2}\\
\cdots & \cdots & \cdots & \cdots\\
\mathbf{A}_{1n} & \mathbf{A}_{2n} & \cdots & \mathbf{A}_{nn}\\
\end{pmatrix}
$$

其中 $\mathbf A_{ij}$ 即代数余子式。

### 逆矩阵

**逆矩阵的定义**。对于 $n$ 阶矩阵 $\mathbf A$，若有 $\mathbf A\mathbf B = \mathbf B\mathbf A = \mathbf E$ ，则称 $\mathbf B$ 为 $\mathbf A$ 的逆矩阵，记作 $\mathbf B=\mathbf A^{-1}$。若 $\vert\mathbf A\vert = 0$，则 $\mathbf A$ 为又称奇异矩阵；若 $\vert\mathbf A\vert \ne 0$，则 $\mathbf A$ 为又称非奇异矩阵。

**逆矩阵的性质**。如下：

- 唯一性。如果矩阵 $\mathbf A$ 可逆，则 $\mathbf A$ 的逆矩阵是唯一的；
- 非奇异性。$\mathbf A$ 可逆 $\iff \vert\mathbf A \vert \ne 0$。

**逆矩阵的计算方法**。知道了逆矩阵的定义和性质后，想要求解一个方阵的逆矩阵就有以下两种求法：

1. 方法一：首先判断 $\mathbf A$ 的行列式是否为 $0$，若 $\vert\mathbf A\vert \ne 0$ ，则说明矩阵 $\mathbf A$ 可逆，那么就有 $\mathbf A^{-1} = \frac{1}{\vert\mathbf A\vert}\mathbf A^*$；
2. 方法二：如果可以找到 $\mathbf A\mathbf B=\mathbf E$ 或 $\mathbf B\mathbf A = \mathbf E$，那么就有 $\mathbf A^{-1}= \mathbf B$。

**逆矩阵的运算规律**。如下：

1. ${(\mathbf A^{-1})}^{-1} = \mathbf A$；
2. $({\lambda \mathbf A})^{-1} = \frac{1}{\lambda} \mathbf A^{-1}$；
3. $({\mathbf A\mathbf B})^{-1} = \mathbf B^{-1}\mathbf A^{-1}$；
4. $(\mathbf A^T)^{-1} = (\mathbf A^{-1})^{T}$；
5. $\vert \mathbf A^{-1}\vert = {\vert \mathbf A\vert}^{-1}$；
6. $\vert \mathbf A^*\vert = {\vert \mathbf A\vert}^{n - 1}$。

### 克拉默法则

克拉默法则是求解一般线性方程组的一个特殊场景，适用于求解「未知数数量和方程个数相等，且系数行列式不为零」的线性方程组。

**克拉默法则的定义**。如下：

如果线性方程组：


$$
\begin{cases}
a_{11}x_{1}+a_{12}x_{2}+&\cdots&+a_{1n}x_{n}= b_{1}\\
a_{21}x_{1}+a_{22}x_{2}+&\cdots&+a_{2n}x_{n}= b_{2}\\
&\vdots&\\
a_{n1}x_{1}+a_{n2}x_{2}+&\cdots&+a_{nn}x_{n}= b_{n}
\end{cases}
$$

的系数矩阵 $\mathbf A$ 的行列式不为零：

$$
\vert\mathbf A\vert =\begin{vmatrix}a_{11}&\cdots&a_{1n}\\\vdots&&\vdots\\a_{n1}&\cdots&a_{nn}\end{vmatrix}\neq0
$$

则方程组有唯一解：

$$
x_1 =\frac{\vert \mathbf A_1\vert}{\vert\mathbf A\vert}, x_2 =\frac{\vert \mathbf A_2\vert}{\vert\mathbf A\vert},\cdots, x_n =\frac{\vert\mathbf A_n\vert}{\mid \mathbf A\vert}
$$

其中 $\mathbf A_j\ (j=1,2,...,n)$ 是把系数矩阵 $\mathbf A$ 中第 $j$ 列的元素用常数向量 $\mathbf b$ 代替后的 $n$ 阶矩阵：

$$
\mathbf A_j =
\begin{pmatrix}
a_{11}&\cdots&a_{1, j-1}&b_1&a_{1, j+1}&\cdots&a_{1n}\\
\vdots&&\vdots&\vdots&\vdots&&\vdots\\
a_{n1}&\cdots&a_{n, j-1}&b_n&a_{n, j+1}&\cdots&a_{nn}
\end{pmatrix}
$$

**克拉默法则的证明**。如下：


首先将方程组转化为矩阵方程：

$$
\mathbf A \mathbf x =\mathbf b,\vert \mathbf A\vert\ne 0
$$

然后应用逆矩阵消元：

$$
\mathbf x =
\begin{pmatrix}x_1\\x_2\\\vdots\\x_n\end{pmatrix}=
\mathbf A^{-1}\mathbf b =
\frac{\mathbf A^*}{\vert\mathbf A\vert}\mathbf b =
\frac{1}{\vert \mathbf A\vert}
\begin{pmatrix}
\mathbf A_{11}&\mathbf A_{21}&\cdots&\mathbf A_{n1}\\
\mathbf A_{12}&\mathbf A_{22}&\cdots&\mathbf A_{n2}\\
\vdots&\vdots&&\vdots\\
\mathbf A_{1n}&\mathbf A_{2n}&\cdots&\mathbf A_{nn}
\end{pmatrix}
\begin{pmatrix}
b_1\\b_2\\
\vdots\\b_n
\end{pmatrix}
$$

最后应用 [行列式的按行/列展开](#行列式的按行列展开) 中补充的性质即可得到最终的结果：

$$
\mathbf x = \cdots =
\frac{1}{\mathbf \vert \mathbf A\vert}
\begin{pmatrix}
\mathbf A_{11}b_1+\mathbf A_{21}b_2+\cdots+\mathbf A_{n1}b_n\\
\mathbf A_{12}b_1+\mathbf A_{22}b_2+\cdots+\mathbf A_{n2}b_n\\
\vdots\\
\mathbf A_{1n}b_1+\mathbf A_{2n}b_2+\cdots+\mathbf A_{nn}b_n
\end{pmatrix}=
\frac{1}{\mathbf \vert \mathbf A\vert}
\begin{pmatrix}
{\mathbf \vert \mathbf A_1\vert}\\
{\mathbf \vert \mathbf A_2\vert}\\
\vdots\\
{\mathbf \vert \mathbf A_n\vert}\\
\end{pmatrix}
$$

### 矩阵分块法

矩阵分块法本质就是将子矩阵看作一个整体进行运算，类似于 [分治](../../../ds-and-algo/topic/base.md#分治) 算法。注意，在对矩阵进行分块计算的时候，有两个注意点，一是两个矩阵一开始的规格要相同，二是两个矩阵分块之后的子矩阵规格也要相同。我们重点关注对角分块矩阵。

**对角分块矩阵的定义**。主对角线为子矩阵：

$$
\mathbf A =
\begin{pmatrix}
\mathbf A_1 & & &\\
& \mathbf A_2 & &\\
& & \ddots &\\
& & & \mathbf A_s
\end{pmatrix}
$$

其中 $\mathbf A_1,\mathbf A_2,...,\mathbf A_s$ 都是方阵。

**对角分块矩阵的运算**。分别介绍幂运算、行列式运算、逆矩阵运算。

幂运算就是主对角线相应子矩阵的幂运算。如下图所示：

<img src="https://cdn.dwj601.cn/images/202406140946437.png" alt="幂运算就是主对角线相应元素的幂运算" style="zoom:50%;" />

行列式运算使用了上三角的性质。如下式：

$$
\vert \mathbf A\vert = \vert \mathbf A_1\vert \vert \mathbf A_2\vert \cdots \vert \mathbf A_s\vert
$$

逆矩阵就是主对角线的子矩阵按位取逆。若 $\vert \mathbf A_i\vert\ne 0\ (i=1,2,\cdots,s)$，则 $\vert \mathbf A\vert\ne 0$，且有：

$$
\mathbf A^{-1} =
\begin{pmatrix}
\mathbf A_1^{-1} & & &\\
& \mathbf A_2^{-1} & &\\
& & \ddots &\\
& & & \mathbf A_s^{-1}
\end{pmatrix}
$$

## 矩阵的初等变换

经过前面的学习，我们了解了线性代数中的基本表示单位：矩阵。但不要忘了，线性代数是为了更方便地求解线性方程组。前文介绍了行列式，其本质就是一种矩阵运算法则（针对方阵），本章将会继续介绍矩阵运算法则（初等变换），从而更方便地求解线性方程组。

### 基本概念

**矩阵初等变换的定义**。矩阵的初等变换分为行变换和列变换，且都是可逆的。以初等行变换为例（将所有的 $r$ 换成 $c$ 就是矩阵的初等列变换），有以下三种：

1. 第 $i$ 行与第 $j$ 行对换，即 $r_i \leftrightarrow r_j$；
2. 第 $i$ 行乘一个常数 $k$，即 $r_i \leftarrow r_i \times k\ (k \neq 0)$；
3. 第 $i$ 行加上第 $j$ 行的 $k$ 倍，即 $r_i \leftarrow r_i + kr_j$​。

为了更方便地表示和书写，我们定义以下矩阵初等变换的符号，对于矩阵 $\mathbf A$ 和矩阵 $\mathbf B$ 而言：

1. $\mathbf A$ 经过有限次「初等行变换」转化为 $\mathbf B$，就称 $\mathbf A$ 与 $\mathbf B$ 行等价，记作 $\mathbf A \stackrel{r}{\sim} \mathbf B$；
2. $\mathbf A$ 经过有限次「初等列变换」转化为 $\mathbf B$，就称 $\mathbf A$ 与 $\mathbf B$ 列等价，记作 $\mathbf A \stackrel{c}{\sim} \mathbf B$；
3. $\mathbf A$ 经过有限次「初等变换」转化为 $\mathbf B$，就称 $\mathbf A$ 与 $\mathbf B$ 等价，记作 $\mathbf A \sim \mathbf B$。

**矩阵初等变换的性质**。初等变换拥有三大特性：

1. 自反性：$\mathbf A \sim \mathbf A$；
2. 对称性：若 $\mathbf A \sim \mathbf B$，则 $\mathbf B \sim \mathbf A$；
3. 传递性：若 $\mathbf A\sim \mathbf B$，$\mathbf B\sim \mathbf C$ ，则 $\mathbf A \sim \mathbf C$。

**矩阵初等变换的产物**。矩阵初等变换的根本目的是将矩阵变换为某种形式，有以下三种目标产物：

1）行阶梯形矩阵。例如：

$$
\begin{pmatrix}
\underline{2} & 4 & -1 & 0 & 4 \\
0 & \underline{5} & -1 & -7 & 3 \\
0 & 0 & 0 & \underline{1} & -3 \\
0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

定义为非零行在零行的上面，并且非零行的首个非零元素（首元）在其上一行（如果存在）首元的右侧。

2）行最简形矩阵。例如：

$$
\begin{pmatrix}
\underline 1 & 0 & -1 & 0 & 4 \\
0 & \underline 1 & -1 & 0 & 3 \\
0 & 0 & 0 & \underline 1 & -3 \\
0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

在满足行阶梯形矩阵的基础上，每行首元为 $1$ 并且其所在列的其他元素都为 $0$。

3）标准形。例如：

$$
\mathbf F = \begin{pmatrix} \mathbf E_r & \mathbf 0 \\ \mathbf 0 & \mathbf 0 \end{pmatrix}_{m \times n}
$$

左上角是一个单位阵，其余元素全是 $0$。$m \times n$ 的矩阵 $\mathbf A$ 总可经过初等变换转换标准形，此标准形由 $m, n, r$ 三个数唯一确定，其中 $r$ 就是行阶梯形矩阵中非零行的行数。

**矩阵初等变换的数学意义**。所有的初等变换都等价于在原矩阵左乘或右乘一个初等矩阵 (elementary matrix)。所谓初等矩阵就是对单位阵进行初等变换后的方阵，所以初等矩阵一定是「可逆」的。那么对于 $\mathbf A_{m\times n}$ 和 $\mathbf B_{m\times n}$ 就有：

1. $\mathbf A \stackrel{r}{\sim} \mathbf B \iff$ 存在 $m$ 阶可逆阵 $\mathbf P$ 使得 $\mathbf P\mathbf A=\mathbf B$；
2. $\mathbf A \stackrel{c}{\sim} \mathbf B \iff$ 存在 $n$ 阶可逆阵 $\mathbf Q$ 使得 $\mathbf A\mathbf Q=\mathbf B$；
3. $\mathbf A \sim \mathbf B \iff$ 存在 $m$ 阶可逆阵 $\mathbf P$ 和 $n$ 阶可逆阵 $\mathbf Q$ 使得 $\mathbf P\mathbf A\mathbf Q=\mathbf B$。

利用该数学性质，结合 [矩阵分块](#矩阵分块法) 的思想，我们可以进行一些很有意思的运算。

1）求解矩阵初等变换中的初等变换矩阵。以求解初等行变换矩阵 $\mathbf P$ 为例：

$$
\mathbf P\mathbf A =\mathbf B \iff
\begin{cases}
\mathbf P\mathbf A =\mathbf B\\
\mathbf P\mathbf E =\mathbf P
\end{cases} \iff
(\mathbf A, \mathbf E) \stackrel{r}{\sim} (\mathbf B ,\mathbf P)
$$

即对 $(\mathbf A , \mathbf E)$ 作初等行变换，当把 $\mathbf A$ 变换为 $\mathbf B$ 时，$\mathbf E$ 就变换为了需要求解的可逆阵 $\mathbf P$。

2）求解方阵 $\mathbf A$ 的逆矩阵。这里介绍 [逆矩阵](#逆矩阵) 的第二种求法，求解过程如下：

$$
\mathbf A\text{ 可逆} \iff
\begin{cases}
\mathbf A^{-1}\mathbf A =\mathbf E\\
\mathbf A^{-1}\mathbf E =\mathbf A^{-1}
\end{cases} \iff
(\mathbf A,\mathbf E) \stackrel{r}{\sim} (\mathbf E,\mathbf A^{-1})
$$

即对 $(\mathbf A,\mathbf E)$ 作初等行变换，当把 $\mathbf A$ 变换为 $\mathbf E$ 时，$\mathbf E$ 就变换为了 $\mathbf A^{-1}$。此法可以在证明一个方阵可逆的同时顺带计算出其逆矩阵。

3）求解线性方程组。已知 $\mathbf A\mathbf X=\mathbf B$，求解 $\mathbf X$。最朴素的做法就是先证明 $\mathbf A$ 可逆，然后计算 $\mathbf A^{-1}\mathbf B$ 即为所求。但这样做有些麻烦，考虑本节学到的知识：求解 $\mathbf A^{-1}\mathbf B$ 的本质是 $\mathbf B$ 进行 $\mathbf A^{-1}$ 的初等行变换，那么仿照上述配凑逻辑，构造 $(\mathbf A,\mathbf B)$ 进行初等行变换：

$$
\begin{cases}
\mathbf A^{-1}\mathbf A =\mathbf E\\
\mathbf A^{-1}\mathbf B =\mathbf X
\end{cases}
\iff
(\mathbf A,\mathbf B)\stackrel{r}{\sim}(\mathbf E,\mathbf X)
$$

### 矩阵的秩

**矩阵秩的定义**。如下：

- 首先定义 $k$ 阶子式。给定一个 $\mathbf A_{m\times n}$，选择其中的 $k$ 行和 $k$ 列 $(k\le\min⁡(m,n)$，由这些行和列的交点组成的 $k\times k$ 子矩阵的行列式，称为 $\mathbf A$ 的一个 $k$ 阶子式；
- 然后定义非零子式。如果一个子式的行列式值不等于零，则称它为非零子式；
- 那么矩阵 $\mathbf A$ 的秩 (rank, R) 就是其非零子式的最高阶数，记作 $R(\mathbf A)$。

**矩阵秩的性质**。如下：

1. 转置不变性：$R(\mathbf A^T)=R(\mathbf A)$；
2. 初等变换不变性：若 $\mathbf P,\mathbf Q$ 可逆，则 $R(\mathbf P\mathbf A\mathbf Q)=R(\mathbf A)$；
3. 上下界：$0 \le R(\mathbf A_{m\times n}) \le \min \{m, n\}$；
4. 配凑性：$\max(R(\mathbf A),R(\mathbf B))\le R(\mathbf A,\mathbf B)\le R(\mathbf A)+R(\mathbf B)$；
5. 加法性：$R(\mathbf A+\mathbf B)\le R(\mathbf A)+R(\mathbf B)$；
6. 压缩性：若 $R(\mathbf A_{m\times n})=r$，则 $\mathbf A$ 一定可以转化为 $\begin{bmatrix}\mathbf B_r & \mathbf 0 \\\mathbf 0 & \mathbf 0\end{bmatrix}$。

### 线性方程组的解

在求解线性方程组前，需要先预判解的数量。对于线性方程组 $\mathbf A\mathbf x=\mathbf b$，其中 $\mathbf A$ 为 $m\times n$，即 $m$ 个方程 $n$ 个未知数。那么根据 $m$ 和 $n$ 的关系共有以下三种情况：

- 超定方程，即 $m>n$，此时「约束过多」导致线性方程组无解；
- 正定方程，即 $m=n$，此时线性方程组有唯一解；
- 欠定方程，即 $m<n$，此时「约束过少」导致线性方程组有无限个解。

对于正定方程，我们已经有以下求解线性方程组 $\mathbf A\mathbf x=\mathbf b$ 的策略了：

1. [逆矩阵](#逆矩阵) 中介绍的。先求逆矩阵 $\mathbf A^{-1}$，再将 $\mathbf A^{-1}$ 与 $\mathbf b$ 相乘（最朴素）；
2. [克拉默法则](#克拉默法则) 中介绍的。求解未知数数量和方程个数相等的线性方程组（有限制）；
3. [矩阵的初等变换](#矩阵的初等变换) 中介绍的。利用矩阵的初等变换求解线性方程组（最通用，其实就是高斯消元法 [^gauss]）。

[^gauss]: [高斯消元法 | 百度百科 - (baike.baidu.com)](https://baike.baidu.com/item/高斯消元法/619561)

对于超定方程和欠定方程，可以参考 *线性方程组解法总结* [^linear-summary] 这篇博客。

[^linear-summary]: [线性方程组解法总结 | Kiritan - (kiritantakechi.github.io)](https://kiritantakechi.github.io/blog/summary-of-linear-system-solutions/)

## 4 向量组的线性相关性

### 4.1 向量组及其线性组合

#### 4.1.1 n 维向量的概念

显然的 $n>3$ 的向量没有直观的几何形象，所谓向量组就是由同维度的列（行）向量所组成的集合。

向量组与矩阵的关系：

![向量组与矩阵的关系](https://cdn.dwj601.cn/images/202406140946454.png)

#### 4.1.2 线性组合和线性表示

定义：

（一）线性组合：

![线性组合定义](https://cdn.dwj601.cn/images/202406140946455.png)

（二）线性表示：

![线性表示定义](https://cdn.dwj601.cn/images/202406140946456.png)

判定：转化为判定方程组有解问题，从而转化为求解矩阵的秩的问题 5

- 判定 **向量** $b$ 能否被 **向量组** $A$ 线性表示：

    ![向量被向量组线性表示](https://cdn.dwj601.cn/images/202406140946457.png)

- 判定 **向量组** $B$ 能否被 **向量组** $A$ 线性表示：
  
    ![向量组被向量组线性表示](https://cdn.dwj601.cn/images/202406140946458.png)

    该判定定理有以下推论：

    ![放缩性质](https://cdn.dwj601.cn/images/202406140946459.png)

- 判定 **向量组** $B$ 与 **向量组** $A$ 等价：

    ![向量组与向量组等价](https://cdn.dwj601.cn/images/202406140946460.png)

### 4.2 向量组的线性相关性

定义：

![线性相关定义](https://cdn.dwj601.cn/images/202406140946461.png)

![注意](https://cdn.dwj601.cn/images/202406140946462.png)

判定：

- 定理一：

    ![定理一](https://cdn.dwj601.cn/images/202406140946463.png)

    证明：按照定义，只需要移项 or 同除，进行构造即可

- 定理二：

    ![定理二](https://cdn.dwj601.cn/images/202406140946464.png)

    证明：按照定义，转化为齐次线性方程组解的问题

    - 有非零解 $\Leftrightarrow$ 无数组解（将解方程取倍数即可），$R(A)=R(A,0)<m$
    - 仅有零解 $\Leftrightarrow$ 唯一解，$R(A)=R(A,0)=m$

结论：

- 结论一：

    ![结论一](https://cdn.dwj601.cn/images/202406140946465.png)

    证明：$R(A)<m \to R(B)\le R(A)+1 <m+1$

- 结论二：

    ![结论二](https://cdn.dwj601.cn/images/202406140946466.png)

    证明：$R(A_{x\times m})=m \to R\binom{A}{b}=m$

- 结论三：

    ![结论三](https://cdn.dwj601.cn/images/202406140946467.png)

    证明：$R(A)\le n <m$

- 结论四：

    ![结论四](https://cdn.dwj601.cn/images/202406140946468.png)

    证明：$R(A)=m,R(A,b)<m+1 \to Ax=b\text{有唯一解}$

    - $\max \{ R(A),R(b) \} \le R(A,b) \le m+1 \to m \le R(A,b) \le m+1$
    - 又 $R(A,b)<m+1$
    - 故 $R(A,b)=m$
    - 因此 $R(A)=R(A,b)=m \to \text{有唯一解}$

### 4.3 向量组的秩

#### 4.3.1 最大无关组的定义

定义一：

![定义一](https://cdn.dwj601.cn/images/202406140946469.png)

注意：

- 最大无关组之间等价
- 最大无关组 $A_0$ 和原向量组 $A$ 等价

定义二：

![定义二](https://cdn.dwj601.cn/images/202406140946470.png)

#### 4.3.2 向量组的秩和矩阵的秩的关系

![向量组的秩和矩阵的秩的关系](https://cdn.dwj601.cn/images/202406140946471.png)

#### 4.3.3 向量组的秩的结论

![向量组的秩的结论 1-2](https://cdn.dwj601.cn/images/202406140946472.png)

![向量组的秩的结论 3-5](https://cdn.dwj601.cn/images/202406140946473.png)

证明：全部可以使用矩阵的秩的性质进行证明

### 4.4 向量空间

**向量空间的概念**。可以从高中学到的平面向量以及空间向量入手进行理解，即平面向量就是一个二维向量空间，同理空间向量就是一个三维向量空间，那么次数就是拓展到 n 维向量空间，道理是一样的，只不过超过三维之后就没有直观的效果展示罢了。

**向量空间的基与维数**。同样可以从高中学到的向量入手，此处的基就是基底，维数就是有几个基底。所有的基之间都是线性无关的，这是显然的。然后整个向量空间中任意一个向量都可以被基线性表示，也就很显然了，此处有三个考点，分别为：

**考点一**：求解空间中的某向量 x 在基 A 下的坐标。

其实就是求解向量 x 在基 A 的各个“轴”上的投影。我们定义列向量 $\lambda$ 为向量 x 在基 A 下的坐标，那么就有如下的表述：

$$
x = A \  \lambda
$$

**考点二**：求解过度矩阵 P。

我们已知一个向量空间中的两个基分别为 A 和 B，若有矩阵 P 满足基变换公式：$B = AP$，我们就称 P 为从基 A 到基 B 的过渡矩阵

**考点三**：已知空间中的某向量 x 在基 A 下坐标为 $\lambda$，以及从基 A 到基 B 的过渡矩阵为 P，求解转换基为 B 之后的坐标 $\gamma$。

![求解过程](https://cdn.dwj601.cn/images/202406140946474.png)

### 4.5 线性方程组的解的结构

本目其实就是 3.3 目的一个知识补充，具体的线性方程组求解方法与 3.3 目几乎完全一致，只不过通过解的结构将解的结构进行了划分从而看似有些不同。但是殊途同归，都是一个东西。下面介绍本目与 3.3 目不同的地方：

我们从 3.3 目可以知道，无论是齐次线性方程组还是非齐次线性方程组，求解步骤都是：将系数矩阵（非齐次就是增广矩阵）进行行等价变换，然后对得到的方程组进行相对应未知变量的赋值即可。区别在于：

$$
\text{非齐次线性方程组的通解}=\text{非齐次线性方程组的一个特解}+\text{齐次线性方程组的通解}
$$

解释：我们将

- 齐次线性方程组记为 $Ax=0$，解为 $\eta$，则有 $A \eta = 0$
- 非齐次线性方程组记为 $Ax=b$，假如其中的一个特解为 $\eta^*$，则 $A\eta^*=b$，假如此时我们又计算出了该方程组的其次线性解 $\eta$，则有 $A\eta=0$。那么显然有 $A(\eta^*+\eta)=b$，此时 $\eta^* + \eta$ 就是该非齐次线性方程组的通解

也就是说本目对 3.3 目的线性方程组的求解给出了进一步的结构上的解释，即非齐次线性方程组的解的结构是基于本身的一个特解与齐次的通解之上的，仅此而已。当然了，本目在介绍齐次线性方程组解的结构时还引入了一个新的定理：若矩阵 $A_{m\times n}$ 的秩为 $r$, 则该矩阵的解空间的维度(基础解系中线性无关向量的个数)就是 $n-r$。即：

$$
dimS = n-r
$$

该定理可以作为一些证明秩相等的证明题的切入点。若想要证明两个 $ n$ 元矩阵 $A$ 和 $B$ 的秩相等，可以转化为证明两个矩阵的基础解析的维度相等，即解空间相等。证明解空间相等进一步转向证明 $Ax=0$ 与 $Bx=0$ 同解，证明同解就很简单了，就是类似于证明一个充要条件，即证明 $Ax=0 \to Bx=0$ 以及 $Bx=0 \to Ax=0$

## 5 相似矩阵及二次型

### 5.1 正交矩阵与正交变换

正交向量。即两向量内积为 0，类似于二维平面中两个垂直的非零向量。

正交向量组。

- 定义：向量组之间的任意两两向量均正交。
- 性质：正交向量组一定线性无关。

标准正交基。

- 定义：是某空间向量的基+正交向量组+每一个向量都是单位向量。

- 求解方法：施密特正交化求解标准正交基。

??? note "施密特正交化求标准正交基 - 详细过程"

    一、正交化
    
    ![正交化](https://cdn.dwj601.cn/images/202406140946478.png)
    
    ![正交化 - 续](https://cdn.dwj601.cn/images/202406140946479.png)
    
    二、单位化
    
    ![单位化](https://cdn.dwj601.cn/images/202406140946480.png)

正交矩阵。

- 定义：满足 $A^TA=E\ \text{or} \ AA^T=E$ 的方阵。
- 定理：正交矩阵的充要条件为矩阵的行/列向量为单位向量且两两正交。

正交变换。

- 定义：对于正交矩阵 $A$，$y=Ax$ 称为称为正交变换。
- 性质：$||y||=\sqrt{y^Ty}=\sqrt{x^TA^TAx}=\sqrt{x^TEx}=\sqrt{x^Tx}=||x||$，即向量经过正交变换之后长度保持不变。

### 5.2 特征值与特征向量

**定义**。对于一个 $n$ 阶方阵 $A$，存在一个复数 $\lambda$ 和一组 $n$ 阶非零向量 $x$ 使得 $Ax =\lambda x$，则称 $x$ 为特征向量，$\lambda$ 为特征值，$|A-\lambda E|$ 为特征多项式。

**特征值的性质**。

- $n$ 阶矩阵 $A$ 在复数范围内含有 $n$ 个特征值，且：
  
    $$
    \begin{aligned}
    \sum_{i = 1}^{n} \lambda _i =& \sum_{i = 1}^{n} a_{ii} \\
    \prod_{i = 1}^{n} \lambda _i =& \left | A \right |
    \end{aligned}
    $$

- 若 $\lambda$ 是 $A$ 的特征值，则 $\phi{(\lambda)}$ 是 $\phi{(A)}$ 的特征值。

**特征向量的性质**。对于同一个矩阵，**不同的** 特征值对应的特征向量之间是 **线性无关** 的。

### 5.3 相似矩阵

#### 5.3.1 定义

对于两个 n 阶方阵 A, B 而言，若存在可逆矩阵 P 使得

$$
PAP^{-1}= B
$$

则称 B 为 A 的相似矩阵，A 与 B 相似，也称对 A 进行相似变换，P 为相似变换矩阵

#### 5.3.2 性质

若矩阵 A 与 B 相似，则 A 与 B 的特征多项式相同，则 A 与 B 的特征值也就相同，A 与 B 的行列式也就相同

#### 5.3.3 矩阵多项式

一个矩阵 A 的多项式 $\phi{(A)}$ 可以通过其相似矩阵 $\Lambda$ 很轻松地计算出来为 $P \phi{(\Lambda)} P^{-1}$，即对角矩阵左乘一个可逆阵，右乘可逆阵的逆矩阵即可，而对角矩阵的幂运算就是对角元素的幂运算，故而非常方便就可以计算一个矩阵的多项式。那么计算的关键在于如何找到一个矩阵的相似矩阵？下面给出判定一个矩阵是否存在相似矩阵（可对角化）的判定定理：

<center> n 阶方阵可对角化的充要条件为该方阵含有 n 个线性无关的特征向量 </center>

### 5.4 对称矩阵的对角化

本目讨论一个 n 阶方阵具备什么条件才能拥有 n 个线性无关的特征向量，从而可对角化。但是对于一般的方阵，情况过于复杂，此处只讨论 n 阶对称矩阵。即：一个 n 阶对角矩阵具备什么条件才能拥有 n 个线性无关的特征向量，从而可对角化。

答案是 n 阶对角矩阵一定是可对角化的。因为有一个定理是这样的：对于一个对称矩阵 A 而言，一定可以找到一个正交矩阵 P 使得 $P^{-1}AP=\Lambda$，又由于正交矩阵一定是可逆矩阵，因此一定可以找到矩阵 A 的 n 个线性无关的特征向量，从而 A 一定可对角化。

对称矩阵的性质如下：

1. 对称矩阵的特征值均为实数
2. 对称矩阵 A 的两个特征值 $\lambda _1$ 与 $\lambda _2$ 对应的两个特征向量分别为 $P_1$ 和 $P_2$，若 $\lambda_1 \ne \lambda_2$，相比于一般的矩阵 $P_1$ 与 $P_2$ 线性无关，此时两者关系更强，即：$P_1$ 与 $P_2$ 正交
3. 对称矩阵的每一个 k 重根，一定对应有 k 个线性无关的特征向量

因此本目相较于 5.3 目其实就是通过可对角化这一个概念，来告诉我们对称矩阵是一定可以求出对角矩阵的。而不用判断当前矩阵是否可对角化了。只不过在此基础之上还附加了一个小定理（也没给出证明），就是对称矩阵的相似变换矩阵一定是一个正交矩阵，那么也就复习回顾了 5.1 目中学到的正交矩阵的概念。为了求解出这个正交矩阵，我们需要在 5.3 目求解特征向量之后再加一个操作，即：对于一个 k 重根，根据上面的性质 3 我们知道当前的根一定有 k 个线性无关的特征向量，为了凑出最终的正交矩阵，我们需要对这 k 个线性无关的特征向量正交化。那么所有的特征值下的特征向量都正交化之后，又由性质 2 可知，不同的特征值下的特征向量又是正交的，于是最终的正交的相似变换矩阵也就求出来了，也就得到了对角矩阵 $\Lambda$

### 5.5 二次型及其标准型（部分）

本目只需要掌握到：将一个二次型转化为标准型，即可。其实就是比 5.4 目多一个将 **二次齐次函数** 的系数取出组成一个二次型的步骤。其中二次型就是一个对称矩阵。接着就是重复 5.4 目中的将对称矩阵转化为对角矩阵的过程了。

## 补

### 对称矩阵和正定性之间的关系

在最优化方法中我们需要通过目标函数海塞矩阵的正定性来判断凸性，显然的海塞矩阵是对称方阵。可以分别从特征值和行列式的角度进行判断。

#### 特征值角度

- 一个对称矩阵 A 是正定的，当且仅当它的所有特征值 $\lambda_i>0$
- 一个对称矩阵 A 是正半定的，当且仅当它的所有特征值 $\lambda_i \ge 0$

#### 行列式角度

- 一个对称矩阵 A 是正定的，当且仅当所有主子矩阵的行列式都大于零
- 一个对称矩阵 A 是正半定的，当且仅当所有主子矩阵的行列式都大于或等于零
