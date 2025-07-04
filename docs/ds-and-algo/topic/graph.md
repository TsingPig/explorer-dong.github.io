---
title: 图论
---

图是一种由顶点和边组成的数据结构。如果边上带有权重，就称该图为网。一些基本概念：

- **可达**：对于无向图，如果每一个顶点之间都有路径可达，就称该图为「连通图」，极大连通子图被称为「连通分量」；而有向图就全部加一个“强”字，其他含义不变，即「强连通图」和「强连通分量」；
- **度**：对于无向图，直接可达的结点数被称为「度」；对于有向图，指出去的直接可达结点数被称为「出度」，指进来的的结点数被称为「入度」。

与树类似，图也可以用邻接表（存储出边，如果存储入边就叫做逆邻接表）或邻接矩阵来存储。本文默认使用邻接表来存储。

## 图的遍历

与 [树的遍历](./ds.md#树的遍历) 类似，图也拥有深度优先和广度优先两种遍历方式。由于图可能存在环路（无论是有向图还是无向图），因此不能像树的遍历那样通过记忆父结点来规避死循环，只能另外创建元素类型为 `bool` 的 `vis` 数组来标记已经遍历过的结点。

=== "C++ DFS"

    ```c++
    vector<int> g[N];
    bool vis[N];
    
    void dfs(int idx) {
        // do something
        
        for (auto& ch: g[idx]) {
            if (vis[ch]) {
                continue;
            }
            vis[ch] = true;
            dfs(ch);
        }
    }
    
    dfs(0);
    ```

=== "C++ BFS"

    ```c++
    vector<int> g[N];
    
    void bfs(int idx) {
        bool vis[N];
        queue<int> q;
    
        // do something
    
        vis[idx] = true;
        q.push(idx)
    
        while (q.size()) {
            auto now = q.front();
            q.pop();
    
            for (auto& ch: g[now]) {
                if (vis[ch]) {
                    continue;
                }
    
                // do something
    
                vis[ch] = true;
                q.push(ch);
            }
        }
    }
    
    bfs(0)
    ```

假设图中结点数为 $n$，边数为 $e$，那么采用邻接表遍历一遍的时间复杂度为 $O(n+e)$，采用邻接矩阵遍历一遍的时间复杂度为 $O(n^2)$。

## 拓扑问题

一张图是可拓扑的当且仅当其是有向无环图 (Directed Acyclic Graph, DAG)。

### 拓扑图的判定

假设图中顶点表示活动，$i$ 指向 $j$ 的边表示 $i$ 需要先于 $j$ 发生，那么这张图是可拓扑的就表明这个活动安排是可行的。这种图也被称为顶点表示活动 (Activity On Vertex, AOV) 网，常用来描述在时间上有先后约束的工程管理问题。

为了判定一张有向图是否可拓扑，即是否无环，有两种方法，要么直接 DFS 判断每一个点是否存在指向其祖宗结点的边，要么利用拓扑排序算法维护出这张图的拓扑序列（线性），看看拓扑序列的元素个数是否和原数组的元素个数相等（若相等则表明该有向图可拓扑），从而间接判断该有向图是否存在环路，这可以通过 BFS 来实现。

以 [课程表 | 力扣 - (leetcode.cn)](https://leetcode.cn/problems/course-schedule/) 这道题为例，给出上述两种方法的实现。题意为给定一个有向图，判断是否存在环，有环输出 `false`，无环输出 `true`，下标从 $0$ 开始。

**直接 DFS 判断每一个点是否存在指向其祖宗结点的边**。开两个 `bool` 数组分别表示全局结点访问情况（记作 `vis`）和路径结点访问情况（记作 `path`），前者用来减少不必要的 DFS，后者用来判断路径上的点是否访问到了祖宗结点（也可以将这两个状态数组合并为一个）。时间复杂度为 $O(n+e)$。有几个注意点：

- 在 DFS 时，如果某个点之前已经被 `vis` 数组标记过，那么环路一定不会经过这个点（反证法，如果环路经过这个点，那么曾经的某一轮 DFS 一定可以判断出来）；
- 在 DFS 时，仍然需要维护 `vis` 数组，但是要等 DFS 结束再将这一轮 DFS 过的结点标记为 `true`，否则无法判断是否存在环路（因为提前标记为 `true` 后，环路上的结点就不会被 `path` 数组标记，也就无法判定环路）。

=== "C++"

    ```c++
    class Solution {
    public:
        bool canFinish(int n, vector<vector<int>>& p) {
            // 建图
            vector<int> g[n];
            for (auto& a: p) {
                g[a[0]].push_back(a[1]);
            }
    
            vector<bool> vis(n);   // 全局访问情况
            vector<bool> path(n);  // 路径访问情况
            bool ok = true;
    
            function<void(int)> dfs = [&](int u) -> void {
                path[u] = true;
                for (int v: g[u]) {
                    if (vis[v]) {
                        continue;
                    }
                    if (path[v]) {
                        ok = false;
                        return;
                    }
                    dfs(v);
                }
                path[u] = false;
                vis[u] = true;  // 注意要在路径遍历完开始回溯时，再标记全局访问情况
            };
    
            for (int i = 0; i < n; i++) {
                if (!vis[i]) {
                    dfs(i);
                }
            }
    
            return ok;
        }
    };
    ```

**通过拓扑排序维护出该有向图的拓扑序列，间接判断有向图是否存在环路**。拓扑序列定义为：对于图中任意一条有向边 $u\to v$，$u$ 在拓扑序列中的顺序都要比 $v$ 在拓扑序列中的顺序更靠前。实现上可以使用 BFS，顶点入队条件是入度为 $0$，顶点出队时需要将被指向点的入度减一。时间复杂度也是 $O(n+e)$。

=== "C++"

    ```c++
    class Solution {
    public:
        bool canFinish(int n, vector<vector<int>>& p) {
            vector<int> g[n];
            vector<int> rd(n);  // 入度
    
            for (auto& a: p) {
                g[a[0]].push_back(a[1]);
                rd[a[1]]++;
            }
    
            queue<int> q;
            vector<int> topo;  // 拓扑序列
            for (int i = 0; i < n; i++) {
                if (!rd[i]) {
                    q.push(i);
                    topo.push_back(i);
                }
            }
    
            while (q.size()) {
                int u = q.front();
                q.pop();
                for (int v: g[u]) {
                    rd[v]--;
                    if (!rd[v]) {
                        q.push(v);
                        topo.push_back(v);
                    }
                }
            }
    
            return topo.size() == n;
        }
    };
    ```

### 拓扑图的最短/长路径

如果 DAG 还有边权，那么该图可以在 $O(n+e)$ 的时间复杂度内，按照结点的拓扑顺序，使用 [动态规划](./dp.md) 算法求出所有可达点到源点的最短/长路径距离。这种性质使得在 DAG 上求最短/长路径时不会受边权正负影响，并且时间复杂度优于一般图的单源最短路算法。

带权 DAG 也被称为边表示活动 (Activity On Edge, AOE) 网，即在 AOV 网的基础上还描述了工程的时间进度，在工程可以并行推进的情况下，AOE 网上的最长路径长度可以用来表示工程预计完成的时间。

*注：最长路径只能出现在 DAG 中，因为无向图或有向有环图的最长路径只需在环上一直走就行。

示例代码 [^oi-wiki-dag-application]：

[^oi-wiki-dag-application]: [DP 求最长（短）路 | OI Wiki - (oi-wiki.org)](https://oi-wiki.org/graph/dag/#dp-求最长短路)

```c++
struct edge {
  int v, w;
};

int n, m;
vector<edge> e[MAXN];
vector<int> L;                               // 存储拓扑排序结果
int max_dis[MAXN], min_dis[MAXN], in[MAXN];  // in 存储每个节点的入度

void toposort() {  // 拓扑排序
  queue<int> S;
  memset(in, 0, sizeof(in));
  for (int i = 1; i <= n; i++) {
    for (int j = 0; j < e[i].size(); j++) {
      in[e[i][j].v]++;
    }
  }
  for (int i = 1; i <= n; i++)
    if (in[i] == 0) S.push(i);
  while (!S.empty()) {
    int u = S.front();
    S.pop();
    L.push_back(u);
    for (int i = 0; i < e[u].size(); i++) {
      if (--in[e[u][i].v] == 0) {
        S.push(e[u][i].v);
      }
    }
  }
}

void dp(int s) {  // 以 s 为起点求单源最长（短）路
  toposort();     // 先进行拓扑排序
  memset(min_dis, 0x3f, sizeof(min_dis));
  memset(max_dis, 0, sizeof(max_dis));
  min_dis[s] = 0;
  for (int i = 0; i < L.size(); i++) {
    int u = L[i];
    for (int j = 0; j < e[u].size(); j++) {
      min_dis[e[u][j].v] = min(min_dis[e[u][j].v], min_dis[u] + e[u][j].w);
      max_dis[e[u][j].v] = max(max_dis[e[u][j].v], max_dis[u] + e[u][j].w);
    }
  }
}
```

### 例：Mad City

> 经典之处：基环树问题
>
> 难度：CF 1700 / 洛谷 黄
>
> OJ：[CF](https://codeforces.com/contest/1873/problem/H)

题意：给定一张含有 $n\ (3\le n\le 2\cdot 10^5)$ 个点和 $n$ 条边的无向连通图，不含重边。图上有 A、B 两个人分别在顶点 $a,b\ (1\le a,b\le n)$ 处，A 能够追上 B 当且仅当到达同一个顶点或在一条边上相向而行。两人一次均只能移动到相邻结点或不移动，都知道对方下一步的行为且足够聪明，问在给定的局面下 B 能否逃脱 A 的追赶，能逃掉输出 "YES"，反之输出 "NO"。

思路：题目给的是一棵基环树（树多了一条边形成的图结构，具有唯一的环路）。显然，B 只有到达环才有逃脱的希望，否则肯定会被抓住。我们记 A 到环的距离为 $d_A$，首次抵达环上的结点为 tag，B 到 tag 的距离为 $d_B$。显然，B 能够逃掉当且仅当 $d_B<d_A$。求解 $d_A,d_B$ 的步骤如下：

1. 首先肯定需要找环。尽管上面介绍的 [判环策略](#拓扑图的判定) 是针对 DAG 的，但对于基环树来说也是可行的。具体地，我们同样利用顶点的入度信息，从图中的边缘点开始拆点删边即可，入队条件为入度为 $1$（DAG 的入队条件为入度为 $0$）。拆掉的点标记为非环，剩余的顶点就都在环上了；
2. 标记了环上的点后，只需要从 B 点遍历一遍即可得出 tag 以及 $d_B$；
3. 求解 $d_A$ 只需要再从 A 点遍历一遍即可。

时间复杂度：$O(n)$

=== "C++"

    ```c++
    #include <iostream>
    #include <queue>
    #include <vector>
    using namespace std;
    
    void solve() {
        int n, a, b;
        cin >> n >> a >> b;
    
        // 建图并维护入度
        vector<int> g[n + 1];
        vector<int> rd(n + 1);
        for (int i = 0; i < n; i++) {
            int u, v;
            cin >> u >> v;
            rd[u]++, rd[v]++;
            g[u].push_back(v);
            g[v].push_back(u);
        }
    
        // 标记环路（DFS、BFS 均可，这里以 BFS 为例）
        vector<bool> loop(n + 1, true);
        queue<int> q;
        for (int i = 1; i <= n; i++) {
            if (rd[i] == 1) {
                loop[i] = false;
                rd[i]--;  // 删边
                q.push(i);
            }
        }
        while (q.size()) {
            int u = q.front();
            q.pop();  // 拆点
            for (int v: g[u]) {
                rd[v]--;  // 删边
                if (rd[v] == 1) {
                    loop[v] = false;
                    rd[v]--;
                    q.push(v);
                }
            }
        }
    
        // 从 B 点开始遍历一遍计算 dB 并找到 tag 点（DFS、BFS 均可，这里以 BFS 为例）
        auto bfs = [&](int b) -> pair<int, int> {
            vector<int> d(n + 1);
            vector<bool> vis(n + 1);
            queue<int> q;
            d[b] = 0;
            vis[b] = true;
            q.push(b);
            while (q.size()) {
                int u = q.front();
                q.pop();
                for (int v: g[u]) {
                    if (!vis[v]) {
                        d[v] = d[u] + 1;
                        vis[v] = true;
                        q.push(v);
                    }
                }
            }
            int dB = n + 1, tag = -1;
            for (int i = 1; i <= n; i++) {
                if (loop[i] && dB > d[i]) {
                    dB = d[i], tag = i;
                }
            }
            return {dB, tag};
        };
        auto [dB, tag] = bfs(b);
    
        // 从 A 点开始遍历一遍计算 dA（DFS、BFS 均可，这里以 BFS 为例）
        auto bfs2 = [&](int a) -> int {
            vector<int> d(n + 1);
            vector<bool> vis(n + 1);
            queue<int> q;
            vis[a] = true;
            d[a] = 0;
            q.push(a);
            while (q.size()) {
                int u = q.front();
                q.pop();
                for (int v: g[u]) {
                    if (!vis[v]) {
                        vis[v] = true;
                        d[v] = d[u] + 1;
                        q.push(v);
                    }
                }
            }
            return d[tag];
        };
        int dA = bfs2(a);
    
        // 最终结果
        cout << (dB < dA ? "YES" : "NO") << "\n";
    }
    
    int main() {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
    
        int T;
        cin >> T;
        while (T--) {
            solve();
        }
    
        return 0;
    }
    ```

## 最短路问题

最短路 (Shortest Path) 顾名思义就是求解图中顶点之间的最短路径。分为单源最短路和多源最短路两种策略。所有的最短路算法都是基于动态规划进行的。

**Dijkstra 算法**。单源最短路算法（无法求解含负边权的单源最短路）。分为朴素版和堆优化版。具体地：

1. 朴素版。采用邻接矩阵存储图。时间复杂度 $O(n^2)$。算法流程如下：

    - 定义 $d[i]$ 表示从起点到当前 $i$ 号点的最短路径的长度；
    - 将顶点分为 $U$ 和 $V-U$ 两个集合，其中 $U$ 表示已经更新了最短路径长度的顶点集合；
    - 枚举集合 $V-U$ 中的结点 $v_i\in V-U$，选择 $U$ 中到当前结点 $v_i$ 最近的顶点 $v_j$ 并更新 `d[i] = d[j] + edges[j][i]`。

2. 堆优化版。采用邻接表存储图。时间复杂度 $O(e \log e)$。

**Bellman-Ford 算法**。单源最短路算法（支持负边权）。

**SPFA 算法**。单源最短路算法（同样支持负边权的单元最短路，属于 Bellman-Ford 算法的优化版）。

**Floyd 算法**。多源最短路算法（支持负边权）。多阶段决策共 $n$ 个阶段，`dp[i][j]` 表示每一个阶段 $k$，从 $i$ 到 $j$ 的选择前 $k$ 个顶点后的最短路径的长度。对于当前阶段 $k$，我们利用阶段 $k-1$ 的状态进行转移更新，其实就是对于新增加的顶点 $v_k$ 是否选择的过程：

- 选择 $v_k$，则 `dp[i][j] = dp[i][k] + dp[k][j]`；
- 不选 $v_k$，则 `dp[i][j]` 就是 $k-1$ 状态下的 `dp[i][j]`。

当然，如果是 DAG，那么可以在  的时间复杂度内求出单源最短路。

### 例：Dijkstra 算法

朴素版：<https://www.acwing.com/problem/content/851/>

堆优化：<https://www.acwing.com/problem/content/852/>

> 题意：给定一个正边权的有向图，可能存在重边与自环，问 $1$ 号点到 $n$ 号点的最短路径长度是多少，如果不可达就输出 $-1$。
>
> 思路一：朴素版。点数 $1\le n \le 500$，边数 $1 \le m\le 10^5$
>
> - 思路：根据数据量，我们采用邻接矩阵的方式存储「点少边多」的稠密图。我们定义 `d[i]` 数组表示起点到 `i` 号点的最短距离。先将起点放入 `SPT (Shortest Path Tree)` 集合，然后更新所有 `V-SPT` 中的点到 `SPT` 集合的最短路径长度。接着循环 `n-1` 次迭代更新剩余的 `n-1` 个点，每次迭代的过程中，首先选择距离起点最近的点 `vex`，然后将该点加入 `SPT` 集合，最后利用该点更新 `V-SPT` 集合中和该点有连边的点到起点的最短距离。最终的 `d[end]` 就是起点 `start` 到终点 `end` 的最短距离。
> - 总结：算法整体采用贪心与动态规划的思路。与 $\text{Prim}$ 算法仔细比对可知，其中的贪心过程几乎一致，即每次选择加入 SPT 集合的点均为当前局面 `V-SPT` 集合中距离起点最近的点。而动态规划的过程体现在，在求解出集合 `V-SPT` 中到集合 `STP` 最短距离的点 `vex` 之后，利用该点对「在 `V-SPT` 集合且和 vex 点有连边的点 `i`」更新 `d[i]` 的过程。更新前的状态都是在之前的子结构下的最优解。
>
> - 时间复杂度：$O(n^2)$
>
> 思路二：堆优化。点数 $1\le n \le 1.5 \times 10^5$，边数 $1 \le m \le 1.5 \times 10^5$
>
> - 思路：根据数据量，我们采用邻接表的方式存储「点多边少」的稀疏图。如果采用上述朴素 Dijkstra 算法进行求解必然会因为点数过多而超时，因此我们利用数据结构「堆」进行时间开销上的优化。不难发现朴素 Dijkstra 算法在迭代过程中主要有三部分：
>
>     1. 选择距离起点最近的点 `vex`。因为需要枚举所有的顶点，因此总的时间复杂度为 $O(n^2)$
>     2. 将该点加入 `SPT` 集合。因为只是简单的打个标记，因此总的时间复杂度为 $O(n)$
>     3. 利用该点更新 `V-SPT` 集合中和该点相连的点到起点的最短距离。因为此时枚举的是该点所有的连边，而邻接表的图存储方式无法进行重边的删除，因此最坏情况下会枚举所有的边，时间复杂度为 $O(m)$
> - 时间复杂度：
>

朴素版 C++：

```cpp
#include <bits/stdc++.h>

using ll = long long;
using namespace std;

int dijkstra_ori(std::vector<std::vector<int>>& g, int start, int end) {
    int n = g.size() - 1;
    std::vector<int> d(n + 1, INT_MAX >> 1);
    std::vector<bool> SPT(n + 1, false);
    
    // update start vex
    d[start] = 0;
    SPT[start] = true;
    for (int i = 1; i <= n; i++) {
        if (!SPT[i] && g[start][i] != INT_MAX >> 1) {
            d[i] = std::min(d[i], d[start] + g[start][i]);
        }
    }
    
    // update remain n-1 vex
    for (int k = 0; k < n - 1; k++) {
        int vex = -1;
        for (int i = 1; i <= n; i++) {
            if (!SPT[i] && (vex == -1 || d[i] < d[vex])) {
                vex = i;
            }
        }
        SPT[vex] = true;
        for (int i = 1; i <= n; i++) {
            if (!SPT[i] && g[vex][i] != INT_MAX >> 1) {
                d[i] = std::min(d[i], d[vex] + g[vex][i]);
            }
        }
    }
    
    return d[end] == INT_MAX >> 1 ? -1 : d[end];
}

void solve() {
    int n, m;
    cin >> n >> m;
    
    vector<vector<int>> g(n + 1, vector<int>(n + 1, INT_MAX >> 1));
    
    while (m--) {
        int u, v, w;
        cin >> u >> v >> w;
        g[u][v] = min(g[u][v], w);
    }
    
    cout << dijkstra_ori(g, 1, n) << "\n";
}

signed main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    int T = 1;
//    std::cin >> T;
    while (T--) solve();
    return 0;
}
```

朴素版 Python：

```python
import heapq
from collections import defaultdict
from typing import List, Tuple
import math
from itertools import combinations

II = lambda: int(input())
FI = lambda: float(input())
MII = lambda: tuple(map(int, input().split()))
LII = lambda: list(map(int, input().split()))


def dijkstra_ori(g: List[List[int]], start: int, end: int) -> int:
    n = len(g) - 1
    d = [10 ** 5] * (n + 1)
    SPT = [False] * (n + 1)
    
    d[start] = 0
    SPT[start] = True
    for i in range(1, n + 1):
        if not SPT[i] and g[start][i] != 10 ** 5:
            d[i] = min(d[i], d[start] + g[start][i])
    
    for _ in range(n - 1):
        vex = -1
        for i in range(1, n + 1):
            if not SPT[i] and (vex == -1 or d[i] < d[vex]):
                vex = i
        SPT[vex] = True
        for i in range(1, n + 1):
            if not SPT[i] and g[vex][i] != 10 ** 5:
                d[i] = min(d[i], d[vex] + g[vex][i])
    
    return -1 if d[end] == 10 ** 5 else d[end]


def solve() -> None:
    n, m = MII()
    g = [[10 ** 5] * (n + 1) for _ in range(n + 1)]
    for _ in range(m):
        u, v, w = MII()
        g[u][v] = min(g[u][v], w)
    print(dijkstra_ori(g, 1, n))


if __name__ == '__main__':
    T = 1
    # T = II()
    while T: solve(); T -= 1
```

### 例：Floyd 算法

<https://www.acwing.com/problem/content/856/>

> 题意：给定一个稠密有向图，可能存在重边与自环，给出多个询问，需要给出每一个询问的两个点之前的最短路径长度
>
> 思路：我们采用动态规划的思路。在此使用多阶段决策的方法，即每一个路径状态为选择 $1\to k$ 个点的情况下的最短路径长度
>
> - 状态表示：`f[k][i][j]` 表示在前 $k$ 个顶点中进行选择（中转），$i$ 号点到 $j$ 号点的最短路径长度
> - 状态转移：对于第 $k$ 个顶点，我们可以选择中转，也可以不中转。
>     - 对于不选择中转的情况：`f[k][i][j] = f[k-1][i][j]`
>     - 对于可选择中转的情况：`f[k][i][j] = f[k-1][i][k] + f[k-1][k][j]`
>     - 在其中取最小值即可，但是有一个注意点：对于第二种情况，选择是有一个约束的：即如果选择了 $k$ 号点进行转移的话，那么 $i$ 号点到 $k$ 号点以及 $k$ 号点到 $j$ 号点都是需要有路径可达的，从而可以选择最小距离
> - 初始化：即选择 0 个站点进行中转时，即 `f[0][i][j]` 的情况中，
>     - 如果 $i$ 号点与 $j$ 号点自环，则取 $0$
>     - 如果 $i$ 号点与 $j$ 号点之间有边，则取重边的最小值
>     - 如果 $i$ 号点与 $j$ 号点之间无边，则初始化为正无穷
> - 答案状态：对于 $a$ 号点到 $b$ 号点之间的最小路径长度，就是 `f[n][a][b]`
> - 时间复杂度：$O(n^3)$
> - 空间复杂度：$O(n^3)$
>
> 空间优化推导：我们尝试优化掉记忆数组的第一维度
>
> - 对于不选择的情况：由于决策局面 $k$ 是从前往后枚举，故当前状态 `f[k][i][j]` 可以 **直接依赖于已经更新出来且不会被当前状态之后的状态再次覆盖的最优子结构 `f[i][j]`**。即上一个局面的选择情况，就是不选择第 $k$ 个顶点的情况
>
> - 对于选择的情况：如果删除第一维度，我们担心的是当前状态 `f[k][i][j]` 依赖的两个状态 `f[i][k]` 与 `f[k][j]` 会不会被后续覆盖掉，即 **我们不确定 `f[i][k]` 与 `f[k][j]` 是否是当前第 k 个局面的最优子结构**。尝试推导：
>
>     > 为了确定 `f[i][k]` 与 `f[k][j]` 是否是当前第 $k$ 个局面的最优子结构，其实就是确定对于当前第 $k$ 个局面，这两个状态会不会在当前状态 `f[i][j]` 之后被更新覆盖，那么我们就看这两个状态是从哪里转移过来进行更新的。如果 `f[i][k]` 与 `f[k][j]` 这两个状态的转移会依赖于当前状态之后的状态，那么删除第一维度就是错误的，反之就是成立的。
>    >
>    > 尝试推导 `f[i][k]` 与 `f[k][j]` 从何转移更新：利用我们未删除维度时正确的状态转移方程进行推演
>    >
>    > 我们知道：`f[k][i][k] = min(f[k-1][i][k], f[k-1][i][k] + f[k-1][k][k])`，其中的 `f[k-1][k][k]` 就是一个自环的路径长度，由于 $floyd$ 算法的约束条件是没有负环，因此 `f[k-1][k][k]` 一定大于零，故 `f[k][i][k]` 一定取前者，即 `f[k][i][k] = f[k-1][i][k]`
>    >
>    > 同理可知：
>    >
>    > `f[k][k][j] = f[k-1][k][j]`
>
>     基于上述推导我们可以知道，当前第 $k$ 个决策局面中的 `f[k][i][k]` 与 `f[k][k][j]` 是依赖于上一个决策局面 $k-1$ 的，也就是说这 **两个状态一定是早于当前状态 `f[i][j]` 被更新覆盖的**，故 `f[i][k]` 与 `f[k][j]` 就是当前第 $k$ 个局面的最优子结构，证毕，可以进行维度的删除
>
> - 时间复杂度：$O(n^3)$
>
> - 空间复杂度：$O(n^2)$

不优化空间

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 210, INF = 0x3f3f3f3f;

int n, m, Q;
int f[N][N][N];

int main() {
    cin >> n >> m >> Q;
    
    // init
    memset(f, INF, sizeof f);
    
    // add edges and generate base
    while (m--) {
        int a, b, w;
        cin >> a >> b >> w;
        if (a == b) continue;                           // 重边就不赋值
        else if (f[0][a][b] == INF) f[0][a][b] = w;     // 第一次加边则直接赋值
        else f[0][a][b] = min(f[0][a][b], w);           // 再次赋边权就取最小值
    }
    
    // generate base again
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            if (i == j) 
                f[0][i][j] = 0;                         // 自环取边权为 0
    
    // dp 
    for (int k = 1; k <= n; k++)
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++) {
                // 不选第k个顶点
                f[k][i][j] = f[k - 1][i][j];
                
                // 选择第k个顶点
                if (f[k - 1][i][k] != INF && f[k - 1][k][j] != INF)
                    f[k][i][j] = min(f[k][i][j], f[k - 1][i][k] + f[k - 1][k][j]);
            }

    // query
    while (Q--) {
        int a, b;
        cin >> a >> b;
        if (f[n][a][b] == INF) cout << "impossible\n";
        else cout << f[n][a][b] << "\n";
    }
    
    return 0;
}
```

优化空间

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 210, INF = 0x3f3f3f3f;

int n, m, Q;
int f[N][N];

int main() {
    cin >> n >> m >> Q;
    
    // init
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            if (i == j) f[i][j] = 0;
            else f[i][j] = INF;
            
    // base
    while (m--) {
        int a, b, w;
        cin >> a >> b >> w;
        if (a == b) continue;
        else if (f[a][b] == INF) f[a][b] = w;
        else f[a][b] = min(f[a][b], w);
    }
    
    // dp
    for (int k = 1; k <= n; k++)
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                if (f[i][k] != INF && f[k][j] != INF)
                    f[i][j] = min(f[i][j], f[i][k] + f[k][j]);
    
    // query
    while (Q--) {
        int a, b;
        cin >> a >> b;
        if (f[a][b] == INF) cout << "impossible\n";
        else cout << f[a][b] << "\n";
    }
    
    return 0;
}
```

### 例：关闭分部的可行集合数目

<https://leetcode.cn/problems/number-of-possible-sets-of-closing-branches/>

> 标签：二进制枚举、最短路
>
> 题意：给定一个含有 $n$ 个顶点的无向图，如何删点可以使得剩余的图中顶点两两可达且最大距离不超过 maxDistance？返回所有删点的方案数。
>
> 思路：由于 $n$ 的数据范围只有 $1 \to 10$，我们可以直接枚举所有的删点方案。那么如何检查一个方案的合法性呢？直接使用最短路算法检查「所有顶点到每一个顶点」的最远距离即可。这里我们采用朴素 dijkstra 算法。
>
> 时间复杂度：$O(2^n \times n^3)$ - 其中枚举需要 $O(2^n)$、计算所有顶点到某个顶点的最远距离需要 $O(n^2)$、检查所有顶点需要 $O(n)$

```cpp
class Solution {
public:
    int numberOfSets(int n, int maxDistance, vector<vector<int>>& roads) {
        vector<vector<int>> g(n, vector<int>(n, INT_MAX >> 1));
        for (auto& r: roads) {
            int u = r[0], v = r[1], w = r[2];
            g[u][v] = g[v][u] = min(g[u][v], w);
        }

        auto get_max_dist = [&](int mask, int v) {
            vector<bool> SPT(n);
            vector<int> d(n, INT_MAX);
            
            d[v] = 0;
            SPT[v] = true;
            
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i) && !SPT[i]) {
                    cnt++;
                    d[i] = min(d[i], d[v] + g[v][i]);
                }
            }

            for (int k = 1; k <= cnt - 1; k++) {
                int vex = -1;
                for (int i = 0; i < n; i++) {
                    if (mask & (1 << i) && !SPT[i] && (vex == -1 || d[i] < d[vex])) {
                        vex = i;
                    }
                }
                SPT[vex] = true;
                for (int i = 0; i < n; i++) {
                    if (mask & (1 << i) && !SPT[i]) {
                        d[i] = min(d[i], d[vex] + g[vex][i]);
                    }
                }
            }
            
            int max_dist = -1;
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) {
                    max_dist = max(max_dist, d[i]);
                }
            }
            
            return max_dist;
        };

        int res = 0;
        for (int mask = 0; mask < 1 << n; mask++) {
            bool ok = true;
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i) && get_max_dist(mask, i) > maxDistance) {
                    ok = false;
                    break;
                }
            }
            res += ok;
        }

        return res;
    }
};
```

```python
class Solution:
    def numberOfSets(self, n: int, maxDistance: int, roads: List[List[int]]) -> int:
        g = [[10 ** 6 for _ in range(n)] for _ in range(n)]
        for u, v, w in roads:
            g[u][v] = g[v][u] = min(g[u][v], w)
        
        def get_max_dist(mask: int, v: int):
            SPT = [False for _ in range(n)]
            d = [10 ** 6 for _ in range(n)]

            SPT[v] = True
            d[v] = 0

            cnt = 0
            for i in range(n):
                if mask & (1 << i) and not SPT[i]:
                    cnt += 1
                    d[i] = min(d[i], d[v] + g[v][i])
            
            for _ in range(cnt - 1):
                vex = -1
                for i in range(n):
                    if mask & (1 << i) and not SPT[i] and (vex == -1 or d[i] < d[vex]):
                        vex = i
                SPT[vex] = True
                for i in range(n):
                    if mask & (1 << i) and not SPT[i]:
                        d[i] = min(d[i], d[vex] + g[vex][i])
                
            max_dist = -1
            for i in range(n):
                if mask & (1 << i):
                    max_dist = max(max_dist, d[i])
            
            return max_dist

        res = 0
        for mask in range(1 << n):
            ok = True
            for i in range(n):
                if mask & (1 << i) and get_max_dist(mask, i) > maxDistance:
                    ok = False
                    break
            res += ok

        return res
```

## 生成树问题

最小生成树 (Minimum Spanning Tree, MST) 即对于一个给定的图结构，选择全部的点和部分的边，使得可以组成一棵树且该树的总权重最小，对应的树就是最小生成树。该算法在很多场景都有实际的应用价值，例如最小化城市之间的道路铺设等。

Prim 算法。这是一种贪心算法。具体地，假设图中包含 $n$ 个顶点，初始时顶点集合 $U$ 含 $1$ 个顶点，顶点集合 $V-U$ 含 $n-1$ 个顶点。我们需要构造 $n-1$ 个「割」的状态并维护两个顶点集合之间的交叉边信息。对于每一个状态，我们将「最小交叉边在集合 $V-U$ 中的顶点」加入到集合 $U$ 中并更新交叉边信息。这样得到的顶点集 $U$ 及其边集就是最终的最小生成树。时间复杂度 $O(n^2)$。

Kruskal 算法。这也是一种贪心算法，并使用了并查集数据结构加速了一些集合操作。具体地，我们初始化 $n$ 个顶点作为 $n$ 个连通分量，接着将所有的边按照权值升序排序，然后枚举所有的边，如果当前边的两个顶点不在同一个集合，则加入最小生成树，如果当前边的两个顶点在同一个集合，则不选择（如果选了就会使得生成树形成回路）。时间复杂度 $O(e\log e)$。

### 例：Kruskal 算法

<https://www.acwing.com/problem/content/861/>

> 题意：给定一个无向图，可能含有重边和自环。试判断能否求解其中的最小生成树，如果可以给出最小生成树的权值
>
> 思路：根据数据量，可以发现顶点数很大，不适用 $Prim$ 算法，只能用 $Kruskal$ 算法，下面简单介绍一下该算法的流程
>
> - 自环首先排除 - 显然这条边连接的“两个”顶点是不可能选进 $MST$ 的
> - 首先将每一个结点看成一个连通分量
> - 接着按照权值将所有的边升序排序后，依次选择
>     - 如果选出的这条边的两个顶点不在一个连通分量中，则选择这条边并将两个顶点所在的连通分量合并
>     - 如果选出的这条边的两个顶点在同一个连通分量中，则不能选择这条边（否则会使得构造的树形成环）
> - 最后统计选择的边的数量 $num$ 进行判断即可
>     - $num=n-1$，则可以生成最小生成树
>     - $num<n-1$，则无法生成最小生成树
> - 时间复杂度：$O(e\log e)$​ - 因为最大的时间开销在对所有的边的权值进行排序上

C++

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

const int N = 100010;

struct edge {
    int a, b;
    int w;
};

int n, m;
vector<edge> edges;
vector<int> p(N);

int Find(int now) {
    if (p[now] != now) {
        p[now] = Find(p[now]);
    }
    return p[now];
}

void solve() {
    cin >> n >> m;
    for (int i = 1; i <= m; i++) {
        int a, b, w;
        cin >> a >> b >> w;
        if (a == b) {
            continue;
        }
        edges.push_back({a, b, w});
    }

    // 按照边权升序排序
    sort(edges.begin(), edges.end(), [&](edge& x, edge& y) {
        return x.w < y.w;
    });

    // 选边
    for (int i = 1; i <= n; i++) {
        p[i] = i;
    }

    int res = 0, num = 0;

    for (auto& e: edges) {
        int pa = Find(e.a), pb = Find(e.b);
        if (pa != pb) {
            num++;
            p[pa] = pb;
            res += e.w;
        }

        if (num == n - 1) {
            break;
        }
    }

    // 特判：选出来的边数无法构成一棵树
    if (num < n - 1) {
        cout << "impossible\n";
        return;
    }

    cout << res << "\n";
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr), cout.tie(nullptr);
    int T = 1;
//    cin >> T;
    while (T--) solve();
    return 0;
}
```

Python

```python
def Find(x: int, p: list) -> int:
    if p[x] != x: p[x] = Find(p[x], p)
    return p[x]

def kruskal(n: int, m: int, edges: list) -> int:
    # 按边权对边进行降序排序
    edges.sort(key=lambda edge: edge[-1])

    # dsu 初始化
    p = [None] + [i for i in range(1, n + 1)]

    # 选边
    cnt = sum = 0
    for edge in edges:
        if cnt == n - 1: break

        pa, pb = Find(edge[0], p), Find(edge[1], p)
        if pa != pb:
            p[pa] = pb
            cnt += 1
            sum += edge[2]

    return sum if cnt == n - 1 else 0


if __name__ == "__main__":
    n, m = map(int, input().split())

    edges = []
    for i in range(m):
        edge = tuple(map(int, input().split()))
        edges.append(edge)

    res = kruskal(n, m, edges)

    if res: print(res)
    else: print("impossible")
```

JavaScript

```javascript
const readline = require('readline');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

let n = null;
let m = null;
const edges = [];

rl.on('line', line => {
    const [a, b, c] = line.split(' ').map(i => Number(i));
    if (n === null) {
        n = a;
        m = b;
    } else {
        edges.push([a, b, c]);
    }
});

rl.on('close', () => {
    const res = kurskal(n, m, edges);
    console.log(res === Infinity ? 'impossible' : res);
});

function Find(x, p) {
    if (p[x] != x) p[x] = Find(p[x], p);
    return p[x];
}

function kurskal(n, m, edges) {
    // 对边进行升序排序
    edges.sort((a, b) => a[2] - b[2]);
    
    // 初始化 dsu
    p = [];
    for (let i = 1; i <= n; i++) p[i] = i;
    
    // 选边
    let cnt = 0, sum = 0;
    for (let [a, b, w] of edges) {
        if (cnt == n - 1) {
            break;
        }
        
        let pa = Find(a, p), pb = Find(b, p);
        if (pa !== pb) {
            cnt++;
            p[pa] = pb;
            sum += w;
        }
    }
    
    if (cnt === n - 1) return sum;
    else return Infinity;
}
```

### 例：Prim 算法

<https://www.acwing.com/problem/content/860/>

> 题意：给定一个稠密无向图，有重边和自环。求出最小生成树
>
> 思路：根据题目的数据量，可以使用邻接矩阵存储的方法配合 $Prim$ 算法求解最小生成树，下面给出该算法的流程
>
> - 首先明确一下变量的定义：
>     - `g[i][j]` 为无向图的邻接矩阵存储结构
>     - `MST[i]` 表示 $i$ 号点是否加入了 $MST$ 集合
>     - `d[i]` 表示 `i` 号点到 $MST$ 集合的最短边长度
> - 自环不存储，重边只保留最短的一条
> - 任选一个点到集合 $MST$ 中，并且更新 $d$ 数组
> - 选择剩余的 $n-1$ 个点，每次选择有以下流程
>     - 找到最短边，记录最短边长度 $e$ 和相应的在 $U-MST$ 集合中对应的顶点序号 $v$
>     - 将 $v$ 号点加入 $MST$ 集合，同时根据此时选出的最短边的长度来判断是否存在最小生成树
>     - 根据 $v$ 号点，更新 $d$ 数组，即更新在集合 $U-MST$ 中的点到 $MST$ 集合中的点的交叉边的最短长度
> - 时间复杂度：$O(n^2)$

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

const int N = 510;

int n, m;
vector<vector<int>> g(N, vector<int>(N, INT_MAX));
vector<int> d(N, INT_MAX); // d[i]表示i号点到MST集合中的最短边长度
bool MST[N];
int res;

void prim() {
    // 选任意一个点到MST中并更新d数组
    MST[1] = true;
    for (int i = 1; i <= n; i++)
        if (!MST[i])
            d[i] = min(d[i], g[i][1]);

    // 选剩下的n-1个点到MST中
    for (int i = 2; i <= n; i++) {
        // 1. 找到最短边
        int e = INT_MAX, v = -1; // e: 最短边长度，v: 最短边不在MST集合中的顶点
        for (int j = 1; j <= n; j++)
            if (!MST[j] && d[j] < e)
                e = d[j], v = j;

        // 2. 加入MST集合
        MST[v] = true;
        if (e == INT_MAX) {
            // 特判无法构造MST的情况
            cout << "impossible\n";
            return;
        } else {
            res += e;
        }

        // 3. 更新交叉边 - 迭代（覆盖更新）
        for (int j = 1; j <= n; j++)
            if (!MST[j])
                d[j] = min(d[j], g[j][v]);
    }

    cout << res << "\n";
}

void solve() {
    cin >> n >> m;
    while (m--) {
        int a, b, w;
        cin >> a >> b >> w;

        if (a == b) {
            continue;
        }

        if (g[a][b] == INT_MAX) {
            g[a][b] = w;
            g[b][a] = w;
        } else {
            g[a][b] = min(g[a][b], w);
            g[b][a] = min(g[b][a], w);
        }
    }

    prim();
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr), cout.tie(nullptr);
    int T = 1;
//    cin >> T;
    while (T--) solve();
    return 0;
}
```

## 二分图

### 二分图判定

染色法：

<https://www.acwing.com/problem/content/862/>

> 题意：给定一个无向图，可能有重边和自环。问是否可以构成二分图。
>
> 二分图的定义：一个图可以被分成两个点集，每个点集内部没有边相连（可以不是连通图）
>
> 思路：利用 **染色法**，遍历每一个连通分量，选择连通分量中的任意一点进行染色扩展
>
> - 如果扩展到的点没有染过色，则染成与当前点相对的颜色
> - 如果扩展到的点已经被染过色了且染的颜色和当前点的颜色相同，则无法构成二分图（奇数环）
>
> 时间复杂度：$O(n+e)$

```cpp
const int N = 100010;

int n, m;
vector<int> G[N], col(N);

bool bfs(int u) {
    queue<int> q;
    q.push(u);
    col[u] = 1;

    while (q.size()) {
        int now = q.front();
        q.pop();
        for (auto& ch: G[now]) {
            if (!col[ch]) {
                col[ch] = -col[now];
                q.push(ch);
            }
            else if (col[ch] == col[now]) {
                return false;
            }
        }
    }

    return true;
}

void solve() {
    cin >> n >> m;
    while (m--) {
        int u, v;
        cin >> u >> v;
        G[u].push_back(v);
        G[v].push_back(u);
    }

    // 遍历每一个连通分量
    for (int i = 1; i <= n; i++) {
        if (!col[i]) {
            bool ok = bfs(i);
            if (!ok) {
                cout << "No\n";
                return;
            }
        }
    }

    cout << "Yes\n";
}
```

### 匈牙利算法

TODO
