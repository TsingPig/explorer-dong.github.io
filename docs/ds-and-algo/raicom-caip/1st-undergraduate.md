---
title: 第 1 届本科组（初赛）
---

## T1 懂的都懂 (20‘/20’)

题意：给定长度为 $n\ (1\le n\le 50)$ 的整数数组 $a\ (0\le a_i <255)$，进行 $k\ (1\le k\le 200)$ 次查询，每次查询输入长度为 $m\ (1\le m\le 200)$ 的整数数组 $b\ (0\le b_i < 255)$，如果 $a$ 与 $b$ 匹配则输出 "Yes"，反之输出 "No"。匹配规则为：$b$ 中所有元素都可以从 $a$ 中任意四个不同位置的元素的平均值计算而来。

思路：模拟题。缓存 $a$ 中所有四个元素的整数均值后，直接处理查询即可。

时间复杂度：$O(n^4)$

=== "C++"

    ```c++
    #include <iostream>
    #include <vector>
    using namespace std;
    
    int main() {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
    
        int n, k;
        cin >> n >> k;
    
        // 缓存整数均值
        vector<int> a(n);
        for (int i = 0; i < n; i++) {
            cin >> a[i];
        }
        vector<bool> vis(256);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                for (int p = j + 1; p < n; p++) {
                    for (int q = p + 1; q < n; q++) {
                        int s = a[i] + a[j] + a[p] + a[q];
                        if (s % 4 == 0) {
                            vis[s / 4] = true;
                        }
                    }
                }
            }
        }
    
        // 处理查询
        while (k--) {
            int x;
            cin >> x;
            bool ok = true;
            for (int i = 0; i < x; i++) {
                int y;
                cin >> y;
                if (!vis[y]) {
                    ok = false;
                }
            }
            cout << (ok ? "Yes" : "No") << "\n";
        }
    
        return 0;
    }
    ```

## T2 芬兰木棋 (25'/25')

题意：玩家在一个二维平面的原点 $(0,0)$ 处，平面有 $n\ (1\le n\le 10^5)$ 个点含有权值 $p\ (1\le p_i\le 10^3)$，点的坐标不超过 $32$ 位整数。玩家的一次操作被描述为：选择任意一个方向并获得该方向上离其最近的任意数量个点的价值。价值计算规则为：如果只选 $1$ 个点，则将该点的权值作为价值；如果选择超过 $1$ 个点，则将点的个数作为价值。输出在玩家获得最大价值收益的情况下，最大收益及其最少操作次数。

思路：模拟题。显然最大价值收益就是所有点的权值之和，只需要考虑最小操作次数。算法上，我们需要预处理出所有方向上的点，接着对于每一个方向的点，按照离原点的距离升序排序，最后对每个方向分别计算操作次数即可。具体地：

- 对于每一个方向，我们都需要维护一个列表用来存储当前方向上所有点的位置及其权值。由于方向需要使用诸如 `pair<int,int>` 等二元数据类型来存储，而这些数据类型往往没有自定义的哈希函数，因此使用基于哈希的 `unordered_map` 就不太合适且容易被卡常，因此这里使用通用的平衡树 `map` 存储每一个方向的点。又由于任意一个方向都可以被唯一的描述为两个互质的数，因此每一个点 $(x_i,y_i,p_i)$ 需要将 $x_i,y_i$ 同除 $\gcd(x_i,y_i)$ 后进行存储。注意 C++ 的 `__gcd()` 函数计算两个数的最大公因数时可能会输出负数，需要取绝对值；
- 对每一个方向的点进行排序时，可以按欧氏距离，但那会进行浮点数运算导致速度下降。直接比较横坐标的值也是可以的，但要在斜率不存在时将比较依据改成纵坐标的值；
- 最后统计每一个方向上排好序的点带来的操作次数。由于题目要求价值优先，操作次数其次，那么显然除了连续的 $1$ 可以一次操作以外，其余每一个点都必须单独操作一次。

时间复杂度：$O(n\log n)$

=== "C++"

    ```c++
    #include <algorithm>
    #include <iostream>
    #include <map>
    #include <vector>
    using namespace std;
    
    struct node {
        int i, j, val;
        bool operator<(node& t) const {
            if (t.i == this->i) {
                return t.j < this->j;
            }
            return t.i < this->i;
        }
    };
    
    int main() {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
    
        int n;
        cin >> n;
    
        // 存储每一个方向的点
        map<pair<int, int>, vector<node>> a;
        int s = 0;  // 价值收益
        for (int i = 0; i < n; i++) {
            int x, y, p;
            cin >> x >> y >> p;
            s += p;
            int gcd = abs(__gcd(x, y));
            a[{x / gcd, y / gcd}].push_back({x, y, p});
        }
    
        // 按方向计算操作次数
        int cnt = 0;  // 操作次数
        for (auto& [_, v]: a) {
            int t = 0;  // 当前方向需要操作的次数
            sort(v.begin(), v.end());
            for (int i = 0; i < v.size();) {  // i 指针始终指向权值大于 1 的元素，或连续 1 区间的首元素
                t++;
                i++;
                while (i > 0 && i < v.size() && v[i].val == 1 && v[i - 1].val == 1) {
                    i++;
                }
            }
            cnt += t;
        }
    
        cout << s << " " << cnt << "\n";
    
        return 0;
    }
    ```

## T3 打怪升级 (25‘/25’)

题意：给定一张无向图，包含 $n\ (1\le n\le10^3)$ 个结点和 $m$ 条边，不包含重边和自环，每条边有 $a\ (1\le a\le 100)$ 和 $b\ (1\le b\le 100)$ 两个属性。给定目标结点列表 $tgt\ (1\le \vert tgt\vert\le n)$，现在需要确定一个起始位置 $x$，使得 $x$ 到 $tgt$ 中每一个结点的路径中最大的 $\sum a$ 最小，最后给出 $x$ 到 $tgt$ 中每一个结点的路径，要求路径的 $\sum a$ 尽可能小同时 $\sum b$ 尽可能大。

思路：

- 题目略微有一些绕弯，主要分为两步，第一步需要找到某一个顶点作为起点，使得该点到所有目标结点的第一权值之和尽可能小。那么我们遍历每一个点分别跑一遍堆优化的 Dijkstra 即可，时间复杂度为 $O(nm\log m)$；
- 确定了起点后，需要求解带有两个边权的最短路及其路径，那么只需要简单修改一下 Dijkstra 的路径更新逻辑即可，即当且仅当新点的第一权值更长或新点的第一权值相等但第二权值更小，就更新新点。至于路径的维护，只需要维护一个链表，当点被更新权重时记录父结点即可，时间复杂度为 $O(m\log m)$。

时间复杂度：$O(nm\log m)$

=== "C++"

    ```c++
    #include <algorithm>
    #include <climits>
    #include <iostream>
    #include <queue>
    #include <vector>
    using namespace std;
    
    const int N = 1010, inf = INT_MAX >> 1;
    using pii = pair<int, int>;
    
    struct edge {
        int v, a, b;
        // 优先队列默认大根堆，至少需要重载小于号，谁小谁就往下 down
        bool operator<(const edge& t) const {
            if (this->a == t.a) {
                return this->b < t.b;
            }
            return this->a > t.a;
        }
    };
    
    int n, m;
    vector<edge> g[N];
    
    int dijkstra(int start) {
        vector<int> d(n + 1, inf);
        vector<bool> vis(n + 1, false);
        priority_queue<pii, vector<pii>, greater<pii>> q;
        d[start] = 0;
        q.push({d[start], start});
        while (q.size()) {
            auto [_, u] = q.top();
            q.pop();
            if (vis[u]) {
                continue;
            }
            vis[u] = true;
            for (auto& [v, a, _]: g[u]) {
                if (!vis[v] && d[v] > d[u] + a) {
                    d[v] = d[u] + a;
                    q.push({d[v], v});
                }
            }
        }
        return *max_element(d.begin() + 1, d.end());
    }
    
    void find_path(vector<int>& tgt, int start) {
        vector<pii> d(n + 1, {inf, -inf});
        vector<bool> vis(n + 1, false);
        vector<int> pre(n + 1);
        priority_queue<edge> q;
        d[start] = {0, 0};
        q.push({start, 0, 0});
        while (q.size()) {
            auto [u, _, __] = q.top();
            q.pop();
            if (vis[u]) {
                continue;
            }
            vis[u] = true;
            for (auto& [v, a, b]: g[u]) {
                if (!vis[v] && (d[v].first > d[u].first + a || 
                                d[v].first == d[u].first + a && d[v].second < d[u].second + b)) {
                    pre[v] = u;
                    d[v] = {d[u].first + a, d[u].second + b};
                    q.push({v, d[v].first, d[v].second});
                }
            }
        }
    
        // 输出
        for (int t: tgt) {
            int tt = t;
            vector<int> path;
            while (t != start) {
                path.push_back(t);
                t = pre[t];
            }
            reverse(path.begin(), path.end());
            cout << start;
            for (int p: path) {
                cout << "->" << p;
            }
            cout << "\n" << d[tt].first << " " << d[tt].second << "\n";
        }
    }
    
    int main() {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
    
        cin >> n >> m;
        for (int i = 0; i < m; i++) {
            int u, v, a, b;
            cin >> u >> v >> a >> b;
            g[u].push_back({v, a, b});
            g[v].push_back({u, a, b});
        }
    
        int k;
        cin >> k;
        vector<int> tgt(k);
        for (int i = 0; i < k; i++) {
            cin >> tgt[i];
        }
    
        // 寻找起点
        int min_energy = inf, start = -1;
        for (int i = 1; i <= n; i++) {
            int energy = dijkstra(i);
            if (energy < min_energy) {
                min_energy = energy;
                start = i;
            }
        }
    
        cout << start << "\n";
    
        // 从起点开始寻找路径
        find_path(tgt, start);
    
        return 0;
    }
    ```

## T4 疫情防控 (30‘/30’)

题意：给定一个含有 $n\ (1\le n\le 5\cdot 10^4)$ 个点和 $m\ (1\le m\le 2\cdot10^5)$ 条边的无向图，进行 $k\ (1\le k\le 10^3)$ 次操作，每次操作需要从图中删除一个结点（题目保证删除的结点之前没有被删除过） 并进行 $q\ (1\le q\le 10^3)$ 次询问，每次询问某两点是否可达。对于每次操作，输出 $q$ 次询问中不可达的次数。

思路：

- 一个似乎比较显然的集合操作。对于每次询问是否可达，本质上就是在询问两个顶点是否在同一个集合中，但由于并查集不支持删除分支结点，因此看上去并查集是不可行的；
- 正难则反，我们不妨从最后一次操作开始思考。正向是并查集的删除操作，那么逆向就是并查集的合并操作。那么本题就变成了最基本的并查集的查询和合并操作了。

时间复杂度：$O(kq)$

=== "C++"

    ```c++
    #include <iostream>
    #include <vector>
    using namespace std;
    
    struct DisjointSetUnion {
        std::vector<int> p;  // p[i] 表示 i 号点的祖先结点编号
        std::vector<int> cnt;  // cnt[i] 表示 i 号点所在集合的元素个数
        int set_cnt;  // 集合的个数
        
        DisjointSetUnion(int n) : p(n), cnt(n) {
            /* 初始化一个含有 n 个元素的并查集，元素下标范围为 [0, n-1] */
            for (int i = 0; i < n; i++) {
                p[i] = i, cnt[i] = 1;
            }
            set_cnt = n;
        }
        
        int find(int a) {
            /* 返回 a 号点的祖先结点 */
            if (p[a] != a) {
                // 路径压缩
                p[a] = find(p[a]);
            }
            return p[a];
        }
        
        void merge(int a, int b) {
            /* 合并结点 a 和结点 b 所在的集合 */
            int pa = find(a), pb = find(b);
            if (pa == pb) {
                return;
            }
            set_cnt--;
            // 按秩合并
            if (cnt[pa] < cnt[pb]) {
                p[pa] = pb;
                cnt[pb] += cnt[pa];
            } else {
                p[pb] = pa;
                cnt[pa] += cnt[pb];
            }
        }
    
        bool same(int a, int b) {
            /* 判断结点 a 和 结点 b 是否在同一个集合 */
            return find(a) == find(b);
        }
        
        int tree_size(int a) {
            /* 返回结点 a 所在集合的元素个数 */
            return cnt[find(a)];
        }
        
        int forest_size() {
            /* 返回集合的个数 */
            return set_cnt;
        }
    };
    
    int main() {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
    
        // 保存输入
        int n, m, k;
        cin >> n >> m >> k;
        vector<int> g[n + 1];
        for (int i = 0; i < m; i++) {
            int u, v;
            cin >> u >> v;
            g[u].push_back(v);
            g[v].push_back(u);
        }
        vector<int> vis(k);  // 待删除的结点
        vector<pair<int, int>> query[k];  // 每次查询的结点对
        for (int i = 0; i < k; i++) {
            int d, q;
            cin >> d >> q;
            vis[i] = d;
            for (int j = 0; j < q; j++) {
                int x, y;
                cin >> x >> y;
                query[i].push_back({x, y});
            }
        }
    
        // 合并所有没有被删除点
        vector<bool> del(n + 1);
        for (int d: vis) {
            del[d] = true;
        }
        DisjointSetUnion dsu(n + 1);
        for (int u = 1; u <= n; u++) {
            if (del[u]) {
                continue;
            }
            for (int v: g[u]) {
                if (!del[v]) {
                    dsu.merge(u, v);
                }
            }
        }
    
        // 逆序处理操作
        vector<int> ans(k);
        for (int i = k - 1; i >= 0; i--) {
            // 统计查询对
            int cnt = 0;
            for (auto& [u, v]: query[i]) {
                cnt += !dsu.same(u, v);
            }
            ans[i] = cnt;
            // 新加边
            int u = vis[i];
            del[u] = false;
            for (int v: g[u]) {
                if (!del[v]) {
                    dsu.merge(u, v);
                }
            }
        }
    
        for (int i = 0; i < k; i++) {
            cout << ans[i] << "\n";
        }
    
        return 0;
    }
    ```
