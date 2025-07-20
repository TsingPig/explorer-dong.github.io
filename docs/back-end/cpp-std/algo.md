---
title: 算法库
---

> 排序：`std::sort`, `std::stable_sort`, `std::partial_sort`
>
> 查找：`std::find`, `std::binary_search`, `std::find_if`
>
> 修改：`std::reverse`, `std::rotate`, `std::replace`
>
> 统计：`std::count`, `std::accumulate`（需 `<numeric>`）
>
> 拷贝/移动：`std::copy`, `std::move`, `std::swap`

## reverse

想要翻转一个容器，大约有以下三种方法，以 string 为例：

### 方法一：使用 `std::reverse()` 原地翻转

```c++
#include <algorithm>
#include <iostream>

using namespace std;

int main() {
    string s = "hello world!";
    cout << s << "\n";

    reverse(s.begin(), s.end());
    cout << s << "\n";

    return 0;
}
```

### 方法二：使用 `std::reverse_copy` 拷贝翻转

```c++
#include <algorithm>
#include <iostream>

using namespace std;

int main() {
    string s = "hello world!";
    cout << s << "\n";

    string t(s.size(), ' ');  // 需要提前申请好内存空间
    reverse_copy(s.begin(), s.end(), t.begin());
    cout << t << "\n";

    return 0;
}
```

### 方法三：重新构造

```c++
#include <algorithm>
#include <iostream>

using namespace std;

int main() {
    string s = "hello world!";
    cout << s << "\n";

    string t = string(s.rbegin(), s.rend());
    cout << t << "\n";

    return 0;
}
```

上述均输出：

```c++
hello world!
!dlrow olleh
```
