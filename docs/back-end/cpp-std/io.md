---
title: I/O 库
---

> 标准输入输出：`std::cin`, `std::cout`
>
> 文件输入输出：`std::ifstream`, `std::ofstream`, `std::fstream`
>
> 格式控制：`std::setw`, `std::setprecision`, `std::left`, `std::right`（头文件 `<iomanip>`）
>
> 字符输入输出：`get`, `put`, `getline`

C++ 封装了三个类用来进行文件 IO 操作，分别为 `ifstream`、`ofstream` 和 `fstream`。其中 `ifstream` 负责读文件，`ofstream` 负责写文件，`fstream` 读写都可以，三个类都需要依赖 `iostream` 和 `fstream` 头文件。下面分三个部分大致介绍一下，后续遇到了继续补充：

### ifstream 只读

```cpp
#include <iostream>
#include <fstream>

int main() {
	std::ifstream fin;

	fin.open("../test.txt", std::ios::in);

	std::string word;
	while (fin >> word) {
		std::cout << word << " ";
	}

	fin.close();

	return 0;
}
```

### ofstream 只写

```cpp
#include <iostream>
#include <fstream>

int main() {
    std::ofstream fout;
    
    fout.open("../test.txt", std::ios::out);
    
    for (int i = 1; i <= 10; i++) {
        fout << i << " ";
    }
    
    fout.close();
    
    return 0;
}
```

### fstream 读写

```cpp

```
