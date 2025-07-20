---
title: 容器库
---

顺序容器

`std::vector<T>`：动态数组

`std::list<T>`：双向链表

`std::array<T, N>`：定长数组

`std::deque<T>`：双端队列

关联容器

`std::set`, `std::map`：有序红黑树容器，O(logN)

`std::unordered_set`, `std::unordered_map`：哈希容器，O(1) 平均查找时间

`std::multiset`, `std::multimap`：允许重复键