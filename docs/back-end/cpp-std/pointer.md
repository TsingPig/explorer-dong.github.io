---
title: 智能指针库
---

`std::unique_ptr<T>`：独占所有权，不能复制

`std::shared_ptr<T>`：引用计数共享资源

`std::weak_ptr<T>`：不影响计数的引用，解决循环引用问题

`make_shared`, `make_unique`：创建智能指针的推荐方式