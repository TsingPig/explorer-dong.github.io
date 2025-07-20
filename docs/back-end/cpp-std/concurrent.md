---
title: 并发库
---

创建线程：`std::thread`

等待线程结束：`.join()`, `.detach()`

同步机制：`std::mutex`, `std::lock_guard`, `std::unique_lock`

原子变量：`std::atomic<T>`

任务异步：`std::async`, `std::future`, `std::promise`