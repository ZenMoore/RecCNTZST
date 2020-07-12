Log of debugging

7.8 weight 在建立时进行初始化，最后在optimizer中尝试使用 global_variables_initializer 对剩余变量进行初始化 

> 但是发现仍然会爆栈，因此得到结论：global_variables_initializer 会一次性加载全部变量

7.9x1 去掉 global_variables_initializer ，对 variables 进行手动初始化

> 定义optimizer后内存溢出，但不是 global_variables_initializer 引起的，而是有两种可能
>
> 1. 可视化 merge_all 的时候 histogram 太多
>
> 2. 训练过程需要加载全部 weight, 加载不下(内存空间不足)，训练不动

7.9x2 去掉所有 weight 和 coordinate 的 histogram, 引入 log 机制 
> optimizer  加载有问题，触发 Windows fatal exception: access violation

7.12x1 部署到实验室 79 服务器