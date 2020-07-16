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

7.12 部署到实验室 79 服务器
> 在 StackOverflow 提出问题(基本运算太慢了)
> CPU 一直满载运行
> 一个奇怪的问题：第一次跑的时候是CPU空余很多MEM拉满，但是之后跑的时候都是MEM空余很多CPU拉满

7.16x1 使用 global_initializer
> 初始化过程很快，两分钟完成
> 张量计算依旧很慢：需要用 gpu
> 优化器构建依旧很慢，但是不超出一个小时了
> 计算图加载依旧超级慢，三个小时以上
> 训练过程出现好多问题：
> 1. Operation 'case_62/cond' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.
> 猜测这个节点的处理有问题，StackOverflow 推荐使用动态图 eager mode
> 2.  Session failed to close after 30 seconds. Continuing after this point may leave your program in an undefined state.
> 未知解决办法
> 3. Node 'gradients/case_132/cond_grad/StatelessIf' : Connecting to invalid output 1 of source node case_132/cond which has 1 outputs. Try using tf.compat.v1.experimental.output_all_intermediates(True).
> tf.compat.v1.experimental.output_all_intermediates(True).
> 4. tensorboard 调用不出来
> 不知道为什么

7.16x2 换用 tensorflow-gpu，output_all_intermediates(True)

