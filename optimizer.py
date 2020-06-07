import config
import numpy as np
import tensorflow as tf
import parse_topo as topoparser
import recursive as loader
import os
import parse_out as outparser

'''相当于后向传播算法'''

sink_delay = []

# 计算总延时
# todo 暂时没有加入组合优化的两个因素：碳纳米管类型和 buffer 类型
def calc_delay():
    # todo 等待 Aida 回信，核实 delay 计算方法, 使用标准的 \sigma(diameter, dop) 的函数关系
    # 自上而下的递归计算，
    # 1. 根据 diameter 和 dop 计算得出电阻率
    # 2. 根据每个节点的 wirlen 得出线延迟
    # 3. 考虑拐点的 contact 以及 sink&merge point 的固有 r/c
    # 4. 将计算的结果以列表的形式保存下来，每个元素为一个sink的延时
    # 5. 返回最大延时
    assert (len(sink_delay) == len(config.sink_set))
    return max(sink_delay)

# 计算引入拉格朗日乘子后的等式约束
def calc_lagrange():
    # 将拉格朗日乘子作为训练参数，梯度下降时候，向对拉格朗日乘子偏导等于零的方向下降
    # todo 如何保证偏导下降到等于零，这是一种 trade-off 吗？trade-off 比例参数在哪里设置？
    lagrangian = tf.get_variable("lagrangian", shape=(1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=True)
    return tf.multiply(lagrangian, (max(sink_delay) - min(sink_delay)))


# 优化算法也就是反向传播算法
def optimize():

    # 加载前向传播的树结构
    loader.load()

    # 计算损失=总延时+等式约束
    delay = calc_delay()
    tf.add_to_collection('losses', delay)
    lagrange = calc_lagrange()
    tf.add_to_collection('losses', lagrange)
    goal = tf.add_n(tf.get_collection('losses'))

    # 定义训练算法
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_dacay(config.learning_rate_base, global_step, 1, config.learning_rate_decay)  # todo
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(goal, global_step=global_step)  # todo
    # 暂时不使用滑动平均
    # variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # # variable_average_op = variable_average.apply(tf.trainable_variables())
    # with tf.control_dependencies([train_step, variable_average_op]): # 暂时不加入滑动平均
    #     train_op = tf.no_op()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(config.num_steps):
            _, goal_value, step = sess.run([train_step, goal, global_step])

            if i%100 == 0:
                print("After %d steps of training, goal value is %g ." %(step, goal_value))
                print("And the total delay of the whole tree is: ", end=None)
                print(sess.run(delay))
                print("And the lagrange-value of equality constraints: ", end=None)
                print(sess.run(lagrange))

                saver.save(sess, os.path.join(config.model_path, config.model_name), global_step=global_step)

    return None

def main(argv = None):
    print('optimizing...')
    optimize()


## 总流程控制
if __name__ == '__main__':

    if not topoparser.parse():
        raise Exception("tree parsing failed.")

    tf.app.run()
    outparser.print()
