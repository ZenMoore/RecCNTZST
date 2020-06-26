import config
import numpy as np
import tensorflow as tf
import parse_topo as topoparser
import recursive as loader
import os
import parse_out as outparser

'''相当于后向传播算法'''

sink_delay = []

def get_tensors_max_min(tensors):
    recur_set = []

    max_result = None
    for e in tensors:
        recur_set.append(e)
    while len(recur_set) > 1:
        a = recur_set.pop()
        b = recur_set.pop()
        max_result = tf.maximum(a, b)
        recur_set.append(max_result)

    min_result = None
    for e in tensors:
        recur_set.append(e)
    while len(recur_set) > 1:
        a = recur_set.pop()
        b = recur_set.pop()
        min_result = tf.minimum(a, b)
        recur_set.append(min_result)

    assert (max_result is not None)
    assert (min_result is not None)
    return max_result, min_result

def N_cnt(bdia, cdia):
    return tf.round(tf.div(tf.multiply(2*config.pi, tf.square(tf.multiply(bdia, 1/2))), tf.multiply(tf.sqrt(3), tf.square(tf.add(cdia, config.delta)))))

def R_c(cdia):
    if tf.less(cdia, 2) and (tf.greater(cdia, 1) or tf.equal(cdia, 1)):
        return tf.multiply(config.R_cnom, tf.div(tf.add(tf.square(cdia), tf.multiply(-2.811, cdia) + 2.538), tf.add(tf.multiply(0.5376, tf.square(cdia)), tf.multiply(-0.8106, cdia) + 0.3934)))
    else:
        return tf.convert_to_tensor(config.R_cnom)

def r_s(cdia, wirelen):
    if tf.greater(wirelen, config.mfp):
        return tf.div(config.R_Q, tf.multiply(config.C_lambda, cdia))
    else:
        return tf.convert_to_tensor(0.0)

# 计算总延时
# todo 暂时没有加入组合优化：buffer 类型
def calc_delay():
    sink_node_set = config.tree.get_sinks()
    for node in sink_node_set:
        while node is not None:
            delay = tf.add(delay, calc_node_delay(node))
            node = node.father
        sink_delay.append(delay)

    assert (len(sink_delay) == len(config.sink_set))
    result, _ = get_tensors_max_min(sink_delay)
    return result  # 必须是tensor数组里面的最大值

def calc_node_delay(node):

    delay = None

    if node.isleaf:
        if node.num_bend == 0:
            return 0.69 * node.obj['c'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])), N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia']))
        elif node.num_bend == 1:
            return tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])), 2* N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
                tf.add(
                    0.38e-6 * tf.div(tf.multiply(node.rec_obj['wirelen'], r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])), N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(config.unit_capacitance, node.rec_obj['wirelen']),
                    0.69 * node.obj['c'] * tf.add(tf.div(1e-6 * tf.multiply(node.rec_obj['wirelen'], r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])), N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])), tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                                  N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])))
                )
            )
        else:
            horizontal_bia = tf.abs(node.father.obj['x'] - node.obj['x'])
            vertical_bia = tf.abs(node.father.obj['y'] - node.obj['y'])


            # todo
            t_horizontal = tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])), 2* N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
                tf.add(
                    0.38e-6 * tf.div(tf.multiply(node.rec_obj['wirelen'], r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])), N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(config.unit_capacitance, node.rec_obj['wirelen']),
                    0.69 * node.obj['c'] * tf.add(tf.div(1e-6 * tf.multiply(node.rec_obj['wirelen'], r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])), N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])), tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                                  N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])))
                )
            )

            t_vertical = tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])), 2* N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
                tf.add(
                    0.38e-6 * tf.div(tf.multiply(node.rec_obj['wirelen'], r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])), N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(config.unit_capacitance, node.rec_obj['wirelen']),
                    0.69 * node.obj['c'] * tf.add(tf.div(1e-6 * tf.multiply(node.rec_obj['wirelen'], r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])), N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])), tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                                  N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])))
                )
            )

            return tf.add(t_horizontal, t_vertical)
            pass

    else:
        pass
    return 0

# 计算引入拉格朗日乘子后的等式约束
def calc_lagrange():
    # 将拉格朗日乘子作为训练参数，梯度下降时候，向对拉格朗日乘子偏导等于零的方向下降
    # todo 如何保证偏导下降到等于零，这是一种 trade-off 吗？trade-off 比例参数在哪里设置？
    lagrangian = tf.get_variable("lagrangian", shape=(1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=True)
    max_delay, min_delay = get_tensors_max_min(sink_delay)
    return tf.multiply(lagrangian, (max_delay - min_delay))


# 优化算法也就是反向传播算法
def optimize(sess):

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

                outparser.point_list(sess)
                outparser.draw()

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
