import numpy as np
import tensorflow as tf
import util
import logging

import config

'''相当于前向传播算法'''

total_num_node = 0
current_node = 0

# 曼哈顿距离
def calc_dist(left, right):
    xl = left.obj['x']
    xr = right.obj['x']
    yl = left.obj['y']
    yr = right.obj['y']
    return tf.add(tf.abs(xl - xr), tf.abs(yl - yr))


# 根据子节点的位置坐标和 wire length 计算父节点的 x, y
# 同时存储从每个节点引出的 wire 的折线段数
def calc_coordinate(left, right):
    x = None
    y = None
    if left.isleaf and type(left.obj['x']) is float:
        left.obj['x'] = tf.convert_to_tensor(
            left.obj['x'])  # todo 这里 converted tensor 会不会梯度变化，即这是个 constant 还是 variable
        left.obj['y'] = tf.convert_to_tensor(left.obj['y'])
    if right.isleaf and type(right.obj['x']) is float:
        right.obj['x'] = tf.convert_to_tensor(right.obj['x'])
        right.obj['y'] = tf.convert_to_tensor(right.obj['y'])

    dist = calc_dist(left, right)

    # 计算左边和右边的节点node
    # state = 0  # 表明两个点的位置关系，但是把水平共线的情况单独设置一个bool变量 hor
    # hor = sess.run(tf.equal(left.obj['y'], right.obj['y']))

    hor = tf.equal(left.obj['y'], right.obj['y'])

    # if sess.run(tf.less(left.obj['x'], right.obj['x'])):
    #     state = 1
    #     # left_p = left
    #     # right_p = right
    # elif sess.run(tf.less(right.obj['x'], left.obj['x'])):
    #     state = 2
    #     # left_p = right
    #     # right_p = left
    # else:
    #     # 下面需要竖直绕线
    #     if sess.run(tf.less(left.obj['y'], right.obj['y'])):
    #         state = 3
    #         # left_p = left
    #         # right_p = right
    #     elif sess.run(tf.less(right.obj['y'], left.obj['y'])):
    #         state = 4
    #         # left_p = right
    #         # right_p = left
    #     else:
    #         raise Exception('Node overlapped.') # 是brother重合了

    # assert(state != 0)

    # state1 = tf.less(left.obj['x'], right.obj['x'])
    # state2 = tf.less(right.obj['x'], left.obj['x'])
    # state3 = tf.less(left.obj['y'], right.obj['y'])
    # state4 = tf.less(right.obj['y'], left.obj['y'])
    # state0: raise Exception

    def case_1():

        def branch_1():

            def state_1():
                return left.obj['x'] + left.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(1), tf.convert_to_tensor(2)

            def state_2():

                return left.obj['x'] - left.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(1), tf.convert_to_tensor(2)

            def state_3():

                return left.obj['x'], left.obj['y'] + left.rec_obj['wirelen'], tf.convert_to_tensor(1), tf.convert_to_tensor(2)

            def state_4():
                return left.obj['x'], left.obj['y'] - left.rec_obj['wirelen'], tf.convert_to_tensor(1), tf.convert_to_tensor(2)

            return tf.case({
                tf.less(left.obj['x'], right.obj['x']) : state_1,
                tf.less(right.obj['x'], left.obj['x']) : state_2,
                tf.less(left.obj['y'], right.obj['y']) : state_3,
                tf.less(right.obj['y'], left.obj['y']) : state_4
            }, default=state_1, exclusive=True)

        def branch_2():

            def state_1():
                return right.obj['x'] - right.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(1), tf.convert_to_tensor(1)

            def state_2():

                return right.obj['x'] + right.rec_obj['wirelen'], right.obj['y'], tf.convert_to_tensor(1), tf.convert_to_tensor(1)

            def state_3():

                return right.obj['x'], right.obj['y'] - right.rec_obj['wirelen'], tf.convert_to_tensor(1), tf.convert_to_tensor(1)

            def state_4():
                return right.obj['x'], right.obj['y'] + right.rec_obj['wirelen'], tf.convert_to_tensor(1), tf.convert_to_tensor(1)

            return tf.case({
                tf.less(left.obj['x'], right.obj['x']) : state_1,
                tf.less(right.obj['x'], left.obj['x']) : state_2,
                tf.less(left.obj['y'], right.obj['y']) : state_3,
                tf.less(right.obj['y'], left.obj['y']) : state_4
            }, default=state_1, exclusive=True)

        def branch_3():

            def state_1():
                return left.obj['x'] + left.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(2), tf.convert_to_tensor(2)

            def state_2():

                return left.obj['x'] - left.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(2), tf.convert_to_tensor(2)

            def state_3():

                return left.obj['x'], left.obj['y'] + left.rec_obj['wirelen'], tf.convert_to_tensor(2), tf.convert_to_tensor(2)

            def state_4():
                return left.obj['x'], left.obj['y'] - left.rec_obj['wirelen'], tf.convert_to_tensor(2), tf.convert_to_tensor(2)

            return tf.case({
                tf.less(left.obj['x'], right.obj['x']): state_1,
                tf.less(right.obj['x'], left.obj['x']): state_2,
                tf.less(left.obj['y'], right.obj['y']): state_3,
                tf.less(right.obj['y'], left.obj['y']): state_4
            }, default=state_1, exclusive=True)

        return tf.case({
            tf.less(left.rec_obj['wirelen'], right.rec_obj['wirelen']) : branch_1,
            tf.equal(left.rec_obj['wirelen'], right.rec_obj['wirelen']) : branch_2
        }, default=branch_3, exclusive=True)

    def case_21():
        return right.obj['x'], right.obj['y'], tf.cond(tf.equal(left.obj['x'], right.obj['x']) | tf.equal(left.obj['y'], right.obj['y']),
                                                       lambda: tf.convert_to_tensor(1),
                                                       lambda: tf.convert_to_tensor(2)), tf.convert_to_tensor(0)

    def case_22():
        return left.obj['x'], left.obj['y'], tf.convert_to_tensor(0), \
               tf.cond(tf.equal(left.obj['x'], right.obj['x']) | tf.equal(left.obj['y'], right.obj['y']),
                                                       lambda: tf.convert_to_tensor(1),
                                                       lambda: tf.convert_to_tensor(2))

    def case_31():

        def state_1():
            return tf.cond(hor, lambda: (right.obj['x'], right.obj['y'] + right.rec_obj['wirelen'],
                                   tf.convert_to_tensor(2), tf.convert_to_tensor(1)),
                    lambda: (right.obj['x'] + right.rec_obj['wirelen'], right.obj['y'], tf.convert_to_tensor(2), tf.convert_to_tensor(1)))

        def state_2():
            return tf.cond(hor, lambda: (right.obj['x'], right.obj['y'] - right.rec_obj['wirelen'],
                                   tf.convert_to_tensor(2), tf.convert_to_tensor(1)),
                    lambda: (right.obj['x'] - right.rec_obj['wirelen'], right.obj['y'], tf.convert_to_tensor(2), tf.convert_to_tensor(1)))

        def state_3():
            return right.obj['x'] + right.rec_obj['wirelen'], right.obj['y'], tf.convert_to_tensor(2), tf.convert_to_tensor(1)

        def state_4():
            return right.obj['x'] - right.rec_obj['wirelen'], right.obj['y'], tf.convert_to_tensor(2), tf.convert_to_tensor(1)

        return tf.case({
            tf.less(left.obj['x'], right.obj['x']): state_1,
            tf.less(right.obj['x'], left.obj['x']): state_2,
            tf.less(left.obj['y'], right.obj['y']): state_3,
            tf.less(right.obj['y'], left.obj['y']): state_4
        }, default=state_1, exclusive=True)

    def case_32():

        def state_1():
            return tf.cond(hor, lambda: (left.obj['x'], left.obj['y'] - left.rec_obj['wirelen'],
                                   tf.convert_to_tensor(1), tf.convert_to_tensor(2)),
                    lambda: (left.obj['x'] - left.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(1), tf.convert_to_tensor(2)))

        def state_2():
            return tf.cond(hor, lambda: (left.obj['x'], left.obj['y'] + left.rec_obj['wirelen'],
                                   tf.convert_to_tensor(1), tf.convert_to_tensor(2)),
                    lambda: (left.obj['x'] + left.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(1), tf.convert_to_tensor(2)))

        def state_3():
            return left.obj['x'] - left.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(1), tf.convert_to_tensor(2)

        def state_4():
            return left.obj['x'] + left.rec_obj['wirelen'], left.obj['y'], tf.convert_to_tensor(1), tf.convert_to_tensor(2)

        return tf.case({
            tf.less(left.obj['x'], right.obj['x']): state_1,
            tf.less(right.obj['x'], left.obj['x']): state_2,
            tf.less(left.obj['y'], right.obj['y']): state_3,
            tf.less(right.obj['y'], left.obj['y']): state_4
        }, default=state_1, exclusive=True)



    x, y, left_num_bend, right_num_bend = tf.case({
        tf.less(left.rec_obj['wirelen'], dist) & tf.less(right.rec_obj['wirelen'], dist): case_1,  # case 1: non detour and non zero wire length
        tf.equal(left.rec_obj['wirelen'], dist): case_21,  # case 2-1: no detour but left-child has a zero wire length
        tf.equal(right.rec_obj['wirelen'], dist): case_22,  # case 2-2: no detour but right-child has a zero wire length
        tf.greater(left.rec_obj['wirelen'], dist): case_31,  # case 3-1: detour and the wirelen of left node is larger
        tf.greater(right.rec_obj['wirelen'], dist): case_32  # case 3-2, detour and the wirelen of right node is larger
    }, default=case_1, exclusive=True)  # unknown case: raise Exception

    # if sess.run(tf.less(left.rec_obj['wirelen'], dist)) and sess.run(tf.less(right.rec_obj['wirelen'],
    #                                                                          dist)):  # if left.rec_obj['wirelen'] < dist and right.rec_obj['wirelen'] < dist:
    #     logging.info()("case 1: non detour and non zero wire length")
    #     if sess.run(tf.less(left.rec_obj['wirelen'], right.rec_obj['wirelen'])):
    #
    #         left.num_bend = 1
    #         right.num_bend = 2
    #
    #         if state == 1:
    #             x = left.obj['x'] + left.rec_obj['wirelen']
    #             y = left.obj['y']
    #         elif state == 2:
    #             x = left.obj['x'] - left.rec_obj['wirelen']
    #             y = left.obj['y']
    #         elif state == 3:
    #             x = left.obj['x']
    #             y = left.obj['y'] + left.rec_obj['wirelen']
    #         else:
    #             x = left.obj['x']
    #             y = left.obj['y'] - left.rec_obj['wirelen']
    #
    #     elif sess.run(tf.equal(left.rec_obj['wirelen'], right.rec_obj['wirelen'])):
    #
    #         left.num_bend = 1
    #         right.num_bend = 1
    #
    #         if state == 1:
    #             x = right.obj['x'] - right.rec_obj['wirelen']
    #             y = left.obj['y']
    #         elif state == 2:
    #             x = right.obj['x'] + right.rec_obj['wirelen']
    #             y = right.obj['y']
    #         elif state == 3:
    #             x = right.obj['x']
    #             y = right.obj['y'] - right.rec_obj['wirelen']
    #         else:
    #             x = right.obj['x']
    #             y = right.obj['y'] + right.rec_obj['wirelen']
    #
    #     else:
    #
    #         left.num_bend = 2
    #         right.num_bend = 1
    #
    #         if state == 1:
    #             x = left.obj['x'] + left.rec_obj['wirelen']
    #             y = left.obj['y']
    #         elif state == 2:
    #             x = left.obj['x'] - left.rec_obj['wirelen']
    #             y = left.obj['y']
    #         elif state == 3:
    #             x = left.obj['x']
    #             y = left.obj['y'] + left.rec_obj['wirelen']
    #         else:
    #             x = left.obj['x']
    #             y = left.obj['y'] - left.rec_obj['wirelen']
    #
    # elif sess.run(tf.equal(left.rec_obj['wirelen'], dist)):  # elif left.rec_obj['wirelen'] == dist:
    #
    #     right.num_bend = 0
    #     if sess.run(tf.equal(left.obj['x'], right.obj['x'])) or sess.run(tf.equal(left.obj['y'], right.obj['y'])):
    #         left.num_bend = 1
    #     else:
    #         left.num_bend = 2
    #
    #     logging.info()("case 2-1: no detour but left-child has a zero wire length")
    #     x = right.obj['x']
    #     y = right.obj['y']
    #
    # elif sess.run(tf.equal(right.rec_obj['wirelen'], dist)):  # right.rec_obj['wirelen'] == dist:
    #
    #     left.num_bend = 0
    #     if sess.run(tf.equal(left.obj['x'], right.obj['x'])) or sess.run(tf.equal(left.obj['y'], right.obj['y'])):
    #         right.num_bend = 1
    #     else:
    #         right.num_bend = 2
    #
    #     logging.info()("case 2-2: no detour but right-child has a zero wire length")
    #     x = left.obj['x']
    #     y = left.obj['y']
    #
    # elif sess.run(tf.greater(left.rec_obj['wirelen'], dist)):  # elif left.rec_obj['wirelen'] > dist:
    #
    #     left.num_bend = 2
    #     right.num_bend = 1
    #
    #     logging.info()('case 3-1: detour and the wirelen of left node is larger')
    #     if state == 1:
    #         if hor:
    #             x = right.obj['x']
    #             y = right.obj['y'] + right.rec_obj['wirelen']
    #         else:
    #             x = right.obj['x'] + right.rec_obj['wirelen']
    #             y = right.obj['y']
    #     elif state == 2:
    #         if hor:
    #             x = right.obj['x']
    #             y = right.obj['y'] - right.rec_obj['wirelen']
    #         else:
    #             x = right.obj['x'] - right.rec_obj['wirelen']
    #             y = right.obj['y']
    #     elif state == 3:
    #         x = right.obj['x'] + right.rec_obj['wirelen']
    #         y = right.obj['y']
    #     else:
    #         x = right.obj['x'] - right.rec_obj['wirelen']
    #         y = right.obj['y']
    #
    # elif sess.run(tf.greater(right.rec_obj['wirelen'] > dist)):  # elif right.rec_obj['wirelen'] > dist:
    #
    #     right.num_bend = 2
    #     left.num_bend = 1
    #
    #     logging.info()('case 3-2, detour and the wirelen of right node is larger')
    #     if state == 1:
    #         if hor:
    #             x = left.obj['x']
    #             y = left.obj['y'] - left.rec_obj['wirelen']
    #         else:
    #             x = left.obj['x'] - left.rec_obj['wirelen']
    #             y = left.obj['y']
    #     elif state == 2:
    #         if hor:
    #             x = left.obj['x']
    #             y = left.obj['y'] + left.rec_obj['wirelen']
    #         else:
    #             x = left.obj['x'] + left.rec_obj['wirelen']
    #             y = left.obj['y']
    #     elif state == 3:
    #         x = left.obj['x'] - left.rec_obj['wirelen']
    #         y = left.obj['y']
    #     else:
    #         x = left.obj['x'] + left.rec_obj['wirelen']
    #         y = left.obj['y']
    #
    # else:
    #     logging.info()('unknown case')

    left.num_bend = left_num_bend
    right.num_bend = right_num_bend
    return x, y


def weight(sess, node, variate, trainable=True):
    # name = node.get_id() + '-' + variate
    if variate == 'wirelen':
        mean = 20.0
        stddev = 10.0
    elif variate == 'cdia':
        mean = 2.0
        stddev = 0.5
    else: # 'bdia'
        mean = 40.0
        stddev = 10.0
    with tf.variable_scope(node.get_id(), reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(variate, shape=[],
                                  initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev),
                                  trainable=trainable, dtype=tf.float32)
        sess.run(weights.initializer)  # 是否需要 if(initialized)
        logging.info(str(weights) + 'initialized.')
        # tf.summary.histogram('weights', weights)
    return weights

# def bia(trainable=True):
#     global scope_id
#     scope_id = scope_id + 1
#     with tf.name_scope(str(scope_id)):
#         bias = tf.get_variable("bias", shape=(1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
#     return bias

def merge(sess, left, right, father):

    global total_num_node
    global current_node
    total_num_node = father.size()

    left = merge_op(sess, left.left_child, left.right_child, left)
    right = merge_op(sess, right.left_child, right.right_child, right)

    logging.info('calculate coordinates of root : ' + str(current_node) + '@' + str(total_num_node) + '.')

    father.obj['x'], father.obj['y'] = calc_coordinate(left, right)

    if type(config.source_point['x']) is float:
        config.source_point['x'] = tf.convert_to_tensor(config.source_point['x'])
    if type(config.source_point['y']) is float:
        config.source_point['y'] = tf.convert_to_tensor(config.source_point['y'])

    logging.info('calculate number of bendings of root : ' + str(current_node) + '@' + str(total_num_node) + '.')
    father.num_bend = tf.cond(tf.equal(father.obj['x'], config.source_point['x']) | tf.equal(father.obj['y'], config.source_point['y']),
            lambda : tf.convert_to_tensor(1),
            lambda : tf.convert_to_tensor(2))
    # if sess.run(tf.equal(father.obj['x'], config.source_point['x'])) or sess.run(
    #         tf.equal(father.obj['y'], config.source_point['y'])):
    #     father.num_bend = 1
    # else:
    #     father.num_bend = 2

    logging.info('assign trainable variates for root : ' + str(current_node) + '@' + str(total_num_node) + '.')
    father.rec_obj['wirelen'] = calc_dist(father, util.Tree(config.source_point))

    father.rec_obj['cdia'] = weight(sess, father, 'cdia')
    father.rec_obj['cdia'] = tf.clip_by_value(father.rec_obj['cdia'], config.cdia_min, config.cdia_max)
    father.rec_obj['bdia'] = weight(sess, father, 'bdia')
    father.rec_obj['bdia'] = tf.clip_by_value(father.rec_obj['bdia'], config.bdia_min, config.bdia_max)

    current_node = current_node + 1
    assert(current_node == total_num_node)
    return father


# 注意 trainable_variables 的分配
def merge_op(sess, left, right, father):
    global total_num_node
    global current_node

    if right is not None:

        left = merge_op(sess, left.left_child, left.right_child, left)
        right = merge_op(sess, right.left_child, right.right_child, right)

        logging.info('assign trainable variates for ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')
        dist = calc_dist(left, right)
        right.rec_obj['wirelen'] = dist - left.rec_obj['wirelen']
        assert ((left.father is right.father) and (right.father is father))
        if not father is father.father.right_child:
            father.rec_obj['wirelen'] = weight(sess, father, 'wirelen')
        father.rec_obj['cdia'] = weight(sess, father, 'cdia')
        father.rec_obj['cdia'] = tf.clip_by_value(father.rec_obj['cdia'], config.cdia_min, config.cdia_max)
        father.rec_obj['bdia'] = weight(sess, father, 'bdia')
        father.rec_obj['bdia'] = tf.clip_by_value(father.rec_obj['bdia'], config.bdia_min, config.bdia_max)

        # 使用参数w的传播计算法
        # right.rec_obj['wirelen'] = dist - left.rec_obj['wirelen']  # 这里去除了right.childs应有的影响，全部右节点由左兄弟的子节点影响
        # father.rec_obj['wirelen'] = tf.add((tf.matmul(left.rec_obj['wirelen'], weight()) + bia()), (tf.matmul(right.rec_obj['wirelen'], weight()) + bia()))
        # father.rec_obj['dop'] =  tf.add((tf.matmul(left.rec_obj['dop'], weight()) + bia()), (tf.matmul(right.rec_obj['dop'], weight()) + bia()))
        # father.rec_obj['dop'] = tf.clip_by_value(father.rec_obj['dop'], config.dop_min, config.dop_max)
        # father.rec_obj['diameter'] = tf.add((tf.matmul(left.rec_obj['diameter'], weight()) + bia()), (tf.matmul(right.rec_obj['diameter'], weight()) + bia()))
        # father.rec_obj['diameter'] = tf.clip_by_value(father.rec_obj['diameter'], config.dia_min, config.dia_max)

        logging.info('calculate coordinate of ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')
        father.obj['x'], father.obj['y'] = calc_coordinate(left, right)
        # with tf.name_scope(father.get_id() + '-' + 'coordinate'):
        #     tf.summary.histogram('x', father.obj['x'])
        #     tf.summary.histogram('y', father.obj['y'])
        # 直接将优化参量作为优化参数：直接计算法

    elif left is not None:
        assert (left.father.right_child is None)
        assert (left.father is father)
        left = merge_op(sess, left.left_child, left.right_child, left)

        logging.info('assign trainable variates for ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')
        if not father is father.father.right_child:
            father.rec_obj['wirelen'] = weight(sess, father, 'wirelen')
        father.rec_obj['cdia'] = weight(sess, father, 'cdia')
        father.rec_obj['cdia'] = tf.clip_by_value(father.rec_obj['cdia'], config.cdia_min, config.cdia_max)
        father.rec_obj['bdia'] = weight(sess, father, 'bdia')
        father.rec_obj['bdia'] = tf.clip_by_value(father.rec_obj['bdia'], config.bdia_min, config.bdia_max)

        # father.rec_obj['wirelen'] = weight(sess)

        # father.rec_obj['dop'] = tf.matmul(left.rec_obj['dop'], weight()) + bia()
        # father.rec_obj['dop'] = tf.clip_by_value(father.rec_obj['dop'], config.dop_min, config.dop_max)
        #
        # father.rec_obj['diameter'] = tf.matmul(left.rec_obj['diameter'], weight()) + bia()
        # father.rec_obj['diameter'] = tf.clip_by_value(father.rec_obj['diameter'], config.dia_min, config.dia_max)
        logging.info('calculate coordinate of ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')
        father.obj['x'], father.obj['y'] = calc_coordinate(left, right)
        # with tf.name_scope(father.get_id() + '-' + 'coordinate'):
        #     tf.summary.histogram('x', father.obj['x'])
        #     tf.summary.histogram('y', father.obj['y'])

    else:

        assert (father.isleaf is True)

        logging.info('assign trainable variates for ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')
        if father is not father.father.right_child:
            # 这时left和right的father是个sink
            # 将father.obj设置为trainable
            father.rec_obj['wirelen'] = weight(sess, father, 'wirelen')
        father.rec_obj['cdia'] = weight(sess, father, 'cdia')
        father.rec_obj['cdia'] = tf.clip_by_value(father.rec_obj['cdia'], config.cdia_min, config.cdia_max)
        father.rec_obj['bdia'] = weight(sess, father, 'bdia')
        father.rec_obj['bdia'] = tf.clip_by_value(father.rec_obj['bdia'], config.bdia_min, config.bdia_max)
        # father.rec_obj[0] = tf.add((tf.matmul(fic_left[0], weight()) + bia()), (tf.matmul(fic_right[0], weight()) + bia()))
        # father.rec_obj[1] = tf.add((tf.matmul(fic_left[1], weight()) + bia()), (tf.matmul(fic_right[1], weight()) + bia()))
        # father.rec_obj[1] = tf.clip_by_value(father.rec_obj[1], config.dop_min, config.dop_max)
        # father.rec_obj[2] = tf.add((tf.matmul(fic_left[2], weight()) + bia()), (tf.matmul(fic_right[2], weight()) + bia()))
        # father.rec_obj[2] = tf.clip_by_value(father.rec_obj[2], config.dia_min, config.dia_max)
        # return father

    current_node = current_node + 1
    return father


# 给整个递归神经网络加载参数
# 在optimizer中调用用来计算损失以及反向传播
def load(sess):
    logging.info('network loading...')
    config.tree = merge(sess, config.tree.left_child, config.tree.right_child, config.tree)
    logging.info('network loaded.')
