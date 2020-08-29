import numpy as np
import torch
import util
import logging
import math

import config

'''相当于前向传播算法'''

total_num_node = 0
current_node = 0
initial = False

# 曼哈顿距离
def calc_dist(left, right):
    if type(left.obj['x']) is float:
        left.obj['x'] = torch.tensor(left.obj['x'])
    if type(left.obj['y']) is float:
        left.obj['y'] = torch.tensor(left.obj['y'])
    if type(right.obj['x']) is float:
        right.obj['x'] = torch.tensor(right.obj['x'])
    if type(right.obj['y']) is float:
        right.obj['y'] = torch.tensor(right.obj['y'])

    xl = left.obj['x']
    xr = right.obj['x']
    yl = left.obj['y']
    yr = right.obj['y']
    return torch.abs(xl - xr) + torch.abs(yl - yr)
    # return tf.add(tf.abs(xl - xr), tf.abs(yl - yr))

# 根据子节点的位置坐标和 wire length 计算父节点的 x, y
# 同时存储从每个节点引出的 wire 的折线段数
def calc_coordinate(left, right):
    x = None
    y = None
    if left.isleaf and type(left.obj['x']) is float:
        # left.obj['x'] = torch.tensor(
        #     left.obj['x'])  # todo 这里 converted tensor 会不会梯度变化，即这是个 constant 还是 variable
        # left.obj['y'] = torch.tensor(left.obj['y'])

        left.obj['x'] = torch.tensor(left.obj['x'], requires_grad=False)
        left.obj['y'] = torch.tensor(left.obj['y'], requires_grad=False)
    if right.isleaf and type(right.obj['x']) is float:
        # right.obj['x'] = torch.tensor(right.obj['x'])
        # right.obj['y'] = torch.tensor(right.obj['y'])
        right.obj['x'] = torch.tensor(right.obj['x'], requires_grad=False)
        right.obj['y'] = torch.tensor(right.obj['y'], requires_grad=False)

    dist = calc_dist(left, right)

    # 计算左边和右边的节点node
    # state = 0  # 表明两个点的位置关系，但是把水平共线的情况单独设置一个bool变量 hor
    # hor = sess.run(tf.equal(left.obj['y'], right.obj['y']))

    # hor = tf.equal(left.obj['y'], right.obj['y'])
    hor = (left.obj['y'] == right.obj['y'])


    state = 0
    if left.obj['x'] < right.obj['x']:
        state = 1
        # left_p = left
        # right_p = right
    elif right.obj['x'] < left.obj['x']:
        state = 2
        # left_p = right
        # right_p = left
    else:
        # 下面需要竖直绕线
        if left.obj['y'] < right.obj['y']:
            state = 3
            # left_p = left
            # right_p = right
        elif right.obj['y'] < left.obj['y']:
            state = 4
            # left_p = right
            # right_p = left
        else:
            raise Exception('Node overlapped.') # 是brother重合了

    assert(state != 0)

    if left.rec_obj['wirelen'] < dist and right.rec_obj['wirelen'] < dist:
        logging.info("case 1: non detour and non zero wire length")

        if left.rec_obj['wirelen'] < right.rec_obj['wirelen']:

            left.num_bend = 1
            right.num_bend = 2

            if state == 1:
                x = left.obj['x'] + left.rec_obj['wirelen']
                y = left.obj['y']
            elif state == 2:
                x = left.obj['x'] - left.rec_obj['wirelen']
                y = left.obj['y']
            elif state == 3:
                x = left.obj['x']
                y = left.obj['y'] + left.rec_obj['wirelen']
            else:
                x = left.obj['x']
                y = left.obj['y'] - left.rec_obj['wirelen']

        elif left.rec_obj['wirelen'] == right.rec_obj['wirelen']:

            left.num_bend = 1
            right.num_bend = 1

            if state == 1:
                x = right.obj['x'] - right.rec_obj['wirelen']
                y = right.obj['y'] # todo 这里到底是 right 还是 left 需要仔细斟酌斟酌
            elif state == 2:
                x = right.obj['x'] + right.rec_obj['wirelen']
                y = right.obj['y']
            elif state == 3:
                x = right.obj['x']
                y = right.obj['y'] - right.rec_obj['wirelen']
            else:
                x = right.obj['x']
                y = right.obj['y'] + right.rec_obj['wirelen']

        else:

            left.num_bend = 2
            right.num_bend = 1

            if state == 1:
                x = left.obj['x'] + left.rec_obj['wirelen']
                y = left.obj['y']
            elif state == 2:
                x = left.obj['x'] - left.rec_obj['wirelen']
                y = left.obj['y']
            elif state == 3:
                x = left.obj['x']
                y = left.obj['y'] + left.rec_obj['wirelen']
            else:
                x = left.obj['x']
                y = left.obj['y'] - left.rec_obj['wirelen']

    elif left.rec_obj['wirelen'] == dist:

        right.num_bend = 0
        if left.obj['x'] == right.obj['x'] or left.obj['y'] == right.obj['y']:
            left.num_bend = 1
        else:
            left.num_bend = 2

        logging.info("case 2-1: no detour but left-child has a zero wire length")

        x = right.obj['x']
        y = right.obj['y']

    elif right.rec_obj['wirelen'] == dist:

        left.num_bend = 0
        if left.obj['x'] == right.obj['x'] or left.obj['y'] == right.obj['y']:
            right.num_bend = 1
        else:
            right.num_bend = 2

        logging.info("case 2-2: no detour but right-child has a zero wire length")

        x = left.obj['x']
        y = left.obj['y']

    elif left.rec_obj['wirelen'] > dist:

        left.num_bend = 2
        right.num_bend = 1

        logging.info('case 3-1: detour and the wirelen of left node is larger')

        if state == 1:
            if hor:
                x = right.obj['x']
                y = right.obj['y'] + right.rec_obj['wirelen']
            else:
                x = right.obj['x'] + right.rec_obj['wirelen']
                y = right.obj['y']
        elif state == 2:
            if hor:
                x = right.obj['x']
                y = right.obj['y'] - right.rec_obj['wirelen']
            else:
                x = right.obj['x'] - right.rec_obj['wirelen']
                y = right.obj['y']
        elif state == 3:
            x = right.obj['x'] + right.rec_obj['wirelen']
            y = right.obj['y']
        else:
            x = right.obj['x'] - right.rec_obj['wirelen']
            y = right.obj['y']

    elif right.rec_obj['wirelen'] > dist:

        right.num_bend = 2
        left.num_bend = 1

        logging.info('case 3-2, detour and the wirelen of right node is larger')

        if state == 1:
            if hor:
                x = left.obj['x']
                y = left.obj['y'] - left.rec_obj['wirelen']
            else:
                x = left.obj['x'] - left.rec_obj['wirelen']
                y = left.obj['y']
        elif state == 2:
            if hor:
                x = left.obj['x']
                y = left.obj['y'] + left.rec_obj['wirelen']
            else:
                x = left.obj['x'] + left.rec_obj['wirelen']
                y = left.obj['y']
        elif state == 3:
            x = left.obj['x'] - left.rec_obj['wirelen']
            y = left.obj['y']
        else:
            x = left.obj['x'] + left.rec_obj['wirelen']
            y = left.obj['y']

    else:
        logging.info('unknown case')


    return x, y


def weight(node, variate, trainable=True):

    def inverse_value(y, min, max):
        return math.atanh((2.0)*(y - min)/(max - min) - (2.0)/2.0)

    name = node.get_id() + '_' + variate
    if variate == 'wirelen':
        # mean = config.rec_ini['wirelen']
        mean = calc_dist(node, node.father).item()
        stddev = config.wirelen_std
    elif variate == 'cdia':
        mean = inverse_value(config.rec_ini['cdia'], config.cdia_min, config.cdia_max)
        stddev = config.cdia_std
    else: # 'bdia'
        mean = inverse_value(config.rec_ini['bdia'], config.bdia_min, config.bdia_max)
        stddev = config.bdia_std

    # with tf.variable_scope(node.get_id(), reuse=tf.AUTO_REUSE):
    #     weights = tf.get_variable(variate, shape=[],
    #                               initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev),
    #                               trainable=trainable, dtype=tf.float32)

    weights = torch.empty([], requires_grad=True)
    torch.nn.init.normal_(tensor=weights, mean=mean, std=stddev)
        # sess.run(weights.initializer)  # 是否需要 if(initialized)
        # config.trainable_variables.append(weights.initializer)
        # logging.info(str(weights) + 'initialized.')
    if variate == 'wirelen':
        config.trainable_wirelens[name] = weights
    elif variate == 'cdia':
        config.trainable_cdias[name] = weights
    else:  # 'bdia'
        config.trainable_bdias[name] = weights
    # config.trainables[name] = weights

    # logging.info(name + '@' + str(weights) + ' created and initialized(normal).')

    return weights

# def bia(trainable=True):
#     global scope_id
#     scope_id = scope_id + 1
#     with tf.name_scope(str(scope_id)):
#         bias = tf.get_variable("bias", shape=(1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
#     return bias

def merge(left, right, father):

    global total_num_node
    global current_node
    global initial
    total_num_node = 0
    current_node = 0

    total_num_node = father.size()

    left = merge_op(left.left_child, left.right_child, left)
    right = merge_op(right.left_child, right.right_child, right)

    logging.info('calculate coordinates of root : ' + str(current_node) + '@' + str(total_num_node) + '.')


    father.obj['x'], father.obj['y'] = calc_coordinate(left, right)

    if type(config.source_point['x']) is float:
        config.source_point['x'] = torch.tensor(config.source_point['x'])
    if type(config.source_point['y']) is float:
        config.source_point['y'] = torch.tensor(config.source_point['y'])

    logging.info('calculate number of bendings of root : ' + str(current_node) + '@' + str(total_num_node) + '.')

    if father.obj['x'] == config.source_point['x'] or father.obj['y'] == config.source_point['y']:
        father.num_bend = torch.tensor(1)
    else:
        father.num_bend = torch.tensor(2)
    # father.num_bend = tf.cond(tf.equal(father.obj['x'], config.source_point['x']) | tf.equal(father.obj['y'], config.source_point['y']),
    #         lambda : torch.tensor(1),
    #         lambda : torch.tensor(2))
    # if sess.run(tf.equal(father.obj['x'], config.source_point['x'])) or sess.run(
    #         tf.equal(father.obj['y'], config.source_point['y'])):
    #     father.num_bend = 1
    # else:
    #     father.num_bend = 2

    # logging.info('assign trainable variates for root : ' + str(current_node) + '@' + str(total_num_node) + '.')

    father.rec_obj['wirelen'] = calc_dist(father, util.Tree(config.source_point))

    if not config.loaded:
        weight(father, 'cdia')
    father.rec_obj['cdia'] = config.cdia_min + (torch.tanh(config.trainable_cdias[father.get_id() + '_' + 'cdia']) - (-1.0)) * ((config.cdia_max - config.cdia_min)/2.0)
    if not config.loaded:
        weight(father, 'bdia')
    father.rec_obj['bdia'] = config.bdia_min + (torch.tanh(config.trainable_bdias[father.get_id() + '_' + 'bdia']) - (-1.0)) * ((config.bdia_max - config.bdia_min)/2.0)

    current_node = current_node + 1
    assert(current_node == total_num_node)
    return father


# 注意 trainable_variables 的分配
def merge_op(left, right, father):
    global total_num_node
    global current_node

    if right is not None:

        left = merge_op(left.left_child, left.right_child, left)
        right = merge_op(right.left_child, right.right_child, right)

        # logging.info('assign trainable variates for ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')

        dist = calc_dist(left, right)
        right.rec_obj['wirelen'] = dist - left.rec_obj['wirelen']
        assert ((left.father is right.father) and (right.father is father))
        if not father is father.father.right_child:
            if not config.loaded:
                father.rec_obj['wirelen'] = weight(father, 'wirelen')
        if not config.loaded:
            weight(father, 'cdia')
        father.rec_obj['cdia'] = config.cdia_min + (torch.tanh(config.trainable_cdias[father.get_id() + '_' + 'cdia']) - (-1.0)) * ((config.cdia_max - config.cdia_min)/2.0)
        if not config.loaded:
            father.rec_obj['bdia'] = weight(father, 'bdia')
        father.rec_obj['bdia'] = config.bdia_min + (torch.tanh(config.trainable_bdias[father.get_id() + '_' + 'bdia']) - (-1.0)) * ((config.bdia_max - config.bdia_min)/2.0)

        # 使用参数w的传播计算法
        # right.rec_obj['wirelen'] = dist - left.rec_obj['wirelen']  # 这里去除了right.childs应有的影响，全部右节点由左兄弟的子节点影响
        # father.rec_obj['wirelen'] = tf.add((tf.matmul(left.rec_obj['wirelen'], weight()) + bia()), (tf.matmul(right.rec_obj['wirelen'], weight()) + bia()))
        # father.rec_obj['dop'] =  tf.add((tf.matmul(left.rec_obj['dop'], weight()) + bia()), (tf.matmul(right.rec_obj['dop'], weight()) + bia()))
        # father.rec_obj['dop'] = torch.tanh(father.rec_obj['dop'], config.dop_min, config.dop_max)
        # father.rec_obj['diameter'] = tf.add((tf.matmul(left.rec_obj['diameter'], weight()) + bia()), (tf.matmul(right.rec_obj['diameter'], weight()) + bia()))
        # father.rec_obj['diameter'] = torch.tanh(father.rec_obj['diameter'], config.dia_min, config.dia_max)

        logging.info('calculate coordinate of ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')

        father.obj['x'], father.obj['y'] = calc_coordinate(left, right)
        # with tf.name_scope(father.get_id() + '-' + 'coordinate'):
        #     tf.summary.histogram('x', father.obj['x'])
        #     tf.summary.histogram('y', father.obj['y'])
        # 直接将优化参量作为优化参数：直接计算法

    elif left is not None:
        assert (left.father.right_child is None)
        assert (left.father is father)
        left = merge_op(left.left_child, left.right_child, left)

        # logging.info('assign trainable variates for ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')

        if not father is father.father.right_child:
            if not config.loaded:
                father.rec_obj['wirelen'] = weight(father, 'wirelen')
        if not config.loaded:
            weight(father, 'cdia')
        father.rec_obj['cdia'] = config.cdia_min + (torch.tanh(config.trainable_cdias[father.get_id() + '_' + 'cdia']) - (-1.0)) * ((config.cdia_max - config.cdia_min)/2.0)
        if not config.loaded:
            weight(father, 'bdia')
        father.rec_obj['bdia'] = config.bdia_min + (torch.tanh(config.trainable_bdias[father.get_id() + '_' + 'bdia']) - (-1.0)) * ((config.bdia_max - config.bdia_min)/2.0)

        # father.rec_obj['wirelen'] = weight(sess)

        # father.rec_obj['dop'] = tf.matmul(left.rec_obj['dop'], weight()) + bia()
        # father.rec_obj['dop'] = torch.tanh(father.rec_obj['dop'], config.dop_min, config.dop_max)
        #
        # father.rec_obj['diameter'] = tf.matmul(left.rec_obj['diameter'], weight()) + bia()
        # father.rec_obj['diameter'] = torch.tanh(father.rec_obj['diameter'], config.dia_min, config.dia_max)
        logging.info('calculate coordinate of ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')

        father.obj['x'], father.obj['y'] = calc_coordinate(left, right)
        # with tf.name_scope(father.get_id() + '-' + 'coordinate'):
        #     tf.summary.histogram('x', father.obj['x'])
        #     tf.summary.histogram('y', father.obj['y'])

    else:

        assert (father.isleaf is True)

        # logging.info('assign trainable variates for ' + father.get_id() + ' : ' + str(current_node) + '@' + str(total_num_node) + '.')

        if father is not father.father.right_child:
            # 这时left和right的father是个sink
            # 将father.obj设置为trainable
            if not config.loaded:
                father.rec_obj['wirelen'] = weight(father, 'wirelen')
        if not config.loaded:
            weight(father, 'cdia')
        father.rec_obj['cdia'] = config.cdia_min + (torch.tanh(config.trainable_cdias[father.get_id() + '_' + 'cdia']) - (-1.0)) * ((config.cdia_max - config.cdia_min)/2.0) # todo 数学性质不是很好，尝试将线性函数转变为平滑的 arctan 函数来进行 clip
        if not config.loaded:
            weight(father, 'bdia')
        father.rec_obj['bdia'] = config.bdia_min + (torch.tanh(config.trainable_bdias[father.get_id() + '_' + 'bdia']) - (-1.0)) * ((config.bdia_max - config.bdia_min)/2.0)
        # father.rec_obj[0] = tf.add((tf.matmul(fic_left[0], weight()) + bia()), (tf.matmul(fic_right[0], weight()) + bia()))
        # father.rec_obj[1] = tf.add((tf.matmul(fic_left[1], weight()) + bia()), (tf.matmul(fic_right[1], weight()) + bia()))
        # father.rec_obj[1] = torch.tanh(father.rec_obj[1], config.dop_min, config.dop_max)
        # father.rec_obj[2] = tf.add((tf.matmul(fic_left[2], weight()) + bia()), (tf.matmul(fic_right[2], weight()) + bia()))
        # father.rec_obj[2] = torch.tanh(father.rec_obj[2], config.dia_min, config.dia_max)
        # return father

    current_node = current_node + 1
    return father


# 给整个递归神经网络加载参数
# 在optimizer中调用用来计算损失以及反向传播
def coordinate_calc():
    logging.info('network loading...')

    config.tree = merge(config.tree.left_child, config.tree.right_child, config.tree)
    logging.info('network loaded.')

