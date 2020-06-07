import numpy as np
import tensorflow as tf
import util

import config

'''相当于前向传播算法'''
# todo 将所有的数值尽量转换为 tensor 的形式，以便生成计算图

scope_id = 0


# 曼哈顿距离
def calc_dist(left, right):
    xl = left.obj['x']
    xr = right.obj['x']
    yl = left.obj['y']
    yr = right.obj['y']
    return tf.add(tf.abs(xl-xr), tf.abs(yl-yr))

# 根据子节点的位置坐标和 wire length 计算父节点的 x, y
def calc_coordinate(left, right):
    x = None
    y = None
    if left.isleaf and type(left.obj['x']) is float:
        left.obj['x'] = tf.convert_to_tensor(left.obj['x'])
        left.obj['y'] = tf.convert_to_tensor(left.obj['y'])
    if right.isleaf and type(right.obj['x']) is float:
        right.obj['x'] = tf.convert_to_tensor(right.obj['x'])
        right.obj['y'] = tf.convert_to_tensor(right.obj['y'])

    dist = calc_dist(left, right)

    # 计算左边和右边的节点node
    state = 0 # 表明两个点的位置关系，但是把水平共线的情况单独设置一个bool变量
    hor = tf.equal(left.obj['y'], right.obj['y'])
    if tf.less(left.obj['x'], right.obj['x']):
        state = 1
        # left_p = left
        # right_p = right
    elif tf.less(right.obj['x'], left.obj['x']):
        state = 2
        # left_p = right
        # right_p = left
    else:
        # 下面需要竖直绕线
        if tf.less(left.obj['y'], right.obj['y']):
            state = 3
            # left_p = left
            # right_p = right
        elif tf.less(right.obj['y'], left.obj['y']):
            state = 4
            # left_p = right
            # right_p = left
        else:
            raise Exception('Node overlapped.') # 是brother重合了

    assert(state != 0)

    if tf.less(left.rec_obj['wirelen'], dist) and tf.less(right.rec_obj['wirelen'], dist): # if left.rec_obj['wirelen'] < dist and right.rec_obj['wirelen'] < dist:
        print("case 1: non detour and non zero wire length")
        if tf.less(left.rec_obj['wirelen'], right.rec_obj['wirelen']):
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

        elif tf.equal(left.rec_obj['wirelen'], right.rec_obj['wirelen']):
            if state == 1:
                x = right.obj['x'] - right.rec_obj['wirelen']
                y = left.obj['y']
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

    elif tf.equal(left.rec_obj['wirelen'], dist): # elif left.rec_obj['wirelen'] == dist:
        print("case 2-1: no detour but left-child has a zero wire length")
        x = right.obj['x']
        y = right.obj['y']

    elif tf.equal(right.rec_obj['wirelen'], dist): # right.rec_obj['wirelen'] == dist:
        print("case 2-2: no detour but right-child has a zero wire length")
        x = left.obj['x']
        y = left.obj['y']

    elif tf.greater(left.rec_obj['wirelen'], dist): # elif left.rec_obj['wirelen'] > dist:
        print('case 3-1: detour and the wirelen of left node is larger')
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

    elif tf.greater(right.rec_obj['wirelen'] > dist): # elif right.rec_obj['wirelen'] > dist:
        print('case 3-2, detour and the wirelen of right node is larger')
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
        print('unknown case')

    return None, None


def weight(trainable=True):
    global scope_id
    scope_id = scope_id + 1
    with tf.name_scope(str(scope_id)):
        weights = tf.get_variable("weights", shape=(1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
    return weights


# def bia(trainable=True):
#     global scope_id
#     scope_id = scope_id + 1
#     with tf.name_scope(str(scope_id)):
#         bias = tf.get_variable("bias", shape=(1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
#     return bias


# 注意 trainable_variables 的分配
def merge(left, right, father):
    if right is not None:

        left = merge(left.left_child, left.right_child, left)
        right = merge(right.left_child, right.right_child, right)
        dist = calc_dist(left, right)
        right.rec_obj['wirelen'] = dist - left.rec_obj['wirelen']
        assert((left.father is right.father) and (right.father is father))
        if not father is father.father.right_child:
            father.rec_obj['wirelen'] = weight()
        father.rec_obj['cdia'] = weight()
        father.rec_obj['cdia'] = tf.clip_by_value(father.rec_obj['cdia'], config.cdia_min, config.cdia_max)
        father.rec_obj['bdia'] = weight()
        father.rec_obj['bdia'] = tf.clip_by_value(father.rec_obj['bdia'], config.bdia_min, config.bdia_max)

        # 使用参数w的传播计算法
        # right.rec_obj['wirelen'] = dist - left.rec_obj['wirelen']  # 这里去除了right.childs应有的影响，全部右节点由左兄弟的子节点影响
        # father.rec_obj['wirelen'] = tf.add((tf.matmul(left.rec_obj['wirelen'], weight()) + bia()), (tf.matmul(right.rec_obj['wirelen'], weight()) + bia()))
        # father.rec_obj['dop'] =  tf.add((tf.matmul(left.rec_obj['dop'], weight()) + bia()), (tf.matmul(right.rec_obj['dop'], weight()) + bia()))
        # father.rec_obj['dop'] = tf.clip_by_value(father.rec_obj['dop'], config.dop_min, config.dop_max)
        # father.rec_obj['diameter'] = tf.add((tf.matmul(left.rec_obj['diameter'], weight()) + bia()), (tf.matmul(right.rec_obj['diameter'], weight()) + bia()))
        # father.rec_obj['diameter'] = tf.clip_by_value(father.rec_obj['diameter'], config.dia_min, config.dia_max)

        # 直接将优化参量作为优化参数：直接计算法

    elif left is not None:
        assert(left.father.right_child is None)
        assert(left.father is father)
        left = merge(left.left_child, left.right_child, left)

        if not father is father.father.right_child:
            father.rec_obj['wirelen'] = weight()
        father.rec_obj['cdia'] = weight()
        father.rec_obj['cdia'] = tf.clip_by_value(father.rec_obj['cdia'], config.cdia_min, config.cdia_max)
        father.rec_obj['bdia'] = weight()
        father.rec_obj['bdia'] = tf.clip_by_value(father.rec_obj['bdia'], config.bdia_min, config.bdia_max)

        # father.rec_obj['wirelen'] = weight()

        # father.rec_obj['dop'] = tf.matmul(left.rec_obj['dop'], weight()) + bia()
        # father.rec_obj['dop'] = tf.clip_by_value(father.rec_obj['dop'], config.dop_min, config.dop_max)
        #
        # father.rec_obj['diameter'] = tf.matmul(left.rec_obj['diameter'], weight()) + bia()
        # father.rec_obj['diameter'] = tf.clip_by_value(father.rec_obj['diameter'], config.dia_min, config.dia_max)

    else:
        if not father is father.father.right_child:
            # 这时left和right的father是个sink
            # 将father.obj设置为trainable
            father.rec_obj['wirelen'] = weight()
        father.rec_obj['cdia'] = weight()
        father.rec_obj['cdia'] = tf.clip_by_value(father.rec_obj['cdia'], config.cdia_min, config.cdia_max)
        father.rec_obj['bdia'] = weight()
        father.rec_obj['bdia'] = tf.clip_by_value(father.rec_obj['bdia'], config.bdia_min, config.bdia_max)
        # father.rec_obj[0] = tf.add((tf.matmul(fic_left[0], weight()) + bia()), (tf.matmul(fic_right[0], weight()) + bia()))
        # father.rec_obj[1] = tf.add((tf.matmul(fic_left[1], weight()) + bia()), (tf.matmul(fic_right[1], weight()) + bia()))
        # father.rec_obj[1] = tf.clip_by_value(father.rec_obj[1], config.dop_min, config.dop_max)
        # father.rec_obj[2] = tf.add((tf.matmul(fic_left[2], weight()) + bia()), (tf.matmul(fic_right[2], weight()) + bia()))
        # father.rec_obj[2] = tf.clip_by_value(father.rec_obj[2], config.dia_min, config.dia_max)
        # return father

    father.obj['x'], father.obj['y'] = calc_coordinate(left, right)
    return father

# 给整个递归神经网络加载参数
# 在optimizer中调用用来计算损失以及反向传播
def load():
        config.tree = merge(config.tree.left_child, config.tree.right_child, config.tree)


