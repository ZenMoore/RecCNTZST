import numpy as np
import tensorflow as tf
import util

import config

'''相当于前向传播算法'''
# todo 将所有的数值尽量转换为 tensor 的形式，以便生成计算图
rec_tree = None

scope_id = 0


# 欧几里得距离
# todo 如何计算 sink 之间的距离，以及如何计算 merge segment 之间的距离
def calc_dist(left, right):
    temL = left.find_in_meta()
    temR = right.find_in_meta()
    dis = 0
    if (temL.isleaf is True) and (temR.isleaf is True):  # point-point distance
        xl = temL.obj[2], yl = temL.obj[3]
        xr = temR.obj[2], yr = temR.obj[3]
        dis = tf.sqrt(tf.add(tf.square(tf.abs(xl - xr)), tf.square(tf.abs(yl, yr))))
    elif (temL.isleaf is False) and (temR.isleaf is True):  # point-ms distance
        # todo 如果是个圆的话，这个是不需要计算TRR的，所以搁置
        # 改变tree中 x,y的值
        pass
    elif (temL.isleaf is True) and (temR.isleaf is False):  # point-ms distance
        # todo 如果是个圆的话，这个是不需要计算TRR的，所以搁置
        # 改变tree中 x,y的值
        pass
    else:  # ms-ms distance
        # todo 如果是个圆的话，这个是不需要计算TRR的，所以搁置
        # 改变tree中 x,y的值
        pass

    return dis


def weight(trainable=True):
    global scope_id
    scope_id = scope_id + 1
    with tf.name_scope(str(scope_id)):
        weights = tf.get_variable("weights", shape=(1,1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
    return weights


def bia(trainable=True):
    bias = tf.get_variable("bias", shape=(2,1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
    return bias


# 注意 trainable_variables 的分配
def merge(left, right, father):
    if right is not None:
        dist = calc_dist(left, right)
        left = merge(left.left_child, left.right_child, left)
        right = merge(right.left_child, right.right_child, right)
        assert((left.father is right.father) and (right.father is father)) # todo 是==还是is
        right.obj['wirelen'] = dist - left.obj['wirelen'] # 这里去除了right.childs应有的影响，全部右节点由左兄弟的子节点影响
        father.obj['wirelen'] = tf.add((tf.matmul(left.obj['wirelen'], weight()) + bia()), (tf.matmul(right.obj['wirelen'], weight()) + bia()))
        father.obj['dop'] =  tf.add((tf.matmul(left.obj['dop'], weight()) + bia()), (tf.matmul(right.obj['dop'], weight()) + bia()))
        father.obj['dop'] = tf.clip_by_value(father.obj['dop'], config.dop_min, config.dop_max)
        father.obj['diameter'] = tf.add((tf.matmul(left.obj['diameter'], weight()) + bia()), (tf.matmul(right.obj['diameter'], weight()) + bia()))
        father.obj['diameter'] = tf.clip_by_value(father.obj['diameter'], config.dia_min, config.dia_max)
        return father
    elif left is not None:
        assert(left.father == father)
        left = merge(left.left_child, left.right_child, left)
        father.obj['wirelen'] = tf.matmul(left.obj['wirelen'], weight()) + bia()

        father.obj['dop'] = tf.matmul(left.obj['dop'], weight()) + bia()
        father.obj['dop'] = tf.clip_by_value(father.obj['dop'], config.dop_min, config.dop_max)

        father.obj['diameter'] = tf.matmul(left.obj['diameter'], weight()) + bia()
        father.obj['diameter'] = tf.clip_by_value(father.obj['diameter'], config.dia_min, config.dia_max)

        return father
    else:
        # 这时left和right的father是个sink
        # 将father.obj设置为trainable
        fic_left = [tf.constant(1), tf.constant(1), tf.constant(1)]
        fic_right = [tf.constant(1), tf.constant(1), tf.constant(1)]
        father.obj[0] = tf.add((tf.matmul(fic_left[0], weight()) + bia()), (tf.matmul(fic_right[0], weight()) + bia()))
        father.obj[1] = tf.add((tf.matmul(fic_left[1], weight()) + bia()), (tf.matmul(fic_right[1], weight()) + bia()))
        father.obj[1] = tf.clip_by_value(father.obj[1], config.dop_min, config.dop_max)
        father.obj[2] = tf.add((tf.matmul(fic_left[2], weight()) + bia()), (tf.matmul(fic_right[2], weight()) + bia()))
        father.obj[2] = tf.clip_by_value(father.obj[2], config.dia_min, config.dia_max)
        # return father


# 给整个递归神经网络加载参数
# 在optimizer中调用用来计算损失以及反向传播
def load():
    global rec_tree
    rec_tree = config.rec_tree
    assert (config.usable is True)
    rec_tree = merge(rec_tree.left_child, rec_tree.right_child, rec_tree)
    # config.rec_tree = rec_tree


