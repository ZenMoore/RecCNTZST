import numpy as np
import tensorflow as tf
import util

import config

'''相当于前向传播算法'''
# todo 将所有的数值尽量转换为tensor的形式，以便生成计算图
meta_tree = None
rec_tree = None

def calc_dist(left, right):
    temL = left.find_in_meta()
    temR = right.find_in_meta()
    xl = temL.obj[2], yl = temL.obj[3]
    xr = temR.obj[2], yr = temR.obj[3]
    return tf.add(tf.abs(xl - xr), tf.abs(yl, yr))


# todo name_scope is necessary
def weight(trainable=True):
    weights = tf.get_variable("weights", shape=(1,1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
    return weights

# todo name_scope is necessary
def bia(trainable=True):
    bias = tf.get_variable("bias", shape=(2,1), initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
    return bias

# 注意 trainable_variables 的分配
def merge(left, right, father):
    if right is not None:
        dist = calc_dist(left, right)
        left = merge(left.left_child, left.right_child, left)
        right = merge(right.left_child, right.right_child, right)
        assert(left.father == right.father == father) # todo 是==还是is
        right.obj[0] = dist - left.obj[0]
        father.obj[0] = tf.add((tf.matmul(left.obj[0], weight()) + bia()), (tf.matmul(right.obj[0], weight()) + bia()))
        father.obj[1] =  tf.add((tf.matmul(left.obj[1], weight()) + bia()), (tf.matmul(right.obj[1], weight()) + bia()))
        father.obj[2] = tf.add((tf.matmul(left.obj[2], weight()) + bia()), (tf.matmul(right.obj[2], weight()) + bia()))
        return father
    elif left is not None:
        assert(left.father == father)
        left = merge(left.left_child, left.right_child, left)
        father.obj[0] = tf.matmul(left.obj[0], weight()) + bia()
        father.obj[1] = tf.matmul(left.obj[1], weight()) + bia()
        father.obj[2] = tf.matmul(left.obj[2], weight()) + bia()
        return father
    else:
        # 这时left和right的father是个sink
        # 将father.obj设置为trainable
        return father


# 给整个递归神经网络加载参数
# 在optimizer中调用用来计算损失以及反向传播
def load():
    global rec_tree
    assert (config.usable is True)
    meta_tree = config.meta_tree
    rec_tree = config.rec_tree
    rec_tree = merge(rec_tree.left_child, rec_tree.right_child, rec_tree)
    config.rec_tree = rec_tree


