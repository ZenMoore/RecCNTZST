import config
import numpy as np
import tensorflow as tf
import parse_topo as topoparser
import recursive as loader

'''相当于后向传播算法'''
meta_tree = None
rec_tree = None
shadow_tree = None

# 计算总延时
def calc_delay():
    return None

# 计算引入拉格朗日乘子后的等式约束
def calc_lagrange():
    return None

# todo 优化算法也就是反向传播算法
def optimize():
    global rec_tree # todo 这些global到底共享地址吗？能不能去掉直接用变量名rec_tree等(不加config.xx, 也不用提前global之后用config.xx赋值)

    for i in range(config.num_steps):
        loader.load()
        rec_tree = config.rec_tree
        # delay = calc_delay()
        # tf.add_to_collection('losses', delay)
        # lagrange = calc_lagrange()
        # tf.add_to_collection('losses', lagrange)
        # goal = tf.add_n(tf.get_collection('losses'))
        

    return None

def main(argv = None):
    optimize()


if __name__ == '__main__':
    global meta_tree
    global rec_tree

    if not config.usable:
        if topoparser.parse():
            config.rec_tree = config.meta_tree.generate_recTree()
            config.shadow_tree = config.meta_tree.generate_shadowTree()
            config.usable = True
        else:
            raise Exception("meta_tree parsing failed.")
    meta_tree = config.meta_tree
    rec_tree = config.rec_tree
    shadow_tree = config.rec_tree
    tf.app.run()
