import config
import numpy as np
import tensorflow as tf
import parse_topo as topoparser

'''相当于后向传播算法'''
meta_tree = None
rec_tree = None
shadow_tree = None

# 计算损失
def calcu_loss():
    return None

# todo 优化算法也就是反向传播算法
def optimize():
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
