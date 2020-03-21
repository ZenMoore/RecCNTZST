import config
import numpy as np
import tensorflow as tf
# 相当于后向传播算法


meta_tree = config.meta_tree
shadow_tree = config.shadow_tree


def run_optimizer():
    assert(config.usable is True)
    meta_tree = config.meta_tree
    shadow_tree = config.shadow_tree