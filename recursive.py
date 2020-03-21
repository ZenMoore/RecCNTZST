import numpy as np
import tensorflow as tf
import util
import parse_topo as topoparser
import config

# 相当于前向传播算法

if not config.usable:
    if topoparser.parse():
        config.rec_tree = config.meta_tree.generate_recTree()
        config.shadow_tree = config.meta_tree.generate_shadowTree()
        config.usable = True
    else:
        raise Exception("meta_tree parsing failed.")

meta_tree = config.meta_tree
rec_tree = config.rec_tree


