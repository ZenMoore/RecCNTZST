import util.Tree as Tree

# input config
sink_set = []
source_dir = "./source.txt" # todo sink RC 的数据文件，换位置

# node config
# todo 这些都是用来表征或者初始化Tree.node中数据的类型或者值的，可以尝试的一个改进是：将单值统一初始化更改为序列部署初始化(即根据节点序号赋予不同初始值)
meta_ini = (0, 0, 0, 0) # (r, c, x, y)
rec_ini = (0, 0.5) # (wirelen, dop) # 可以尝试DME算法生成序列进行初始化
shadow_ini = (int(0), int(0)) # (cnt_type, buffer_type)


# topo config
meta_tree = Tree()
rec_tree = meta_tree.generate_recTree() # todo 其实这种初始化的设法很占用空间而且没有必要嗷
shadow_tree = meta_tree.generate_shadowTree()
usable = False # 这些默认的树结构是否可以使用

# net config
learning_rate = 0.01
# lambda_op =
