# 在 parse_topo 中构造一棵二叉树
#    二叉树的边表示merge对与merge segment关系
#    二叉树的节点表示一个sink或者merge segment(R, C)
# 在 recursive 中读取这颗二叉树
#    二叉树的边将来作为神经网络连接(w, b)
#    二叉树的节点将来作为状态(wirelen, dop, process)
# ----for future----
# 在 optimizer 中复制一个影子参数
#    二叉树的边表示merge对与merge segment关系
#    二叉树的节点除了R, C外，新加cnt type和buffer type
# ----for future----
class Tree:

    # root_obj 表示 tuple(R,C)
    def __init__(self, root_obj, father, num_leaf):
        self.father = father
        self.key = root_obj
        self.left_child = None
        self.right_child = None
        self.num_leaf = num_leaf # read_source读取的sink set的大小

    # todo 在插入前先检测左子节点是否为空，若不为空判断右子节点
    def insert_left(self, new_node):
        assert(self.left_child == None)
        self.left_child = Tree(new_node, self)

    # todo 在插入前先检测右子节点是否为空，若不为空则必须以两个子节点为根递归地构造子拓扑
    def insert_right(self, new_node):
        assert (self.right_child == None)
        self.right_child = Tree(new_node)

    def get_right(self):
        return self.right_child

    def get_left(self):
        return self.left_child

    def get_father(self):
        return self.father

    def get_num_leaf(self):
        return self.num_leaf

    # todo 返回sink+merge segment的总数量
    def size(self):
        num = len(self.get_num_leaf())
        temp = int(num / 2)
        while(temp != 0):
            num += temp
            temp = int(temp/2)
        return num

    # 在 optimizer 中控制 cnts, buffers 两个序列，并且定义搜索轨迹，每次试验时生成影子树并在影子树上训练
    # def shadow(self, cnts, buffers):




