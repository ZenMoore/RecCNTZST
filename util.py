# 在 parse_topo 中构造一棵二叉树 元树Tree
# #    二叉树的边表示merge对与merge segment关系
# #    二叉树的节点表示一个sink或者merge segment{'r', 'c', 'x', 'y'}
# # 在 recursive 中复制该树的 练树 RecTree
# #    二叉树的边将来作为神经网络连接(w, b)
# #    二叉树的节点将来作为状态{'wirelen', 'diameter_cnt', 'diameter_bundle'}
# # ----for future----
# # 在 optimizer 中复制该树的 影树 ShadowTree
# #    二叉树的边表示merge对与merge segment关系
# #    二叉树的节点表示cnt type和buffer type
# # ----for future----
import config


# todo 为防止optimizer在读取值的时候误更新，可以加一层保护机制，比如，新加只读或者读写的控制参数
# todo 新增数据域 name_scope, 方便修正 variable/tensor 的名称体系，从而方便可视化等等
class Tree:

    # obj = {'r', 'c', 'x', 'y'}
    # rec_obj = {'wirelen','cdia','ddia'}
    # shadow_obj = {'buffer_type'}
    def __init__(self, root_obj=config.meta_ini, father=None):
        self.father = father
        self.obj = root_obj
        if type(self.obj['c']) is str:
            self.obj['c'] = float(self.obj['c'])
        self.rec_obj = config.rec_ini
        self.shadow_obj = config.shadow_ini
        self.left_child = None
        self.right_child = None  # todo 感觉这里设置成 Tree(root_obj=None, father=self)更好，对子树是否为空的判断通过判断root_obj
        # self.id = 0
        self.isleaf = True
        self.num_bend = 0 # 从该点引出的 wire 的折线段数
        self.overlapped = False  # 在wire length=0的时候，该父节点与其中一个子节点重合

    '''树构建函数群'''

    # 自顶向下构建(配合merge函数，用于局部的自顶向下构建)

    # 在调用该函数前先检测左子节点是否为空，若不为空判断右子节点
    def insert_left(self, new_obj):
        self.isleaf = False
        assert (self.left_child is None)
        self.left_child = Tree(new_obj, father=self)

    # 在调用该函数前先检测右子节点是否为空，若不为空则必须以两个子节点为根递归地构造子拓扑
    def insert_right(self, new_obj):
        self.isleaf = False
        assert (self.left_child is not None)
        assert (self.right_child is None)
        self.right_child = Tree(new_obj, father=self)

    # 自底向上构建
    @staticmethod
    def merge(left, right, father):
        father = Tree(father)
        if left is not None:
            father.insert_left(left)
        else:
            raise Exception('error in NS-algo: null point')
        if right is not None:
            father.insert_right(right)
        else:
            raise Exception('error in NS-algo: null point')

        return father

    '''树读取函数群'''

    def get_right(self):
        return self.right_child

    def get_left(self):
        return self.left_child

    def get_father(self):
        return self.father

    def get_bro(self):
        if self.father == None:
            return None
        else:
            if self == self.father.left_child:
                return self.father.right_child
            elif self == self.father.right_child:
                return self.father.left_child
            else:
                raise Exception("Dual-non paradox: ", self)

    # 返回根的值为 obj 的子树的根
    def find(self, obj):
        this = (self.obj['x'], self.obj['y'])
        that = (obj['x'], obj['y'])

        if this == that:
            return self
        elif self.right_child is not None:
            resultL = self.left_child.find(obj)
            resultR = self.right_child.find(obj)
            if resultL is not None:
                assert (resultR is None)
                return resultL
            elif resultR is not None:
                return resultR
            else:
                return None
        elif self.left_child is not None:
            return self.left_child.find(obj)
        else:
            return None

    # 返回 sink+merge point 的总数量
    def size(self):
        count = 1  # count root
        if self.right_child is not None:
            assert (self.left_child is not None)
            count = count + self.left_child.size() + self.right_child.size()
            return count
        elif self.left_child is not None:
            # assert(self.right_child == None)
            count = count + self.left_child.size()
            return count
        else:
            return count

    # 返回所有的sink叶子节点
    def get_sinks(self):
        leaf_set = []
        for sink in config.sink_set:
            leaf_set.append(self.find(sink))
        assert len(leaf_set) == len(config.sink_set)
        return leaf_set

    # '''树复制函数群'''
    # def generate_recTree(self):
    #
    #     return RecTree(self)
    #
    # def generate_shadowTree(self):
    #     return ShadowTree(self)

    '''树扩展功能函数群'''

    def load_node_set(self):
        config.node_set = []

        if self is None:
            return

        config.node_set.append((self.obj['x'], self.obj['y']))

        self.left_child.load_node_set()
        self.right_child.load_node_set()

    # overlapped的变量可以判断父节点与子节点的重合，但是除此之外其他的节点重合是不被允许的
    def check_invalid_overlapped_node(self):
        self.load_node_set()
        self.check_invalid_overlapped_node_op()

    def check_invalid_overlapped_node_op(self):

        if self is None:
            return

        element = (self.obj['x'], self.obj['y'])

        if config.node_set.count(element) == 0:
            raise Exception("A node is not in the node set.")
        elif config.node_set.count(element) == 1:
            pass
        elif config.node_set.count(element) == 2:
            valid = False
            if self.left_child is not None and self.right_child is not None:
                if (self.left_child.obj['x'], self.left_child.obj['y']) == element or (self.right_child.obj['x'], self.right_child.obj['y']) == element:
                    valid = True
            elif self.father is not None:
                if element == (self.father.obj['x'], self.father.obj['y']):
                    valid = True
            else:
                raise Exception("There is an invalid overlapped node in the tree.")

            if not valid:
                raise Exception("There is an invalid overlapped node in the tree.")
        else:
            raise Exception("There is an invalid overlapped node in the tree.")


        self.left_child.check_invalid_overlapped_node_op()
        self.right_child.check_invalid_overlapped_node_op()

    def print(self):  # todo realize
        print("文字打印树")

    def paint(self):  # todo realize
        print("图像打印树")

    # # 这里仅仅比较元树节点的相等与否
    # def equals(self, corres):
    #     return self.obj['x'] == corres.obj['x'] and self.obj['y'] == corres.obj['y']

# # 练树
# class RecTree(Tree):
#
#     # obj = {'wirelen', 'dop', 'diameter'}
#     def __init__(self, tree, father=None):
#         self.father = father
#         self.obj = config.rec_ini  # initialize the obj values
#         if tree.right_child is not None:
#             self.left_child = RecTree(tree.left_child, father= self)
#             self.right_child = RecTree(tree.right_child, father= self)
#         elif tree.left_child is not None:
#             self.left_child = RecTree(tree.left_child)
#         else:
#             self.left_child = None
#             self.right_child = None
#
#     # 计算其在meta_tree中的对应树
#     def find_in_meta(self):
#         path = []
#         temp = self.father
#         while(temp is not None):
#             if(self is temp.left_child):
#                 path.append(0)
#             if (self is temp.right_child):
#                 path.append(1)
#             temp = temp.father
#         result = None
#         path = path.reverse()
#         for i in path:
#             if(i == 0):
#                 result = config.meta_tree.left_child
#             else:
#                 result = config.meta_tree.right_child
#         return result
#
#
# # 影树
# class ShadowTree(Tree):
#
#
#     # obj = (buffer_type)
#     def __init__(self, tree, father=None):
#         self.father = father
#         self.obj = config.shadow_ini  # initialize the obj values
#         if tree.right_child is not None:
#             self.left_child = RecTree(tree.left_child, father= self)
#             self.right_child = RecTree(tree.right_child, father= self)
#         elif tree.left_child is not None:
#             self.left_child = RecTree(tree.left_child)
#         else:
#             self.left_child = None
#             self.right_child = None
#
#     # todo 返回 cnt_types 和 buffer_types 的新的组合，即定义搜索轨迹
#     def next_combination(self):
#         # 初步的想法是一个散列函数，函数值尽量表现某组合的预期收益特征值(即预期收益不同，特征值也不同)
#         # 然后 hash(i) = combination_i, next_combination = hash(i+1)
#         # combination 直接操作 self 树，更新所有节点值 obj
#         pass
#
#     # 计算其在meta_tree中的对应树
#     def find_in_meta(self):
#         return None
