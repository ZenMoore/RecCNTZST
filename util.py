# 在 parse_topo 中构造一棵二叉树 元树Tree
# #    二叉树的边表示merge对与merge segment关系
# #    二叉树的节点表示一个sink或者merge segment{'r', 'c', 'x', 'y'}
# # 在 recursive 中复制该树的 练树 RecTree
# #    二叉树的边将来作为神经网络连接(w, b)
# #    二叉树的节点将来作为状态{'wirelen', 'dop', 'diameter'}
# # ----for future----
# # 在 optimizer 中复制该树的 影树 ShadowTree
# #    二叉树的边表示merge对与merge segment关系
# #    二叉树的节点表示cnt type和buffer type
# # ----for future----
import config

# todo 为防止optimizer在读取值的时候误更新，可以加一层保护机制，比如，新加只读或者读写的控制参数
# 元树
class Tree:

    # obj = tuple(r, c, x, y)
    def __init__(self, root_obj=config.meta_ini, father=None):
        self.father = father
        self.obj = root_obj
        self.left_child = None
        self.right_child = None # todo 感觉这里设置成 Tree(root_obj=None, father=self)更好，对子树是否为空的判断通过判断root_obj
        # self.id = 0
        self.isleaf = True

    '''树构建函数群'''
    # todo 在调用该函数前先检测左子节点是否为空，若不为空判断右子节点
    def insert_left(self, new_obj):
        self.isleaf = False
        assert(self.left_child is None)
        self.left_child = Tree(new_obj, father= self)

    # todo 在调用该函数前先检测右子节点是否为空，若不为空则必须以两个子节点为根递归地构造子拓扑
    def insert_right(self, new_obj):
        self.isleaf = False
        assert (self.right_child is None)
        self.right_child = Tree(new_obj, father= self)

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

    # todo 返回sink+merge segment的总数量
    def size(self):
        count = 1 # count root
        if self.right_child is not None:
            assert(self.left_child is not None)
            count = count + self.left_child.size() + self.right_child.size()
            return count
        elif self.left_child is not None:
            # assert(self.right_child == None)
            count = count + self.left_child.size()
            return count
        else:
            return count

    # todo 返回所有的sink叶子节点, 这个在optimizer中尤其有用
    def get_sinks(self):
        pass

    '''树复制函数群'''
    def generate_recTree(self):
        return RecTree(self)

    def generate_shadowTree(self):
        return ShadowTree(self)

    '''树扩展功能函数群'''
    def print(self):  # todo realize
        print("文字打印树")

    def paint(self):  # todo realize
        print("图像打印树")


# 练树
class RecTree(Tree):

    # obj = tuple(wirelen, dop, diameter)
    def __init__(self, tree, father=None):
        self.father = father
        self.obj = config.rec_ini  # initialize the obj values
        if tree.right_child is not None:
            self.left_child = RecTree(tree.left_child, father= self)
            self.right_child = RecTree(tree.right_child, father= self)
        elif tree.left_child is not None:
            self.left_child = RecTree(tree.left_child)
        else:
            self.left_child = None
            self.right_child = None

    # 计算其在meta_tree中的对应树
    def find_in_meta(self):
        path = []
        temp = self.father
        while(temp is not None):
            if(self is temp.left_child):
                path.append(0)
            if (self is temp.right_child):
                path.append(1)
            temp = temp.father
        result = None
        path = path.reverse()
        for i in path:
            if(i == 0):
                result = config.meta_tree.left_child
            else:
                result = config.meta_tree.right_child
        return result


# 影树
class ShadowTree(Tree):


    # obj = (cnt_type, buffer_type)
    def __init__(self, tree, father=None):
        self.father = father
        self.obj = config.shadow_ini  # initialize the obj values
        if tree.right_child is not None:
            self.left_child = RecTree(tree.left_child, father= self)
            self.right_child = RecTree(tree.right_child, father= self)
        elif tree.left_child is not None:
            self.left_child = RecTree(tree.left_child)
        else:
            self.left_child = None
            self.right_child = None

    # todo 返回 cnt_types 和 buffer_types 的新的组合，即定义搜索轨迹
    def next_combination(self):
        # 初步的想法是一个散列函数，函数值尽量表现某组合的预期收益特征值(即预期收益不同，特征值也不同)
        # 然后 hash(i) = combination_i, next_combination = hash(i+1)
        # combination 直接操作 self 树，更新所有节点值 obj
        pass

    # 计算其在meta_tree中的对应树
    def find_in_meta(self):
        return None

