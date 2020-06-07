import read_source as reader
import util
import numpy as np
import config


# 计算点到点的距离: 曼哈顿距离
def calc_dist(left, right):
    xl = left['x']
    yl = left['y']
    xr = right['x']
    yr = right['y']
    return np.add(np.abs(xl-xr), np.abs(yl-yr))


def get_nearest(recur_set):
    result = (recur_set[0], recur_set[1])
    minlen = calc_dist(result[0], result[1])
    for left in recur_set:
        for right in recur_set:
            if left == right:
                continue
            temp_len = calc_dist(left, right)
            if temp_len < minlen:
                minlen = temp_len
                result = (left, right)
    return minlen, result[0], result[1]


# 计算中点，根据重点进行计算，改进方法见 generate 函数注释
def merge_point(left, right):
    point = {'r': None,
             'c': None,
             'x': (left['x']+right['x'])/2,
             'y': (left['y']+right['y'])/2}
    return point


# 根据路径递归的构建树结构
def construct(construct_path):

    # 确定根部

    temp = construct_path[-1]
    root = util.Tree.merge(temp[0], temp[1], temp[2])
    construct_path.remove(temp) # todo test

    # corres = root.find(construct_path[-1][2])
    # assert (corres is not None) # 一定在前面的树中出现过
    # corres.insert_left()
    while len(construct_path) != 0:
        print('___')
        temp = construct_path[-1]
        print(temp)
        corres = root.find(temp[2])
        print(corres)
        assert (corres is not None)  # 一定在前面的树中出现过
        assert (corres.left_child is None)
        corres.insert_left(temp[0])
        assert (corres.right_child is None)
        corres.insert_right(temp[1])
        construct_path.remove(temp)
    return root


def print_path(construct_path):
    for e in construct_path:
        print(e)


# 返回是否生成成功
# using nearest neighbor selection
# todo 改进，先按照 MMM-Mode 构建初始拓扑，跑完前向收敛后按照相应参数重新构建拓扑，循环往复几遍，认为可达到最优拓扑
def generate():
    sink_set = config.sink_set
    recur_set = []
    for i in sink_set:
        recur_set.append(i)
    root = None
    # merge_points = [] # 由于 merge_point 仍然在 recur_set 中，存在重复生成 tree node 的风险，所以维护一个列表核对存在性,元素为tuple(x, y)
    # merge_nodes = [] # merge_points 一一对应的元素为 tree node 的表
    construct_path = [] # 存储树的构建路径，元素为tuple(left, right, merge_point)
    merging_point = None

    while len(recur_set) > 1:
        # nearest neighbor
        _, left, right = get_nearest(recur_set)

        # print(str(left['x']) +', ' + str(left['y']))
        # print(str(right['x']) + ', ' + str(right['y']))
        # update recur_set
        recur_set.remove(left)

        recur_set.remove(right)
        merging_point = merge_point(left, right)
        recur_set.append(merging_point)

        # merge tree topo
        # merge_points.append((merge_point['x'], merge_point['y']))
        # if ((left['x'], left['y']) not in merge_points) and ((right['x'], right['y']) not in merge_points):
        #     root = Tree.merge(left, right, merge_point)
        #     merge_points.append(root)
        # elif ((left['x'], left['y']) in merge_points) and ((right['x'], right['y']) not in merge_points):
        #     pass
        # elif ((left['x'], left['y']) not in merge_points) and ((right['x'], right['y']) in merge_points):
        #     pass
        # else:
        #     pass

        construct_path.append((left, right, merging_point))

    print_path(construct_path)
    root = construct(construct_path)

    # 递归地生成拓扑
    config.tree = root
    return True  # always


# 返回是否解析成功
def parse():
    if reader.read():
        return generate()
    else:
        raise Exception("reading failed: ", config.source_dir)


if __name__ == '__main__':
    if parse():
        root = config.tree

    print("The element number in the sink set is "+str(len(config.sink_set)))
    print("The size of built tree is "+root.size())
