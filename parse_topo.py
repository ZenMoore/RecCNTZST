import read_source as reader
import util.Tree as Tree

source_dir = "./source.txt" # todo sink RC 的数据文件，换位置

def generate(sink_set):
    # 递归地生成拓扑
    return Tree(root_Obj= None, father= None, num_leaf= len(sink_set))


def parse():
    sink_set = reader.read(source_dir)
    topo = generate(sink_set)
    return topo
