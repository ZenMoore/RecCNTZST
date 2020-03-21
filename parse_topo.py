import read_source as reader
import util.Tree as Tree
import config


# 返回是否生成成功
def generate():
    sink_set = config.sink_set
    # 递归地生成拓扑
    config.meta_tree = Tree(root_Obj= None, father= None)
    return True

# 返回是否解析成功
def parse():
    if reader.read():
        return generate()
    else:
        raise Exception("reading failed: ", config.source_dir)
