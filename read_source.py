import config

# 返回是否成功读取
def read():
    sink_set = []
    # 从source_dir中读取全部sink并存在sink_set里
    # 每个元素是tuple(r, c, x, y)
    config.sink_set = sink_set
    return True