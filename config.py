
# input config
source_point = None
sink_set = []
headers = []
source_dir = "./benchmark/source.txt"  # todo sink RC 的数据文件

# node config
# todo 这些都是用来表征或者初始化Tree.node中数据的类型或者值的，可以尝试的一个改进是：将单值统一初始化更改为序列部署初始化(即根据节点序号赋予不同初始值)
meta_ini = {'r':0,
            'c':0,
            'x':0,
            'y':0}  # (r, c, x, y)
rec_ini = {'wirelen':0,
           'cdia':2,
           'bdia':40}  # (wirelen, diameter_cnt, diameter_bundle) # 可以尝试DME算法生成序列进行初始化
shadow_ini = int(0)  # buffer_type


# topo config
tree = None

# net config
learning_rate_base = 0.01
learning_rate_decay = 0.8
num_steps = 1000  # 最大迭代轮数

# technique limitation
# unit=nm
cdia_max = 4
cdia_min = 1
ddia_max = 60
ddia_min = 20

# constants
unit_capacitance = 0.16

# output config
node_set = None # element = (x, y) # 另外这个变量有时候可能只是子树的 node set
model_path = './models'
model_name = "model.ckpt"

# temporary variables
