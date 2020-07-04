import tensorflow as tf

# input config
source_point = None
sink_set = []
headers = []
source_dir = "./benchmark/source.txt"  # todo sink RC 的数据文件

# node config
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

#training configuration
gpu_options = tf.GPUOptions(allow_growth=True)
train_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)

# net config
learning_rate_base = 0.01
learning_rate_decay = 0.8
num_steps = 1000  # 最大迭代轮数
# initialized_weights = []

# technique limitation
# unit=nm
cdia_max = 10
cdia_min = 1
bdia_max = 60
bdia_min = 20

# constants
unit_capacitance = 0.16e-3
R_Q = 6453.2
C_lambda = 888.9
R_cnom = 20000.0
delta = 0.32
pi = 3.1415926
mfp = 1.0

# check config
node_set = None # element = (x, y) # 另外这个变量有时候可能只是子树的 node set

#topo-update
topo_step = 0
max_topo_step = 10

#output config
model_path = './models'
model_name = "model.ckpt"
result_path = './models/results'
tensorboard_dir = './models/visualization'

# temporary variables
