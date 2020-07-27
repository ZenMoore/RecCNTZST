import logging
import datetime

# input config
source_point = None
sink_set = []
headers = []
source_dir = "./benchmark/source.txt"

# node config
meta_ini = {'r':None,
            'c':None,
            'x':0.0,
            'y':0.0}  # (r, c, x, y)
rec_ini = {'wirelen':20.0,
           'cdia':2.0,
           'bdia':40.0}  # (wirelen, diameter_cnt, diameter_bundle) # 可以尝试DME算法生成序列进行初始化
shadow_ini = int(0)  # buffer_type
wirelen_std = 10.0
cdia_std = 0.5
bdia_std = 10.0


# topo config
tree = None

# state config
scalar_tree = True
loaded = True
lagranger = 0.0
sink_delay = []

# net config
learning_rate_base = 0.1
# learning_rate_decay = 0.8
learning_rate_ending = 0.01
learning_rate_T = 10
# learning_rate_power = 0.5
lagrangian_ini = 10.0
lagrangian_std = 1.0
num_steps = 1000  # 最大迭代轮数
# initialized_weights = []

# forprop variables
trainables = {}

# technique limitation
# unit=nm
cdia_max = 10.0
cdia_min = 1.0
bdia_max = 60.0
bdia_min = 20.0

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

# topo-update
topo_step = 0
max_topo_step = 0

# output config
model_path = './models'
visual_path = './models/visualization'

# log config
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='./log/console-' + datetime.datetime.now().strftime('%m%d-%H%M%S.log'), level=logging.INFO, format=LOG_FORMAT)

# visualization config
p_area = 20.0
p_color_s = 'b'
p_color_m = 'g'
p_color_b = 'r'
l_width_max = 5.0
l_width_min = 1.0
l_op_max = 1.0
l_op_min = 0.1
l_color = 'k'