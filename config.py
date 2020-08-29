import logging
import datetime
import time
import os
from collections import deque
import warnings

# input config
source_point = None
sink_set = []
headers = []
source_dir = "./benchmark/r1"

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

# steps
num_steps = 600  # 最大迭代轮数
max_topo_step = 30
local_step = 100
skew_separate_step = 1000
delay_separate_step = 1000
conver_judge_step = 100

# state config
scalar_tree = True
loaded = True
lagranger = 1.0
sink_delay = []
between_skew = []
# between_goal = []
# constraints = []
delays = {}
mindelays = deque(maxlen=conver_judge_step)
maxdelays = deque(maxlen=conver_judge_step)
paths = []
ad_cont = [] # todo visualization of additional contact

# net config
learning_rate_base = {
    'wirelen': 100.0,
    'cdia': 0.1, #0.1
    'bdia': 1.0, #1.0`
    'lag': 0.00001
}
# learning_rate_decay = 0.8
# learning_rate_ending = {
#     'wirelen':10,
#     'cdia':0.01,
#     'bdia':0.1
# }
# learning_rate_base = 1
learning_rate_ending = 0.1
learning_rate_T = 50
# learning_rate_power = 0.5
lagrangian_ini = 0.0
lagrangian_std = 0.01
# initialized_weights = []

# forprop variables
trainable_wirelens = {}
trainable_cdias = {}
trainable_bdias = {}
trainable_helpers = {}

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
epsilon = 100

# check config
node_set = None # element = (x, y) # 另外这个变量有时候可能只是子树的 node set

# topo-update
topo_step = 0
k = 3

# output config
model_path = './models/pointlist'
visual_path = './models/visualization'
img_path = time.strftime('./models/embedding/%m%d%H%M%S', time.localtime())

# log config
# warnings.filterwarnings('error')
if not os.path.exists(time.strftime('./log/%m%d', time.localtime())):
    os.makedirs(time.strftime('./log/%m%d', time.localtime()))
logger = logging.getLogger()
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=time.strftime('./log/%m%d', time.localtime()) + '/' + datetime.datetime.now().strftime('%H%M%S.log'), level=logging.INFO, format=LOG_FORMAT)
console = logging.StreamHandler()
logger.addHandler(console)


# visualization config
nodes_to_show = []
between_goal_to_show = []
# cdia_histos = [temp + '_cdia' for temp in nodes_to_show]
# bdia_histos = [temp + '_bdia' for temp in nodes_to_show]
# wirelen_histos = [temp + '_wirelen' for temp in nodes_to_show]


# plot config
p_area = 20.0
p_color_s = 'b'
p_color_m = 'g'
p_color_b = 'r'
l_width_max = 5.0
l_width_min = 1.0
l_op_max = 1.0
l_op_min = 0.1
l_color = 'k'


# hyperparams search
hyper_search = False
increase = False
former = None
topo_tree = None
max_hyper_step = 21
judgements = []
judge_step = 50
hyper_step = 0

# launch configuration
post_embed = True
lag_adagrad = True
use_nng = False
skew_separate = False
delay_separate = True
# local_optimize = False

def print_hyperparams(step):
    logging.info(
        '\n--------------hyperparams system %d--------------\n' % step +
        '* initialization=normal(except lag=constant=mean): \n' +
        'means:cdia=%g, bdia=%g, lag=%g\n' % (rec_ini['cdia'], rec_ini['bdia'], lagrangian_ini) +
        'stds: wirelen=%g, cdia=%g, bdia=%g, lag=%g\n' % (wirelen_std, cdia_std, bdia_std, lagrangian_std) +
        '* value bounds: \n' +
        'cdia=%g~%g, bdia=%g~%g\n' % (cdia_min, cdia_max, bdia_min, bdia_max) +
        '* training\n' +
        'learning_rate_schedule=CosineAnnealing\n' +
        'learning_rate_base(except lag=not trainable): wirelen=%g, cdia=%g, bdia=%g, lag=%g\n' % (
        learning_rate_base['wirelen'], learning_rate_base['cdia'], learning_rate_base['bdia'],
        learning_rate_base['lag']) +
        'learning_rate_ending=%g\n' % (learning_rate_ending) +
        'learning_rate_T=%g\n' % (learning_rate_T) +
        '* steps\n' +
        'max_topo_step=%d\n' % (max_topo_step) +
        'training_step=%d\n' % (num_steps) +
        '* other info\n' +
        '|x|	adagrad	L1	maxT+λΦ(Ε)	exact=0	skew_separate\n' +
        '--------------hyperparams system %d--------------' % step
    )