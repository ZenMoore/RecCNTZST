import config
import logging
import numpy as np
# import tensorflow.compat.v1 as tf
import torch
import parse_topo as topoparser
import recursive as network
import os
import parse_out as outparser
import tensorboardX as board
import random
from torchviz import make_dot
import warnings
import hyperparams_searcher as hyperparam

# import tensorboardX as board

'''相当于后向传播算法'''


# todo no_grad?
def get_tensors_max_min(tensors):
    recur_set = []

    max_result = None
    for e in tensors:
        recur_set.append(e)
    while len(recur_set) > 1:
        a = recur_set.pop()
        b = recur_set.pop()
        max_result = torch.where(a >= b, a, b)
        recur_set.append(max_result)

    min_result = None
    for e in tensors:
        recur_set.append(e)
    while len(recur_set) > 1:
        a = recur_set.pop()
        b = recur_set.pop()
        min_result = torch.where(a >= b, b, a)
        recur_set.append(min_result)

    assert (max_result is not None)
    assert (min_result is not None)
    return max_result, min_result


def N_cnt(bdia, cdia):

    return torch.round(torch.div(2.0 * config.pi * torch.square(bdia * 1.0 / 2.0),
                           torch.sqrt(torch.tensor(3.0)) * torch.square(cdia + config.delta)))


def R_c(cdia):

    if cdia < torch.tensor(2.0):
        return config.R_cnom * (torch.div(torch.square(cdia) + -2.811 * cdia + 2.538,
                                          0.5376 * torch.square(cdia) + -0.8106 * cdia + 0.3934))
    else:
        return torch.tensor(config.R_cnom)


def r_s(cdia, wirelen):

    if wirelen > torch.tensor(config.mfp):
        return torch.div(torch.tensor(config.R_Q), config.C_lambda * cdia)
    else:
        return torch.tensor(0.0)


def contact_wire_contact(wirelen, cdia, bdia):
    Rc = R_c(cdia)
    rs = r_s(cdia, wirelen)
    Ncnt = N_cnt(bdia, cdia)
    # div = torch.div(wirelen * r_s(cdia, wirelen), N_cnt(bdia, cdia))
    return 2 * 0.69 * config.unit_capacitance * wirelen * torch.div(config.R_Q + Rc, 2 * Ncnt) \
           + 0.38e+3 * torch.div(wirelen * rs, Ncnt) * config.unit_capacitance * wirelen


def contact_wire_sink(wirelen, cdia, bdia, cap):
    Rc = R_c(cdia)
    Ncnt = N_cnt(bdia, cdia)
    rs = r_s(cdia, wirelen)
    return 0.69 * config.unit_capacitance * wirelen * torch.div(config.R_Q + Rc, 2 * Ncnt) \
            + 0.38e+3 * torch.div(wirelen * rs, Ncnt) * config.unit_capacitance * wirelen \
            + 0.69e+12 * cap * (torch.div(1e3 * wirelen * rs, Ncnt) + torch.div(config.R_Q + Rc, Ncnt))


def sum(tensors):
    sum = torch.tensor(0.0)
    for tensor in tensors:
        sum = torch.add(sum, tensor)
    return sum


# 计算总延时
def calc_delays():

    sink_node_set = config.tree.get_sinks()

    for node in sink_node_set:
        logging.info('calculating delay of sink%d, there remain %d.' % (
            sink_node_set.index(node), len(sink_node_set) - sink_node_set.index(node)))
        # delay = tf.Variable(0, dtype=tf.float32)
        delay = torch.tensor(0.0)
        while node.father is not None:
            delay = delay + calc_node_delay(node)
            node = node.father
        delay = torch.add(delay, calc_root_delay(node))
        config.sink_delay.append(delay)

    assert (len(config.sink_delay) == len(config.sink_set))


def calc_root_delay(node):

    def with_bending():

        horizontal_bia = torch.abs(config.source_point['x'] - node.obj['x'])
        vertical_bia = torch.abs(config.source_point['y'] - node.obj['y'])

        t_horizontal = contact_wire_sink(horizontal_bia, node.rec_obj['cdia'], node.rec_obj['bdia'], config.source_point['c'])

        t_vertical = contact_wire_contact(vertical_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])

        return torch.add(t_horizontal, t_vertical)

    if node.num_bend == torch.tensor(0):
        return torch.tenor(0.0)
    elif node.num_bend == torch.tensor(1):
        return contact_wire_sink(node.rec_obj['wirelen'], node.rec_obj['cdia'], node.rec_obj['bdia'], config.source_point['c'])
    elif node.num_bend == torch.tensor(2):
        return with_bending()
    else:
        raise Exception('bending number of node ' + node.get_id() + ' is abnormal.')


def calc_node_delay(node):
    if node.isleaf:

        def with_bending_sink():
            horizontal_bia = torch.abs(node.father.obj['x'] - node.obj['x'])
            vertical_bia = torch.abs(node.father.obj['y'] - node.obj['y'])

            t_horizontal = contact_wire_sink(horizontal_bia, node.rec_obj['cdia'], node.rec_obj['bdia'], node.obj['c'])

            t_vertical = contact_wire_contact(vertical_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])

            return torch.add(t_horizontal, t_vertical)

        if node.num_bend == torch.tensor(0):
            return 0.69e+12 * node.obj['c'] * torch.div(torch.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                                 N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia']))

        elif node.num_bend == torch.tensor(1):
            return contact_wire_sink(node.rec_obj['wirelen'], node.rec_obj['cdia'], node.rec_obj['bdia'], node.obj['c'])

        elif node.num_bend == torch.tensor(2):
            return with_bending_sink()
        else:
            raise Exception('bending number of node ' + node.get_id() + ' is abnormal.')

    else:
        def with_bending_mp():
            horizontal_bia = torch.abs(node.father.obj['x'] - node.obj['x'])
            vertical_bia = torch.abs(node.father.obj['y'] - node.obj['y'])

            t_horizontal = contact_wire_contact(horizontal_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])

            t_vertical = contact_wire_contact(vertical_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])

            return torch.add(t_horizontal, t_vertical)

        if node.num_bend == torch.tensor(0):
            return torch.tensor(0.0)
        elif node.num_bend == torch.tensor(1):
            return contact_wire_contact(node.rec_obj['wirelen'],node.rec_obj['cdia'], node.rec_obj['bdia'])
        elif node.num_bend == torch.tensor(2):
            return with_bending_mp()
        else:
            raise Exception('bending number of node ' + node.get_id() + ' is abnormal.')


def lag(name):
    weight = torch.empty([], dtype=torch.float)
    # # torch.nn.init.normal_(weight, mean=config.lagrangian_ini, std=config.lagrangian_std)
    torch.nn.init.constant_(weight, config.lagrangian_ini)
    # logging.info(name + ' created and initialized(constant).')
    return weight


# 计算引入拉格朗日乘子后的等式约束
def calc_constraint(tensor):

    # 将拉格朗日乘子作为训练参数，梯度下降时候，向对拉格朗日乘子偏导等于零的方向下降
    # lagrangian = tf.get_variable("lagrangian_multiplier", shape=(), initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                              trainable=True)
    if not config.loaded:
        config.lagranger = lag('lag_multiplier')

    return config.lagranger * tensor


def embed(loss, sigma_delay, sigma_skew, max_delay, lag, max_min_skew, step):
    outparser.draw(loss, sigma_delay, sigma_skew, max_delay, lag, max_min_skew, step)
    return None


def set_nodes_to_show():
    config.nodes_to_show = ['root_left_left']
    logging.info('It will show nodes: ' + str(config.nodes_to_show)[1:-1])
    return None


def calc_between_skews():
    i = 0
    former = None
    for delay in config.sink_delay:
        if i == 0:
            former = delay
        else:
            logging.info('calculate between skew %d' % i)
            skew = torch.abs(delay - former)
            config.between_skew.append(skew)
            logging.info('calculate between skew %d' % i)
            former = delay
        i += 1


def forprop():

    config.sink_delay = []
    config.between_skew = []
    network.coordinate_calc()
    # 计算损失=总延时+等式约束
    logging.info('delay and skew calculating...')

    calc_delays()
    calc_between_skews()

    assert(len(config.sink_delay) == len(config.between_skew)+1)
    sum_delay = sum(config.sink_delay)
    sum_skew = sum(config.between_skew)

    logging.info('delay and skew calculated.')


    logging.info('calculating constraint...')

    lag_constraint = calc_constraint(sum_skew)

    logging.info('constraint calculated.')


    logging.info('calculating goal...')

    # goal = tf.add_n(tf.get_collection('losses'))
    goal = torch.add(sum_delay, lag_constraint)
    logging.info('goal calculated.')

    config.loaded = True
    return goal, sum_delay, sum_skew


# dynamic update algorithm for lag
# return \Delta\lambda
def dua(delta_L, delta_E):
    return torch.tensor(float(-delta_L)/float(delta_E))


# when \Delta\L == 0, it should relax/oscillate \lambda
# return \delta\lambda
def relax(lag):
    if random.randint(0, 1) == 0:
        return -lag/2.0
    else:
        return lag/2.0


# 优化算法也就是反向传播算法
def optimize():

    # tf.reset_default_graph()
    config.trainable_wirelens = {}
    config.trainable_bdias = {}
    config.trainable_cdias = {}
    config.scalar_tree = False
    goal, sum_delay, sum_skew = forprop()

    # 定义训练算法
    logging.info('defining optimizer...')

    train = torch.optim.Adagrad(
        [{'params': list(config.trainable_wirelens.values()), 'lr': config.learning_rate_base['wirelen']},
         {'params': list(config.trainable_cdias.values()), 'lr': config.learning_rate_base['cdia']},
         {'params': list(config.trainable_bdias.values()), 'lr': config.learning_rate_base['bdia']},])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(train, T_max=config.learning_rate_T, eta_min=config.learning_rate_ending) # todo 如何设置不同的 learning_rate_ending
    # scheduler = torch.optim.lr_scheduler.StepLR(train, step_size=1, gamma=1.0)  # todo 如何设置不同的 learning_rate_ending
    logging.info('optimizer defined.')


    # visualization handler
    if not os.path.exists(config.visual_path + '/topo-' + str(config.topo_step)):
        os.makedirs(config.visual_path + '/topo-' + str(config.topo_step))

    writer = board.SummaryWriter(config.visual_path + '/topo-' + str(config.topo_step))
    set_nodes_to_show()
    # g = make_dot(goal, params=config.trainables)
    # g.view(filename='graph-' + str(i) + '.pdf', directory=config.visual_path + '/topo-' + str(config.topo_step))
    # g.view(filename='graph' + '.pdf', directory=config.visual_path + '/topo-' + str(config.topo_step))


    for i in range(config.num_steps):
        logging.info('step-%d@topo-%d backproping...' % (i+1, config.topo_step))

        train.zero_grad()
        # _, goal_value, step = sess.run([train_step, goal, global_step])
        goal.backward(retain_graph=True)
        # delay.backward(retain_graph=True)
        # skew.backward(retain_graph=True)

        train.step()
        scheduler.step()
        logging.info('step-%d@topo-%d backproped...' % (i+1, config.topo_step))

        goal, sum_delay, sum_skew = forprop()

        loss = goal.item()
        logging.info("After %d steps of training, loss value@topo-%d is: %g;" % (i+1, config.topo_step, loss))

        sigma_delay = sum_delay.item()
        logging.info("and the sum delay of the whole tree is: %g;" % sigma_delay)

        lag_multiplier = config.lagranger.item()
        logging.info("and the lagrangian multiplier value is: %g;" % lag_multiplier)

        sigma_skew = sum_skew.item()
        logging.info("and the sum skew is: %g;" % sigma_skew)

        max_delay, min_delay = get_tensors_max_min(config.sink_delay)
        clock_delay = max_delay.item()
        max_min_skew = (max_delay - min_delay).item()
        logging.info("and the max delay is: %g;" % clock_delay)
        logging.info("and the max-min skew is: %g." % max_min_skew)

        writer.add_scalar('loss', loss, i+1)
        writer.add_scalar('sum_delay', sigma_delay, i+1)
        writer.add_scalar('lag', lag_multiplier, i+1)
        writer.add_scalar('sum_skew', sigma_skew, i+1)
        writer.add_scalar('max_delay', clock_delay, i + 1)
        writer.add_scalar('max_min_skew', max_min_skew, i + 1)

        # dynamic update algorithm of lagrangian multiplier
        if i==0:
            former_L = loss
            former_T = sigma_delay
            former_E = sigma_skew
        else:
            # dua lag
            delta_T = sigma_delay - former_T
            delta_E = sigma_skew - former_E
            delta_L = loss - former_L
            if delta_T < 0 and delta_E > 0:
                config.lagranger = config.lagranger + dua(delta_L, delta_E)
            elif delta_T > 0 and delta_E < 0:
                config.lagranger = config.lagranger + dua(delta_L, delta_E)
            elif delta_L == 0 and i != 0:
                config.lagranger = config.lagranger + relax(config.lagranger)
            else:
                config.lagranger = config.lagranger
            config.lagranger = torch.clamp(config.lagranger, min=0.0)
            logging.info('update lagrange multiplier to %g.' % config.lagranger.item())
            former_L = loss
            former_T = sigma_delay
            former_E = sigma_skew

        # todo 这段不知道为什么不能运行，不过影响不大，可以不处理
        # for node in [temp + '_cdia' for temp in config.nodes_to_show]:
        #     cdia = float(config.trainable_cdias[node].item())
        #     writer.add_scalar(node, cdia, i+1)
        #     writer.add_histogram(node, cdia, i+1)
        # for node in [temp + '_bdia' for temp in config.nodes_to_show]:
        #     bdia = float(config.trainable_bdias[node].item())
        #     writer.add_scalar(node, bdia, i+1)
        #     writer.add_histogram(node, bdia, i+1)
        # for node in [temp + '_wirelen' for temp in config.nodes_to_show]:
        #     wirelen = float(config.trainable_wirelens[node].item())
        #     writer.add_scalar(node, wirelen, i+1)
        #     writer.add_histogram(node, wirelen, i+1)

        if i % 10 == 0:
            logging.info('saving model(step-%d@topo-%d)...' % (i + 1, config.topo_step))

            if config.post_embed:
                outparser.point_list_detail(loss, sigma_delay, sigma_skew, clock_delay, lag_multiplier, max_min_skew,
                                            i + 1, config.topo_step)
                logging.info('model(step-%d@topo-%d) saved.' % (i + 1, config.topo_step))
            else:
                outparser.point_list(i + 1)
                logging.info('model(step-%d@topo-%d) saved.' % (i + 1, config.topo_step))
                # embed(loss,sum_delay, sum_skew, final_delay, lag_multiplier, skew_constraint, i + 1)
                embed(loss, sigma_delay, sigma_skew, clock_delay, lag_multiplier, max_min_skew,
                                            i + 1, config.topo_step)

        # if i%1000 == 0:
            # logging.info('making operation graph dot of loss...')
            # g = make_dot(goal, params=config.trainables)
            # logging.info('operation graph dot made, waiting pdf file...')
            # g.view(filename='graph%d@%d'%(), directory=config.visual_path + '/topo-' + str(config.topo_step))
            # logging.info('operation graph of step-%d@topo-%d is shown as pdf.' % (i+1, config.topo_step))

    return True


# 总流程控制
if __name__ == '__main__':
    logging.info('optimizing...')

    for dirname in os.listdir(config.model_path):
        for filename in os.listdir(config.model_path+ '/' + dirname):
            os.remove(config.model_path + '/' + dirname + '/' + filename)

    for dirname in os.listdir(config.visual_path):
        for filename in os.listdir(config.visual_path + '/' + dirname):
            if 'events' in filename:
                os.remove(config.visual_path + '/' + dirname + '/' + filename)

    try:
        # with hyperparam search
        config.print_hyperparams(0)
        while config.topo_step <= config.max_topo_step:
            if config.topo_step == 0:
                if topoparser.parse():
                    optimize()
                    config.topo_step = config.topo_step + 1
                else:
                    raise Exception("tree parsing failed.")
            else:
                if topoparser.update():
                    optimize()
                    config.topo_step = config.topo_step + 1
                else:
                    raise Exception("tree updating failed.")
    except Exception as e:
        logging.exception(e)
