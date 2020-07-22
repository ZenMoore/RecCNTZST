import config
import logging
import numpy as np
# import tensorflow.compat.v1 as tf
import torch
import parse_topo as topoparser
import recursive as network
import os
import parse_out as outparser

'''相当于后向传播算法'''

sink_delay = []

# todo no_grad?
def get_tensors_max_min(tensors):
    recur_set = []

    max_result = None
    for e in tensors:
        recur_set.append(e)
    while len(recur_set) > 1:
        a = recur_set.pop()
        b = recur_set.pop()
        max_result = torch.max(a, b)
        recur_set.append(max_result)

    min_result = None
    for e in tensors:
        recur_set.append(e)
    while len(recur_set) > 1:
        a = recur_set.pop()
        b = recur_set.pop()
        min_result = torch.min(a, b)
        recur_set.append(min_result)

    assert (max_result is not None)
    assert (min_result is not None)
    return max_result, min_result


def N_cnt(bdia, cdia):

    return torch.round(torch.div(2.0 * config.pi * torch.square(bdia * 1.0 / 2.0),
                           torch.sqrt(torch.tensor(3.0)) * torch.square(cdia + config.delta)))


def R_c(cdia):
    # if sess.run(tf.less(cdia, torch.tensor(2.0)))[0] and sess.run(tf.greater(cdia, torch.tensor(1.0)))[0] or sess.run(tf.equal(cdia, torch.tensor(1.0)))[0]:
    #     return torch.mul()(config.R_cnom, tf.div(tf.add(tf.square(cdia), torch.mul()(-2.811, cdia) + 2.538), tf.add(torch.mul()(0.5376, tf.square(cdia)), torch.mul()(-0.8106, cdia) + 0.3934)))
    # else:
    #     return torch.tensor(config.R_cnom)
    if cdia < torch.tensor(2.0):
        return config.R_cnom * (torch.div(torch.square(cdia) + -2.811 * cdia + 2.538,
                                          0.5376 * torch.square(cdia) + -0.8106 * cdia + 0.3934))
    else:
        return torch.tensor(config.R_cnom)
    # return tf.cond(cdia < torch.tensor(2.0),
    #                lambda: torch.mul()(config.R_cnom, tf.div(tf.add(tf.square(cdia), torch.mul()(-2.811, cdia) + 2.538),
    #                                                          tf.add(torch.mul()(0.5376, tf.square(cdia)),
    #                                                                 torch.mul()(-0.8106, cdia) + 0.3934))),
    #                lambda: torch.tensor(config.R_cnom))  # 默认下界被限定


def r_s(cdia, wirelen):
    # if sess.run(tf.greater(wirelen, torch.tensor(config.mfp))):
    #     return tf.div(config.R_Q, torch.mul()(config.C_lambda, cdia))
    # else:
    #     return torch.tensor(0.0)
    if wirelen > torch.tensor(config.mfp):
        return torch.div(torch.tensor(config.R_Q), config.C_lambda * cdia)
    else:
        return torch.tensor(0.0)
    # return tf.cond(wirelen > torch.tensor(config.mfp),
    #                lambda: tf.div(config.R_Q, torch.mul()(config.C_lambda, cdia)),
    #                lambda: torch.tensor(0.0))


def contact_wire_contact(wirelen, cdia, bdia):
    Rc = R_c(cdia)
    rs = r_s(cdia, wirelen)
    Ncnt = N_cnt(bdia, cdia)
    # div = torch.div(wirelen * r_s(cdia, wirelen), N_cnt(bdia, cdia))
    return 2 * 0.69 * config.unit_capacitance * wirelen * torch.div(config.R_Q + Rc, 2 * Ncnt) \
           + 0.38e-6 * torch.div(wirelen * rs, Ncnt) * config.unit_capacitance * wirelen


def contact_wire_sink(wirelen, cdia, bdia, cap):
    Rc = R_c(cdia)
    Ncnt = N_cnt(bdia, cdia)
    rs = r_s(cdia, wirelen)
    return 0.69 * config.unit_capacitance * wirelen * torch.div(config.R_Q + Rc, 2 * Ncnt) \
            + 0.38e-6 * torch.div(wirelen * rs, Ncnt) * config.unit_capacitance * wirelen \
            + 0.69 * cap * torch.div(1e-6 * wirelen * rs, Ncnt) \
            + torch.div(config.R_Q + Rc, Ncnt)


# 计算总延时
def calc_delay():
    global sink_delay

    sink_node_set = config.tree.get_sinks()

    for node in sink_node_set:
        logging.info('calculating delay of sink%d, there remain %d.' % (
            sink_node_set.index(node), len(sink_node_set) - sink_node_set.index(node)))
        # delay = tf.Variable(0, dtype=tf.float32)
        delay = torch.tensor(0.0)
        while node.father is not None:
            delay = torch.add(delay, calc_node_delay(node))
            node = node.father
        delay = torch.add(delay, calc_root_delay(node))
        sink_delay.append(delay)

    assert (len(sink_delay) == len(config.sink_set))
    result, _ = get_tensors_max_min(sink_delay)

    return result  # 必须是tensor数组里面的最大值


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
        return  contact_wire_sink(node.rec_obj['wirelen'], node.rec_obj['cdia'], node.rec_obj['bdia'])
    elif node.num_bend == torch.tensor(2):
        return with_bending()
    else:
        raise Exception('bending number of node ' + node.get_id() + ' is abnormal.')

    # return tf.case({
    #     tf.equal(node.num_bend, torch.tensor(0)): lambda: torch.tensor(0.0),
    #     tf.equal(node.num_bend, torch.tensor(1)): lambda: tf.add(
    #         0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
    #             tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
    #             2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
    #         tf.add(
    #             0.38e-6 * tf.div(
    #                 torch.mul(node.rec_obj['wirelen'],
    #                             r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
    #                 N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul(config.unit_capacitance,
    #                                                                                  node.rec_obj['wirelen']),
    #             0.69 * config.source_point['c'] * tf.add(tf.div(1e-6 * torch.mul(node.rec_obj['wirelen'],
    #                                                                                r_s(node.rec_obj['cdia'],
    #                                                                                    node.rec_obj[
    #                                                                                        'wirelen'])),
    #                                                             N_cnt(node.rec_obj['bdia'],
    #                                                                   node.rec_obj['cdia'])),
    #                                                      tf.div(tf.add(config.R_Q,
    #                                                                    R_c(node.rec_obj['cdia'])),
    #                                                             N_cnt(node.rec_obj['bdia'],
    #                                                                   node.rec_obj['cdia'])))
    #         )),
    #     tf.equal(node.num_bend, torch.tensor(2)): with_bending
    # }, default=lambda: torch.tensor(0.0), exclusive=True)
    # if node.num_bend == 0:
    #     return torch.tensor(0.0)
    #
    # elif node.num_bend == 1:
    #     return tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
    #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
    #         2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
    #                   tf.add(
    #                       0.38e-6 * tf.div(
    #                           torch.mul()(node.rec_obj['wirelen'],
    #                                       r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
    #                           N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(config.unit_capacitance,
    #                                                                                            node.rec_obj['wirelen']),
    #                       0.69 * config.source_point['c'] * tf.add(tf.div(1e-6 * torch.mul()(node.rec_obj['wirelen'],
    #                                                                                          r_s(node.rec_obj['cdia'],
    #                                                                                              node.rec_obj[
    #                                                                                                  'wirelen'])),
    #                                                                       N_cnt(node.rec_obj['bdia'],
    #                                                                             node.rec_obj['cdia'])),
    #                                                                tf.div(tf.add(config.R_Q,
    #                                                                              R_c(node.rec_obj['cdia'])),
    #                                                                       N_cnt(node.rec_obj['bdia'],
    #                                                                             node.rec_obj['cdia'])))
    #                   )
    #                   )
    # else:
    #     horizontal_bia = tf.abs(config.source_point['x'] - node.obj['x'])
    #     vertical_bia = tf.abs(config.source_point['y'] - node.obj['y'])
    #
    #     t_horizontal = tf.add(
    #         0.69 * config.unit_capacitance * horizontal_bia * tf.div(
    #             tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
    #             2 * N_cnt(node.rec_obj['bdia'],
    #                       node.rec_obj['cdia'])),
    #         tf.add(
    #             0.38e-6 * tf.div(torch.mul()(horizontal_bia, r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
    #                              N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(
    #                 config.unit_capacitance, horizontal_bia),
    #             0.69 * config.source_point['c'] * tf.add(
    #                 tf.div(1e-6 * torch.mul()(horizontal_bia, r_s(node.rec_obj['cdia'], horizontal_bia)),
    #                        N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
    #                 tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
    #                        N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])))
    #         )
    #     )
    #
    #     t_vertical = tf.add(
    #         2 * 0.69 * config.unit_capacitance * vertical_bia * tf.div(
    #             tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
    #             2 * N_cnt(node.rec_obj['bdia'],
    #                       node.rec_obj['cdia'])),
    #         0.38e-6 * tf.div(torch.mul()(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
    #                          N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(config.unit_capacitance,
    #                                                                                           vertical_bia),
    #     )
    #
    #     return tf.add(t_horizontal, t_vertical)


def calc_node_delay(node):
    if node.isleaf:

        def with_bending_sink():
            horizontal_bia = torch.abs(node.father.obj['x'] - node.obj['x'])
            vertical_bia = torch.abs(node.father.obj['y'] - node.obj['y'])

            t_horizontal = contact_wire_sink(horizontal_bia, node.rec_obj['cdia'], node.rec_obj['bdia'], node.obj['c'])

            t_vertical = contact_wire_contact(vertical_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])

            return torch.add(t_horizontal, t_vertical)

        if node.num_bend == torch.tensor(0):
            return 0.69 * node.obj['c'] * torch.div(torch.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                                 N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia']))

        elif node.num_bend == torch.tensor(1):
            return contact_wire_sink(node.rec_obj['wirelen'], node.rec_obj['cdia'], node.rec_obj['bdia'], node.obj['c'])

        elif node.num_bend == torch.tensor(2):
            return with_bending_sink()
        else:
            raise Exception('bending number of node ' + node.get_id() + ' is abnormal.')

        # return tf.case({
        #     tf.equal(node.num_bend, torch.tensor(0)):lambda :0.69 * node.obj['c'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #                                          N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
        #     tf.equal(node.num_bend, torch.tensor(1)):lambda :tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
        #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
        #                   tf.add(0.38e-6 * tf.div(torch.mul(node.rec_obj['wirelen'],
        #                                                       r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
        #                                           N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul(
        #                       config.unit_capacitance, node.rec_obj['wirelen']),
        #                          0.69 * node.obj['c'] * tf.add(tf.div(1e-6 * torch.mul(node.rec_obj['wirelen'],
        #                                                                                  r_s(node.rec_obj['cdia'],
        #                                                                                      node.rec_obj['wirelen'])),
        #                                                               N_cnt(node.rec_obj['bdia'],
        #                                                                     node.rec_obj['cdia'])),
        #                                                        tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #                                                               N_cnt(node.rec_obj['bdia'],
        #                                                                     node.rec_obj['cdia'])))
        #                          )
        #                   ),
        #     tf.equal(node.num_bend, torch.tensor(2)): with_bending_sink
        # }, default=lambda :0.69 * node.obj['c'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #                                          N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])), exclusive=True)
        # if node.num_bend == 0:
        #     return 0.69 * node.obj['c'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #                                          N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia']))
        # elif node.num_bend == 1:
        #     return tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
        #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
        #                   tf.add(0.38e-6 * tf.div(torch.mul()(node.rec_obj['wirelen'],
        #                                                       r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
        #                                           N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(
        #                       config.unit_capacitance, node.rec_obj['wirelen']),
        #                          0.69 * node.obj['c'] * tf.add(tf.div(1e-6 * torch.mul()(node.rec_obj['wirelen'],
        #                                                                                  r_s(node.rec_obj['cdia'],
        #                                                                                      node.rec_obj['wirelen'])),
        #                                                               N_cnt(node.rec_obj['bdia'],
        #                                                                     node.rec_obj['cdia'])),
        #                                                        tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #                                                               N_cnt(node.rec_obj['bdia'],
        #                                                                     node.rec_obj['cdia'])))
        #                          )
        #                   )
        # else:
        #     horizontal_bia = tf.abs(node.father.obj['x'] - node.obj['x'])
        #     vertical_bia = tf.abs(node.father.obj['y'] - node.obj['y'])
        #
        #     t_horizontal = tf.add(0.69 * config.unit_capacitance * horizontal_bia * tf.div(
        #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
        #                           tf.add(
        #                               0.38e-6 * tf.div(torch.mul()(horizontal_bia, r_s(node.rec_obj['cdia'],
        #                                                                                node.rec_obj['wirelen'])),
        #                                                N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(
        #                                   config.unit_capacitance, horizontal_bia),
        #                               0.69 * node.obj['c'] * tf.add(
        #                                   tf.div(1e-6 * torch.mul()(horizontal_bia, r_s(node.rec_obj[
        #                                                                                     'cdia'],
        #                                                                                 horizontal_bia)),
        #                                          N_cnt(node.rec_obj['bdia'],
        #                                                node.rec_obj['cdia'])), tf.div(
        #                                       tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #                                       N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])))
        #                           )
        #                           )
        #
        #     t_vertical = tf.add(2 * 0.69 * config.unit_capacitance * vertical_bia * tf.div(
        #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
        #                         0.38e-6 * tf.div(
        #                             torch.mul()(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
        #                             N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(
        #                             config.unit_capacitance, vertical_bia),
        #                         )
        #
        #     return tf.add(t_horizontal, t_vertical)
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

        # return torch.case({
        #     torch.equal(node.num_bend, torch.tensor(0)): lambda :torch.tensor(0.0),
        #     torch.equal(node.num_bend, torch.tensor(1)): lambda : torch.add(2 * 0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * torch.div(
        #         torch.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
        #                   0.38e-6 * torch.div(torch.mul(node.rec_obj['wirelen'],
        #                                                r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
        #                                    N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul(
        #                       config.unit_capacitance, node.rec_obj['wirelen']),
        #                   ),
        #     torch.equal(node.num_bend, torch.tensor(2)): with_bending_mp
        # }, default=lambda: torch.tensor(0.0), exclusive=True)
        # if node.num_bend == 0:
        #     return torch.tensor(0)
        # elif node.num_bend == 1:
        #     return torch.add(2 * 0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * torch.div(
        #         torch.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
        #                   0.38e-6 * torch.div(torch.mul()(node.rec_obj['wirelen'],
        #                                                r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
        #                                    N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(
        #                       config.unit_capacitance, node.rec_obj['wirelen']),
        #                   )
        # else:
        #     horizontal_bia = torch.abs(node.father.obj['x'] - node.obj['x'])
        #     vertical_bia = torch.abs(node.father.obj['y'] - node.obj['y'])
        #
        #     t_horizontal = tf.add(
        #         0.69 * config.unit_capacitance * horizontal_bia * tf.div(
        #             tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #             2 * N_cnt(node.rec_obj['bdia'],
        #                       node.rec_obj['cdia'])),
        #         tf.add(
        #             0.38e-6 * tf.div(
        #                 torch.mul()(horizontal_bia, r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
        #                 N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(
        #                 config.unit_capacitance, horizontal_bia),
        #             0.69 * config.unit_capacitance * tf.add(
        #                 tf.div(1e-6 * torch.mul()(horizontal_bia, r_s(node.rec_obj['cdia'], horizontal_bia)),
        #                        N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
        #                 tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #                        N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])))
        #         )
        #     )
        #
        #     t_vertical = tf.add(2 * 0.69 * config.unit_capacitance * vertical_bia * tf.div(
        #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
        #                         0.38e-6 * tf.div(
        #                             torch.mul()(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
        #                             N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * torch.mul()(
        #                             config.unit_capacitance, vertical_bia),
        #                         )
        #
        #     return tf.add(t_horizontal, t_vertical)


# 计算引入拉格朗日乘子后的等式约束
def calc_lagrange():

    # 将拉格朗日乘子作为训练参数，梯度下降时候，向对拉格朗日乘子偏导等于零的方向下降
    # todo 如何保证偏导下降到等于零，这是一种 trade-off 吗？trade-off 比例参数在哪里设置？
    # lagrangian = tf.get_variable("lagrangian_multiplier", shape=(), initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                              trainable=True)
    if not config.loaded:
        config.lagranger = torch.empty([], requires_grad=True)
        torch.nn.init.normal_(config.lagranger, mean=config.lagrangian_ini, std=config.lagrangian_std)
        # sess.run(lagrangian.initializer)
        config.trainables.append(config.lagranger)
        logging.info('lagrangian multiplier created and initialized(normal).')
    max_delay, min_delay = get_tensors_max_min(sink_delay)

    return config.lagranger * (max_delay - min_delay), max_delay - min_delay

def visualize(loss, delay, lag, skew, step):
    outparser.draw(loss, delay, lag, skew, step)
    return None


def forprop():
    global sink_delay
    sink_delay = []
    network.coordinate_calc()
    # 计算损失=总延时+等式约束
    logging.info('delay calculating...')
    delay = calc_delay()
    # tf.add_to_collection('losses', delay)
    # tf.summary.scalar('delay', delay)
    logging.info('all delay calculated.')

    logging.info('calculating skew and lagrangian multiplier...')
    lag_constraint, skew = calc_lagrange()
    # tf.add_to_collection('losses', lagrange)
    # tf.summary.scalar('lagrangian_multiplier', lagrange)
    # tf.summary.scalar('skew', skew)
    logging.info('skew and lagrangian multiplier calculated.')

    logging.info('calculating goal...')
    # goal = tf.add_n(tf.get_collection('losses'))
    goal = torch.add(delay, lag_constraint)
    logging.info('goal calculated.')
    config.loaded = True
    return goal, delay, skew


# 优化算法也就是反向传播算法
def optimize():

    # tf.reset_default_graph()
    config.trainables = []
    config.scalar_tree = False
    goal, delay, skew = forprop()
    # todo 清空之前的计算图，这里默认新的计算图与旧的计算图之间没有连接相互独立而不影响训练
    # tf.compat.v1.experimental.output_all_intermediates(True)
    # config.trainable_variables = []
    # tensorflow.executing_eagerly()
    # tf.disable_eager_execution()

    # 定义训练算法
    # global_step = torch.tensor(0, name='global_step', requires_grad=False)
    # sess.run(global_step.initializer)
    # logging.info('global step initialized.')

    # sess.run(config.trainable_variables)

    logging.info('defining optimizer...')
    train = torch.optim.Adam(config.trainables, lr=config.learning_rate_base)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(train, T_max=config.learning_rate_T, eta_min=config.learning_rate_ending)
    train.zero_grad()
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(goal, global_step=global_step)
    # print(train_step)
    logging.info('optimizer defined.')


    # logging.info('define learning rate.')
    #
    # learning_rate = tf.train.polynomial_decay(config.learning_rate_base, global_step, 1,
    #                                            config.learning_rate_ending, config.learning_rate_power, cycle = True)  # todo

    # 暂时不使用滑动平均
    # variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # # variable_average_op = variable_average.apply(tf.trainable_variables())
    # with tf.control_dependencies([train_step, variable_average_op]): # 暂时不加入滑动平均
    #     train_op = tf.no_op()

    # logging.info('define saver.')
    # saver = tf.train.Saver()

    # for variable in tf.get_collection(tf.GraphKeys.VARIABLES):
    #     logging.info('initializing ' + str(variable))
    #     variable.initializer.run()

    # logging.info('initializing global variables...')
    # # tf.global_variables_initializer().run()
    # logging.info('global variables initialized.')

    # visualization
    # logging.info('merging all figure...')
    # painter = tf.summary.FileWriter(config.tensorboard_dir + '/topo-' + str(config.topo_step))
    # logging.info('add operation graph.')
    # # painter.add_graph(sess.graph)
    # # all_fig = tf.summary.merge_all()
    # # logging.info('all figure merged.')
    logging.info('graph ignored.')


    for i in range(config.num_steps):
        logging.info('step-' + str(i) + ' backproping...')
        # _, goal_value, step = sess.run([train_step, goal, global_step])
        goal.backward()
        train.step()
        scheduler.step()
        logging.info('step-' + str(i) + ' backproped.')
        goal, delay, skew = forprop()

        if i % 100 == 0:
            loss = goal.item()
            logging.info("After %d steps of training, goal value is: %g ." % (i, loss))
            final_delay = delay.item()
            logging.info("And the total delay of the whole tree is: %g ." % final_delay)
            lag_multiplier = config.lagranger.item()
            logging.info("And the lagrangian multiplier value is: %g ." % lag_multiplier)
            skew_constraint = skew.item()
            logging.info("And the max-min skew: %g ." % skew_constraint)

            logging.info('saving model...')
            outparser.point_list(i)
            logging.info('model saved.')

            # dynamic visualization
            # painter.add_summary(all_fig, i)
            visualize(loss, final_delay, lag_multiplier, skew_constraint, i)
    return None

# 总流程控制
if __name__ == '__main__':
    logging.info('optimizing...')
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
