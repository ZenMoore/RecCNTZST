import config
import logging
import numpy as np
import tensorflow.compat.v1 as tf
import parse_topo as topoparser
import recursive as loader
import os
import parse_out as outparser

'''相当于后向传播算法'''

sink_delay = []


def get_tensors_max_min(tensors):
    recur_set = []

    max_result = None
    for e in tensors:
        recur_set.append(e)
    while len(recur_set) > 1:
        a = recur_set.pop()
        b = recur_set.pop()
        max_result = tf.maximum(a, b)
        recur_set.append(max_result)

    min_result = None
    for e in tensors:
        recur_set.append(e)
    while len(recur_set) > 1:
        a = recur_set.pop()
        b = recur_set.pop()
        min_result = tf.minimum(a, b)
        recur_set.append(min_result)

    assert (max_result is not None)
    assert (min_result is not None)
    return max_result, min_result


def N_cnt(bdia, cdia):
    return tf.round(tf.div(tf.multiply(2.0 * config.pi, tf.square(tf.multiply(bdia, 1.0 / 2.0))),
                           tf.multiply(tf.sqrt(3.0), tf.square(tf.add(cdia, config.delta)))))


def R_c(cdia):
    # if sess.run(tf.less(cdia, tf.convert_to_tensor(2.0)))[0] and sess.run(tf.greater(cdia, tf.convert_to_tensor(1.0)))[0] or sess.run(tf.equal(cdia, tf.convert_to_tensor(1.0)))[0]:
    #     return tf.multiply(config.R_cnom, tf.div(tf.add(tf.square(cdia), tf.multiply(-2.811, cdia) + 2.538), tf.add(tf.multiply(0.5376, tf.square(cdia)), tf.multiply(-0.8106, cdia) + 0.3934)))
    # else:
    #     return tf.convert_to_tensor(config.R_cnom)
    return tf.cond(cdia < tf.convert_to_tensor(2.0),
                   lambda: tf.multiply(config.R_cnom, tf.div(tf.add(tf.square(cdia), tf.multiply(-2.811, cdia) + 2.538),
                                                             tf.add(tf.multiply(0.5376, tf.square(cdia)),
                                                                    tf.multiply(-0.8106, cdia) + 0.3934))),
                   lambda: tf.convert_to_tensor(config.R_cnom))  # 默认下界被限定


def r_s(cdia, wirelen):
    # if sess.run(tf.greater(wirelen, tf.convert_to_tensor(config.mfp))):
    #     return tf.div(config.R_Q, tf.multiply(config.C_lambda, cdia))
    # else:
    #     return tf.convert_to_tensor(0.0)
    return tf.cond(wirelen > tf.convert_to_tensor(config.mfp),
                   lambda: tf.div(config.R_Q, tf.multiply(config.C_lambda, cdia)),
                   lambda: tf.convert_to_tensor(0.0))


# 计算总延时
def calc_delay():

    sink_node_set = config.tree.get_sinks()

    for node in sink_node_set:
        logging.info('calculating delay of sink%d, there remain %d.' % (
            sink_node_set.index(node), len(sink_node_set) - sink_node_set.index(node)))
        delay = tf.Variable(0, dtype=tf.float32)
        while node.father is not None:
            delay = tf.add(delay, calc_node_delay(node))
            node = node.father
        delay = tf.add(delay, calc_root_delay(node))
        sink_delay.append(delay)

    assert (len(sink_delay) == len(config.sink_set))
    result, _ = get_tensors_max_min(sink_delay)

    return result  # 必须是tensor数组里面的最大值


def calc_root_delay(node):

    def with_bending():
        horizontal_bia = tf.abs(config.source_point['x'] - node.obj['x'])
        vertical_bia = tf.abs(config.source_point['y'] - node.obj['y'])

        t_horizontal = tf.add(
            0.69 * config.unit_capacitance * horizontal_bia * tf.div(
                tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                2 * N_cnt(node.rec_obj['bdia'],
                          node.rec_obj['cdia'])),
            tf.add(
                0.38e-6 * tf.div(tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
                                 N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
                    config.unit_capacitance, horizontal_bia),
                0.69 * config.source_point['c'] * tf.add(
                    tf.div(1e-6 * tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'], horizontal_bia)),
                           N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
                    tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                           N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])))
            )
        )

        t_vertical = tf.add(
            2 * 0.69 * config.unit_capacitance * vertical_bia * tf.div(
                tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                2 * N_cnt(node.rec_obj['bdia'],
                          node.rec_obj['cdia'])),
            0.38e-6 * tf.div(tf.multiply(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
                             N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(config.unit_capacitance,
                                                                                              vertical_bia),
        )

        return tf.add(t_horizontal, t_vertical)

    return tf.case({
        tf.equal(node.num_bend, tf.convert_to_tensor(0)): lambda: tf.convert_to_tensor(0.0),
        tf.equal(node.num_bend, tf.convert_to_tensor(1)): lambda: tf.add(
            0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
                tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
            tf.add(
                0.38e-6 * tf.div(
                    tf.multiply(node.rec_obj['wirelen'],
                                r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
                    N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(config.unit_capacitance,
                                                                                     node.rec_obj['wirelen']),
                0.69 * config.source_point['c'] * tf.add(tf.div(1e-6 * tf.multiply(node.rec_obj['wirelen'],
                                                                                   r_s(node.rec_obj['cdia'],
                                                                                       node.rec_obj[
                                                                                           'wirelen'])),
                                                                N_cnt(node.rec_obj['bdia'],
                                                                      node.rec_obj['cdia'])),
                                                         tf.div(tf.add(config.R_Q,
                                                                       R_c(node.rec_obj['cdia'])),
                                                                N_cnt(node.rec_obj['bdia'],
                                                                      node.rec_obj['cdia'])))
            )),
        tf.equal(node.num_bend, tf.convert_to_tensor(2)): with_bending
    }, default=lambda: tf.convert_to_tensor(0.0), exclusive=True)
    # if node.num_bend == 0:
    #     return tf.convert_to_tensor(0.0)
    #
    # elif node.num_bend == 1:
    #     return tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
    #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
    #         2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
    #                   tf.add(
    #                       0.38e-6 * tf.div(
    #                           tf.multiply(node.rec_obj['wirelen'],
    #                                       r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
    #                           N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(config.unit_capacitance,
    #                                                                                            node.rec_obj['wirelen']),
    #                       0.69 * config.source_point['c'] * tf.add(tf.div(1e-6 * tf.multiply(node.rec_obj['wirelen'],
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
    #             0.38e-6 * tf.div(tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
    #                              N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
    #                 config.unit_capacitance, horizontal_bia),
    #             0.69 * config.source_point['c'] * tf.add(
    #                 tf.div(1e-6 * tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'], horizontal_bia)),
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
    #         0.38e-6 * tf.div(tf.multiply(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
    #                          N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(config.unit_capacitance,
    #                                                                                           vertical_bia),
    #     )
    #
    #     return tf.add(t_horizontal, t_vertical)


def calc_node_delay(node):
    if node.isleaf:

        def with_bending_sink():
            horizontal_bia = tf.abs(node.father.obj['x'] - node.obj['x'])
            vertical_bia = tf.abs(node.father.obj['y'] - node.obj['y'])

            t_horizontal = tf.add(0.69 * config.unit_capacitance * horizontal_bia * tf.div(
                tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                2 * N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
                                  tf.add(
                                      0.38e-6 * tf.div(tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'],
                                                                                       node.rec_obj['wirelen'])),
                                                       N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
                                          config.unit_capacitance, horizontal_bia),
                                      0.69 * node.obj['c'] * tf.add(
                                          tf.div(1e-6 * tf.multiply(horizontal_bia, r_s(node.rec_obj[
                                                                                            'cdia'],
                                                                                        horizontal_bia)),
                                                 N_cnt(node.rec_obj['bdia'],
                                                       node.rec_obj['cdia'])), tf.div(
                                              tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                              N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])))
                                  )
                                  )

            t_vertical = tf.add(2 * 0.69 * config.unit_capacitance * vertical_bia * tf.div(
                tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
                                0.38e-6 * tf.div(
                                    tf.multiply(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
                                    N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
                                    config.unit_capacitance, vertical_bia),
                                )

            return tf.add(t_horizontal, t_vertical)

        return tf.case({
            tf.equal(node.num_bend, tf.convert_to_tensor(0)):lambda :0.69 * node.obj['c'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                                 N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
            tf.equal(node.num_bend, tf.convert_to_tensor(1)):lambda :tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
                tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                2 * N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
                          tf.add(0.38e-6 * tf.div(tf.multiply(node.rec_obj['wirelen'],
                                                              r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
                                                  N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
                              config.unit_capacitance, node.rec_obj['wirelen']),
                                 0.69 * node.obj['c'] * tf.add(tf.div(1e-6 * tf.multiply(node.rec_obj['wirelen'],
                                                                                         r_s(node.rec_obj['cdia'],
                                                                                             node.rec_obj['wirelen'])),
                                                                      N_cnt(node.rec_obj['bdia'],
                                                                            node.rec_obj['cdia'])),
                                                               tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                                                      N_cnt(node.rec_obj['bdia'],
                                                                            node.rec_obj['cdia'])))
                                 )
                          ),
            tf.equal(node.num_bend, tf.convert_to_tensor(2)): with_bending_sink
        }, default=lambda :0.69 * node.obj['c'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                                                 N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])), exclusive=True)
        # if node.num_bend == 0:
        #     return 0.69 * node.obj['c'] * tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #                                          N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia']))
        # elif node.num_bend == 1:
        #     return tf.add(0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
        #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia'])),
        #                   tf.add(0.38e-6 * tf.div(tf.multiply(node.rec_obj['wirelen'],
        #                                                       r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
        #                                           N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
        #                       config.unit_capacitance, node.rec_obj['wirelen']),
        #                          0.69 * node.obj['c'] * tf.add(tf.div(1e-6 * tf.multiply(node.rec_obj['wirelen'],
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
        #                               0.38e-6 * tf.div(tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'],
        #                                                                                node.rec_obj['wirelen'])),
        #                                                N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
        #                                   config.unit_capacitance, horizontal_bia),
        #                               0.69 * node.obj['c'] * tf.add(
        #                                   tf.div(1e-6 * tf.multiply(horizontal_bia, r_s(node.rec_obj[
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
        #                             tf.multiply(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
        #                             N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
        #                             config.unit_capacitance, vertical_bia),
        #                         )
        #
        #     return tf.add(t_horizontal, t_vertical)
    else:

        def with_bending_mp():
            horizontal_bia = tf.abs(node.father.obj['x'] - node.obj['x'])
            vertical_bia = tf.abs(node.father.obj['y'] - node.obj['y'])

            t_horizontal = tf.add(
                0.69 * config.unit_capacitance * horizontal_bia * tf.div(
                    tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                    2 * N_cnt(node.rec_obj['bdia'],
                              node.rec_obj['cdia'])),
                tf.add(
                    0.38e-6 * tf.div(
                        tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
                        N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
                        config.unit_capacitance, horizontal_bia),
                    0.69 * config.unit_capacitance * tf.add(
                        tf.div(1e-6 * tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'], horizontal_bia)),
                               N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
                        tf.div(tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                               N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])))
                )
            )

            t_vertical = tf.add(2 * 0.69 * config.unit_capacitance * vertical_bia * tf.div(
                tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
                                0.38e-6 * tf.div(
                                    tf.multiply(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
                                    N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
                                    config.unit_capacitance, vertical_bia),
                                )

            return tf.add(t_horizontal, t_vertical)

        return tf.case({
            tf.equal(node.num_bend, tf.convert_to_tensor(0)): lambda :tf.convert_to_tensor(0.0),
            tf.equal(node.num_bend, tf.convert_to_tensor(1)): lambda : tf.add(2 * 0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
                tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
                2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
                          0.38e-6 * tf.div(tf.multiply(node.rec_obj['wirelen'],
                                                       r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
                                           N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
                              config.unit_capacitance, node.rec_obj['wirelen']),
                          ),
            tf.equal(node.num_bend, tf.convert_to_tensor(2)): with_bending_mp
        }, default=lambda: tf.convert_to_tensor(0.0), exclusive=True)
        # if node.num_bend == 0:
        #     return tf.convert_to_tensor(0)
        # elif node.num_bend == 1:
        #     return tf.add(2 * 0.69 * config.unit_capacitance * node.rec_obj['wirelen'] * tf.div(
        #         tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #         2 * N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])),
        #                   0.38e-6 * tf.div(tf.multiply(node.rec_obj['wirelen'],
        #                                                r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
        #                                    N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
        #                       config.unit_capacitance, node.rec_obj['wirelen']),
        #                   )
        # else:
        #     horizontal_bia = tf.abs(node.father.obj['x'] - node.obj['x'])
        #     vertical_bia = tf.abs(node.father.obj['y'] - node.obj['y'])
        #
        #     t_horizontal = tf.add(
        #         0.69 * config.unit_capacitance * horizontal_bia * tf.div(
        #             tf.add(config.R_Q, R_c(node.rec_obj['cdia'])),
        #             2 * N_cnt(node.rec_obj['bdia'],
        #                       node.rec_obj['cdia'])),
        #         tf.add(
        #             0.38e-6 * tf.div(
        #                 tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'], node.rec_obj['wirelen'])),
        #                 N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
        #                 config.unit_capacitance, horizontal_bia),
        #             0.69 * config.unit_capacitance * tf.add(
        #                 tf.div(1e-6 * tf.multiply(horizontal_bia, r_s(node.rec_obj['cdia'], horizontal_bia)),
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
        #                             tf.multiply(vertical_bia, r_s(node.rec_obj['cdia'], vertical_bia)),
        #                             N_cnt(node.rec_obj['bdia'], node.rec_obj['cdia'])) * tf.multiply(
        #                             config.unit_capacitance, vertical_bia),
        #                         )
        #
        #     return tf.add(t_horizontal, t_vertical)


# 计算引入拉格朗日乘子后的等式约束
def calc_lagrange():

    # 将拉格朗日乘子作为训练参数，梯度下降时候，向对拉格朗日乘子偏导等于零的方向下降
    # todo 如何保证偏导下降到等于零，这是一种 trade-off 吗？trade-off 比例参数在哪里设置？
    lagrangian = tf.get_variable("lagrangian_multiplier", shape=(), initializer=tf.truncated_normal_initializer(stddev=0.1),
                                 trainable=True)

    # sess.run(lagrangian.initializer)
    # logging.info('lagrangian multiplier initialized.')
    max_delay, min_delay = get_tensors_max_min(sink_delay)

    return tf.multiply(lagrangian, (max_delay - min_delay)), max_delay - min_delay


# 优化算法也就是反向传播算法
def optimize():

    tf.reset_default_graph()
    # tf.compat.v1.experimental.output_all_intermediates(True)
    # config.trainable_variables = []
    # tensorflow.executing_eagerly()
    tf.disable_eager_execution()

    with tf.Session(config=config.train_config) as sess:
    # with tf.Session() as sess:
        # 加载前向传播的树结构
        loader.load()
        config.scalar_tree = False

        # 计算损失=总延时+等式约束
        logging.info('delay calculating...')
        delay = calc_delay()
        tf.add_to_collection('losses', delay)
        tf.summary.scalar('delay', delay)
        logging.info('all delay calculated.')

        logging.info('calculating skew and lagrangian multiplier...')
        lagrange, skew = calc_lagrange()
        tf.add_to_collection('losses', lagrange)
        tf.summary.scalar('lagrangian_multiplier', lagrange)
        tf.summary.scalar('skew', skew)
        logging.info('skew and lagrangian multiplier calculated.')

        logging.info('calculating goal...')
        goal = tf.add_n(tf.get_collection('losses'))
        logging.info('goal calculated.')

        # 定义训练算法
        global_step = tf.Variable(0, name='global_step')
        # sess.run(global_step.initializer)
        # logging.info('global step initialized.')

        # sess.run(config.trainable_variables)

        logging.info('initializing...')
        sess.run(tf.global_variables_initializer())
        logging.info('initialized.')

        logging.info('define learning rate.')
        learning_rate = tf.train.polynomial_decay(config.learning_rate_base, global_step, 1,
                                                   config.learning_rate_ending, config.learning_rate_power, cycle = True)  # todo
        logging.info('defining optimizer...')
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(goal, global_step=global_step)  # todo
        print(train_step)
        logging.info('optimizer defined.')
        # 暂时不使用滑动平均
        # variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        # # variable_average_op = variable_average.apply(tf.trainable_variables())
        # with tf.control_dependencies([train_step, variable_average_op]): # 暂时不加入滑动平均
        #     train_op = tf.no_op()

        logging.info('define saver.')
        saver = tf.train.Saver()

        # for variable in tf.get_collection(tf.GraphKeys.VARIABLES):
        #     logging.info('initializing ' + str(variable))
        #     variable.initializer.run()

        # logging.info('initializing global variables...')
        # # tf.global_variables_initializer().run()
        # logging.info('global variables initialized.')

        # visualization
        logging.info('merging all figure...')
        painter = tf.summary.FileWriter(config.tensorboard_dir + '/topo-' + str(config.topo_step))
        logging.info('add operation graph.')
        # painter.add_graph(sess.graph)
        # all_fig = tf.summary.merge_all()
        # logging.info('all figure merged.')
        logging.info('graph ignored.')

        for i in range(config.num_steps):
            logging.info('step-' + str(i) + ' training...')
            _, goal_value, step = sess.run([train_step, goal, global_step])
            logging.info('step-' + str(i) + ' trained.')

            if i % 100 == 0:
                logging.info("After %d steps of training, goal value is %g ." % (step, goal_value))
                logging.info("And the total delay of the whole tree is: ", end=None)
                final_delay = sess.run(delay)
                logging.info(str(final_delay))
                logging.info("And the lagrange-value of equality constraints: ", end=None)
                lag_multiplier = sess.run(lagrange)
                logging.info(str(lag_multiplier))

                logging.info('saving model...')
                saver.save(sess, os.path.join(config.model_path + '/topo-' + str(config.topo_step), config.model_name),
                           global_step=global_step)
                logging.info('model saved.')

                # dynamic visualization
                # painter.add_summary(all_fig, i)

                outparser.point_list(sess, i)
                outparser.draw(final_delay, lag_multiplier, i)

    return None


def main(argv=None):
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


# 总流程控制
if __name__ == '__main__':
    tf.app.run()
    # todo drawer with cdia&bdia
