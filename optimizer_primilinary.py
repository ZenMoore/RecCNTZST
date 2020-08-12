# import config
# import logging
# import numpy as np
# # import tensorflow.compat.v1 as tf
# import torch
# import parse_topo as topoparser
# import recursive as network
# import os
# import parse_out as outparser
# import tensorboardX as board
# from torchviz import make_dot
# import warnings
# import hyperparams_searcher as hyperparam
#
# # import tensorboardX as board
#
# '''相当于后向传播算法'''
#
# # todo no_grad?
# def get_tensors_max_min(tensors):
#     recur_set = []
#
#     max_result = None
#     for e in tensors:
#         recur_set.append(e)
#     while len(recur_set) > 1:
#         a = recur_set.pop()
#         b = recur_set.pop()
#         max_result = torch.where(a >= b, a, b)
#         recur_set.append(max_result)
#
#     min_result = None
#     for e in tensors:
#         recur_set.append(e)
#     while len(recur_set) > 1:
#         a = recur_set.pop()
#         b = recur_set.pop()
#         min_result = torch.where(a >= b, b, a)
#         recur_set.append(min_result)
#
#     assert (max_result is not None)
#     assert (min_result is not None)
#     return max_result, min_result
#
#
# def N_cnt(bdia, cdia):
#
#     return torch.round(torch.div(2.0 * config.pi * torch.square(bdia * 1.0 / 2.0),
#                            torch.sqrt(torch.tensor(3.0)) * torch.square(cdia + config.delta)))
#
#
# def R_c(cdia):
#     # if sess.run(tf.less(cdia, torch.tensor(2.0)))[0] and sess.run(tf.greater(cdia, torch.tensor(1.0)))[0] or sess.run(tf.equal(cdia, torch.tensor(1.0)))[0]:
#     #     return torch.mul()(config.R_cnom, tf.div(tf.add(tf.square(cdia), torch.mul()(-2.811, cdia) + 2.538), tf.add(torch.mul()(0.5376, tf.square(cdia)), torch.mul()(-0.8106, cdia) + 0.3934)))
#     # else:
#     #     return torch.tensor(config.R_cnom)
#     if cdia < torch.tensor(2.0):
#         return config.R_cnom * (torch.div(torch.square(cdia) + -2.811 * cdia + 2.538,
#                                           0.5376 * torch.square(cdia) + -0.8106 * cdia + 0.3934))
#     else:
#         return torch.tensor(config.R_cnom)
#     # return tf.cond(cdia < torch.tensor(2.0),
#     #                lambda: torch.mul()(config.R_cnom, tf.div(tf.add(tf.square(cdia), torch.mul()(-2.811, cdia) + 2.538),
#     #                                                          tf.add(torch.mul()(0.5376, tf.square(cdia)),
#     #                                                                 torch.mul()(-0.8106, cdia) + 0.3934))),
#     #                lambda: torch.tensor(config.R_cnom))  # 默认下界被限定
#
#
# def r_s(cdia, wirelen):
#     # if sess.run(tf.greater(wirelen, torch.tensor(config.mfp))):
#     #     return tf.div(config.R_Q, torch.mul()(config.C_lambda, cdia))
#     # else:
#     #     return torch.tensor(0.0)
#     if wirelen > torch.tensor(config.mfp):
#         return torch.div(torch.tensor(config.R_Q), config.C_lambda * cdia)
#     else:
#         return torch.tensor(0.0)
#     # return tf.cond(wirelen > torch.tensor(config.mfp),
#     #                lambda: tf.div(config.R_Q, torch.mul()(config.C_lambda, cdia)),
#     #                lambda: torch.tensor(0.0))
#
#
# def contact_wire_contact(wirelen, cdia, bdia):
#     Rc = R_c(cdia)
#     rs = r_s(cdia, wirelen)
#     Ncnt = N_cnt(bdia, cdia)
#     # div = torch.div(wirelen * r_s(cdia, wirelen), N_cnt(bdia, cdia))
#     return 2 * 0.69 * config.unit_capacitance * wirelen * torch.div(config.R_Q + Rc, 2 * Ncnt) \
#            + 0.38e+3 * torch.div(wirelen * rs, Ncnt) * config.unit_capacitance * wirelen
#
#
# def contact_wire_sink(wirelen, cdia, bdia, cap):
#     Rc = R_c(cdia)
#     Ncnt = N_cnt(bdia, cdia)
#     rs = r_s(cdia, wirelen)
#     return 0.69 * config.unit_capacitance * wirelen * torch.div(config.R_Q + Rc, 2 * Ncnt) \
#             + 0.38e+3 * torch.div(wirelen * rs, Ncnt) * config.unit_capacitance * wirelen \
#             + 0.69e+12 * cap * (torch.div(1e3 * wirelen * rs, Ncnt) + torch.div(config.R_Q + Rc, Ncnt))
#
#
# # 计算总延时
# def calc_delay():
#
#     sink_node_set = config.tree.get_sinks()
#
#     for node in sink_node_set:
#         logging.info('calculating delay of sink%d, there remain %d.' % (
#             sink_node_set.index(node), len(sink_node_set) - sink_node_set.index(node)))
#         # delay = tf.Variable(0, dtype=tf.float32)
#         delay = torch.tensor(0.0)
#         while node.father is not None:
#             delay = delay + calc_node_delay(node)
#             node = node.father
#         delay = torch.add(delay, calc_root_delay(node))
#         config.sink_delay.append(delay)
#
#     assert (len(config.sink_delay) == len(config.sink_set))
#     result, _ = get_tensors_max_min(config.sink_delay)
#
#     return result  # 必须是tensor数组里面的最大值
#
#
# def calc_sink_delay(node):
#     # delay = tf.Variable(0, dtype=tf.float32)
#     delay = torch.tensor(0.0)
#     while node.father is not None:
#         delay = delay + calc_node_delay(node)
#         node = node.father
#     delay = torch.add(delay, calc_root_delay(node))
#     return delay
#
#
# def calc_root_delay(node):
#
#     def with_bending():
#
#         horizontal_bia = torch.abs(config.source_point['x'] - node.obj['x'])
#         vertical_bia = torch.abs(config.source_point['y'] - node.obj['y'])
#
#         t_horizontal = contact_wire_sink(horizontal_bia, node.rec_obj['cdia'], node.rec_obj['bdia'], config.source_point['c'])
#
#         t_vertical = contact_wire_contact(vertical_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])
#
#         return torch.add(t_horizontal, t_vertical)
#
#     if node.num_bend == torch.tensor(0):
#         return torch.tenor(0.0)
#     elif node.num_bend == torch.tensor(1):
#         return contact_wire_sink(node.rec_obj['wirelen'], node.rec_obj['cdia'], node.rec_obj['bdia'], config.source_point['c'])
#     elif node.num_bend == torch.tensor(2):
#         return with_bending()
#     else:
#         raise Exception('bending number of node ' + node.get_id() + ' is abnormal.')
#
#
# def calc_node_delay(node):
#     if node.isleaf:
#
#         def with_bending_sink():
#             horizontal_bia = torch.abs(node.father.obj['x'] - node.obj['x'])
#             vertical_bia = torch.abs(node.father.obj['y'] - node.obj['y'])
#
#             t_horizontal = contact_wire_sink(horizontal_bia, node.rec_obj['cdia'], node.rec_obj['bdia'], node.obj['c'])
#
#             t_vertical = contact_wire_contact(vertical_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])
#
#             return torch.add(t_horizontal, t_vertical)
#
#         if node.num_bend == torch.tensor(0):
#             return 0.69e+12 * node.obj['c'] * torch.div(torch.add(config.R_Q, R_c(node.rec_obj['cdia'])),
#                                                  N_cnt(node.father.rec_obj['bdia'], node.father.rec_obj['cdia']))
#
#         elif node.num_bend == torch.tensor(1):
#             return contact_wire_sink(node.rec_obj['wirelen'], node.rec_obj['cdia'], node.rec_obj['bdia'], node.obj['c'])
#
#         elif node.num_bend == torch.tensor(2):
#             return with_bending_sink()
#         else:
#             raise Exception('bending number of node ' + node.get_id() + ' is abnormal.')
#
#     else:
#         def with_bending_mp():
#             horizontal_bia = torch.abs(node.father.obj['x'] - node.obj['x'])
#             vertical_bia = torch.abs(node.father.obj['y'] - node.obj['y'])
#
#             t_horizontal = contact_wire_contact(horizontal_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])
#
#             t_vertical = contact_wire_contact(vertical_bia, node.rec_obj['cdia'], node.rec_obj['bdia'])
#
#             return torch.add(t_horizontal, t_vertical)
#
#         if node.num_bend == torch.tensor(0):
#             return torch.tensor(0.0)
#         elif node.num_bend == torch.tensor(1):
#             return contact_wire_contact(node.rec_obj['wirelen'],node.rec_obj['cdia'], node.rec_obj['bdia'])
#         elif node.num_bend == torch.tensor(2):
#             return with_bending_mp()
#         else:
#             raise Exception('bending number of node ' + node.get_id() + ' is abnormal.')
#
#
# def weight(name):
#     weight = torch.empty([], requires_grad=True)
#     torch.nn.init.normal_(weight, mean=config.lagrangian_ini, std=config.lagrangian_std)
#     config.trainable_helpers.update({name: weight})
#     logging.info('lagrangian multiplier ' + name + ' created and initialized(normal).')
#     return weight
#
#
# # 计算引入拉格朗日乘子后的等式约束
# def calc_lagrange():
#
#     # 将拉格朗日乘子作为训练参数，梯度下降时候，向对拉格朗日乘子偏导等于零的方向下降
#     # lagrangian = tf.get_variable("lagrangian_multiplier", shape=(), initializer=tf.truncated_normal_initializer(stddev=0.1),
#     #                              trainable=True)
#     if not config.loaded:
#         config.lagranger = weight('lag_multiplier')
#
#     max_delay, min_delay = get_tensors_max_min(config.sink_delay)
#
#     return config.lagranger * (max_delay - min_delay), max_delay - min_delay
#
#
# def embed(loss, delay, lag, skew, step):
#     outparser.draw(loss, delay, lag, skew, step)
#     return None
#
#
# def set_nodes_to_show():
#     config.nodes_to_show = ['root_left_left']
#     logging.info('It will show nodes: ' + str(config.nodes_to_show)[1:-1])
#     return None
#
#
# def set_between_goal_to_show():
#     config.between_goal_to_show = [0]
#     logging.info('It will show between goals: ' + str(config.between_goal_to_show)[1:-1])
#     return None
#
#
# def calc_between_skew():
#     i = 0
#     former = None
#     for delay in config.sink_delay:
#         if i == 0:
#             former = delay
#         else:
#             logging.info('calculate between skew %d' % i)
#             skew = torch.abs(delay - former)
#             config.between_skew.append(skew)
#             name = "lag_%d" % i
#             if not config.loaded:
#                 weight(name)
#             logging.info('calculate between skew %d' % i)
#             config.constraints.append(config.trainable_helpers[name] * skew)
#             former = delay
#         i += 1
#
#
# def calc_between_goal():
#     i = 1
#     for skew in config.between_skew:
#         delay = config.sink_delay[i]
#         name = 'lag_%d' % i
#         logging.info('calculate goal with ' + name)
#         config.between_goal.append(
#             (delay + config.trainable_helpers[name] * skew,
#              delay,
#              config.trainable_helpers[name],
#              skew))
#         i += 1
#
#
# def forprop():
#
#     config.sink_delay = []
#     config.between_skew = []
#     config.constraints = []
#     config.between_goal = []
#     network.coordinate_calc()
#     # 计算损失=总延时+等式约束
#     logging.info('delay calculating...')
#
#     delay = calc_delay()
#     if config.local_optimize:
#         calc_between_skew()
#         calc_between_goal()
#     # tf.add_to_collection('losses', delay)
#     # tf.summary.scalar('delay', delay)
#     logging.info('all delay calculated.')
#
#
#     logging.info('calculating skew and lagrangian multiplier...')
#
#     lag_constraint, skew = calc_lagrange()
#     # tf.add_to_collection('losses', lagrange)
#     # tf.summary.scalar('lagrangian_multiplier', lagrange)
#     # tf.summary.scalar('skew', skew)
#     logging.info('skew and lagrangian multiplier calculated.')
#
#
#     logging.info('calculating goal...')
#
#     # goal = tf.add_n(tf.get_collection('losses'))
#     goal = torch.add(delay, lag_constraint)
#     logging.info('goal calculated.')
#
#     config.loaded = True
#     return goal, delay, skew
#
#
# # 优化算法也就是反向传播算法
# def optimize():
#
#     # tf.reset_default_graph()
#     config.trainable_helpers = {}
#     config.trainable_wirelens = {}
#     config.trainable_bdias = {}
#     config.trainable_cdias = {}
#     config.scalar_tree = False
#     goal, delay, skew = forprop()
#     # todo 清空之前的计算图，这里默认新的计算图与旧的计算图之间没有连接相互独立而不影响训练
#     # tf.compat.v1.experimental.output_all_intermediates(True)
#     # config.trainable_variables = []
#     # tensorflow.executing_eagerly()
#     # tf.disable_eager_execution()
#
#     # 定义训练算法
#     logging.info('defining optimizer...')
#
#
#     train = torch.optim.Adagrad(
#         [{'params': list(config.trainable_wirelens.values()), 'lr': config.learning_rate_base['wirelen']},
#          {'params': list(config.trainable_cdias.values()), 'lr': config.learning_rate_base['cdia']},
#          {'params': list(config.trainable_bdias.values()), 'lr': config.learning_rate_base['bdia']},
#          {'params': list(config.trainable_helpers.values()), 'lr': config.learning_rate_base['lag']}])
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(train, T_max=config.learning_rate_T, eta_min=config.learning_rate_ending) # todo 如何设置不同的 learning_rate_ending
#     # scheduler = torch.optim.lr_scheduler.StepLR(train, step_size=1, gamma=1.0)  # todo 如何设置不同的 learning_rate_ending
#     logging.info('optimizer defined.')
#
#
#     # visualization handler
#     if not config.hyper_search:
#         if not os.path.exists(config.visual_path + '/topo-' + str(config.topo_step)):
#             os.makedirs(config.visual_path + '/topo-' + str(config.topo_step))
#
#         writer = board.SummaryWriter(config.visual_path + '/topo-' + str(config.topo_step))
#         set_nodes_to_show()
#         set_between_goal_to_show()
#     # g = make_dot(goal, params=config.trainables)  # todo
#     # g.view(filename='graph-' + str(i) + '.pdf', directory=config.visual_path + '/topo-' + str(config.topo_step))
#     # g.view(filename='graph' + '.pdf', directory=config.visual_path + '/topo-' + str(config.topo_step))
#
#     for i in range(config.num_steps):
#         logging.info('step-%d@topo-%d backproping...' % (i, config.topo_step))
#
#         train.zero_grad()
#         # _, goal_value, step = sess.run([train_step, goal, global_step])
#         if config.local_optimize:
#             for between_goal in config.between_goal:
#                 logging.info('backprop ' + str(between_goal))
#                 for j in range(config.local_step):
#                     between_goal[0].backward(retain_graph = True)
#                     for index in config.between_goal_to_show:
#                         if config.between_goal.index(between_goal)  == index:
#                             writer.add_scalar('goal' + str(index), between_goal[0], j+config.local_step*i)
#                             writer.add_scalar('delay' + str(index), between_goal[1], j+config.local_step*i)
#                             writer.add_scalar('lag' + str(index), between_goal[2], j+config.local_step*i) # todo 这个 lag 不下降
#                             writer.add_scalar('skew' + str(index), between_goal[3], j+config.local_step*i)
#                     index = config.between_goal.index(between_goal)
#                     d = calc_sink_delay(config.tree.get_sinks()[index])
#                     config.sink_delay[index] = d
#                     e = torch.abs(d - calc_sink_delay(config.tree.get_sinks()[index + 1]))
#                     between_goal = (d + between_goal[2] * e, d, between_goal[2], e)
#                     config.between_goal[index] = between_goal
#
#         else:
#             goal.backward(retain_graph=True)
#         # delay.backward(retain_graph=True)
#         # skew.backward(retain_graph=True)
#
#         train.step()
#         scheduler.step()
#         logging.info('step-%d@topo-%d backproped...' % (i, config.topo_step))
#
#         goal, delay, skew = forprop()
#
#         loss = goal.item()
#         logging.info("After %d steps of training, loss value@topo-%d is: %g;" % (i, config.topo_step, loss))
#
#         final_delay = delay.item()
#         logging.info("And the total delay of the whole tree is: %g;" % final_delay)
#
#         lag_multiplier = config.lagranger.item()
#         logging.info("And the lagrangian multiplier value is: %g;" % lag_multiplier)
#
#         skew_constraint = skew.item()
#         logging.info("And the max-min skew: %g." % skew_constraint)
#
#
#         # hyperparams search
#         if config.hyper_search:
#             if config.former is not None:
#                 if not hyperparam.judge(config.former, skew):
#                     return False
#             config.former = skew
#
#
#         # visualization
#         if not config.hyper_search:
#             writer.add_scalar('loss', loss, i)
#             writer.add_scalar('delay', final_delay, i)
#             writer.add_scalar('lag', lag_multiplier, i)
#             writer.add_scalar('skew', skew_constraint, i)
#
#             # todo 这段不知道为什么不能运行，不过影响不大，可以不处理
#             # for node in [temp + '_cdia' for temp in config.nodes_to_show]:
#             #     cdia = float(config.trainable_cdias[node].item())
#             #     writer.add_scalar(node, cdia, i)
#             #     writer.add_histogram(node, cdia, i)
#             # for node in [temp + '_bdia' for temp in config.nodes_to_show]:
#             #     bdia = float(config.trainable_bdias[node].item())
#             #     writer.add_scalar(node, bdia, i)
#             #     writer.add_histogram(node, bdia, i)
#             # for node in [temp + '_wirelen' for temp in config.nodes_to_show]:
#             #     wirelen = float(config.trainable_wirelens[node].item())
#             #     writer.add_scalar(node, wirelen, i)
#             #     writer.add_histogram(node, wirelen, i)
#
#
#         if i % 10 == 0:
#             logging.info('saving model(step-%d@topo-%d)...' % (i, config.topo_step))
#
#             if config.post_embed:
#                 outparser.point_list_detail(loss, final_delay, lag_multiplier, skew_constraint, i + 1, config.topo_step)
#                 logging.info('model(step-%d@topo-%d) saved.' % (i, config.topo_step))
#             else:
#                 outparser.point_list(i+1)
#                 logging.info('model(step-%d@topo-%d) saved.' % (i, config.topo_step))
#                 embed(loss, final_delay, lag_multiplier, skew_constraint, i + 1)
#
#         # if i%1000 == 0: # todo 本应该是没过100轮画一个为好，但是画一次时间太长，先只作一个图简单看下运算过程
#             # logging.info('making operation graph dot of loss...')
#             # g = make_dot(goal, params=config.trainables)  # todo
#             # logging.info('operation graph dot made, waiting pdf file...')
#             # g.view(filename='graph%d@%d'%(), directory=config.visual_path + '/topo-' + str(config.topo_step))
#             # logging.info('operation graph of step-%d@topo-%d is shown as pdf.' % (i, config.topo_step))
#
#     return True
#
#
# # 总流程控制
# if __name__ == '__main__':
#     logging.info('optimizing...')
#
#     for dirname in os.listdir(config.model_path):
#         for filename in os.listdir(config.model_path+ '/' + dirname):
#             os.remove(config.model_path + '/' + dirname + '/' + filename)
#
#     for dirname in os.listdir(config.visual_path):
#         for filename in os.listdir(config.visual_path + '/' + dirname):
#             if 'events' in filename:
#                 os.remove(config.visual_path + '/' + dirname + '/' + filename)
#
#     try:
#         # with hyperparam search
#         if config.hyper_search:
#             while config.topo_step <= config.max_topo_step:
#                 if config.topo_step == 0:
#                     if topoparser.parse():
#                         for i in range(config.max_hyper_step):
#                             hyperparam.update(i)
#                             config.print_hyperparams(i)
#                             if optimize():
#                                 logging.info('find a good hyperparams system which is in step-%d.' % i)
#                                 break
#                             else:
#                                 logging.info('step-%d is a bad system.' % i)
#                                 continue
#                         config.topo_step = config.topo_step + 1
#                     else:
#                         raise Exception("tree parsing failed.")
#                 else:
#                     if topoparser.update():
#                         for i in range(config.max_hyper_step):
#                             hyperparam.update(i)
#                             config.print_hyperparams(i)
#                             if optimize():
#                                 logging.info('find a good hyperparams system which is in step-%d.' % i)
#                                 break
#                             else:
#                                 logging.info('step-%d is a bad system.' % i)
#                                 continue
#                         config.topo_step = config.topo_step + 1
#                     else:
#                         raise Exception("tree updating failed.")
#         else:
#         # without hyperparam search
#             config.print_hyperparams(0)
#             while config.topo_step <= config.max_topo_step:
#                 if config.topo_step == 0:
#                     if topoparser.parse():
#                         optimize()
#                         config.topo_step = config.topo_step + 1
#                     else:
#                         raise Exception("tree parsing failed.")
#                 else:
#                     if topoparser.update():
#                         optimize()
#                         config.topo_step = config.topo_step + 1
#                     else:
#                         raise Exception("tree updating failed.")
#     except Exception as e:
#         logging.exception(e)
