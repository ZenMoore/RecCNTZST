import config
import util
import matplotlib.pyplot as plt
import torch
import os.path
import logging
import time
import read_source as reader

ptlst = []  # point list

'''Model Saving'''


def generate_no_tensor():
    global ptlst
    ptlst = []

    ptlst.append((config.tree.left_child.obj['x'], config.tree.left_child.obj['y'],
                  config.tree.left_child.rec_obj['cdia'], config.tree.left_child.rec_obj['bdia']))
    ptlst.append((config.tree.obj['x'], config.tree.obj['y'], config.tree.rec_obj['cdia'], config.tree.rec_obj['bdia']))
    ptlst.append((config.tree.right_child.obj['x'], config.tree.right_child.obj['y'],
                  config.tree.right_child.rec_obj['cdia'], config.tree.right_child.rec_obj['bdia']))
    generate_no_tensor_op(config.tree)


def generate_no_tensor_op(node):
    global ptlst

    if node.left_child is not None:
        generate_no_tensor_op(node.left_child)
    if node.right_child is not None:
        generate_no_tensor_op(node.right_child)

    if node.father is None:
        return
    if node.father.father is not None:
        ptlst.append((node.obj['x'], node.obj['y'], node.rec_obj['cdia'], node.rec_obj['bdia']))
        ptlst.append((node.father.obj['x'],
                      node.father.obj['y'],
                      node.father.rec_obj['cdia'],
                      node.father.rec_obj['bdia']))
        ptlst.append(
            (node.get_bro().obj['x'], node.get_bro().obj['y'], node.get_bro().rec_obj['cdia'],
             node.get_bro().rec_obj['bdia']))


def point_list_no_tensor():
    global ptlst

    if not os.path.exists(config.model_path + '/topo-' + str(config.topo_step)):
        os.makedirs(config.model_path + '/topo-' + str(config.topo_step))
    with open(config.model_path + '/topo-' + str(config.topo_step) + '/result-' + 'topo' + '.ptlst', 'w') as file:
        generate_no_tensor()
        while len(ptlst) != 0:
            data = ptlst.pop()
            file.write(str(data[0])), file.write(', ')
            file.write(str(data[1])), file.write(', ')
            file.write(str(data[2])), file.write(', ')
            file.write(str(data[3]))
            file.write('\n')


def generate_ptlst():
    global ptlst
    ptlst = []

    ptlst.append((config.tree.left_child.obj['x'].item(),
                  config.tree.left_child.obj['y'].item(),
                  config.tree.left_child.rec_obj['cdia'].item(),
                  config.tree.left_child.rec_obj['bdia'].item()))
    ptlst.append((config.tree.obj['x'].item(),
                  config.tree.obj['y'].item(),
                  config.tree.rec_obj['cdia'].item(),
                  config.tree.rec_obj['bdia'].item()))
    ptlst.append((config.tree.right_child.obj['x'].item(),
                  config.tree.right_child.obj['y'].item(),
                  config.tree.right_child.rec_obj['cdia'].item(),
                  config.tree.right_child.rec_obj['bdia'].item()))

    generate_op(config.tree)


def generate_op(node):
    global ptlst
    if node.left_child is not None:
        generate_op(node.left_child)
    if node.right_child is not None:
        generate_op(node.right_child)

    if node.father is None:
        return
    if node.father.father is not None:
        ptlst.append((node.obj['x'].item(),
                      node.obj['y'].item(),
                      node.rec_obj['cdia'].item(),
                      node.rec_obj['bdia'].item()))
        ptlst.append((node.father.obj['x'].item(),
                      node.father.obj['y'].item(),
                      node.father.rec_obj['cdia'].item(),
                      node.father.rec_obj['bdia'].item()))
        ptlst.append((node.get_bro().obj['x'].item(),
                      node.get_bro().obj['y'].item(),
                      node.get_bro().rec_obj['cdia'].item(),
                      node.get_bro().rec_obj['bdia'].item()))


def point_list(step):
    global ptlst
    logging.info('generating ptlst...')

    if not os.path.exists(config.model_path + '/topo-' + str(config.topo_step)):
        os.makedirs(config.model_path + '/topo-' + str(config.topo_step))

    with open(config.model_path + '/topo-' + str(config.topo_step) + '/result-' + str(step) + '.ptlst', 'w') as file:
        generate_ptlst()
        while len(ptlst) != 0:
            
            data = ptlst.pop()
            file.write(str(data[0])), file.write(', ') # x
            file.write(str(data[1])), file.write(', ') # y
            file.write(str(data[2])), file.write(', ') # cdia
            file.write(str(data[3])) # bdia
            file.write('\n')
    logging.info('ptlst generated.')


def point_list_no_tensor_detail(topo_step):
    global ptlst

    if not os.path.exists(config.model_path + '/topo-' + str(config.topo_step)):
        os.makedirs(config.model_path + '/topo-' + str(config.topo_step))
    with open(config.model_path + '/topo-' + str(config.topo_step) + '/result-' + 'topo' + '.ptlst', 'w') as file:
        generate_no_tensor()
        file.write('topo=%d\n' % topo_step)
        while len(ptlst) != 0:
            data = ptlst.pop()
            file.write(str(data[0])), file.write(', ')
            file.write(str(data[1])), file.write(', ')
            file.write(str(data[2])), file.write(', ')
            file.write(str(data[3]))
            file.write('\n')


def point_list_detail(loss, sum_delay, sum_skew, delay, lag, skew, step, topo_step):
    global ptlst
    logging.info('generating ptlst...')

    if not os.path.exists(config.model_path + '/topo-' + str(config.topo_step)):
        os.makedirs(config.model_path + '/topo-' + str(config.topo_step))

    with open(config.model_path + '/topo-' + str(config.topo_step) + '/result-' + str(step) + '.ptlst', 'w') as file:
        generate_ptlst()
        file.write('loss=%g, sum_delay=%g, sum_skew=%g, max_delay=%g, lag=%g, max_min_skew=%g, step=%d, topo=%d\n'%(loss, sum_delay, sum_skew, delay, lag, skew, step, topo_step))
        while len(ptlst) != 0:
            data = ptlst.pop()
            file.write(str(data[0])), file.write(', ')  # x
            file.write(str(data[1])), file.write(', ')  # y
            file.write(str(data[2])), file.write(', ')  # cdia
            file.write(str(data[3]))  # bdia
            file.write('\n')
    logging.info('ptlst generated.')


def l_width(bdia):
    return config.l_width_min + (config.l_width_max - config.l_width_min) / (config.bdia_max - config.bdia_min) * (bdia - config.bdia_min)


def l_op(cdia):
    return config.l_op_min + (config.l_op_max - config.l_op_min) / (config.cdia_max - config.cdia_min) * (cdia - config.cdia_min)


def draw(loss=None, sum_delay=None, sum_skew=None, delay=None, lag=None, skew=None, step=None):

    global ptlst
    ptlst = []
    name = ''
    first_line = True

    logging.info('drawing...')

    with open(config.model_path + '/topo-' + str(config.topo_step) + '/result-' + str(step) + '.ptlst', 'r') as file:
        for line in file:
            if first_line and config.post_embed:
                if loss is None:
                    assert (float(line.split('=')[1]) == config.topo_step)
                else:
                    for phrase in line.split(', '):
                        if phrase.split('=')[0] == 'loss':
                            loss = float(phrase.split('=')[1])
                        elif phrase.split('=')[0] == 'delay':
                            delay = float(phrase.split('=')[1])
                        elif phrase.split('=')[0] == 'lag':
                            lag = float(phrase.split('=')[1])
                        elif phrase.split('=')[0] == 'skew':
                            skew = float(phrase.split('=')[1])
                        elif phrase.split('=')[0] == 'step':
                            assert (step == float(phrase.split('=')[1]))
                        elif phrase.split('=')[0] == 'topo':
                            assert (config.topo_step == float(phrase.split('=')[1]))
                        else:
                            raise Exception('unknown point list head data item.')
                first_line = False
            else:
                ptlst.append((float(line.split(', ')[0]), float(line.split(', ')[1]), float(line.split(', ')[2]),
                              float(line.split(', ')[3])))  # (x, y, cdia, bdia)

    assert (len(ptlst) % 3 == 0)

    root = ptlst[1]  # root 与 source point 单独处理
    num = 0

    fig = plt.figure(1)

    while len(ptlst) != 0:
        left = ptlst.pop()
        mid = ptlst.pop()
        right = ptlst.pop()

        if loss is None:
            plt.title('topo=%d, step=topo' % config.topo_step)
        else:
            plt.title('loss=%g, lagrange_multiplier=%g\nsum_delay=%g, max_delay=%g\nsum_skew=%g, max_min_skew=%g\ntopo=%d, step=%d' % (loss, lag, sum_delay, delay, sum_skew, skew, config.topo_step, step))
        plt.plot([left[0], mid[0]], [left[1], left[1]], color='k', alpha=l_op(left[2]), linewidth=l_width(left[3]))
        plt.plot([mid[0], mid[0]], [left[1], mid[1]], color='k', alpha=l_op(left[2]), linewidth=l_width(left[3]))

        plt.plot([right[0], mid[0]], [right[1], right[1]], color='k', alpha=l_op(right[2]), linewidth=l_width(right[3]))
        plt.plot([mid[0], mid[0]], [right[1], mid[1]], color='k', alpha=l_op(right[2]), linewidth=l_width(right[3]))
        # sink points and merging points # sink points 之后会被 p_color_s 覆盖，因此两者实际分开
        plt.scatter(left[0], left[1], s=config.p_area, c=config.p_color_m)
        plt.scatter(right[0], right[1], s=config.p_area, c=config.p_color_m)
        plt.scatter(mid[0], mid[1], s=config.p_area, c=config.p_color_m)

        # bending points
        plt.scatter(mid[0], left[1], s=config.p_area, c=config.p_color_b)
        plt.scatter(mid[0], right[1], s=config.p_area, c=config.p_color_b)

        # sink points
    for point in config.sink_set:
        plt.scatter(point['x'], point['y'], s=config.p_area, c=config.p_color_s)

    if not os.path.exists(config.img_path):
        os.makedirs(config.img_path)
    if loss is None:
        plt.savefig(config.img_path + '/topo@%d.jpg' % (config.topo_step))
    else:
        plt.savefig(config.img_path + '/%d@%d=L%gST%gSE%gT%glag%gE%g.jpg'%(step, config.topo_step, loss, sum_delay, sum_skew, delay, lag, skew))
    plt.close(fig)
    # plt.show()
    logging.info('waiting next figure...')


def post_embed():
    if reader.read():
        logging.info('read source.')
    else:
        try:
            raise Exception('read failed.')
        except Exception as e:
            logging.exception(e)

    logging.info('post embedding...')
    for dirname in os.listdir(config.model_path):
        logging.info('embedding ' + dirname + '...')
        config.topo_step = int(dirname.split('-')[1])
        for filename in os.listdir(config.model_path + '/' + dirname):
            logging.info('embedding ' + filename + '...')
            step = filename.split('-')[1].split('.')[0]
            try:
                if step == 'topo':
                    draw(step=step)
                else:
                    step = int(step)
                    draw(loss=-1, step=step)
            except Exception as e:
                logging.exception(e)

if __name__ == '__main__':
    post_embed()
