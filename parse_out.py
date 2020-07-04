import config
import util
import matplotlib.pyplot as plt

ptlst = []  # point list


def generate_ptlst(sess):
    ptlst.append((sess.run(config.tree.left_child.obj['x']),
                  sess.run(config.tree.left_child.obj['y']),
                  sess.run(config.tree.left_child.rec_obj['cdia']),
                  sess.run(config.tree.left_child.rec_obj['bdia'])))
    ptlst.append((sess.run(config.tree.obj['x']),
                  sess.run(config.tree.obj['y']),
                  sess.run(config.tree.rec_obj['cdia']),
                  sess.run(config.tree.rec_obj['bdia'])))
    ptlst.append((sess.run(config.tree.right_child.obj['x']),
                  sess.run(config.tree.right_child.obj['y']),
                  sess.run(config.tree.right_child.rec_obj['cdia']),
                  sess.run(config.tree.right_child.rec_obj['bdia'])))

    generate_op(sess, config.tree)


def generate_op(sess, node):
    if node.left_child is not None:
        generate_op(sess, node.left_child)
    if node.right_child is not None:
        generate_op(sess, node.right_child)
    if node.father.father is not None:
        ptlst.append((sess.run(node.obj['x']),
                      sess.run(node.obj['y']),
                      sess.run(node.rec_obj['cdia']),
                      sess.run(node.rec_obj['bdia'])))
        ptlst.append((sess.run(node.father.obj['x']),
                      sess.run(node.father.obj['y']),
                      sess.run(node.father.rec_obj['cdia']),
                      sess.run(node.father.rec_obj['bdia'])))
        ptlst.append((sess.run(node.get_bro.obj['x']),
                      sess.run(node.get_bro.obj['y']),
                      sess.run(node.get_bro.rec_obj['cdia']),
                      sess.run(node.get_bro.rec_obj['bdia'])))


def generate_without_sess():
    ptlst.append((config.tree.left_child.obj['x'], config.tree.left_child.obj['y'],
                  config.tree.left_child.rec_obj['cdia'], config.tree.left_child.rec_obj['bdia']))
    ptlst.append((config.tree.obj['x'], config.tree.obj['y'], config.tree.rec_obj['cdia'], config.tree.rec_obj['bdia']))
    ptlst.append((config.tree.right_child.obj['x'], config.tree.right_child.obj['y'],
                  config.tree.right_child.rec_obj['cdia'], config.tree.right_child.rec_obj['bdia']))
    generate_without_sess_op(config.tree)


def generate_without_sess_op(node):
    if node.left_child is not None:
        generate_without_sess_op(node.left_child)
    if node.right_child is not None:
        generate_without_sess_op(node.right_child)

    if node.father is None:
        return
    if node.father.father is not None :
        ptlst.append((node.obj['x'], node.obj['y'], node.rec_obj['cdia'], node.rec_obj['bdia']))
        ptlst.append((node.father.obj['x'],
                      node.father.obj['y'],
                      node.father.rec_obj['cdia'],
                      node.father.rec_obj['bdia']))
        ptlst.append(
            (node.get_bro().obj['x'], node.get_bro().obj['y'], node.get_bro().rec_obj['cdia'], node.get_bro().rec_obj['bdia']))


def point_list(sess, step):
    with open(config.result_path + '/topo-' + str(config.topo_step) + '/result-' + str(step) + '.ptlst', 'w') as file:
        generate_ptlst(sess)
        while len(ptlst) != 0:
            data = ptlst.pop()
            file.write(str(data[0])), file.write(', ')
            file.write(str(data[1])), file.write(', ')
            file.write(str(data[2])), file.write(', ')
            file.write(str(data[3]))
            file.write('\n')

def point_list_without_sess():
    with open(config.result_path + '/topo-' + str(config.topo_step) + '/result-' + 'topo' + '.ptlst', 'w') as file:
        generate_without_sess()
        while len(ptlst) != 0:
            data = ptlst.pop()
            file.write(str(data[0])), file.write(', ')
            file.write(str(data[1])), file.write(', ')
            file.write(str(data[2])), file.write(', ')
            file.write(str(data[3]))
            file.write('\n')


def draw(final_delay, lagrangian, step):
    print('drawing...')
    with open(config.result_path + '/topo-' + str(config.topo_step) + '/result-' + str(step) + '.ptlst', 'r') as file:
        for line in file:
            ptlst.append((float(line.split(', ')[0]), float(line.split(', ')[1]), float(line.split(', ')[2]),
                          float(line.split(', ')[3]))) # (x, y, cdia, bdia)

    assert(len(ptlst) % 3 == 0)

    root = ptlst[1]  # root 与 source point 单独处理
    num = 0

    # plt.ion()

    while len(ptlst) != 0:
        left = ptlst.pop()
        mid = ptlst.pop()
        right = ptlst.pop()

        print([left[0], left[1]])
        print([mid[0], mid[1]])
        print([right[0], right[1]])

        fig = plt.figure(1)
        # plt.plot([left[0], mid[0]], [left[1], mid[1]], color='r')
        plt.plot([left[0], mid[0]], [left[1], left[1]], color='r')
        plt.plot([mid[0], mid[0]], [left[1], mid[1]], color='r')

        plt.plot([right[0], mid[0]], [right[1], right[1]], color='r')
        plt.plot([mid[0], mid[0]], [right[1], mid[1]], color='r')

        # plt.pause(0.001)  # todo 引入随 training_step 动态更新图像的机制

    plt.pause(5)
    plt.close(fig)
    #     plt.scatter(left[0], left[1])
    #     plt.scatter(mid[0], mid[1])
    #     plt.scatter(right[0], right[1])
    #
    # plt.show()
    print('waiting next figure...')

