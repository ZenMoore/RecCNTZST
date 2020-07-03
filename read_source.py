import config
import re
import matplotlib.pyplot as plt


def read():

    print('reading benchmark...')

    FILENAME = config.source_dir
    with open(FILENAME, 'r') as f:
        lines = f.readlines()
    HEADER_NAMES = ["NumPins", "PerUnitResistance", "PerUnitCapacitance", "TsvBound", "SrcZ", "TsvR", "TsvC"]
    headers = {}
    sinks = []

    regex_num = "(-*\d+.*\d*e*-*\d*)\s*"
    regex_sink = "Sink : (\d+)\n"
    # regex_coordinate = "\s*Coordinate : (\d+) (\d+) (\d+)\s*" #(x,y,z)
    regex_coordinate = "\s*Coordinate : (\d+) (\d+) \s*" # (x, y)
    regex_capacitive_load = '\s*Capacitive Load : ' + regex_num
    for line in lines:
        if line[0] == '#' or line == '\n':
            continue
        for name in HEADER_NAMES:
            if name in line:
                pattern = re.compile(name + " : " + regex_num)
                matcher = pattern.match(line)
                headers[name] = matcher.group(1)
        if 'Sink' in line:
            pattern = re.compile(regex_sink)
            matcher = pattern.match(line)
            sinks.append({'id': matcher.group(1)})
        if 'Coordinate' in line:
            pattern = re.compile(regex_coordinate)
            matcher = pattern.match(line)
            sinks[-1]['x'] = float(matcher.group(1))
            sinks[-1]['y'] = float(matcher.group(2))
            # sinks[-1]['z'] = int(matcher.group(3))
        if 'Capacitive Load' in line:
            pattern = re.compile(regex_capacitive_load)
            matcher = pattern.match(line)
            sinks[-1]['cap'] = matcher.group(1)


    for e in sinks:
        temp = {'r': None,
                'c': e['cap'],
                'x': e['x'],
                'y': e['y']}
        config.sink_set.append(temp)

    config.source_point = config.sink_set[0]
    config.sink_set.remove(config.sink_set[0])
    config.headers = headers

    print('benchmark read.')

    return True


def draw_sink_nodes():
    headers = config.headers
    sinks = config.sink_set

    xlow, xhi = -36667, 12821
    ylow, yhi = 70742, 112456
    N_index = 4
    width = (xhi - xlow)
    height = (yhi - ylow)
    bucket_width = width / N_index
    bucket_height = height / N_index
    axis_width = max(width, height)
    epsilon = 1000

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    # ax.axis([xlow-epsilon, xlow+axis_width+epsilon, ylow-epsilon, ylow+axis_width+epsilon])

    # x = [sink['x'] - sink['y'] for sink in sinks]
    # y = [sink['x'] + sink['y'] for sink in sinks]
    x = [sink['x'] for sink in sinks]
    y = [sink['y'] for sink in sinks]
    ax.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(sinks[i]['x'], xy=(x[i], y[i]))

    # for i in range(N_index):
    #     for j in range(N_index):
    #         x1, y1 = xlow + bucket_width * i, ylow + bucket_height * j
    #         x1, y1 = (y1 + x1) / 2, (y1 - x1) / 2
    #         x2, y2 = xlow + bucket_width * (i + 1), ylow + bucket_height * j
    #         x2, y2 = (y2 + x2) / 2, (y2 - x2) / 2
    #         x3, y3 = xlow + bucket_width * (i + 1), ylow + bucket_height * (j + 1)
    #         x3, y3 = (y3 + x3) / 2, (y3 - x3) / 2
    #         x4, y4 = xlow + bucket_width * i, ylow + bucket_height * (j + 1)
    #         x4, y4 = (y4 + x4) / 2, (y4 - x4) / 2
    #         x5, y5 = x1, y1
    #
    #         ax.plot([x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5])
            # rect = plt.Rectangle((x_start, y_start), bucket_width/2, bucket_height/2, fill=False)
            # ax.add_patch(rect)

    plt.show()

if __name__ == '__main__':
    read()
    draw_sink_nodes()