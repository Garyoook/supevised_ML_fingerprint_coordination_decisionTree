import sys
import time

import numpy as np

CLASS_NUM = 4
ATTR_NUM = 7


def calc_entropy(data):
    """
    :param data: P1 ... Pk
    :return: the entropy of the given list of Pk.
    """
    if not data:
        return 0
    result = 0
    for p in data:
        if p != 0:
            result -= p * np.log2(p)
    return result


def find_split_points(dataset):
    """
    Optimize the splitting process by
    sorting the values of each attribute.
    :param dataset: the training dataset.
    :return: the list(2d-array) of splitting points for each attribute.
    """
    split_points = []
    for col in range(ATTR_NUM):
        column = []
        col_split_points = []
        for row in dataset:
            column.append(row[col])
        column = list(set(column))
        column.sort()
        for i in range(len(column) - 1):
            col_split_points.append((column[i] + column[i + 1]) / 2)
        split_points.append(col_split_points)
    return split_points


def find_split(training_dataset):
    """
    :return: a dictionary of 2 dataset 'left_split': ... and 'right_split': ...
    (format(matrix) like the original dataset) and a 'wifi_number': xx
    e.g. {'left_split': None,'right_split': None,'wifi_number': 0}
    """

    label_sample_size = [0] * CLASS_NUM  # store the number of signal data in each room

    ratios_for_entropy_calc = []  # this is P1, P2, ..., Pk

    total_size = 0  # sample size
    for rowi in training_dataset:
        room_id = int(rowi[-1])
        label_sample_size[room_id - 1] += 1  # record the sample size of signals of each room.

    for sub_sample_size in label_sample_size:  # calculate the total sample size
        total_size += sub_sample_size

    for roomi in range(CLASS_NUM):
        pk = label_sample_size[roomi] / total_size
        ratios_for_entropy_calc.append(pk)  # record Pk for calculating entropy

    current_entropy = calc_entropy(ratios_for_entropy_calc)  # calculate H(A)

    cur_max_ig = 0  # tracer of maximum information gain.
    cur_split_point = 0  # tracer of the split point with largest ig
    left_branch = list()  # tracer of the left split with largest ig
    right_branch = list()  # tracer of the right split with largest ig
    curr_wifi_number = 0  # tracer of the current attribute name
    split_points = find_split_points(training_dataset)

    for wifi_i in range(ATTR_NUM):
        for split_signal in split_points[wifi_i]:
            # clear split branches
            left_split = []
            right_split = []

            # List of numbers of data for rooms 1-4 with wifi signal larger than split signal, CLASS_NUM is 4 here.
            larger_data_num = [0] * CLASS_NUM

            # List of numbers of data for rooms 1-4 with wifi signal smaller than split signal
            smaller_data_num = [0] * CLASS_NUM

            for rowi in training_dataset:
                roomi = int(rowi[-1])

                signal = rowi[wifi_i]  # fetch signal from a row of the dataset and do the stats.
                if signal > split_signal:
                    larger_data_num[roomi - 1] += 1
                    left_split.append(rowi)
                else:
                    smaller_data_num[roomi - 1] += 1
                    right_split.append(rowi)

            larger_data_total = sum(larger_data_num)
            smaller_data_total = sum(smaller_data_num)
            all_data_total = larger_data_total + smaller_data_total

            # lists to store pk for calculating entropy
            larger_data_pk = []
            smaller_data_pk = []
            for larger_data in larger_data_num:
                if larger_data_total != 0:
                    larger_data_pk.append(larger_data / larger_data_total)
            for smaller_data in smaller_data_num:
                if smaller_data_total != 0:
                    smaller_data_pk.append(smaller_data / smaller_data_total)

            larger_entropy = calc_entropy(larger_data_pk)
            smaller_entropy = calc_entropy(smaller_data_pk)
            remainder = larger_entropy * larger_data_total / all_data_total + smaller_entropy * smaller_data_total / all_data_total

            # calculate the information gain:
            ig = current_entropy - remainder

            # find the maximum ig and corresponding nodes.
            if ig > cur_max_ig:
                cur_max_ig = ig
                cur_split_point = split_signal
                left_branch = list(left_split)
                right_branch = list(right_split)
                curr_wifi_number = wifi_i

    return {'left_split': left_branch, 'right_split': right_branch}, cur_split_point, curr_wifi_number + 1


def decision_tree_learning(training_dataset, depth):
    label = []
    for rowi in training_dataset:
        label.append(rowi[-1])

    # check if all labels are the same
    isUniLabel = label.count(label[0]) == len(label)

    if isUniLabel:
        # return a pair of node and depth
        return {'attribute': 'Room: ', 'value': label[0], 'left': None, 'right': None, 'leaf': True}, depth
    else:
        split = find_split(training_dataset)
        signal_strength = split[1]
        wifi_number = split[2]
        l_dataset = split[0]['left_split']
        r_dataset = split[0]['right_split']
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)
        node = {'attribute': 'wifi_' + str(wifi_number) + '_signal > ', 'value': signal_strength, 'left': l_branch,
                'right': r_branch, 'leaf': False}
        return (node, max(l_depth, r_depth))


def tree_toString(node, depth):
    for d in range(depth + 2):
        printTree(node, d)


def printTree(node, level):
    if not node:
        return
    elif level == 1:
        print(' ' + node['attribute'] + str(node['value']))
    elif level > 1:
        printTree(node['left'], level - 1)
        printTree(node['right'], level - 1)


if __name__ == '__main__':
    inputfile = sys.argv[1]
    # decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
    training_dataset = np.loadtxt(inputfile)
    np.random.shuffle(training_dataset)
    depth = 0

    start_time = time.perf_counter()
    (d_tree, depth) = decision_tree_learning(training_dataset, depth)
    end_time = time.perf_counter()
    print('Time used: ' + str(end_time - start_time) + ' s\n')
    print('The maximum depth of the tree is ' + str(depth))
    print(tree_toString(d_tree, depth))
