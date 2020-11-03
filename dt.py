import os
import time

import numpy as np
import random


def calc_entropy(data):
    result = 0
    for i in data:
        if data[i] != 0:
            result -= data[i] * np.log2(data[i])
    return result


# data is a list
def calc_entropy_list(data):
    if not data:
        return 0
    result = 0
    for p in data:
        if p != 0:
            result -= p * np.log2(p)
    return result


def find_split_points(dataset):
    split_points = []
    for col in range(7):
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
    # need to return a dictionary of 2 dataset 'left_split': ... and 'right_split': ...
    # (format(matrix) like the original dataset) and a 'wifi_number': xx
    # e.g. {'left_split': None,'right_split': None,'wifi_number': 0}

    num_of_room = [0] * 4

    ratios_for_entropy_calc = dict()

    number_total = 0
    for rowi in training_dataset:
        room_id = int(rowi[-1])
        num_of_room[room_id - 1] += 1

    for num in num_of_room:
        number_total += num
    for roomi in range(1, 5):
        ratio_target_room = num_of_room[roomi - 1] / number_total
        ratios_for_entropy_calc['room' + str(roomi)] = ratio_target_room

    current_entropy = calc_entropy(ratios_for_entropy_calc)

    cur_max_ig = 0
    cur_split_point = 0
    left_split = []
    right_split = []

    split_points = find_split_points(training_dataset)

    for wifi_i in range(7):
        for split_signal in split_points[wifi_i]:
        # for split_signal in range(-100, 0):
            left_split.clear()
            right_split.clear()
            # List of numbers of data for rooms 1-4 with wifi signal larger than split signal
            larger_data_num = [0] * 4
            # List of numbers of data for rooms 1-4 with wifi signal smaller than split signal
            smaller_data_num = [0] * 4

            for rowi in training_dataset:
                roomi = int(rowi[-1])
                signal_roomi = rowi[wifi_i]
                if signal_roomi > split_signal:
                    larger_data_num[roomi - 1] += 1
                    left_split.append(rowi)
                else:
                    smaller_data_num[roomi - 1] += 1
                    right_split.append(rowi)

            larger_data_total = sum(larger_data_num)
            smaller_data_total = sum(smaller_data_num)
            all_data_total = larger_data_total + smaller_data_total
            larger_data_ratio = []
            smaller_data_ratio = []
            for larger_data in larger_data_num:
                if larger_data_total != 0:
                    larger_data_ratio.append(larger_data / larger_data_total)
            for smaller_data in smaller_data_num:
                if smaller_data_total != 0:
                    smaller_data_ratio.append(smaller_data / smaller_data_total)

            larger_entropy = calc_entropy_list(larger_data_ratio)
            smaller_entropy = calc_entropy_list(smaller_data_ratio)
            remainder = larger_entropy * larger_data_total / all_data_total + smaller_entropy * smaller_data_total / all_data_total
            ig = current_entropy - remainder
            if ig > cur_max_ig:
                cur_max_ig = ig
                cur_split_point = split_signal
                left_branch = list(left_split)
                right_branch = list(right_split)
                curr_wifi_number = wifi_i

    # print('wifi ' + str(curr_wifi_number) + 'split at: ' + str(cur_split_point))
    return {'left_split': left_branch, 'right_split': right_branch}, cur_split_point, curr_wifi_number + 1


# TODO: then loop through the remainder to get Information Gain:


def decision_tree_learning(training_dataset, depth):
    tags = []
    for rowi in training_dataset:
        tags.append(rowi[-1])

    # check if all tags are the same
    isUniTag = tags.count(tags[0]) == len(tags)

    if isUniTag:
        # COMMENT: return a pair of node and depth
        # print('leaf: room' + str(tags[0]))
        return ({'attribute': 'Room: ', 'value': tags[0], 'left': None, 'right': None, 'leaf': True}, depth)
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
    inputfile = './wifi_db/clean_dataset.txt'

    # COMMENT: decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
    # COMMENT: split rule: trial0: split by room numbers.
    training_dataset = np.loadtxt(inputfile)
    random.shuffle(training_dataset)
    depth = 0

    start_time = time.perf_counter()
    (d_tree, depth) = decision_tree_learning(training_dataset, depth)
    end_time = time.perf_counter()
    print('time used: ' + str(end_time - start_time) + ' s\n')

    print(tree_toString(d_tree, depth))
