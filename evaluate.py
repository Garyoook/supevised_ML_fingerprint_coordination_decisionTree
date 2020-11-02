import numpy as np
import random
from matplotlib import pyplot as plt

import dt


def get_result_class(test_data, d_tree):
    if d_tree['leaf']:
        return d_tree['value']
    wifi_number = int(d_tree['attribute'].split('_')[1])
    signal_value = d_tree['value']
    if test_data[wifi_number - 1] > signal_value:
        return get_result_class(test_data, d_tree['left'])
    else:
        return get_result_class(test_data, d_tree['right'])


def evaluate(test_db, trained_tree):
    # COMMENT: decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
    wifi_number = int(trained_tree['attribute'].split('_')[1])
    signal_value = trained_tree['value']
    confusion_matrix = [[0] * 4 for _ in range(4)]

    for rowi in test_db:
        actual_signal = rowi[wifi_number - 1]
        actual_room = int(rowi[-1])
        if actual_signal > signal_value:
            predicted_room = int(get_result_class(rowi, trained_tree['left']))
        else:
            predicted_room = int(get_result_class(rowi, trained_tree['right']))
        confusion_matrix[actual_room - 1][predicted_room - 1] += 1
    return confusion_matrix


# return tp, fp, tn, fn in order in a list
def get_tp_fp_tn_fn(test_db, trained_tree, class_num):
    confusion_matrix = evaluate(test_db, trained_tree)
    tp = confusion_matrix[class_num - 1][class_num - 1]
    fp = sum(confusion_matrix[i][class_num - 1] for i in range(4) if i != class_num - 1)
    tn = sum(confusion_matrix[i][i] for i in range(4) if i != class_num - 1)
    fn = sum(confusion_matrix[class_num - 1][i] for i in range(4) if i != class_num - 1)
    return [tp, fp, tn, fn]


def get_precision(test_db, trained_tree, class_num):
    attributes = get_tp_fp_tn_fn(test_db, trained_tree, class_num)
    tp = attributes[0]
    fp = attributes[1]
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        print("tp + fp result in a sum of 0, please check the classifier:")


def get_recall(test_db, trained_tree, class_num):
    attributes = get_tp_fp_tn_fn(test_db, trained_tree, class_num)
    tp = attributes[0]
    fn = attributes[3]
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        print("tp + fn result in a sum of 0, please check the classifier:")


def get_f1(test_db, trained_tree, class_num):
    precision = get_precision(test_db, trained_tree, class_num)
    recall = get_recall(test_db, trained_tree, class_num)
    try:
        return 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        print("precision and recall are both 0, please check the classifier:")


# accuracy:
def get_classification_rate(test_db, trained_tree, class_num):
    attributes = get_tp_fp_tn_fn(test_db, trained_tree, class_num)
    tp = attributes[0]
    tn = attributes[2]
    try:
        return (tp + tn) / sum(attributes)
    except ZeroDivisionError:
        print("tp + tn + fp + fn result in a sum of 0, please check the classifier:")


if __name__ == '__main__':
    all_db = np.loadtxt('./wifi_db/clean_dataset.txt')
    all_db_list = []
    for row in all_db:
        all_db_list.append(row)
    random.shuffle(all_db_list)
    test_db = np.concatenate((all_db_list[:200], all_db_list[1800:]), axis=0)
    training_db = np.concatenate((all_db_list[:350], all_db_list[650:]), axis=0)
    d_tree, depth = dt.decision_tree_learning(training_db, 0)

    # d_tree = {'attribute': 'wifi_1_signal > ', 'value': -55.0, 'left': {'attribute': 'wifi_1_signal > ', 'value': -45.0, 'left': {'attribute': 'wifi_4_signal > ', 'value': -48.0, 'left': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_1_signal > ', 'value': -43.0, 'left': {'attribute': 'wifi_3_signal > ', 'value': -48.5, 'left': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'leaf': False}, 'right': {'attribute': 'wifi_5_signal > ', 'value': -71.0, 'left': {'attribute': 'wifi_4_signal > ', 'value': -40.0, 'left': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_5_signal > ', 'value': -53.5, 'left': {'attribute': 'Room: ', 'value': 4.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_3_signal > ', 'value': -54.0, 'left': {'attribute': 'wifi_7_signal > ', 'value': -73.0, 'left': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'wifi_2_signal > ', 'value': -55.0, 'left': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_6_signal > ', 'value': -78.0, 'left': {'attribute': 'wifi_5_signal > ', 'value': -67.0, 'left': {'attribute': 'wifi_7_signal > ', 'value': -78.0, 'left': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'wifi_4_signal > ', 'value': -49.0, 'left': {'attribute': 'wifi_7_signal > ', 'value': -78.0, 'left': {'attribute': 'wifi_6_signal > ', 'value': -78.5, 'left': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'wifi_4_signal > ', 'value': -47.0, 'left': {'attribute': 'wifi_6_signal > ', 'value': -86.0, 'left': {'attribute': 'wifi_6_signal > ', 'value': -79.5, 'left': {'attribute': 'wifi_1_signal > ', 'value': -49.0, 'left': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'leaf': False}, 'right': {'attribute': 'wifi_7_signal > ', 'value': -77.0, 'left': {'attribute': 'wifi_2_signal > ', 'value': -58.5, 'left': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'right': {'attribute': 'wifi_4_signal > ', 'value': -49.0, 'left': {'attribute': 'wifi_3_signal > ', 'value': -55.5, 'left': {'attribute': 'wifi_7_signal > ', 'value': -79.5, 'left': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'wifi_3_signal > ', 'value': -59.0, 'left': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_1_signal > ', 'value': -50.5, 'left': {'attribute': 'Room: ', 'value': 2.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'right': {'attribute': 'wifi_5_signal > ', 'value': -60.0, 'left': {'attribute': 'wifi_5_signal > ', 'value': -56.5, 'left': {'attribute': 'Room: ', 'value': 4.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_4_signal > ', 'value': -58.5, 'left': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 4.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'leaf': False}, 'right': {'attribute': 'wifi_4_signal > ', 'value': -56.0, 'left': {'attribute': 'wifi_2_signal > ', 'value': -51.0, 'left': {'attribute': 'wifi_1_signal > ', 'value': -59.0, 'left': {'attribute': 'Room: ', 'value': 1.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'right': {'attribute': 'wifi_3_signal > ', 'value': -56.0, 'left': {'attribute': 'wifi_7_signal > ', 'value': -86.0, 'left': {'attribute': 'Room: ', 'value': 1.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_5_signal > ', 'value': -63.0, 'left': {'attribute': 'Room: ', 'value': 4.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_6_signal > ', 'value': -85.5, 'left': {'attribute': 'Room: ', 'value': 1.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'wifi_1_signal > ', 'value': -58.0, 'left': {'attribute': 'Room: ', 'value': 3.0, 'left': None, 'right': None, 'leaf': True}, 'right': {'attribute': 'Room: ', 'value': 4.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'right': {'attribute': 'Room: ', 'value': 1.0, 'left': None, 'right': None, 'leaf': True}, 'leaf': False}, 'leaf': False}, 'leaf': False}, 'leaf': False}
    print('confusion matrix is: ' + str(evaluate(test_db, d_tree)))
    print('accuracy: ' + str(get_accuracy(test_db, d_tree, 4)))
    print('precision: ' + str(get_precision(test_db, d_tree, 4)))
    print('recall: ' + str(get_recall(test_db, d_tree, 4)))
    print('f1: ' + str(get_f1(test_db, d_tree, 4)))
