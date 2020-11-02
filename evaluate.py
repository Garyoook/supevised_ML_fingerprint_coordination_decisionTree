import numpy as np
import random
from matplotlib import pyplot as plt

import dt


# confusion matrix, recall, precision, f1 = 2*precision*recall/(precision+recall), classification = accuracy = (tp + tn) / (tp + tn + fp + fn)
def get_result(test_data, d_tree):
    if d_tree['leaf']:
        return d_tree['value']
    wifi_number = int(d_tree['attribute'].split('_')[1])
    signal_value = d_tree['value']
    if test_data[wifi_number - 1] > signal_value:
        return get_result(test_data, d_tree['left'])
    else:
        return get_result(test_data, d_tree['right'])


def evaluate(test_db, trained_tree):
    # COMMENT: decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
    wifi_number = int(trained_tree['attribute'].split('_')[1])
    signal_value = trained_tree['value']
    confusion_matrix = [[0] * 4 for _ in range(4)]
    # true_positive = 0
    # false_positive = 0
    # true_negative = 0
    # false_negative = 0

    for rowi in test_db:
        actual_signal = rowi[wifi_number - 1]
        actual_room = int(rowi[-1])
        if actual_signal > signal_value:
            predicted_room = int(get_result(rowi, trained_tree['left']))
        else:
            predicted_room = int(get_result(rowi, trained_tree['right']))
        confusion_matrix[actual_room - 1][predicted_room - 1] += 1
    print(confusion_matrix)
    return confusion_matrix


# return tp, fp, tn, fn in order in a list
def get_tp_fp_tn_fn(test_db, trained_tree, class_num):
    confusion_matrix = eval(test_db, trained_tree)
    tp = confusion_matrix[class_num - 1][class_num - 1]
    fp = sum(confusion_matrix[i][class_num - 1] for i in range(4) if i != class_num - 1)
    tn = sum(confusion_matrix[i][i] for i in range(4) if i != class_num - 1)
    fn = sum(confusion_matrix[class_num - 1][i] for i in range(4) if i != class_num - 1)
    return [tp, fp, tn, fn]


def get_precision(test_db, trained_tree, class_num):
    attributes = get_tp_fp_tn_fn(test_db, trained_tree, class_num)
    tp = attributes[0]
    fp = attributes[1]
    return tp / (tp + fp)


def get_recall(test_db, trained_tree, class_num):
    attributes = get_tp_fp_tn_fn(test_db, trained_tree, class_num)
    tp = attributes[0]
    fn = attributes[3]
    return tp / (tp + fn)


def get_f1(test_db, trained_tree, class_num):
    precision = get_precision(test_db, trained_tree, class_num)
    recall = get_recall(test_db, trained_tree, class_num)
    return 2 * precision * recall / (precision + recall)


def get_classification_rate(test_db, trained_tree, class_num):
    attributes = get_tp_fp_tn_fn(test_db, trained_tree, class_num)
    tp = attributes[0]
    tn = attributes[2]
    return (tp + tn) / sum(attributes)


if __name__ == '__main__':
    all_db = np.loadtxt('./wifi_db/clean_dataset.txt')
    all_db_list = []
    for row in all_db:
        all_db_list.append(row)
    random.shuffle(all_db_list)
    test_db = all_db_list[450:550]
    training_db = np.concatenate((all_db_list[:450], all_db_list[550:]), axis=0)
    d_tree, depth = dt.decision_tree_learning(training_db, 0)

    evaluate(test_db, d_tree)
