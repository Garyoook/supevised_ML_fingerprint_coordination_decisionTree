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


def cross_validation(all_db_list):
    for roomi in range(1, 5):
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        for start in range(0, 2000, 200):
            end = start + 100
            test_db = all_db_list[start:end]
            if start == 0:
                training_db = all_db_list[end:]
            elif end == 2000:
                training_db = np.concatenate((all_db_list[:start], all_db_list[end:]))
            else:
                training_db = all_db_list[:start]
            d_tree, depth = dt.decision_tree_learning(training_db, 0)
            accuracy = get_classification_rate(test_db, d_tree, roomi)
            precision = get_precision(test_db, d_tree, roomi)
            recall = get_recall(test_db, d_tree, roomi)
            f1 = get_f1(test_db, d_tree, roomi)
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            print('confusion matrix for room ' + str(roomi) + ' is: ' + str(evaluate(test_db, d_tree)))
            print('accuracy for room ' + str(roomi) + ' is: ' + str(accuracy))
            print('precision for room ' + str(roomi) + ' is: ' + str(precision))
            print('recall for room ' + str(roomi) + ' is: ' + str(recall))
            print('f1 for room ' + str(roomi) + ' is: ' + str(f1))
        print('average accuracy for room ' + str(roomi) + ' is: ' + str(total_accuracy / 10))
        print('average precision for room ' + str(roomi) + ' is: ' + str(total_precision / 10))
        print('average recall for room ' + str(roomi) + ' is: ' + str(total_recall / 10))
        print('average f1 for room ' + str(roomi) + ' is: ' + str(total_f1 / 10))


if __name__ == '__main__':
    all_db_clean = np.loadtxt('./wifi_db/clean_dataset.txt')
    all_db_clean_list = []
    for row in all_db_clean:
        all_db_clean_list.append(row)
    random.shuffle(all_db_clean_list)
    cross_validation(all_db_clean_list)
    all_db_noisy = np.loadtxt('./wifi_db/noisy_dataset.txt')
    all_db_noisy_list = []
    for row in all_db_noisy:
        all_db_noisy_list.append(row)
    random.shuffle(all_db_noisy_list)
    cross_validation(all_db_noisy_list)

