import random
import sys

import numpy as np

import dt

import sys

from texttable import Texttable

FOLD_NUM = 10
CLASS_NUM = 4


def predict(test_data, d_tree):
    """
    :param test_data: test set
    :param d_tree: decision tree
    :return: classified label
    """
    if d_tree['leaf']:
        return d_tree['value']
    wifi_number = int(d_tree['attribute'].split('_')[1])
    signal_value = d_tree['value']
    if test_data[wifi_number - 1] > signal_value:
        return predict(test_data, d_tree['left'])
    else:
        return predict(test_data, d_tree['right'])


def evaluate(test_db, trained_tree):
    """
    generate confusion matrix
    :param test_db: test set
    :param trained_tree: decision tree generated by training set
    :return: confusion matrix
    """
    # decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
    wifi_number = int(trained_tree['attribute'].split('_')[1])
    signal_value = trained_tree['value']
    confusion_matrix = [[0] * CLASS_NUM for _ in range(CLASS_NUM)]

    for rowi in test_db:
        actual_signal = rowi[wifi_number - 1]
        actual_room = int(rowi[-1])
        if actual_signal > signal_value:
            predicted_room = int(predict(rowi, trained_tree['left']))
        else:
            predicted_room = int(predict(rowi, trained_tree['right']))
        confusion_matrix[actual_room - 1][predicted_room - 1] += 1
    return confusion_matrix


# return tp, fp, tn, fn in order in a list
def get_tp_fp_tn_fn(test_db, trained_tree, class_num):
    confusion_matrix = evaluate(test_db, trained_tree)
    tp = confusion_matrix[class_num - 1][class_num - 1]
    fp = sum(confusion_matrix[i][class_num - 1] for i in range(CLASS_NUM) if i != class_num - 1)
    tn = sum(confusion_matrix[i][i] for i in range(CLASS_NUM) if i != class_num - 1)
    fn = sum(confusion_matrix[class_num - 1][i] for i in range(CLASS_NUM) if i != class_num - 1)
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


def get_accuracy(test_db, trained_tree, class_num):
    attributes = get_tp_fp_tn_fn(test_db, trained_tree, class_num)
    tp = attributes[0]
    tn = attributes[2]
    try:
        return (tp + tn) / len(test_db)
    except ZeroDivisionError:
        print("tp + tn + fp + fn result in a sum of 0, please check the classifier:")


def cross_validation(all_db_list):  
    label_list = ["index", "accuracy", "precision", "recall", "f1"]  # set up heading for evaluation result table
    for roomi in range(1, CLASS_NUM + 1):
        # total accuracy, precision, recall, f1 scores for all 10 folds of validation
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        db_size = len(all_db_list)
        step = db_size // FOLD_NUM
        arr = []
        arr.append(label_list)
        for start in range(0, db_size, step):
            # start and end position of test data
            end = start + step
            test_db = all_db_list[start:end]
            # set training set
            if start == 0:
                training_db = all_db_list[end:]
            elif end == db_size:
                training_db = all_db_list[:start]
            else:
                training_db = np.concatenate((all_db_list[:start], all_db_list[end:]))
            d_tree, depth = dt.decision_tree_learning(training_db, 0)
            accuracy = get_accuracy(test_db, d_tree, roomi)
            precision = get_precision(test_db, d_tree, roomi)
            recall = get_recall(test_db, d_tree, roomi)
            f1 = get_f1(test_db, d_tree, roomi)
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            col=[str(start),str(accuracy),str(precision),str(recall),str(f1)]
            arr.append(col)
            data = evaluate(test_db, d_tree)
            class_list = ["room1", "room2", "room3", "room4"]  # set up heading for the confusion matrix 
            data.insert(0,class_list)
            matrix = Texttable()
            matrix.add_rows(data)
            print('confusion matrix for room ' + str(roomi) + ' in fold ' + str(start) +' is: ')
            print(matrix.draw()) 
        t = Texttable()
        t.add_rows(arr)
        print('Evaluation result for room' + str(roomi) + ' is: ')
        average_result = ["average", str(total_accuracy / FOLD_NUM), str(total_precision / FOLD_NUM), str(total_recall / FOLD_NUM), str(total_f1 / FOLD_NUM)]
        t.add_row(average_result)
        print(t.draw())           
        


if __name__ == '__main__':
    inputfile = sys.argv[1]
    all_db = np.loadtxt(inputfile)
    all_db_list = []
    for row in all_db:
        all_db_list.append(row)
    random.shuffle(all_db_list)
    cross_validation(all_db_list)
    
