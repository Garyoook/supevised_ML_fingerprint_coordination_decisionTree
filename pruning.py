import random
import sys
from collections import deque

import numpy as np
from texttable import Texttable

import dt
from evaluate import evaluate, get_confusion_matrix, get_recall, get_precision, get_f1, get_accuracy, FOLD_NUM, \
    CLASS_NUM


def prune(test_data, d_tree):
    layers = get_layers(d_tree)
    accuracy = evaluate(test_data, d_tree)
    while layers:
        layer = layers.pop()
        for node in layer:
            if node["left"]["leaf"] and node["right"]["leaf"]:
                prev_node = node
                node = prev_node["left"]
                if evaluate(test_data, d_tree) < accuracy:
                    node = prev_node["right"]
                    if evaluate(test_data, d_tree) < accuracy:
                        node = prev_node
                else:
                    prev_node = node
                    node = prev_node["right"]
                    if evaluate(test_data, d_tree) < accuracy:
                        node = prev_node
    return d_tree


def get_layers(d_tree):
    layers = deque()
    queue = deque()
    queue.append(d_tree)
    while queue:
        row = []
        row_size = len(queue)
        while row_size > 0:
            current_node = queue.popleft()
            if not current_node["left"]["leaf"]:
                queue.append(current_node["left"])
            if not current_node["right"]["leaf"]:
                queue.append(current_node["right"])
            row.append(current_node)
            row_size = row_size - 1
        layers.append(row)
    return layers


def cross_validation(all_db_list):
    label_list = ["index", "accuracy", "precision", "recall", "f1"]  # set up heading for evaluation result table
    class_list = ["room1", "room2", "room3", "room4"]  # set up heading for the confusion matrix
    for roomi in range(1, CLASS_NUM + 1):
        # total accuracy, precision, recall, f1 scores for all 10 folds of validation
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_matrix = np.zeros((4, 4))
        db_size = len(all_db_list)
        step = db_size // FOLD_NUM
        arr = [label_list]
        for start in range(0, db_size, step):
            # start and end position of test data
            end = start + step
            test_db = all_db_list[start:end]
            # set training set
            if start == 0:
                training_validation_db = all_db_list[end:]
            elif end == db_size:
                training_validation_db = all_db_list[:start]
            else:
                training_validation_db = np.concatenate((all_db_list[:start], all_db_list[end:]))
            training_validation_db_size = len(training_validation_db)
            for nested_start in range(0, training_validation_db_size, step):
                # start and end position of test data
                nested_end = nested_start + step
                validation_db = all_db_list[start:end]
                # set training set
                if start == 0:
                    training_db = all_db_list[end:]
                elif end == db_size:
                    training_db = all_db_list[:start]
                else:
                    training_db = np.concatenate((all_db_list[:start], all_db_list[end:]))
                d_tree, depth = dt.decision_tree_learning(training_db, 0)
                prune(validation_db, d_tree)
                accuracy = get_accuracy(test_db, d_tree, roomi)
                precision = get_precision(test_db, d_tree, roomi)
                recall = get_recall(test_db, d_tree, roomi)
                f1 = get_f1(test_db, d_tree, roomi)
                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                data = get_confusion_matrix(test_db, d_tree)
                total_matrix = np.array(data) + np.array(total_matrix)
                col = [str(start / step), str(accuracy), str(precision), str(recall), str(f1)]
                arr.append(col)
                data.insert(0, class_list)
        t = Texttable()
        t.add_rows(arr)
        print('Evaluation result for room' + str(roomi) + ' is: ')
        average_result = ["average", str(total_accuracy / FOLD_NUM), str(total_precision / FOLD_NUM),
                          str(total_recall / FOLD_NUM), str(total_f1 / FOLD_NUM)]
        t.add_row(average_result)
        print(t.draw())  # print "index", "accuracy", "precision", "recall", "f1" of each fold
        average_matrix = np.array(total_matrix) / FOLD_NUM
        m = Texttable()
        m.header(class_list)
        for i in range(CLASS_NUM):
            m.add_row(average_matrix[i])
        print('average confusion matrix for room ' + str(roomi) + ' in fold ' + str(start / step) + ' is: ')
        print(m.draw())  # print average confusion matrix


if __name__ == '__main__':
    inputfile = sys.argv[1]
    all_db = np.loadtxt(inputfile)
    all_db_list = []
    for row in all_db:
        all_db_list.append(row)
    random.shuffle(all_db_list)
    cross_validation(all_db_list)