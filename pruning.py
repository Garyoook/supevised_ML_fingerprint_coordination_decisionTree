import random
import sys
from collections import deque

import numpy as np
from texttable import Texttable

import dt
from evaluate import evaluate, get_confusion_matrix, get_recall, get_precision, get_f1, get_accuracy, \
    separate_data, FOLD_NUM, CLASS_NUM
from visualise_dtree import visualise_decision_tree


def prune(test_data, d_tree):
    prune_helper(test_data, d_tree, d_tree)


def prune_helper(test_data, node, d_tree):
    if not node["left"]["leaf"]:
        prune_helper(test_data, node["left"], d_tree)
    if not node["right"]["leaf"]:
        prune_helper(test_data, node["right"], d_tree)
    if node["left"]["leaf"] and node["right"]["leaf"]:
        accuracy = evaluate(test_data, d_tree)
        curr_node = node.copy()
        node.update(curr_node["left"].copy())
        if evaluate(test_data, d_tree) < accuracy:
            node.update(curr_node.copy())
        else:
            left_node = node.copy()
            node.update(curr_node["right"].copy())
            if evaluate(test_data, d_tree) < accuracy:
                node.update(left_node.copy())


# def prune(test_data, d_tree):
#     layers = get_layers(d_tree)
#     accuracy = evaluate(test_data, d_tree)
#     while layers:
#         layer = layers.pop()
#         for node in layer:
#             if node["left"]["leaf"] and node["right"]["leaf"]:
#                 curr_node = node.copy()
#                 node.update(curr_node["left"].copy())
#                 if evaluate(test_data, d_tree) < accuracy:
#                     node.update(curr_node.copy())
#                     node.update(curr_node["right"].copy())
#                     if evaluate(test_data, d_tree) < accuracy:
#                         node.update(curr_node.copy())
#                 else:
#                     left_node = node.copy()
#                     node.update(curr_node["right"].copy())
#                     if evaluate(test_data, d_tree) < accuracy:
#                         node.update(left_node.copy())
#     return d_tree


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
    header_list = ["index", "accuracy", "precision", "recall", "f1"]  # set up heading for evaluation result table
    class_list = ["room1", "room2", "room3", "room4"]  # set up heading for the confusion matrix
    macro_table = Texttable()
    macro_table.header(header_list)

    d_tree_max_accuracy = dict()

    for roomi in range(CLASS_NUM):  # validate for each class (room)
        max_accuracy = 0
        # total accuracy, precision, recall, f1 scores and confusion matrix for all 10 folds of validation
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_matrix = np.zeros((CLASS_NUM, CLASS_NUM))
        # maximum depth of all decision trees generated
        max_depth = 0
        # calculate step size
        db_size = len(all_db_list)
        step = db_size // FOLD_NUM
        training_validation_db_size = 0
        # initialise list for result output
        output = [header_list]

        for start in range(0, db_size, step):
            # separate data into (training + validation) data and test data
            end = start + step
            test_db, training_validation_db = separate_data(all_db_list, start, end, db_size)

            training_validation_db_size = len(training_validation_db)

            for nested_start in range(0, training_validation_db_size, step):
                # separate (training + validation) data into training data and validation data
                nested_end = nested_start + step
                validation_db, training_db = separate_data(training_validation_db, nested_start, nested_end,
                                                           training_validation_db_size)

                # training
                d_tree, depth = dt.decision_tree_learning(training_db, 0)

                # pruning
                prune(validation_db, d_tree)

                # calculate metrics
                confusion_matrix = get_confusion_matrix(test_db, d_tree)
                precision = get_precision(roomi, confusion_matrix)
                recall = get_recall(roomi, confusion_matrix)
                f1 = get_f1(roomi, confusion_matrix)
                accuracy = get_accuracy(roomi, confusion_matrix)
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_accuracy += accuracy
                total_matrix = np.array(confusion_matrix) + np.array(total_matrix)

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    d_tree_max_accuracy[roomi - 1] = (d_tree, depth)

                # print results
                col = [str(start // step + 1), str(accuracy), str(precision), str(recall), str(f1)]
                output.append(col)
        t = Texttable()
        t.add_rows(output)
        print('Evaluation result for room' + str(roomi + 1) + ' is: ')
        stats_denom = np.ceil((training_validation_db_size / step))
        average_result = ["average of room" + str(roomi + 1), str(total_accuracy / (stats_denom * FOLD_NUM)),
                          str(total_precision / (stats_denom * FOLD_NUM)),
                          str(total_recall / (stats_denom * FOLD_NUM)), str(total_f1 / (stats_denom * FOLD_NUM))]
        macro_table.add_row(average_result)
        t.add_row(average_result)
        print(t.draw())  # print "index", "accuracy", "precision", "recall", "f1" of each fold
        average_matrix = np.array(total_matrix) / (stats_denom * FOLD_NUM)
        m = Texttable()
        m.header(class_list)
        for i in range(CLASS_NUM):
            m.add_row(average_matrix[i])
        print('average confusion matrix for room ' + str(roomi + 1) + ' is: ')
        print(m.draw())  # print average confusion matrix
    print(macro_table.draw())
    for key in d_tree_max_accuracy:
        visualise_decision_tree(d_tree_max_accuracy[key][0], d_tree_max_accuracy[key][1])


if __name__ == '__main__':
    inputfile = sys.argv[1]
    all_db = np.loadtxt(inputfile)
    all_db_list = []
    for row in all_db:
        all_db_list.append(row)
    random.shuffle(all_db_list)
    cross_validation(all_db_list)
