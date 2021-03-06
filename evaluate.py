import sys

import numpy as np
from texttable import Texttable  # used for formatting the output, not in evaluation implementation.

import dt
from dt import SEED_CONST

FOLD_NUM = 10
CLASS_NUM = 4


def evaluate(test_db, trained_tree):
    """
    :param test_db: The test dataset
    :param trained_tree: The decision tree to evaluate
    :return: the accuracy of the confusion matrix generated by the decision tree
    """
    confusion_matrix = get_confusion_matrix(test_db, trained_tree)
    total_accuracy = 0
    for roomi in range(CLASS_NUM):
        total_accuracy += get_accuracy(roomi, confusion_matrix)
    total_accuracy /= CLASS_NUM
    return total_accuracy


def cross_validation(all_db_list):
    """
    Generated cross-validation results for decision tree learning
    This is a single loop cross-validation process
    :param all_db_list: input data set
    """
    header_list = ["index", "accuracy", "precision", "recall", "f1",
                   "maximal depth"]  # set up heading for evaluation result table
    class_list = ["room1", "room2", "room3", "room4"]  # set up heading for the confusion matrix
    macro_table = Texttable()
    macro_table.header(header_list)

    # total accuracy, precision, recall, f1 scores and confusion matrix for all 10 folds of validation
    total_accuracy = [0] * CLASS_NUM
    total_precision = [0] * CLASS_NUM
    total_recall = [0] * CLASS_NUM
    total_f1 = [0] * CLASS_NUM
    total_matrix = np.zeros((CLASS_NUM, CLASS_NUM))

    # maximum depth of all decision trees generated
    max_depth = 0

    # calculate step size
    db_size = len(all_db_list)
    step = db_size // FOLD_NUM

    # initialise 4 charts for result output
    metric_charts_display = []
    for i in range(CLASS_NUM):
        t = Texttable()
        t.add_row(header_list)
        metric_charts_display.append(t)

    for start in range(0, db_size, step):  # permute training data set and test data set
        # separate data into training data and test data
        end = start + step
        test_db, training_db = separate_data(all_db_list, start, end, db_size)

        # training
        d_tree, depth = dt.decision_tree_learning(training_db, 0)

        # update maximum depth
        if depth > max_depth:
            max_depth = depth

        # get confusion matrix
        confusion_matrix = get_confusion_matrix(test_db, d_tree)
        total_matrix = np.array(confusion_matrix) + np.array(total_matrix)

        # display confusion matrix
        matrix_display = Texttable()
        matrix_display.header(class_list)
        for i in range(CLASS_NUM):
            matrix_display.add_row(confusion_matrix[i])
        fold_num = np.int(np.ceil(start / step)) + 1
        print('Confusion matrix of fold ' + str(fold_num) + ' is: ')
        print(matrix_display.draw())  # print average confusion matrix
        print()

        for roomi in range(CLASS_NUM):  # validate for each class (room)
            # calculate metrics
            precision = get_precision(roomi, confusion_matrix)
            recall = get_recall(roomi, confusion_matrix)
            f1 = get_f1(roomi, confusion_matrix)
            accuracy = get_accuracy(roomi, confusion_matrix)
            total_precision[roomi] += precision
            total_recall[roomi] += recall
            total_f1[roomi] += f1
            total_accuracy[roomi] += accuracy

            # add result of each fold to the text-table of each room
            col = [str(fold_num), str(accuracy), str(precision), str(recall), str(f1), str(depth)]
            metric_charts_display[roomi].add_row(col)

    for roomi in range(CLASS_NUM):  # display results for each room
        print('Evaluation result for room ' + str(roomi + 1) + ' is: ')
        average_result = ["average of room " + str(roomi + 1), str(total_accuracy[roomi] / FOLD_NUM),
                          str(total_precision[roomi] / FOLD_NUM),
                          str(total_recall[roomi] / FOLD_NUM), str(total_f1[roomi] / FOLD_NUM),
                          str(max_depth) + ' (Note: this is max depth rather than avg depth)']
        macro_table.add_row(average_result)
        metric_charts_display[roomi].add_row(average_result)
        # print "index", "accuracy", "precision", "recall", "f1" of each fold for each room
        print(metric_charts_display[roomi].draw())
        print()

    # display confusion matrix
    average_matrix = np.array(total_matrix) / FOLD_NUM
    matrix_display = Texttable()
    matrix_display.header(class_list)
    for i in range(CLASS_NUM):
        matrix_display.add_row(average_matrix[i])
    print('Average confusion matrix is: ')
    print(matrix_display.draw())  # print average confusion matrix
    print()

    # display average results in all folds for each room
    print('Average metrics for each room is:')
    print(macro_table.draw())
    print()


def separate_data(all_db_list, start, end, size):
    """
    Separates the data into (training + validation) set and test set / training set and test set
    :param all_db_list: all the data
    :param start: start index of test/validation set
    :param end: end index of test/validation set
    :param size: size of all data
    :return: a pair (test_set, training_set) or (test_set, training_validation_set)
    """
    test_set = all_db_list[start:end]  # test or validation set
    # training set or (training + validation) set
    if start == 0:
        training_set = all_db_list[end:]
    elif end == size:
        training_set = all_db_list[:start]
    else:
        training_set = np.concatenate((all_db_list[:start], all_db_list[end:]))
    return test_set, training_set


def predict(test_data_row, d_tree):
    """
    :param: test_data: one-dimensional, a row of test data
    :param: d_tree: trained tree
    :return: classified label (room)
    """
    # leaf (base case)
    if d_tree['leaf']:
        return d_tree['value']

    # node (recursion case)
    attr_number = int(d_tree['attribute'].split('_')[1])  # attribute (can be 1-7) at this node
    split_value = d_tree['value']  # split value at this node
    if test_data_row[attr_number - 1] > split_value:
        return predict(test_data_row, d_tree['left'])
    else:
        return predict(test_data_row, d_tree['right'])


def get_confusion_matrix(test_db, trained_tree):
    """
    :param: test_db: two-dimensional, all test data
    :param: trained_tree: a python dictionary {'attribute', 'value', 'left', 'right', 'leaf'}
    :return: confusion matrix generated by the decision tree and the test dataset
    """

    confusion_matrix = [[0] * CLASS_NUM for _ in range(CLASS_NUM)]  # 4*4 confusion matrix

    for rowi in test_db:
        actual_room = int(rowi[-1])  # class (room) value from test data
        predicted_room = int(predict(rowi, trained_tree))
        confusion_matrix[actual_room - 1][predicted_room - 1] += 1
    return confusion_matrix


def get_recall(class_num, confusion_matrix):
    attributes = get_tp_fp_tn_fn(class_num, confusion_matrix)
    tp = attributes[0]
    fn = attributes[3]
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        print("tp + fn result in a sum of 0, please check the classifier:")


def get_precision(class_num, confusion_matrix):
    attributes = get_tp_fp_tn_fn(class_num, confusion_matrix)
    tp = attributes[0]
    fp = attributes[1]
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        print("tp + fp result in a sum of 0, please check the classifier:")


def get_f1(class_num, confusion_matrix):
    precision = get_precision(class_num, confusion_matrix)
    recall = get_recall(class_num, confusion_matrix)
    try:
        return 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        print("precision and recall are both 0, please check the classifier:")


def get_accuracy(class_num, confusion_matrix):
    metrics = get_tp_fp_tn_fn(class_num, confusion_matrix)
    tp = metrics[0]
    tn = metrics[2]
    try:
        return (tp + tn) / sum(metrics)
    except ZeroDivisionError:
        print("tp + tn + fp + fn result in a sum of 0, please check the classifier:")


def get_tp_fp_tn_fn(class_num, confusion_matrix):
    tp = confusion_matrix[class_num][class_num]
    fp = sum(confusion_matrix[i][class_num] for i in range(CLASS_NUM) if i != class_num)
    fn = sum(confusion_matrix[class_num][i] for i in range(CLASS_NUM) if i != class_num)
    tn = sum(confusion_matrix[i][j] for i in range(CLASS_NUM) for j in range(CLASS_NUM)) - tp - fp - fn
    return [tp, fp, tn, fn]


if __name__ == '__main__':
    inputfile = sys.argv[1]
    all_db = np.loadtxt(inputfile)
    all_db_list = []
    for row in all_db:
        all_db_list.append(row)
    # np.random.seed(SEED_CONST)  # random seed used for consistent output during implementation
    np.random.shuffle(all_db_list)
    cross_validation(all_db_list)
