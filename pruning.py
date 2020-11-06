import sys

import numpy as np
from texttable import Texttable  # only used for formatting print results

import dt
from dt import SEED_CONST
from evaluate import evaluate, get_confusion_matrix, get_recall, get_precision, get_f1, get_accuracy, \
    separate_data, FOLD_NUM, CLASS_NUM
from visualise_dtree import get_tree_depth


def prune(validation_data, d_tree):
    """
    Calls the helped method below and recursively prune the decision tree in a post-order traversal
    :param validation_data: validation data set for pruning
    :param d_tree: decision tree generated from learning data set
    :return: pruned decision tree
    """
    prune_helper(validation_data, d_tree, d_tree)


def prune_helper(validation_data, node, d_tree):
    if not node["left"]["leaf"]:
        prune_helper(validation_data, node["left"], d_tree)
    if not node["right"]["leaf"]:
        prune_helper(validation_data, node["right"], d_tree)
    if node["left"]["leaf"] and node["right"]["leaf"]:
        accuracy = evaluate(validation_data, d_tree)
        curr_node = node.copy()
        node.update(curr_node["left"].copy())
        if evaluate(validation_data, d_tree) < accuracy:
            node.update(curr_node.copy())
            node.update(curr_node["right"].copy())
            if evaluate(validation_data, d_tree) < accuracy:
                node.update(curr_node.copy())
        else:
            left_node = node.copy()
            node.update(curr_node["right"].copy())
            if evaluate(validation_data, d_tree) < accuracy:
                node.update(left_node.copy())


def cross_validation(all_db_list):
    """
    Generates cross-validation results for decision tree learning and pruning
    This is a double loop cross-validation process
    :param all_db_list: input data set
    """
    # set up heading for evaluation result table
    header_list = ["index", "accuracy", "precision", "recall", "f1",
                   "maximal depth before pruning", "maximal depth after pruning"]
    # set up heading for the confusion matrix
    class_list = ["room1", "room2", "room3", "room4"]
    # set up text-table for average results of all metrics
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
    max_depth_after_pruning = 0

    # calculate step size
    db_size = len(all_db_list)
    step = db_size // FOLD_NUM

    # initialise 4 charts for result output
    metric_charts_display = []
    for i in range(CLASS_NUM):
        t = Texttable()
        t.add_row(header_list)
        metric_charts_display.append(t)

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

            # update maximum depth before pruning
            if depth > max_depth:
                max_depth = depth

            # pruning
            prune(validation_db, d_tree)
            depth_after_pruning = get_tree_depth(d_tree)  # update depth after pruning

            # update maximum depth after pruning
            if depth_after_pruning > max_depth_after_pruning:
                max_depth_after_pruning = depth_after_pruning

            # get confusion matrix
            confusion_matrix = get_confusion_matrix(test_db, d_tree)
            total_matrix = np.array(confusion_matrix) + np.array(total_matrix)

            # display confusion matrix
            matrix_display = Texttable()
            matrix_display.header(class_list)
            for i in range(CLASS_NUM):
                matrix_display.add_row(confusion_matrix[i])
            fold_num = np.int(np.ceil(start / step)) + 1
            inner_fold_num = np.int(np.ceil(nested_start / step)) + 1
            print('Confusion matrix of fold ' + str(fold_num) + '-' + str(inner_fold_num) + ' is: ')
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
                col = [str(fold_num) + '-' + str(inner_fold_num), str(accuracy), str(precision), str(recall), str(f1),
                       str(depth), str(depth_after_pruning)]
                metric_charts_display[roomi].add_row(col)

    total_fold_num = FOLD_NUM * (FOLD_NUM - 1)

    for roomi in range(CLASS_NUM):  # display results for each room
        print('Evaluation result for room' + str(roomi + 1) + ' is: ')
        average_result = ["average of room" + str(roomi + 1), str(total_accuracy[roomi] / total_fold_num),
                          str(total_precision[roomi] / total_fold_num),
                          str(total_recall[roomi] / total_fold_num), str(total_f1[roomi] / total_fold_num),
                          str(max_depth) + ' (Note: depths are maximal rather than avg value)',
                          str(max_depth_after_pruning)]
        macro_table.add_row(average_result)
        metric_charts_display[roomi].add_row(average_result)
        # print "index", "accuracy", "precision", "recall", "f1" of each fold for each room
        print(metric_charts_display[roomi].draw())
        print()

    # display confusion matrix
    average_matrix = np.array(total_matrix) / (total_fold_num)
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


if __name__ == '__main__':
    inputfile = sys.argv[1]
    all_db = np.loadtxt(inputfile)
    all_db_list = []
    for row in all_db:
        all_db_list.append(row)
    # np.random.seed(SEED_CONST)  # random seed used for consistent output during implementation
    np.random.shuffle(all_db_list)
    cross_validation(all_db_list)
