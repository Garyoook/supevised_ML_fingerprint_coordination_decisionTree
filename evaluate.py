import numpy as np
from matplotlib import pyplot as plt

import dt


def getResult(test_data, d_tree):
    if d_tree['leaf']:
        return d_tree['value']
    wifi_number = int(d_tree['attribute'].split('_')[1])
    signal_value = d_tree['value']
    if test_data[wifi_number - 1] > signal_value:
        return getResult(test_data, d_tree['left'])
    else:
        return getResult(test_data, d_tree['right'])


def evaluate(test_db, trained_tree):
    # COMMENT: decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
    wifi_number = int(trained_tree['attribute'].split('_')[1])
    signal_value = trained_tree['value']
    confusion_matrix = [[0] * 4] * 4
    # true_positive = 0
    # false_positive = 0
    # true_negative = 0
    # false_negative = 0

    predicted_room = 0
    for rowi in test_db:
        actual_signal = rowi[wifi_number - 1]
        actual_room = int(rowi[-1])
        if actual_signal > signal_value:
            predicted_room = int(getResult(rowi, trained_tree['left']))
        else:
            predicted_room = int(getResult(rowi, trained_tree['right']))
        print(predicted_room)
        if predicted_room == actual_room:
            confusion_matrix[actual_room-1][predicted_room-1] += 1
        else:
            confusion_matrix[actual_room-1][predicted_room-1] += 1
    print(confusion_matrix)


if __name__ == '__main__':
    all_db = np.loadtxt('./wifi_db/clean_dataset.txt')
    test_db = all_db[450:550]
    training_db = np.concatenate((all_db[0:450], all_db[550:2001]), axis=0)
    d_tree, depth = dt.decision_tree_learning(training_db, 0)

    evaluate(test_db, d_tree)
