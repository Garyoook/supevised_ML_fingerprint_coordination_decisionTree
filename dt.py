import numpy as np
from matplotlib import pyplot as plt

# util imports (MUST REMOVE when submitted)
import pandas as pd


def create_csv_data_from_txt(inputfile):
    outputfile = './wifi_db/clean_dataset.csv'
    print('generating csv...')
    data = pd.read_csv(inputfile, sep='\t',
                       names=['rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'room'])
    data.to_csv(outputfile, header=['rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'room'], index=0)


def decision_tree_learning(traning_dataset, depth):
    pass

def calc_entropy(data):
    result = 0
    for i in data:
        result -= np.log2(data[i])
    return result


if __name__ == '__main__':
    inputfile = './wifi_db/clean_dataset.txt'
    # create_csv_data_from_txt(inputfile)


# decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
# split rule: trial0: split by room numbers.
    training_dataset = np.loadtxt(inputfile)
    depth = 0

# do statistic for the sample size of different rooms
    rooms = dict()
    for rowi in training_dataset:
        room_id = rowi[-1]
        if room_id in rooms:
            rooms[room_id] = rooms[room_id] + 1
        else:
            rooms[room_id] = 1
# turns out that there are 500 samples for each room.

# then calculate the proportion of samples in each room
    total_num = len(training_dataset)
    room_ratio = dict()
    for i in range(len(rooms)):
        curr_num = rooms[i+1]
        room_ratio['room'+ str(i+1)] = curr_num/total_num

# after getting the ratio, we can calculate the entropy:
    entropy_all = calc_entropy(room_ratio)
    print(entropy_all)

# TODO: then loop through the remainder to get Information Gain:


# TODO: complete this function after
    decision_tree_learning(training_dataset, depth)
