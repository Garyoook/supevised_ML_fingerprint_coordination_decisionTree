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


def calc_entropy(data):
    result = 0
    for i in data:
        if data[i] != 0:
            result -= data[i] * np.log2(data[i])
    return result


def find_split(training_dataset):
    # COMMENT do statistic for the sample size of different rooms
    rooms = dict()
    for rowi in training_dataset:
        room_id = rowi[-1]
        if room_id in rooms:
            rooms[room_id] = rooms[room_id] + 1
        else:
            rooms[room_id] = 1
    # COMMENT: turns out that there are 500 samples for each room.

    # COMMENT: then calculate the proportion of samples in each room
    total_num = len(training_dataset)
    room_ratio = dict()
    for i in range(1, len(rooms) + 1):
        curr_num = rooms[i]
        room_ratio['room' + str(i)] = curr_num / total_num

    # COMMENT: after getting the ratio, we can calculate the entropy H(A):
    entropy_all = calc_entropy(room_ratio)
    print(entropy_all)

    # COMMENT: select wifi data with each room number:
    room_ids = []
    for id in rooms.keys():
        room_ids.append(id)
    # wifi 1:
    rssi_rooms = dict()
    for i in range(1, len(room_ids)):
        rssi_roomi = []
        for rowi in training_dataset:
            rssi_roomi.append(rowi[0:7])
            # COMMENT from here we have rssi for all wifi routers in each room.

    # COMMENT: making a dictionary of each wifi spot.
    rssi_split_by_wifi = []
    for i in range(1, 8):
        rssi_wifii = []
        rssi_wifij = []
        for rowi in training_dataset:
            rssi_wifii.append(rowi[i])
        rssi_split_by_wifi.append(rssi_wifii)
    # COMMENT: so the first array in array rssi_split_by_wifi is rssi data from wifi1... etc.

    # COMMENT: added corresponding rooms number, but not used in the below WARNING section yet
    rooms_with_wifidata = dict()
    for rowi in training_dataset:
        room_id = rowi[-1]
        if room_id in rooms_with_wifidata:
            rooms_with_wifidata[room_id].append(rowi)
        else:
            rooms_with_wifidata[room_id] = [rowi]

    # WARNING: MISSING information about the out come in this section, so the ratio obtained is wrong in this version.
    entropy_splitted = dict()
    for i in range(len(rssi_split_by_wifi)):
        wifii_data = rssi_split_by_wifi[i]
        wifii_ratio = dict()
        entropy_room1_wifii = 0
        for split_rssi in range(-100, 1):
            rssi_greater = []
            rssi_smaller = []
            num_greater = 0
            num_smaller = 0
            for rssi in wifii_data:
                if rssi > split_rssi:
                    rssi_greater.append(rssi)
                else:
                    rssi_smaller.append(rssi)
                num_greater = len(rssi_greater)
                num_smaller = len(rssi_smaller)
                num_total = len(wifii_data)
                # print('greater: ' + str(num_greater))
                # print('smaller: ' + str(num_smaller))
            wifii_ratio['greater than ' + str(split_rssi)] = num_greater / num_total
            wifii_ratio['smallerOrEqual than ' + str(split_rssi)] = num_smaller / num_total
            # COMMENT: calculate entropy here:
            entropy_room1_wifii = calc_entropy(wifii_ratio)
            wifii_ratio.clear()
            # print(entropy_room1_wifii)
        entropy_splitted['H(room, wifi' + str(i)] = entropy_room1_wifii


# TODO: then loop through the remainder to get Information Gain:


def decision_tree_learning(traning_dataset, depth):
    tags = []
    for rowi in training_dataset:
        tags.append(rowi[-1])

    # check if all tags are the same
    isUniTag = tags.count(tags[0]) == len(tags)

    if isUniTag:
        # COMMENT: return a pair of node and depth
        return ({'attribute': 'leaf', 'value': 0, 'left': None, 'right': None, 'leaf': True}, depth)
    else:
        split = find_split(training_dataset)
        l_dataset = split[0]
        r_dataset = split[1]
        node = {'attribute': '', 'value': 0, 'left': None, 'right': None, 'leaf': False}
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)
        return (node, max(l_depth, r_depth))


if __name__ == '__main__':
    inputfile = './wifi_db/clean_dataset.txt'
    # create_csv_data_from_txt(inputfile)

    # COMMENT: decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
    # COMMENT: split rule: trial0: split by room numbers.
    training_dataset = np.loadtxt(inputfile)
    depth = 0

    # TODO: complete this function after
    decision_tree_learning(training_dataset, depth)

