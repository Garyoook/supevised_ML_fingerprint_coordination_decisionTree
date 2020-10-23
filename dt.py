import numpy
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


if __name__ == '__main__':
    inputfile = './wifi_db/clean_dataset.txt'
    # create_csv_data_from_txt(inputfile)


# decision_tree format: python dictionary: {'attribute', 'value', 'left', 'right', 'leaf'}
# split rule: trial0: split by room numbers.
    training_dataset = numpy.loadtxt(inputfile)

    decision_tree_learning(training_dataset, depth)


    print('initial commit.')
