import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

from sklearn.datasets import load_iris

inputfile = './wifi_db/clean_dataset.txt'
df = pd.read_table(inputfile, names=['wifi1', 'wifi2', 'wifi3', 'wifi4', 'wifi5', 'wifi6', 'wifi7', 'room'])
data = df.loc[:, lambda df: ['wifi1', 'wifi2', 'wifi3', 'wifi4', 'wifi5', 'wifi6', 'wifi7']]
targets = df['room']
X_train, X_test, Y_train, Y_test = train_test_split(data,
                                                    df['room'], random_state=0)

# Step 2: Make an instance of the Model
clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
# Step 3: Train the model on the data
clf.fit(X_train, Y_train)

tree.plot_tree(clf)

fn = ['wifi1', 'wifi2', 'wifi3', 'wifi4', 'wifi5', 'wifi6', 'wifi7']
cn = ['room1', 'room2', 'room3', 'room4']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=1024)
tree.plot_tree(clf,
               feature_names=fn,
               class_names=cn,
               filled=True)
plt.show()
# fig.savefig('imagename.png')
