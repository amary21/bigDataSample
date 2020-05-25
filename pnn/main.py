import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats

from peceptron import Perceptron

dataset = pd.read_excel("transfusion_data_training.xlsx")
dataset = dataset.drop('No', axis=1)
dataset_test = pd.read_excel("transfusion_data_testing.xlsx")

# plotting a graph to see class imbalance
# dataset[dataset.columns[4]].value_counts().plot(kind="barh")
# plt.xlabel("Count")
# plt.ylabel("Classes")
# plt.show()

from sklearn.preprocessing import MinMaxScaler

#X = np.array(dataset.drop(dataset.columns[4], axis=1))
#y = np.array(dataset[dataset.columns[4]])

X = np.array(dataset_test.drop(dataset_test.columns[4], axis=1))
y = np.array(dataset_test[dataset_test.columns[4]])


mnscaler = MinMaxScaler()
#
X = mnscaler.fit_transform(X)
# X = stats.zscore(X, axis=1)
# X = np.array(pd.DataFrame(X, columns=data.drop("class", axis=1).columns))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

perceptron = Perceptron()

#50000 0.001 80%
wt_matrix = perceptron.fit(X_train, Y_train, 300, 0.001)

Y_pred_test = perceptron.predict(X_test)

print("Acuracy " + str(accuracy_score(Y_pred_test, Y_test)))

print("Y pred test", Y_pred_test)
print("Y Test", Y_test)

print(len(Y_pred_test))
print(len(Y_test))

a = 0

for i in Y_pred_test:
    for j in Y_test:
        if i == j :
            a += 1

print("True",a)
        