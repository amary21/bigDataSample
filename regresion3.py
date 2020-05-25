# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:20:52 2020

@author: Muhammad Imam Zunaidi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("german_credit_data.csv", delimiter=",")

print(data.groupby('Purpose').size())

data_baru = data.select_dtypes(include=['object']).copy()
jumlahbaris, jumlahkolom = data_baru.shape
for myIndex in range(0,jumlahkolom):
    headerName = data_baru.columns[myIndex]
    data_baru[headerName] = data_baru[headerName].astype("category")
    data_baru[headerName] = data_baru[headerName].cat.codes
    data[headerName] = data_baru[headerName]
data_numeric = data

x = data_numeric.drop(["Purpose"], axis = 1)
y = data_numeric["Purpose"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 20, random_state = 12)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

LogisticRegression(C =10, class_weight = None, dual = False, fit_intercept = True,
                   intercept_scaling =1, l1_ratio = None, max_iter = 100, multi_class = 'warn',
                   n_jobs = None, penalty = '12', random_state = None, solver = 'warn', tol = 0.0001, 
                   verbose = 0, warm_start = False)

y_pred = logistic_regression.predict(x_test)
y_pred

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
print("Data Test : ",+ accuracy_percentage, "%")