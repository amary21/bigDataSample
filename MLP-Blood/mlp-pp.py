#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 03:46:32 2020

@author: amary
@Email: taufik.amary@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.pipeline import Pipeline


#memanggil data set train
data_train = pd.read_excel('transfusion_data.xlsx', sheet_name='training')
data_train = data_train.drop('No', axis=1)

#memanggil data set test
data_test = pd.read_excel('transfusion_data.xlsx', sheet_name='testing')
data_test = data_test.drop('No', axis=1)

#membaca jumlah feature    
n_feature = len(data_train.columns[0:-1])

#menentukan input & output data train
x_data_train = data_train.drop('whether he/she donated blood in March 2007', axis=1)
y_data_train = data_train['whether he/she donated blood in March 2007']

#menentukan input & output data test 
x_data_test = data_test.drop('whether he/she donated blood in March 2007', axis=1)
y_data_test = data_test['whether he/she donated blood in March 2007']

#menggabungkan dataset
all_x_dataset = [x_data_train, x_data_test]
all_y_dataset = [y_data_train, y_data_test]
x_dataset = pd.concat(all_x_dataset)
y_dataset = pd.concat(all_y_dataset)

#correlation matrix
sns.heatmap(
    data=x_dataset.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn'
)
fig = plt.gcf()
fig.set_size_inches(30, 16)
plt.show()

# #mencari best parameter MLP
# pipe = Pipeline(steps=[
#     ('preprocess', StandardScaler()),
#     ('classification', MLPClassifier())
# ])

# random_state = 42
# mlp_activation = ['identity', 'logistic', 'tanh', 'relu']
# mlp_solver = ['lbfgs', 'sgd', 'adam']
# mlp_max_iter = range(1000, 10000, 1000)
# mlp_alpha = [1e-4, 1e-3, 0.01, 0.1, 1]
# mlp_hidden_layer_sizes = np.arange(1, 30)
# preprocess = [Normalizer(), MinMaxScaler(), StandardScaler(), RobustScaler(), QuantileTransformer()]

# mlp_param_grid = [
#     {
#         'preprocess': preprocess,
#         'classification__hidden_layer_sizes': mlp_hidden_layer_sizes,
#         'classification__activation': mlp_activation,
#         'classification__solver': mlp_solver,
#         'classification__random_state': [random_state],
#         'classification__max_iter': mlp_max_iter,
#         'classification__alpha': mlp_alpha
#     }
# ]

# strat_k_fold = StratifiedKFold(
#     n_splits=10,
#     random_state=42
# )

# mlp_grid = GridSearchCV(
#     pipe,
#     param_grid=mlp_param_grid,
#     cv=strat_k_fold,
#     scoring='f1',
#     n_jobs=-1,
#     verbose=2
# )

# mlp_grid.fit(x_dataset, y_dataset)

# # Best MLPClassifier parameters
# print(mlp_grid.best_params_)
# # Best score for MLPClassifier with best parameters
# print('\nBest F1 score for MLP: {:.2f}%'.format(mlp_grid.best_score_ * 100))
# best_params = mlp_grid.best_params_

scaler = StandardScaler()
print('\nData preprocessing with {scaler}\n'.format(scaler=scaler))

x_train_scaler = scaler.fit_transform(x_data_train)
x_test_scaler = scaler.fit_transform(x_data_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(3,),
    max_iter=1000,
    alpha=0.01,
    activation='relu',
    verbose=True,
    learning_rate_init = 0.01,
    solver='lbfgs',
    random_state=42
)

mlp.fit(x_train_scaler, y_data_train)

mlp_predict = mlp.predict(x_test_scaler)
mlp_predict_proba = mlp.predict_proba(x_test_scaler)[:, 1]

MSE = mean_squared_error(y_data_test, mlp_predict)

print('MLP report:\n\n', classification_report(y_data_test, mlp_predict))
print('MLP Training set score: {:.2f}%'.format(mlp.score(x_train_scaler, y_data_train) * 100))
print('MLP Testing set score: {:.2f}%'.format(mlp.score(x_test_scaler, y_data_test) * 100))
print('\nMLP Accuracy: {:.2f}%'.format(accuracy_score(y_data_test, mlp_predict) * 100))
print("Root Mean Square Error ", MSE)

# outcome_labels = sorted(y_dataset.unique())
# # Confusion Matrix for MLPClassifier
# sns.heatmap(
#     confusion_matrix(y_data_test, mlp_predict),
#     annot=True,
#     fmt="d",
#     xticklabels=outcome_labels,
#     yticklabels=outcome_labels
# )

# # ROC for MLPClassifier
# fpr, tpr, thresholds = roc_curve(y_data_test, mlp_predict_proba)

# plt.plot([0,1],[0,1],'k--')
# plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.rcParams['font.size'] = 12
# plt.title('ROC curve for MLPClassifier')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.grid(True)

# strat_k_fold = StratifiedKFold(
#     n_splits=10,
#     random_state=42
# )

# scaler = StandardScaler()

# X_std = scaler.fit_transform(x_dataset)

# fe_score = cross_val_score(
#     mlp,
#     X_std,
#     y_dataset,
#     cv=strat_k_fold,
#     scoring='f1'
# )

# print("MLP: F1 after 10-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
#     fe_score.mean() * 100,
#     fe_score.std() * 2
# ))