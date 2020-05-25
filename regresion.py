import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def main():
    dataset_csv = pd.read_csv('german_credit_data.csv')
    dataset_csv.head()
    dataset_csv.fillna('0', inplace=True)

    label_encode = preprocessing.LabelEncoder()
    sex = label_encode.fit_transform(dataset_csv["Sex"])
    housing = label_encode.fit_transform(dataset_csv["Housing"])
    saving = label_encode.fit_transform(dataset_csv["Saving accounts"])
    checking = label_encode.fit_transform(dataset_csv["Checking account"])

    dataset_csv["Sex"] = sex
    dataset_csv["Housing"] = housing
    dataset_csv["Saving accounts"] = saving
    dataset_csv["Checking account"] = checking

    x = dataset_csv.drop("Purpose", axis=1)
    y = dataset_csv.Purpose

    # min_max = preprocessing.MinMaxScaler()
    # dataset_normal = min_max.fit_transform(dataset_csv)
    # dataset_normal = np.array(dataset_normal)

    # rows = range(0, 1000)
    # columns = range(0, 8)

    # x = dataset_normal[rows].T[columns].T
    # y = dataset_normal[rows].T[8].T

    # label_encoder = preprocessing.LabelEncoder()
    # encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4, test_size=0.25)

    logical_rgs = LogisticRegression()

    logical_rgs.fit(x_train, y_train)

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000, multi_class='warn',
                       n_jobs=None, penalty='l2', random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)

    y_prediction = logical_rgs.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_prediction)
    accuracy_percentage = 100 * accuracy
    print(accuracy)
    print(accuracy_percentage)


if __name__ == '__main__':
    main()
