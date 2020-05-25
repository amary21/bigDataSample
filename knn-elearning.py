from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd

data_set = pd.read_csv('german_credit_data.csv')
#print("Data Awal :\n",data_set)

le = preprocessing.LabelEncoder()
data_set['Sex']= le.fit_transform(data_set['Sex'])
data_set['Housing']=le.fit_transform(data_set['Housing'])
#print("\nData Edit :\n",data_set)

x = data_set.drop(["Purpose","Checking account","Saving accounts","Unnamed: 0"],axis = 1)
y = data_set['Purpose']
#print("\nVariabel independen :\n",x)
#print("\nVariabel dependen :\n",y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 20, random_state = 12)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

logical_rgs = LogisticRegression()
logical_rgs.fit(x_train, y_train)

LogisticRegression(C =1.0, class_weight = None, dual = False, fit_intercept = True,
                   intercept_scaling =1, l1_ratio = None, max_iter = 100, multi_class = 'warn',
                   n_jobs = None, penalty = '12', random_state = None, solver = 'warn', tol = 0.0001, 
                   verbose = 0, warm_start = False)

y_pred_test = logical_rgs.predict(x_test)
y_pred_train = logical_rgs.predict(x_train)

conf_test = confusion_matrix(y_test, y_pred_test)
conf_train = confusion_matrix(y_train, y_pred_train)

clas_test = classification_report(y_test, y_pred_test)
clas_train = classification_report(y_train, y_pred_train)

#data train
print(conf_train)
print(clas_train)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_percent_train = 100 * accuracy_train

print("Akurasi Train : ", accuracy_percent_train , "% \n\n")

#data test
print(conf_test)
print(clas_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_percent_test = 100 * accuracy_test

print("Akurasi Test : ", accuracy_percent_test , "% \n\n")


