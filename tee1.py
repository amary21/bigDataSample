import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix 

data_sample = pd.read_csv('german_credit_data.csv')
lEncode = preprocessing.LabelEncoder()

data_sample['Sex'] = lEncode.fit_transform(data_sample['Sex'])
data_sample['Housing'] = lEncode.fit_transform(data_sample['Housing'])


# Drop String character data non label
x = data_sample.drop(["Purpose","Checking account","Saving accounts"], axis=1)
y = data_sample["Purpose"]

# data tidak di normalisasi
x_train, x_test = train_test_split(x, test_size=0.2)
y_train, y_test = train_test_split(y, test_size=0.2)

# Normalisasi data X
scaler= MinMaxScaler() 
scaler.fit_transform(x_train)

x_scaled = scaler.fit_transform(x)
df_xscaled = pd.DataFrame(x_scaled)
print(df_xscaled)
#Untuk data ternormalisasi
x_scaled_train, x_scaled_test = train_test_split(df_xscaled, test_size=0.2)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test) 

# model Test 1
array_hasil=[]
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    #value prediksi
    pred=knn.predict(x_test)
    #nilaiPrediksi
    hasil=accuracy_score(y_test, pred)
    array_hasil.append(hasil)

print(array_hasil)

# model Test 2
array_norm=[]
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_scaled_train,y_train)
    #value prediksi
    pred1=knn.predict(x_scaled_test)
    #nilaiPrediksi
    hasil1=accuracy_score(y_test, pred1)
    array_norm.append(hasil1)
    
print(array_norm)

import matplotlib.pyplot as plt
import numpy as np
plt.plot(array_hasil)
plt.ylabel('nilai akurasi')
plt.xlabel('nilai K')
plt.xticks(np.arange(10),('1','2','3','4','5','6','7','8','9','10'))
plt.show()