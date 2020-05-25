import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

read_data = pd.read_csv('german_credit_data.csv')

business = read_data['Job']
car = read_data['Credit amount']
target = read_data['Purpose']

le = preprocessing.LabelEncoder()
business_encoded = le.fit_transform(business)
car_encoded = le.fit_transform(car)
label = le.fit_transform(target)

features = list(zip(business_encoded,car_encoded))

from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.99)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

predicted= model.predict([[0,5]])
print(predicted)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
