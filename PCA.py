from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import  numpy as np

import csv

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.iloc[: , 0: -1]
y_train = train_data.iloc[: , -1]
X_test = test_data.iloc[: , 0:]

#standart derivationı 0 - 1 
var=np.std(X_train,axis=0)
count=0
for i in var:
    if i==0 or i>1:
        X_train = X_train.drop(X_train.columns[[count]], axis=1)
        X_test = X_test.drop(X_test.columns[[count]], axis=1)
    count=count+1

train_x = StandardScaler().fit_transform(X_train)
test_x = StandardScaler().fit_transform(X_test)

pca = PCA(n_components=23)

principalComponents_train = pca.fit_transform(train_x)
cover_train=pca.explained_variance_ratio_.cumsum()

principalComponents_test = pca.fit_transform(test_x)
cover_test=pca.explained_variance_ratio_.cumsum()


classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(principalComponents_train , y_train)
y_pred  = classifier.predict(principalComponents_test)

for i in range(0 , len(y_pred)+1):
    if i == 0:
        with open('output.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(["ID", "Predicted"])
        continue
    row = [i, int(y_pred[i - 1])]
    with open('output.csv', 'a') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(row)




