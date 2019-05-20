import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.feature_selection import SelectFromModel

def randomForest(X_train , y_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train, y_train)
    X_test = sc.fit_transform(X_test)

    sel = SelectFromModel(RandomForestClassifier(n_estimators=5))
    sel.fit(X_train, y_train)


    selected_features = X_train[: , (sel.get_support())]

    print(selected_features.shape[1])



    classifier = RandomForestClassifier(n_estimators = 5)
    classifier.fit(selected_features , y_train)

    selected_features = X_test[: , (sel.get_support())]
    print(selected_features.shape[1])

    y_pred = classifier.predict(selected_features)

    writeBack(y_pred)
    print(y_pred)

    count = 0

    for i in y_pred:
        if i == 1:
            count +=1

    print(count)





def adaBoostClassifier(X_train , y_train , X_test):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)

    X_test = pca.fit_transform(X_test)

    abc = AdaBoostClassifier(n_estimators = 100 , learning_rate = 1)
    model = abc.fit(X_train , y_train)

    y_pred = model.predict(X_test)
    print(y_pred)
    writeBack(y_pred)



def writeBack(y_pred):


    for i in range(0 , len(y_pred)+1):
        if i == 0:
            with open('output.csv', 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(["ID","Predicted"])
            continue
        row = [i,int(y_pred[i-1])]
        with open('output.csv' , 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(row)





if __name__ == '__main__':

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    X_train = train_data.iloc[: , 0: -1]
    y_train = train_data.iloc[: , -1]
    X_test = test_data.iloc[: , 0:]

    #X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=1)


    randomForest(X_train , y_train , X_test ) #Bu haliyle %60 aldi

