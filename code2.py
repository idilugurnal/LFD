from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sklearn.tree as tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2, f_regression




def writeBack(y_pred):
    for i in range(0, len(y_pred) + 1):
        if i == 0:
            with open('output.csv', 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(["ID", "Predicted"])
            continue
        row = [i, int(y_pred[i - 1])]
        with open('output.csv', 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(row)



def feature_selection(X_train , y_train , X_test):


    bestfeatures = SelectKBest(score_func=chi2 , k = 6)

    fit = bestfeatures.fit(X_train , y_train)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)

    featureScores = pd.concat([dfcolumns,dfscores] , axis = 1)
    featureScores.columns = ['Specs' , 'Score']

    ###


    selected_features = featureScores.sort_values(['Score'] , ascending=0).iloc[0:6,:]



    newX_train = X_train.loc[:,selected_features['Specs']]

    newX_test = X_test.loc[:,selected_features['Specs']]


    return newX_train , newX_test


def randomForest(X_train , y_train , X_test):
    model = RandomForestClassifier(n_estimators=5)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(y_pred)

    count = 0

    for i in y_pred:
        if i == 1:
            count += 1
    print(count)

    writeBack(y_pred)







if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    X_train = train_data.iloc[:, 0: -1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, 0:]

    # X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=1)


    X_train , X_test = feature_selection(X_train, y_train , X_test)

    randomForest(X_train , y_train ,X_test)
