import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import sklearn.tree as tree
import sklearn.neighbors as neighbors
import csv

def randomForest(X_train , y_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    pca = PCA(n_components = 2)
    X_scaled = pca.fit_transform(X_train)

    X_test_scaled = pca.fit_transform(X_test)

    ''' Burasi kac tane feature a indirgememiz gerektigini anlamak icin var daha detayli anlamak icin
    #https://towardsdatascience.com/dive-into-pca-principal-component-analysis-with-python-43ded13ead21 '''
    ex_variance = np.var(X_scaled, axis=0)
    ex_variance_ratio = ex_variance / np.sum(ex_variance)
    print(ex_variance_ratio)



    classifier = RandomForestClassifier(n_estimators = 100)
    classifier.fit(X_scaled , y_train)
    y_pred  = classifier.predict(X_test_scaled)

    writeBack(y_pred)
    print(y_pred)





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

    X_train = train_data.iloc[: , 1: -1]
    y_train = train_data.iloc[: , -1]
    X_test = test_data.iloc[: , 1:]

    randomForest(X_train , y_train , X_test)

    #Not: Hangi classifieri denersem deneyeyim pca yi 2 vererek maks a ulastim (o da 55) baska seyler denemek gerekebilir
    #Issuelara bak

