import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import csv





if __name__ == '__main__':

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    X_train = train_data.iloc[: , 1: -1]
    y_train = train_data.iloc[: , -1]
    X_test = test_data.iloc[: , 1:]



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



    regressor = RandomForestRegressor(n_estimators = 20 , random_state = 0)
    regressor.fit(X_scaled , y_train)
    y_pred  = regressor.predict(X_test_scaled)

    #print(y_pred)

    count = 1

    for i in range(len(y_pred)):
        if(y_pred[i] < 0.5):
            y_pred[i] = 0
        else:
            y_pred[i] = 1
        count+=1


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


