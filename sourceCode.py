import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.feature_selection import SelectKBest, chi2, f_regression

from sklearn.preprocessing import StandardScaler




def writeBack(y_pred):

    #Function to write back

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



def preprocessing(X_train , y_train , X_test):

    #Feature selection is done here


    bestfeatures = SelectKBest(score_func=chi2 , k = 50)

    fit = bestfeatures.fit(X_train , y_train)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)

    featureScores = pd.concat([dfcolumns,dfscores] , axis = 1)
    featureScores.columns = ['Specs' , 'Score']



    selected_features = featureScores.sort_values(['Score'] , ascending=0).iloc[0:50,:]


    #5 , 6 , 11 , 7 , 9 are deducted


    newX_train = X_train.loc[:,selected_features['Specs']]

    newX_test = X_test.loc[:,selected_features['Specs']]

    plt.matshow(newX_train.corr().abs())
    plt.show()

    newX_train = newX_train.drop(['X584' , 'X579' , 'X404', 'X528' , 'X318'] , axis = 1)
    newX_test = newX_test.drop(['X584', 'X579', 'X404', 'X528', 'X318'] , axis = 1)

    sc = StandardScaler()
    newX_train = sc.fit_transform(newX_train)
    newX_test = sc.fit_transform(newX_test)


    return newX_train , newX_test

def plot_decision_regions(X, y):

    #This is for plotting decision regions

    print(X)

    newX = pd.concat([X , y], axis = 1)

    print(newX)

    plt.figure()

    a = newX.plot.scatter(x = 'X558' , y = 'X168' , c = 'class', colormap = 'viridis')

    plt.show()

def trainModel(X_train , y_train , X_test):

    #training

    model = AdaBoostClassifier(n_estimators=10 )


    model.fit(X_train , y_train)

    return model

def predict(model):

    #prediction
    y_pred = model.predict(X_test)

    return y_pred



def loadData():
    # Read train and test data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # Take useful parts
    X_train = train_data.iloc[:, 0: -1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, 0:]


    return X_train , y_train , X_test







if __name__ == '__main__':


    X_train , y_train , X_test = loadData()


    #Do feature selection on data
    X_train , X_test = preprocessing(X_train, y_train , X_test)


    #Classify test set
    model = trainModel(X_train , y_train ,X_test)
    y_pred = predict(model)
    writeBack(y_pred)
