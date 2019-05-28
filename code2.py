import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler




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

    


    bestfeatures = SelectKBest(score_func=chi2 , k = 50)

    fit = bestfeatures.fit(X_train , y_train)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)

    featureScores = pd.concat([dfcolumns,dfscores] , axis = 1)
    featureScores.columns = ['Specs' , 'Score']

    ###


    selected_features = featureScores.sort_values(['Score'] , ascending=0).iloc[0:50,:]


    #5 , 6 , 11 , 7 , 9,


    newX_train = X_train.loc[:,selected_features['Specs']]

    newX_test = X_test.loc[:,selected_features['Specs']]

    plt.matshow(newX_train.corr().abs())
    plt.show()

    newX_train = newX_train.drop(['X584' , 'X579' , 'X404', 'X528' , 'X318'] , axis = 1)
    newX_test = newX_test.drop(['X584', 'X579', 'X404', 'X528', 'X318'] , axis = 1)

    sc = StandardScaler()
    newX_train = sc.fit_transform(newX_train)
    newX_test = sc.fit_transform(newX_test)



    #pca = PCA(n_components=8)

    #newX_train = pca.fit_transform(newX_train , y_train)
    #newX_test = pca.fit_transform(newX_test)

    print(newX_train)



    #draw_x = newX_train.drop(['X243', 'X66', 'X38', 'X474', 'X450', 'X503'], axis=1)

    #plot_decision_regions(draw_x , y_train)
    return newX_train , newX_test

def plot_decision_regions(X, y):

    print(X)

    newX = pd.concat([X , y], axis = 1)

    print(newX)

    plt.figure()

    a = newX.plot.scatter(x = 'X558' , y = 'X168' , c = 'class', colormap = 'viridis')

    plt.show()

def randomForest(X_train , y_train , X_test):
    model = BaggingClassifier(n_estimators=100, random_state=3)

    # 3 ile %60 aldik


    model.fit(X_train , y_train)

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
