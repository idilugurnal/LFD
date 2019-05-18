import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv
from keras import Sequential
from keras.layers import Dense

def randomForest(X_train , y_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)


    classifier = RandomForestClassifier(n_estimators = 5)
    classifier.fit(X_train , y_train)
    y_pred  = classifier.predict(X_test)

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


def NeuralNetwork(X_train , y_train , X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    pca = PCA(n_components=20)
    X_scaled = pca.fit_transform(X_train)

    X_test_scaled = pca.fit_transform(X_test)

    #First Layer

    classifier = Sequential()
    classifier.add(Dense(10 , activation = 'relu' , kernel_initializer='random_normal' , input_dim = 20))

    #Second Layer

    classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal'))


    #Output Layer

    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

    # Compiling the neural network
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the data to the training dataset
    classifier.fit(X_scaled, y_train, batch_size=5, epochs=400)

    y_pred = classifier.predict(X_test_scaled)

    '''for i in range(0 , len(y_pred)):
        if(y_pred[i] > 0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0


    print(y_pred)

    result = 0

    #print(len(y_pred))
    #print(y_test)


    for i in range(0 , len(y_pred)):
        if(int(y_pred[i]) == y_test.iloc[i]):
            result+=1

    print(result / len(y_pred))'''
    writeBack(y_pred)




if __name__ == '__main__':

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    X_train = train_data.iloc[: , 0: -1]
    y_train = train_data.iloc[: , -1]
    X_test = test_data.iloc[: , 0:]

    #X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=1)


    randomForest(X_train , y_train , X_test ) #Bu haliyle %60 aldi

