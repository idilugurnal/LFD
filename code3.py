import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import csv

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.iloc[: , 0: -1]
Y_train = train_data.iloc[: , -1]
X_test = test_data.iloc[: , 0:]




bestfeatures=SelectKBest(score_func=chi2 , k = 50)
bestFeatures = bestfeatures.fit(X_train , Y_train)

dfscores = pd.DataFrame(bestFeatures.scores_)
dfcolumns = pd.DataFrame(X_train.columns)

selected_features = pd.concat([dfcolumns, dfscores], axis=1)
selected_features.columns = ['Specs', 'Score']
selected_features=selected_features.sort_values(['Score'] , ascending=0).iloc[0:50,:]

newX_train = X_train.loc[:,selected_features['Specs']]
newX_test = X_test.loc[:,selected_features['Specs']]



df = pd.DataFrame(newX_train)
corr=df.corr()


plt.matshow(corr.abs())
plt.show()

var=np.std(newX_train,axis=0)
mean=np.mean(var)

newX_train = StandardScaler().fit_transform(newX_train)
train_data=pd.concat([pd.DataFrame(newX_train),Y_train], axis=1)

sum0 = 0
sum1 = 0
count = 0

print(Y_train[0])

for i in X_train:
    if Y_train[count] == 1:
        sum1+= X_train[i]
    else:
        sum0+= X_train[i]
    count +=1
    print(count)

print(sum0)
print(sum1)

plt.matshow(pd.DataFrame(train_data).loc[train_data['class'] == 1])
plt.show()

plt.matshow(pd.DataFrame(train_data).loc[train_data['class'] == 0])
plt.show()

plt.matshow(corr.abs())
plt.show()



