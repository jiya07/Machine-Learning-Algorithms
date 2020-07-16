import numpy as numpy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
#Downloading data
#wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv
df=pd.read_csv('drug200.csv')
print(df.head())
print(df.size)

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

#Since decision tree doesn't work with categorical values, we convert them to numerical values
numSex=preprocessing.LabelEncoder()
numBP=preprocessing.LabelEncoder()
numChol=preprocessing.LabelEncoder()
numSex.fit(['F','M'])
numBP.fit(['HIGH','LOW','NORMAL'])
numChol.fit(['HIGH','NORMAL'])

X[:,1]=numSex.transform(X[:,1])
X[:,2]=numBP.transform(X[:,2])
X[:,3]=numChol.transform(X[:,3])
# print(X[0:5])

y=df['Drug']
# print(y.head())

#Spiliting data for training and testing
X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=0.3,random_state=5)
decTree=DecisionTreeClassifier(criterion='entropy',max_depth=4)
# print(decTree)

decTree.fit(X_train,Y_train)
predTree=decTree.predict(X_test)
print(predTree[0:5])
print(Y_test[0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(Y_test, predTree))
