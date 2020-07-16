import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Downloading data
#wget -O ChurnData.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv
df = pd.read_csv('ChurnData.csv')
# print(df.head())
df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
df['churn'] = df['churn'].astype('int')
print(df.head())

X = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
#Standardize dataset
X = preprocessing.StandardScaler().fit(X).transform(X)
Y = np.asarray(df['churn'])
# print(X[0:5])

#Splittling training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)

#Here,C is the inverse of regularization strength which is always a positive float value
#Solver specifies the optimization technique to be used 
#Log Loss, We can change values of C and solver to minimize the log loss
lr= LogisticRegression(C=0.05, solver='liblinear').fit(X_train,Y_train)
# print(lr)

yhat = lr.predict(X_test)
yhat_prob = lr.predict_proba(X_test)
print(yhat_prob[0])
print("Accuracy: ",accuracy_score(Y_test,yhat))
print("Log Loss: ",log_loss(Y_test, yhat_prob))



