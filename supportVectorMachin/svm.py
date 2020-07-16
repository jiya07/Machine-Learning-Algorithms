import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import f1_score

#Loading data
df = pd.read_csv("cell_samples.csv")
print(df.dtypes)

#we drop the rows where the values of BareNuc are not numerical
df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')
# print(df.dtypes)

ax = df[df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Red', label='malignant');
df[df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

X= np.asarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
df['Class'] = df['Class'].astype('int')
Y= np.asarray(df['Class'])

#Spliting training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)

clf = svm.SVC(kernel='rbf').fit(X_train, Y_train) 

#Predicton
yhat = clf.predict(X_test)

#Calculating accuracy
print(f1_score(Y_test, yhat, average='weighted')) 
