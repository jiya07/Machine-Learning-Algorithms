import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

#Downloading data
#wget -nv -O china_gdp.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv
df = pd.read_csv("china_gdp.csv")
# print(df.head(10))

#Plotting data
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
# print(plt.show())

#Logistic function used for initializing the paramters
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

#Normalize data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

#Training and testing dataset
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]

train_y = ydata[msk]
test_y = ydata[~msk]

#Build model
from scipy.optimize import curve_fit

#For optimization of parameters Beta1 and Beta 2 we use curve_fit
popt, pcov = curve_fit(sigmoid, train_x, train_y)

#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

#Predicted Values
y = sigmoid(test_x, *popt)
#Plotting both the graphs
plt.figure(figsize=(8,5))
plt.plot(test_x, test_y, 'ro', label='data')
plt.plot(test_x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
print(plt.show())

#Calculated error
print("Mean absolute error: %.2f" % np.mean(np.absolute(y - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y - test_y) ** 2))
print("R2-score: %.2f" % r2_score(y , test_y) )
