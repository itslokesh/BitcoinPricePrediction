import csv
import os
import numpy as np
import pandas as pd
import talib
import time
from keras.layers import Dense, Activation
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dropout


dataset = pd.read_csv('data.csv')
dataset = dataset.dropna()
dataset = dataset[['Open', 'High', 'Low', 'Close']]
np.random.seed(7)
layers=[1,50,100,1]
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day MA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day MA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30day MA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev']= dataset['Close'].rolling(5).std()
dataset['RSI'] = talib.RSI(dataset['Close'].values, timeperiod = 9)
dataset['Williams %R'] = talib.WILLR(dataset['High'].values, dataset['Low'].values, dataset['Close'].values, 7)
dataset['Price_Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)

dataset = dataset.dropna()
X = dataset #.iloc[:, 1:3]
y = dataset.iloc[:, 4]

split=int(len(dataset)*0.1)
X_train, X_test, y_train, y_test= X[:split],X[split:], y[:split], y[split:]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 250)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
trade_dataset = dataset.dropna()

trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns'])

trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

trade_dataset['ind']=np.arange(1,len(trade_dataset)+1,1)

dataset['ind']=np.arange(1,len(dataset)+1,1)
trade_dataset['Return Rate']=(trade_dataset['Close']-trade_dataset['Close'].shift(1))/trade_dataset['Close']
dataset['Return Rate']=(dataset['Close']-dataset['Close'].shift(1))/dataset['Close'].shift(1)
mean_return=np.mean(trade_dataset['Return Rate'])
mean_data=np.mean(dataset['Return Rate'])
variance_return=np.var(trade_dataset['Return Rate'])
variance_data=np.var(dataset['Return Rate'])
trade_dataset['Predicted']=5.97*np.exp((mean_return-variance_return/2)*trade_dataset['ind'] + np.sqrt(variance_return)*np.random.normal(0,1))
dataset['Predicted']=5.97*np.exp((mean_data-variance_data/2)*dataset['ind']+np.sqrt(variance_data)*np.random.normal(0,1))
latest=[]
for i in range(len(dataset)):
	latest.append(5.97*np.exp((mean_data-variance_data/2)*i+np.sqrt(variance_data)*np.random.normal(0,1)))


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(latest,color='g', label='Geometric Brownian Motion')
plt.plot(dataset['Close'], color='r', label='Close Values')
#plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
#plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()
