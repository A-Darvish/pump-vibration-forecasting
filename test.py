# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 00:41:54 2023

@author: Arvand
"""

# load data
import pandas as pd

file = 'D:/Arvand/uni/projects/New folder/art_daily_jumpsup.csv'
train = pd.read_csv(file)

train.head()
train.describe()

from sklearn.preprocessing import MinMaxScaler
# scale
s1 = MinMaxScaler(feature_range=(-1, 1))
Xs = s1.fit_transform(train[['pump_vibration']])
#for i, x in enumerate(Xs):
#    if x > 0 :
#        print(i)


#print(len(Xs))
# scale predicted value
s2 = MinMaxScaler(feature_range=(-1, 1))
Ys = s2.fit_transform(train[['pump_vibration']])
window = 65 
X = []
Y = []
for i in range(window, len(Xs)):
    X.append(Xs[i-window:i])
    Y.append(Ys[i])





# Reshape data to format accepted by LSTM
import numpy as np
X, Y = np.array(X), np.array(Y)

# create and train LSTM model

# For LSTM model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

# Initialize LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1],X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
model.summary()

# Allow for early exit 

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',mode='min',verbose=1,patience=10)

# Fit (and time) LSTM model

import time
t0 = time.time()
history = model.fit(X, Y, epochs = 12, batch_size = 50, callbacks=[es], verbose=1)
t1 = time.time()
print('Runtime: %.2f s' %(t1-t0))

# Plot loss 

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.semilogy(history.history['loss'])
plt.xlabel('epoch'); plt.ylabel('loss')
plt.savefig('model_loss.png')
model.save('model.h5')

# Verify the fit of the model
Yp = model.predict(X)

# un-scale outputs
Yu = s2.inverse_transform(Yp)
Ym = s2.inverse_transform(Y)

plt.figure(figsize=(8,4))
plt.plot(train['time'][window:],Yu,'r-',label='LSTM')
plt.plot(train['time'][window:],Ym,'k--',label='Measured')

plt.xlabel('Time (sec)'); plt.ylabel('pump vibration')

plt.legend()
plt.show()
plt.savefig('model_fit.png')
