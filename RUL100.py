import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import time
from PIL import Image
import cv2
#from keras import LSTM
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv(r'C:\all\data\NotchData\Drexel_fatigue.csv')

df

df["RUL_norm"][:1187].plot(figsize=(16,4))
df["RUL_norm"][1187:].plot(figsize=(16,4))
plt.legend(['Training','Test set '])
plt.title('RUL')
plt.ylabel('Normalized RUL')
plt.xlabel('Cycles')
plt.show()

df=df.set_index(df.index)
df1=df.drop(columns=['cycles','RUL'])
print("data shape is ", df1.shape)
#print("training data size: ", df.shape[0]*0.94)
#train_value = int(df.shape[0]*0.94)
#print(train_value)
df1.head()

# Open price visualiztion
hits = df1.hits.values
hits = hits.reshape(-1, 1)
RUL_norm = df1.RUL_norm.values
RUL_norm = RUL_norm.reshape(-1, 1)
print("hits: ",hits.shape)
print("RUL_norm: ",RUL_norm)
plt.plot(RUL_norm)
plt.show()

train_value = 1187

# Today's forcast value depends on last 20-day values
window_size=100
# Only using "Open" feature for training
train_data=df1.iloc[:(train_value+window_size)]
train_data.shape

train_data

#train_data= np.array(train_data)

# Append window-sized data to training dataset
x_train, y_train = [], []
train_len = len(train_data)
for i in range(train_len-window_size):
    x_train.append(np.array(train_data['hits'][i:i+window_size]))
    y_train.append(np.array(train_data['RUL_norm'][i+window_size]))

x_train=np.expand_dims(np.array(x_train),-1)
y_train=np.array(y_train)
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout


model=Sequential(
    [LSTM(128,return_sequences=True,input_shape=( x_train.shape[1],x_train.shape[2])),
     Dropout(0.2),
     LSTM(50,return_sequences=True),
     Dropout(0.2),
     LSTM(50),
     Dropout(0.2),
     Dense(1),
     Activation('linear')]
    )


model.summary()
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

start = time.time()

print ('compilation time : ', time.time() - start)
from keras.models import load_model

# Load the model from the .keras directory
model = load_model(r'C:\all\data\NotchData\AE_RUL.keras')


train_df = df1.iloc[:train_value]
test_df = df1.iloc[train_value:]


data = pd.concat((train_df['hits'], test_df['hits']), axis=0)

test_input = data[len(train_df) - window_size:].values
test_input = test_input.reshape(-1,1)

x_test = []

for i in range(window_size, len(test_df)+window_size):
    x_test.append(test_input[i-window_size:i, 0])

x_test = np.array(x_test)

#Transform x_test values compatible with LSTM
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_test.shape)
y_test = df1['RUL_norm'].iloc[train_value:].values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#              Predicting the future
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tr=[]
pr=[]
for i in range (1, len(x_test)-1):
    from numpy import newaxis
    # We sorted descending
    curr_frame = x_test[i]
    future = []

    # Quick plot of the frame we're predicting from
    #plt.plot(curr_frame, label='History')
    
    points_to_predict = 1
    for j in range(points_to_predict):
          # append the prediction to our empty future list
         future.append(model.predict(curr_frame[newaxis,:,:])[0,0])
          # insert our predicted point to our current frame
         curr_frame = np.insert(curr_frame, len(x_test[0]), future[-1], axis=0)
          # push the frame up one to make it progress into the future
         curr_frame = curr_frame[1:]
    fig = plt.figure(figsize=(8,6))
    plt.scatter(i,y_test[i+1], label='True',color='green')      
    plt.scatter(i,future, label='prediction',color='red')
    tr.append(y_test[i+1])
    pr.append(future)
    plt.plot(tr, color='green')
    plt.plot(pr, color='red')
    plt.title(f"Prediction at Step {i}")
    plt.xlabel('Cycles')
    plt.ylabel('RUL')
    plt.ylim(0,1.2)
    plt.xlim(0,i+5)
    plt.legend()
    
    a = plt.savefig(f"RUL100_fix/fig{i}")
        
 #   c = np.array(Image.open(f"fig{i}.png"))

  #  b = cv2.resize(c,(450,350))
   
    
    
    #plt.legend()
  
        # Display the plot and pause briefly to visualize
    plt.pause(0.5)  # Pauses for 0.5 seconds
    

# After the loop, keep the final plot on screen
plt.show()
