#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install yfinance')
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Flatten,Input
import matplotlib.pyplot as plt
from keras.layers import LSTM,Dense,Dropout,MaxPooling1D,TimeDistributed,Conv1D
from keras.models import load_model
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import yfinance as yf


# In[7]:


# Load the stock price data
stock_data = yf.download('AAPL', start='2010-01-01', end='2020-12-31')


# In[8]:


# Convert the data to a pandas dataframe
df = pd.DataFrame(stock_data)


# In[10]:


df.head(), df.tail()


# In[12]:


# Set the date as the index
df.set_index('Close', inplace=True)


# In[15]:


df['Close_Scaled'] = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))


# In[16]:


print(df.columns)


# In[17]:


# Create a function to create the training and testing datasets
def create_dataset(dataset, time_step):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:i+time_step, 0]
        X.append(a)
        Y.append(dataset[i + time_step - 1, 0])
    return np.array(X), np.array(Y)


# In[18]:


# Create the training and testing datasets
time_step = 60
X_train, y_train = create_dataset(df['Close_Scaled'].values.reshape(-1, 1), time_step)
X_test, y_test = create_dataset(df['Close_Scaled'].values.reshape(-1, 1), time_step)


# In[19]:


# Reshape the data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[20]:


# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))


# In[21]:


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[22]:


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)


# In[23]:


# Make predictions
predictions = model.predict(X_test)


# In[24]:


# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f'MSE: {mse}')


# In[25]:


# Plot the results
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()


# In[26]:


# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()


# In[ ]:




