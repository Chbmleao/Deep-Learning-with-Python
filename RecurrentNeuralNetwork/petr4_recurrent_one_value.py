from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

database = pd.read_csv('petr4_treinamento.csv')
database = database.dropna()
train_database = database.iloc[:, 1:2].values

normalizer = MinMaxScaler(feature_range=(0,1))
train_normalized_database = normalizer.fit_transform(train_database)

predictors = []
real_price = []
for i in range(90, 1242):
    predictors.append(train_normalized_database[i-90:i, 0])
    real_price.append(train_normalized_database[i, 0])
    
predictors, real_price = np.array(predictors), np.array(real_price)
predictors = np.reshape(predictors, 
                        (predictors.shape[0], predictors.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units = 100,
                   return_sequences = True,
                   input_shape = (predictors.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50,
                   return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50,
                   return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1,
                    activation = 'linear'))

regressor.compile(optimizer = 'rmsprop',
                  loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

regressor.fit(predictors, real_price,
              epochs = 100, batch_size = 32)
