from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

database = pd.read_csv('petr4_treinamento.csv')
database = database.dropna()
train_database = database.iloc[:, 1:2].values
max_value_database = database.iloc[:, 2:3].values

normalizer = MinMaxScaler(feature_range=(0,1))
train_normalized_database = normalizer.fit_transform(train_database)
max_value_normalized_database = normalizer.fit_transform(max_value_database)

predictors = []
real_price = []
max_price = []
for i in range(90, 1242):
    predictors.append(train_normalized_database[i-90:i, 0])
    real_price.append(train_normalized_database[i, 0])
    max_price.append(max_value_normalized_database[i, 0])
    
predictors = np.array(predictors)
real_price = np.array(real_price)
max_price = np.array(max_price)
predictors = np.reshape(predictors, 
                        (predictors.shape[0], predictors.shape[1], 1))

real_and_max_prices = np.column_stack((real_price, max_price))

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

regressor.add(Dense(units = 2,
                    activation = 'linear'))

regressor.compile(optimizer = 'rmsprop',
                  loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

regressor.fit(predictors, real_and_max_prices,
              epochs = 100, batch_size = 32)

# stock prices forecast
test_database = pd.read_csv('petr4_teste.csv')
test_real_price_open = test_database.iloc[:, 1:2].values
test_real_price_high = test_database.iloc[:, 2:3].values

complete_database = pd.concat((database['Open'], test_database['Open']), axis = 0)
inputs = complete_database[len(complete_database) - len(test_database) - 90:].values
inputs = inputs.reshape(-1, 1)
inputs = normalizer.transform(inputs)

x_test = []
for i in range(90, 112):
    x_test.append(inputs[i-90:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))   
 
predictions = regressor.predict(x_test)
predictions = normalizer.inverse_transform(predictions)  

# stock price chart
plt.plot(test_real_price_open, color = 'red', label = 'Real open price')
plt.plot(test_real_price_high, color = 'black', label = 'Real high price')

plt.plot(predictions[:, 0], color = 'blue', label = 'Open predictions')
plt.plot(predictions[:, 1], color = 'orange', label = 'High predictions')

plt.title('Stock prices forecast')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()





