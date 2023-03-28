from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

database = pd.read_csv('petr4_treinamento_ex.csv')
database = database.dropna()
train_database = database.iloc[:, 1:2].values

normalizer = MinMaxScaler(feature_range=(0,1))
train_normalized_database = normalizer.fit_transform(train_database)

predictors = []
real_price = []
for i in range(90, 1342):
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

# stock prices forecast
test_database = pd.read_csv('petr4_teste_ex.csv')
test_real_price = test_database.iloc[:, 1:2].values
complete_database = pd.concat((database['Open'], test_database['Open']), axis = 0)
inputs = complete_database[len(complete_database) - len(test_database) - 90:].values
inputs = inputs.reshape(-1, 1)
inputs = normalizer.transform(inputs)

x_test = []
for i in range(90, 109):
    x_test.append(inputs[i-90:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))    
predictions = regressor.predict(x_test)
predictions = normalizer.inverse_transform(predictions)   

predictions.mean()
test_real_price.mean()

# stock price chart
plt.plot(test_real_price, color = 'red', label = 'Real price')
plt.plot(predictions, color = 'blue', label = 'Predictions')
plt.title('Stock prices forecast')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()
