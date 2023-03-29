from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

database = pd.read_csv('petr4_treinamento.csv')
database = database.dropna()
train_database = database.iloc[:, 1:7].values

normalizer = MinMaxScaler(feature_range=(0,1))
train_normalized_database = normalizer.fit_transform(train_database)

prediction_normalizer = MinMaxScaler(feature_range=(0,1))
prediction_normalizer.fit_transform(train_database[:, 0:1])

predictors = []
real_price = []
for i in range(90, 1242):
    predictors.append(train_normalized_database[i-90:i, 0:6])
    real_price.append(train_normalized_database[i, 0])
    
predictors, real_price = np.array(predictors), np.array(real_price)


regressor = Sequential()
regressor.add(LSTM(units = 100,
                   return_sequences = True,
                   input_shape = (predictors.shape[1], 6)))
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
                    activation = 'sigmoid'))

regressor.compile(optimizer = 'adam',
                  loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss',
                   min_delta = 1e-10,
                   patience = 10,
                   verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss',
                        factor = 0.2,
                        patience = 5,
                        verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.h5',
                      monitor = 'loss',
                      save_best_only = True,
                      verbose = 1)

regressor.fit(predictors, real_price,
              epochs = 100, batch_size = 32,
              callbacks = [es, rlr, mcp])


test_database = pd.read_csv('petr4_teste.csv')
test_real_price = test_database.iloc[:, 1:2].values
frames = [database, test_database]
complete_database = pd.concat(frames)
complete_database = complete_database.drop('Date', axis = 1)

inputs = complete_database[len(complete_database) - len(test_database) - 90:].values
inputs = normalizer.transform(inputs)

x_test = []
for i in range(90, 112):
    x_test.append(inputs[i-90:i, 0:6])
x_test = np.array(x_test) 

predictions = regressor.predict(x_test)
predictions = prediction_normalizer.inverse_transform(predictions)   


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






