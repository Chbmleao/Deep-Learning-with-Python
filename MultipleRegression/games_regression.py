import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

database = pd.read_csv('games.csv')
database = database.drop('Other_Sales', axis = 1)
database = database.drop('Global_Sales', axis = 1)
database = database.drop('Developer', axis = 1)

database = database.dropna(axis = 0)
database = database.loc[database['NA_Sales'] > 1]
database = database.loc[database['EU_Sales'] > 1]

database['Name'].value_counts()
gamesName = database.Name
database = database.drop('Name', axis = 1)

predictors = database.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
naSales = database.iloc[:, 4].values
euSales = database.iloc[:, 5].values
jpSales = database.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
predictors[:, 0] = labelEncoder.fit_transform(predictors[:, 0])
predictors[:, 2] = labelEncoder.fit_transform(predictors[:, 2])
predictors[:, 3] = labelEncoder.fit_transform(predictors[:, 3])
predictors[:, 8] = labelEncoder.fit_transform(predictors[:, 8])

from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(),[0, 2, 3, 8])], remainder="passthrough")
predictors = ct.fit_transform(predictors).toarray()    


# neural network

inputLayer = Input(shape = (61,))
ocultLayer1 = Dense(units = 32, activation = 'sigmoid')(inputLayer)
ocultLayer2 = Dense(units = 32, activation = 'sigmoid')(ocultLayer1)
outputLayer1 = Dense(units = 1, activation = 'linear')(ocultLayer2)
outputLayer2 = Dense(units = 1, activation = 'linear')(ocultLayer2)
outputLayer3 = Dense(units = 1, activation = 'linear')(ocultLayer2)

regressor = Model(inputs = inputLayer,
                  outputs = [outputLayer1, outputLayer2, outputLayer3])

regressor.compile(optimizer = 'adam',
                  loss = 'mse')
regressor.fit(predictors, [naSales, euSales, jpSales],
              epochs = 5000, batch_size = 100)

naPrevision, euPrevision, jpPrevision = regressor.predict(predictors)












