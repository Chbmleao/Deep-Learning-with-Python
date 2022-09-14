import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

database = pd.read_csv('games.csv')
database = database.drop('NA_Sales', axis = 1)
database = database.drop('EU_Sales', axis = 1)
database = database.drop('JP_Sales', axis = 1)
database = database.drop('Other_Sales', axis = 1)
database = database.drop('Developer', axis = 1)

database = database.dropna(axis = 0)
database = database.loc[database['Global_Sales'] > 1]

database['Name'].value_counts()
gamesName = database.Name
database = database.drop('Name', axis = 1)

predictors = database.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].values
globalSales = database.iloc[:, 4].values

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

inputLayer = Input(shape = (99,))
myActivation = Activation(activation = 'sigmoid')
ocultLayer1 = Dense(units = 50, activation = myActivation)(inputLayer)
ocultLayer2 = Dense(units = 50, activation = myActivation)(ocultLayer1)
outputLayer = Dense(units = 1, activation = 'linear')(ocultLayer2)

regressor = Model(inputs = inputLayer,
                  outputs = [outputLayer])

regressor.compile(optimizer = 'adam',
                  loss = 'mse')
regressor.fit(predictors, globalSales,
              epochs = 5000, batch_size = 100)

globalPrevision = regressor.predict(predictors)