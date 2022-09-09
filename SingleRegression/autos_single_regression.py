import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

database = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')
database = database.drop('dateCrawled', axis = 1)
database = database.drop('dateCreated', axis = 1)
database = database.drop('nrOfPictures', axis = 1)
database = database.drop('postalCode', axis = 1)
database = database.drop('lastSeen', axis = 1)

# initial database analisys
database['name'].value_counts()
database = database.drop('name', axis = 1)
database['seller'].value_counts()
database = database.drop('seller', axis = 1)
database['offerType'].value_counts()
database = database.drop('offerType', axis = 1)

# treatment of inconsistents
inconsistents1 = database.loc[database.price <= 10]
database = database[database.price > 10]
inconsistents2 = database.loc[database.price > 350000]
database = database[database.price < 350000]

# checking null attributes
database.loc[pd.isnull(database['vehicleType'])]
database['vehicleType'].value_counts() # limousine
database.loc[pd.isnull(database['gearbox'])]
database['gearbox'].value_counts() # manuell
database.loc[pd.isnull(database['model'])]
database['model'].value_counts() # golf
database.loc[pd.isnull(database['fuelType'])]
database['fuelType'].value_counts() # benzin
database.loc[pd.isnull(database['notRepairedDamage'])]
database['notRepairedDamage'].value_counts() # nein

# values to replace in null attributes
values = {'vehicleType': 'limousine',
          'gearbox': 'manuell',
          'model': 'golf',
          'fuelType': 'benzin',
          'notRepairedDamage': 'nein'}
database = database.fillna(value = values)


predictors = database.iloc[:, 1:13].values
realPrices = database.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderPredictors = LabelEncoder()
predictors[:, 0] = labelEncoderPredictors.fit_transform(predictors[:, 0])
predictors[:, 1] = labelEncoderPredictors.fit_transform(predictors[:, 1])
predictors[:, 3] = labelEncoderPredictors.fit_transform(predictors[:, 3])
predictors[:, 5] = labelEncoderPredictors.fit_transform(predictors[:, 5])
predictors[:, 8] = labelEncoderPredictors.fit_transform(predictors[:, 8])
predictors[:, 9] = labelEncoderPredictors.fit_transform(predictors[:, 9])
predictors[:, 10] = labelEncoderPredictors.fit_transform(predictors[:, 10])


from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(),[0, 1, 3, 5, 8, 9, 10])], 
                       remainder="passthrough")
predictors = ct.fit_transform(predictors).toarray()    


# neural network structure
regressor = Sequential()
# first layer
regressor.add(Dense(units = 158, 
                    activation = 'relu',
                    input_dim = 316))
#second layer
regressor.add(Dense(units = 158, 
                    activation = 'relu'))
# output layer
regressor.add(Dense(units = 1,
                    activation = 'linear'))

regressor.compile(loss = 'mean_absolute_error',
                  optimizer = 'adam',
                  metrics = ['mean_absolute_error'])
regressor.fit(predictors, realPrices, batch_size=300, epochs = 100)

previsions = regressor.predict(predictors)
realPrices.mean()
previsions.mean()