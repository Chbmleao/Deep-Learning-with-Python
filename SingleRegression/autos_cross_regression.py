import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

database = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

database = database.drop('dateCrawled', axis = 1)
database = database.drop('dateCreated', axis = 1)
database = database.drop('nrOfPictures', axis = 1)
database = database.drop('postalCode', axis = 1)
database = database.drop('lastSeen', axis = 1)
database = database.drop('name', axis = 1)
database = database.drop('seller', axis = 1)
database = database.drop('offerType', axis = 1)

database = database[database.price > 10]
database = database.loc[database.price < 350000]

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
ct = ColumnTransformer([("Column_Transformer", OneHotEncoder(),[0, 1, 3, 5, 8, 9, 10])], 
                       remainder="passthrough")
predictors = ct.fit_transform(predictors).toarray()    


def createNetwork():
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
    return regressor

regressor = KerasRegressor(build_fn = createNetwork,
                           epochs = 100,
                           batch_size = 300)
results = cross_val_score(estimator = regressor,
                          X = predictors, y = realPrices,
                          cv = 10, scoring = 'neg_median_absolute_error')

mean = results.mean()
resultsStandardDeviation = results.std()



















