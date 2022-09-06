import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

dataBase = pd.read_csv('iris.csv')
predictors = dataBase.iloc[:, 0:4].values
rank = dataBase.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
rank = labelEncoder.fit_transform(rank)
dummyClass = np_utils.to_categorical(rank)

def createNetwork(optimizer, kernel_initializer, activation, neurons):
    # classifier structure
    classifier = Sequential()
    
    # first layer
    classifier.add(Dense(units = neurons, 
                         activation = activation,
                         kernel_initializer = kernel_initializer, 
                         input_dim = 4))
    # second layer
    classifier.add(Dense(units = neurons,
                         activation = activation,
                         kernel_initializer = kernel_initializer, ))
    # output layer
    classifier.add(Dense(units = 3,
                         activation = 'softmax'))
    
    classifier.compile(optimizer = optimizer,
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = createNetwork)
parameters = {'batch_size': [10, 30],
              'epochs': [500, 1000],
              'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [3, 4, 5]
              }

gridSearch = GridSearchCV(estimator = classifier,
                        param_grid = parameters,
                        cv = 5)

gridSearch = gridSearch.fit(predictors, rank)
bestParameters = gridSearch.best_params_