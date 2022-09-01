import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

predictors = pd.read_csv('input_breast.csv')
rank = pd.read_csv('output_breast.csv')

def createNetwork(optimizer, loss, kernel_initializer, activation, neurons):
    classifier = Sequential()
    
    # first layer
    classifier.add(Dense(units = neurons, activation = activation, 
                         kernel_initializer = kernel_initializer, 
                         input_dim = 30))
    classifier.add(Dropout(0.2))
    
    # second layer
    classifier.add(Dense(units = neurons, activation = activation, 
                         kernel_initializer = kernel_initializer))
    classifier.add(Dropout(0.2))
    
    # output layer
    classifier.add(Dense(units = 1, 
                         activation = 'sigmoid'))
    
    classifier.compile(optimizer = optimizer, 
                       loss = loss,
                       metrics = ['binary_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = createNetwork)
parameters = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}

gridSearch = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 5)

gridSearch = gridSearch.fit(predictors, rank)
bestParameters = gridSearch.best_params_
bestPrecision = gridSearch.best_score_




