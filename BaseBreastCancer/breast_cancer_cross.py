import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

predictors = pd.read_csv('input_breast.csv')
rank = pd.read_csv('output_breast.csv')

def createNetwork():
    classifier = Sequential()
    # first layer
    classifier.add(Dense(units = 32, activation = 'relu', 
                         kernel_initializer = 'normal', 
                         input_dim = 30))
    classifier.add(Dropout(0.2))
    
    # second layer
    classifier.add(Dense(units = 32, activation = 'relu', 
                         kernel_initializer = 'normal'))
    classifier.add(Dropout(0.2))
    
    # third layer
    classifier.add(Dense(units = 16, activation = 'relu', 
                         kernel_initializer = 'normal'))
    classifier.add(Dropout(0.2))
    
    # fourth layer
    classifier.add(Dense(units = 16, activation = 'relu', 
                         kernel_initializer = 'normal'))
    classifier.add(Dropout(0.2))
    
    # output layer
    classifier.add(Dense(units = 1, 
                         activation = 'sigmoid'))
    
    # custom optimizer
    myOptimizer = keras.optimizers.Adam(learning_rate = 0.001, 
                                        decay = 0.0001, 
                                        clipvalue = 0.5)
    classifier.compile(optimizer = myOptimizer, 
                       loss = 'binary_crossentropy',
                       metrics = ['binary_accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = createNetwork, 
                             epochs = 500, 
                             batch_size = 10)

results = cross_val_score(estimator = classifier,
                          X = predictors, y = rank,
                          cv = 10, scoring = 'accuracy')

resultsMean = results.mean()
resultsStandardDeviation = results.std()
