import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

predictors = pd.read_csv('input_breast.csv')
rank = pd.read_csv('output_breast.csv')

def createNetwork():
    classifier = Sequential()
    # first layer
    classifier.add(Dense(units = 16, activation = 'relu', 
                         kernel_initializer = 'random_uniform', 
                         input_dim = 30))
    # second layer
    classifier.add(Dense(units = 16, activation = 'relu', 
                         kernel_initializer = 'random_uniform'))
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
                             epochs = 100, 
                             batch_size = 10)

results = cross_val_score(estimator = classifier,
                          X = predictors, y = rank,
                          cv = 10, scoring = 'accuracy')

resultsMean = results.mean()
resultsStandardDeviation = results.std()
