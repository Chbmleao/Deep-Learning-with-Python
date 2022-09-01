import pandas as pd

predictors = pd.read_csv('input_breast.csv')
rank = pd.read_csv('output_breast.csv')

from sklearn.model_selection import train_test_split
trainingPredictors, testPredictors, trainingRank, testRank = train_test_split(predictors, rank, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# first layer
classifier.add(Dense(units = 16, activation = 'relu', 
                     kernel_initializer = 'random_uniform', input_dim = 30))
# second layer
classifier.add(Dense(units = 16, activation = 'relu', 
                     kernel_initializer = 'random_uniform'))
# output
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# custom optimizer
myOptimizer = keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0001, clipvalue = 0.5)
classifier.compile(optimizer = myOptimizer, loss = 'binary_crossentropy',
                   metrics = ['binary_accuracy'])
# normal adam optimizer
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
#                    metrics = ['binary_accuracy'])

# training
classifier.fit(trainingPredictors, trainingRank, 
               batch_size = 10, epochs = 100)

# connections weights 
weights0 = classifier.layers[0].get_weights()
print(weights0)
print(len(weights0))
weights1 = classifier.layers[1].get_weights()
weights2 = classifier.layers[2].get_weights()

predictions = classifier.predict(testPredictors)
predictions = (predictions > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

precision = accuracy_score(testRank, predictions)
matrix = confusion_matrix(testRank, predictions)

result = classifier.evaluate(testPredictors, testRank)
