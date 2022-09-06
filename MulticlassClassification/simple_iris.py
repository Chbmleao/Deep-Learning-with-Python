import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

dataBase = pd.read_csv('iris.csv')

predictors = dataBase.iloc[:, 0:4].values
rank = dataBase.iloc[:, 4].values

# convert output
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
rank = labelEncoder.fit_transform(rank)
dummyRank = np_utils.to_categorical(rank)
# iris setosa       1 0 0
# iris virginica    0 1 0
# iris versicolor   0 0 1

from sklearn.model_selection import train_test_split
trainingPredictors, testPredictors, trainingRank, testRank = train_test_split(predictors, dummyRank, test_size = 0.25)

# classifier structure
classifier = Sequential()
# first layer
classifier.add(Dense(units = 4, 
                     activation = 'relu', 
                     input_dim = 4))
# second layer
classifier.add(Dense(units = 4,
                     activation = 'relu'))
# output layer
classifier.add(Dense(units = 3,
                     activation = 'softmax'))

classifier.compile(optimizer = 'adam',
                   loss = 'categorical_crossentropy',
                   metrics = ['categorical_accuracy'])

classifier.fit(trainingPredictors, trainingRank,
               batch_size = 10,
               epochs = 1000)

results = classifier.evaluate(testPredictors, testRank)
previsions = classifier.predict(testPredictors)
previsions = (previsions > 0.5)
import numpy as np
testRank2 = [np.argmax(t) for t in testRank]
previsions2 = [np.argmax(t) for t in previsions]


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(previsions2, testRank2)
