import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

database = pd.read_csv('personagens.csv')
predictors = database.drop("classe", axis = 'columns')
classes = database["classe"]
# convert output
labelEncoder = LabelEncoder()
classes = labelEncoder.fit_transform(classes)

trainPredictors, testPredictors, trainClass, testClass = train_test_split(predictors, classes, test_size=0.25)

classifier = Sequential()
# first layer
classifier.add(Dense(units = 64, activation = 'relu', 
                     kernel_initializer = 'random_uniform', input_dim = 6))
# second layer
classifier.add(Dense(units = 64, activation = 'relu', 
                     kernel_initializer = 'random_uniform'))
# output
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# normal adam optimizer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['binary_accuracy'])

# training
classifier.fit(trainPredictors, trainClass, 
               batch_size = 10, epochs = 2000)

# results
results = classifier.evaluate(testPredictors, testClass)