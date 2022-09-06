import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

dataBase = pd.read_csv('iris.csv')
predictors = dataBase.iloc[:, 0:4].values
rank = dataBase.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
rank = labelEncoder.fit_transform(rank)
dummyRank = np_utils.to_categorical(rank)

# classifier structure
classifier = Sequential()
    
# first layer
classifier.add(Dense(units = 3, 
                     activation = 'relu',
                     kernel_initializer = 'random_uniform', 
                     input_dim = 4))
# second layer
classifier.add(Dense(units = 3,
                     activation = 'relu',
                     kernel_initializer = 'random_uniform', ))
# output layer
classifier.add(Dense(units = 3,
                     activation = 'softmax'))

classifier.compile(optimizer = 'adam',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])

classifier.fit(predictors, dummyRank,
               batch_size = 10,
               epochs = 1000)

classifierJson = classifier.to_json()
with open('iris_classifier.json', 'w') as json_file:
    json_file.write(classifierJson)
classifier.save_weights('iris_classifier.h5')
