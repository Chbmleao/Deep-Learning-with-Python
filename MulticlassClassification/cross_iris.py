import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

dataBase = pd.read_csv('iris.csv')
predictors = dataBase.iloc[:, 0:4].values
rank = dataBase.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
rank = labelEncoder.fit_transform(rank)
dummyClass = np_utils.to_categorical(rank)

def createNetwork():
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

    return classifier

classifier = KerasClassifier(build_fn = createNetwork,
                             epochs = 1000,
                             batch_size = 10)

results = cross_val_score(estimator = classifier,
                          X = predictors, y = rank,
                          cv = 10, scoring = 'accuracy')

resultsMean = results.mean()
resultsStandardDeviation = results.std()