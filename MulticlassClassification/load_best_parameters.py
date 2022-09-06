import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.utils import np_utils

file = open('iris_classifier.json', 'r')
networkStructure = file.read()
file.close()

classifier = model_from_json(networkStructure)
classifier.load_weights('iris_classifier.h5')

# test
newRegister = np.array([[6.1, 4.5, 2.4, 0.5]])

prevision = classifier.predict(newRegister)
prevision = (prevision > 0.5)

dataBase = pd.read_csv('iris.csv')
predictors = dataBase.iloc[:, 0:4].values
rank = dataBase.iloc[:, 4].values

# convert output
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
rank = labelEncoder.fit_transform(rank)
dummyRank = np_utils.to_categorical(rank)

classifier.compile(optimizer = 'adam',
                   loss = 'categorical_crossentropy',
                   metrics = ['categorical_accuracy'])

result = classifier.evaluate(predictors, dummyRank)