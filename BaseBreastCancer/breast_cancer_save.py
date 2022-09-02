import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

predictors = pd.read_csv('input_breast.csv')
rank = pd.read_csv('output_breast.csv')

classifier = Sequential()
    
# first layer
classifier.add(Dense(units = 8, activation = 'relu', 
                     kernel_initializer = 'normal', 
                     input_dim = 30))
classifier.add(Dropout(0.2))

# second layer
classifier.add(Dense(units = 8, activation = 'relu', 
                     kernel_initializer = 'normal'))
classifier.add(Dropout(0.2))

# output layer
classifier.add(Dense(units = 1, 
                     activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['binary_accuracy'])

classifier.fit(predictors, rank, batch_size = 10, epochs = 100)

classifierJson = classifier.to_json()
with open('classifier_breast_cancer.json', 'w') as json_file:
    json_file.write(classifierJson)
classifier.save_weights('classifier_breast_cancer.h5')
