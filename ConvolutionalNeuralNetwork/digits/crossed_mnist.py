from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(X, y), (X_test, y_test) = mnist.load_data()
predictors = X.reshape(X.shape[0], 28, 28, 1)
predictors = predictors.astype('float32')
predictors /= 255
classe = np_utils.to_categorical(y, 10)

kfold = StratifiedKFold(n_splits = 5, 
                        shuffle = True, 
                        random_state = seed)
results = []

for train_index, test_index in kfold.split(predictors, np.zeros(shape=(classe.shape[0], 1))):
    # print('Train indexes: ', train_index, ' Test indexes: ', test_index)
    classifier = Sequential()
    classifier.add(Conv2D(32, 
                          (3,3), 
                          input_shape=(28,28,1), 
                          activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128,
                        activation='relu'))
    classifier.add(Dense(units=10,
                         activation='softmax'))
    classifier.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    classifier.fit(predictors[train_index],
                   classe[train_index],
                   batch_size=128,
                   epochs=5)
    precision = classifier.evaluate(predictors[test_index], classe[test_index])
    results.append(precision[1])
    
# average = results.mean()
average = sum(results) / len(results)