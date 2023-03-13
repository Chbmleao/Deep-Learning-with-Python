import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np

# database preprocessing
(xTraining, yTraining), (xTest, yTest) = mnist.load_data()
plt.imshow(xTraining[3], cmap = 'gray')
plt.title('Class ' + str(yTraining[3]))

trainingPredictors = xTraining.reshape(xTraining.shape[0],
                                       28, 28, 1)
testPredictors = xTest.reshape(xTest.shape[0],
                               28, 28, 1)
trainingPredictors = trainingPredictors.astype('float32')
testPredictors = testPredictors.astype('float32')

trainingPredictors /= 255
testPredictors /= 255

trainingRank = np_utils.to_categorical(yTraining, 10)
testRank = np_utils.to_categorical(yTest, 10)

# neural network structure
classifier = Sequential()
# first step
classifier.add(Conv2D(32, (3,3), input_shape=(28, 28, 1),
                      activation = 'relu'))
classifier.add(BatchNormalization())
# second step - pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))
# third step - flattening
#classifier.add(Flatten())

classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

# fourth step - dense neural network
classifier.add(Dense(units = 128,
                     activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 10,
                     activation = 'softmax'))
classifier.compile(loss = 'categorical_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy'])
classifier.fit(trainingPredictors, trainingRank,
               batch_size = 128, epochs = 5,
               validation_data = (testPredictors, testRank))

results = classifier.evaluate(testPredictors, testRank)


# predict just one image
test_image = xTest[2]
plt.imshow(test_image)
test_image= test_image.reshape(1, 28, 28, 1)
test_image = test_image.astype('float32')
test_image /= 255
prevision = classifier.predict(test_image)

for i in range(10):
    if prevision[0][i] > 0.5:
        print(i)
