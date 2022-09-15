import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

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

