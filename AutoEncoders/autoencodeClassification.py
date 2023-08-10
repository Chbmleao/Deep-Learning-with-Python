import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils

(trainPredictors, trainClass), (testPredictors, testClass) = mnist.load_data()
trainPredictors = trainPredictors.astype("float32") / 255
testPredictors = testPredictors.astype("float32") / 255

trainDummyClass = np_utils.to_categorical(trainClass)
testDummyClass = np_utils.to_categorical(testClass)

trainPredictors = trainPredictors.reshape((len(trainPredictors), np.prod(trainPredictors.shape[1:])))
testPredictors = testPredictors.reshape((len(testPredictors), np.prod(testPredictors.shape[1:])))

# 784 - 32 - 784
autoencoder = Sequential()
autoencoder.add(Dense(units=32, 
                      activation="relu", 
                      input_dim=784))
autoencoder.add(Dense(units=784, activation="sigmoid"))
autoencoder.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])
autoencoder.fit(trainPredictors, 
                trainPredictors,
                epochs=100,
                batch_size=256,
                validation_data=(testPredictors, testPredictors))

originalDimension = Input(shape=(784,))
encoderLayer = autoencoder.layers[0]
encoder = Model(originalDimension, encoderLayer(originalDimension))

trainEncodedPredictors = encoder.predict(trainPredictors)
testEncodedPredictors = encoder.predict(testPredictors)