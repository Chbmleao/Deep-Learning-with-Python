import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense

(trainPredictors, _), (testPredictors, _) = mnist.load_data()
trainPredictors = trainPredictors.astype("float32") / 255
testPredictors = testPredictors.astype("float32") / 255

trainPredictors = trainPredictors.reshape((len(trainPredictors), np.prod(trainPredictors.shape[1:])))

testPredictors = testPredictors.reshape((len(testPredictors), np.prod(testPredictors.shape[1:])))

# 784 - 32 - 784
compressionFactor = 784 / 32

autoencoder = Sequential()
autoencoder.add(Dense(units=32, 
                      activation="relu", 
                      input_dim=784))
autoencoder.add(Dense(units=784, activation="sigmoid"))
autoencoder.summary()
autoencoder.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])
autoencoder.fit(trainPredictors, 
                trainPredictors,
                epochs=100,
                batch_size=256,
                validation_data=(testPredictors, testPredictors))

