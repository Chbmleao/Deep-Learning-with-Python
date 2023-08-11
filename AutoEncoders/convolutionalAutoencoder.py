import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

(trainPredictors, _), (testPredictors, _) = mnist.load_data()
trainPredictors = trainPredictors.reshape((len(trainPredictors), 28, 28, 1))
testPredictors = testPredictors.reshape((len(testPredictors), 28, 28, 1))

trainPredictors = trainPredictors.astype("float32") / 255
testPredictors = testPredictors.astype("float32") / 255

autoencoder = Sequential()

# Encoder
autoencoder.add(Conv2D(filter=16,
                       kernel_size=(3,3),
                       activation="relu",
                       input_shape=(28, 28, 1)))
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

autoencoder.add(Conv2D(filter=8,
                       kernel_size=(3,3),
                       activation="relu", 
                       padding="same"))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

# 4, 4, 8
autoencoder.add(Conv2D(filter=8,
                       kernel_size=(3,3),
                       activation="relu",
                       padding="same",
                       strides=(2, 2)))

autoencoder.add(Flatten())

autoencoder.add(Reshape((4, 4, 8)))











