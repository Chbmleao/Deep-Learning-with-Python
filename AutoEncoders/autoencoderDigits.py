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





