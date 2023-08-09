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

originalDimension = Input(shape=(784,))
encoderLayer = autoencoder.layers[0]
encoder = Model(originalDimension, encoderLayer(originalDimension))
encoder.summary()

encodedImages = encoder.predict(testPredictors)
decodedImages = autoencoder.predict(testPredictors)

numImages = 10
testImages = np.random.randint(testPredictors.shape[0], size=numImages)
plt.figure(figsize=(18,18))
for i, imageId in enumerate(testImages):
    # print(i)
    # print(imageId)

    # original images
    axis = plt.subplot(10, 10, i+1)
    plt.imshow(testPredictors[imageId].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
    
    # encoded images
    axis = plt.subplot(10, 10, i+1+numImages)
    plt.imshow(encodedImages[imageId].reshape(8, 4))
    plt.xticks(())
    plt.yticks(())
    
    # decoded images
    axis = plt.subplot(10, 10, i+1+numImages*2)
    plt.imshow(decodedImages[imageId].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
