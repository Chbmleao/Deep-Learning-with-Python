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
autoencoder.add(Conv2D(filters=16,
                       kernel_size=(3,3),
                       activation="relu",
                       input_shape=(28, 28, 1)))
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

autoencoder.add(Conv2D(filters=8,
                       kernel_size=(3,3),
                       activation="relu", 
                       padding="same"))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

# 4, 4, 8
autoencoder.add(Conv2D(filters=8,
                       kernel_size=(3,3),
                       activation="relu",
                       padding="same",
                       strides=(2, 2)))

autoencoder.add(Flatten())

autoencoder.add(Reshape((4, 4, 8)))

# Decoder
autoencoder.add(Conv2D(filters=8,
                       kernel_size=(3,3),
                       activation="relu", 
                       padding="same"))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=8,
                       kernel_size=(3,3),
                       activation="relu", 
                       padding="same"))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=16,
                       kernel_size=(3,3),
                       activation="relu"))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=1,
                       kernel_size=(3,3),
                       activation="sigmoid", 
                       padding="same"))

autoencoder.summary()


autoencoder.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])
autoencoder.fit(trainPredictors,
                trainPredictors,
                epochs=50,
                batch_size=256,
                validation_data=(testPredictors, testPredictors))

encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.get_layer("flatten_2").output)
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
    plt.imshow(encodedImages[imageId].reshape(16, 8))
    plt.xticks(())
    plt.yticks(())
    
    # decoded images
    axis = plt.subplot(10, 10, i+1+numImages*2)
    plt.imshow(decodedImages[imageId].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())




