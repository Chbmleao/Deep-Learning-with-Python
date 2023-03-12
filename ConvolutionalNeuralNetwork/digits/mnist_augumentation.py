from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

(trainX, trainY), (testX, testY) = mnist.load_data()
trainPredictors = trainX.reshape(trainX.shape[0],
                                 28, 28, 1)
testPredictors = testX.reshape(testX.shape[0],
                               28, 28, 1)
trainPredictors = trainPredictors.astype('float32')
testPredictors = testPredictors.astype('float32')
trainPredictors /= 255
testPredictors /= 255
trainClass = np_utils.to_categorical(trainY, 10)
testClass = np_utils.to_categorical(testY, 10)

classifier = Sequential()
classifier.add(Conv2D(32, 
                      (3,3),
                      input_shape=(28, 28, 1),
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))
classifier.compile(loss = 'categorical_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy'])

trainGenerator = ImageDataGenerator(rotation_range = 7,
                                    horizontal_flip = True,
                                    shear_range = 0.2,
                                    height_shift_range = 0.07,
                                    zoom_range = 0.2)
testGenerator = ImageDataGenerator()

trainDatabase = trainGenerator.flow(trainPredictors, 
                                    trainClass,
                                    batch_size = 128)
testDatabase = testGenerator.flow(testPredictors,
                                  testClass,
                                  batch_size = 128)

classifier.fit_generator(trainDatabase, 
                         steps_per_epoch = 60000 / 128,
                         epochs = 5,
                         validation_data = testDatabase,
                         validation_steps = 10000 / 128)