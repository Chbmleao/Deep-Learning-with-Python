import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

# database preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

plt.imshow(x_train[0])
plt.title('Classe ' + str(y_train[0]))

train_predictors = x_train.reshape(x_train.shape[0],
                                   32, 32, 3)  
test_predictors = x_test.reshape(x_test.shape[0],
                                 32, 32, 3)
train_predictors = train_predictors.astype('float32')
test_predictors = test_predictors.astype('float32')

test_predictors /= 255
train_predictors /= 255

train_class = np_utils.to_categorical(y_train, 10)
test_class = np_utils.to_categorical(y_test, 10)

classifier = Sequential()
classifier.add(Conv2D(64, (3,3),
                      input_shape=(32, 32, 3),
                      activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(64, (3, 3), 
                      activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units = 256,
                     activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 256, 
                     activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 10,
                     activation = 'softmax'))
classifier.compile(loss = 'categorical_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy'])
classifier.fit(train_predictors, train_class,
               batch_size = 128, epochs = 5,
               validation_data = (test_predictors, test_class))

result = classifier.evaluate(test_predictors, test_class)