from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization

classifier = Sequential()

classifier.add(Conv2D(64, (3,3),
                      input_shape = (64, 64, 3),
                      activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(64, (3,3),
                      input_shape = (64, 64, 3),
                      activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 64,
                     activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 64,
                     activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1,
                     activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

train_generator = ImageDataGenerator(rescale = 1./255,
                                     rotation_range = 7,
                                     horizontal_flip = True,
                                     shear_range = 0.2,
                                     height_shift_range = 0.07,
                                     zoom_range = 0.2)
test_generator = ImageDataGenerator(rescale = 1./255)

train_database = train_generator.flow_from_directory('dataset_personagens/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 10,
                                                     class_mode = 'binary')
test_database = test_generator.flow_from_directory('dataset_personagens/test_set',
                                                   target_size = (64, 64),
                                                   batch_size = 10,
                                                   class_mode = 'binary')

classifier.fit_generator(train_database,
                         steps_per_epoch = 196 / 10,
                         epochs = 100,
                         validation_data = test_database,
                         validation_steps = 73 / 10)