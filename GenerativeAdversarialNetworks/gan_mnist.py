import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten, Reshape
from keras.regularizers import L1L2
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

(trainPredictors, _), (_, _) = mnist.load_data()
trainPredictors = trainPredictors.astype("float32") / 255

# Generator
generator = Sequential()
generator.add(Dense(units=500,
                    input_dim=100,
                    activation="relu",
                    kernel_regularizer=L1L2(1e-5, 1e-5)))
generator.add(Dense(units=500,
                    input_dim=100,
                    activation="relu",
                    kernel_regularizer=L1L2(1e-5, 1e-5)))
generator.add(Dense(units=784,
                    activation="sigmoid",
                    kernel_regularizer=L1L2(1e-5, 1e-5)))
generator.add(Reshape((28, 28)))

# Discriminator
discriminator = Sequential()
discriminator.add(InputLayer(input_shape=(28, 28)))
discriminator.add(Flatten())
discriminator.add(Dense(units=500,
                        activation="relu",
                        kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminator.add(Dense(units=500,
                        activation="relu",
                        kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminator.add(Dense(units=1,
                        activation="sigmoid",
                        kernel_regularizer=L1L2(1e-5, 1e-5)))


gan = simple_gan(generator, discriminator, normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan,
                         player_params=[generator.trainable_weights,
                                        discriminator.trainable_weights])

model.adversarial_compile(adversarial_optimizer = AdversarialOptimizerSimultaneous,
                          player_optimizers = ["adam", "adam"],
                          loss = "binary_crossentropy")

model.fit(x = trainPredictors, 
          y = gan_targets(60000), 
          epochs = 100, 
          batch_size = 256)

samples = np.random_normal(size = (10, 100))
prediction = generator.predict(samples)

for i in range(prediction.shape[0]):
    plt.imshow(prediction[i, :], cmap="gray")
    plt.show()
