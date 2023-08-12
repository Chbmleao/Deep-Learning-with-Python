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
