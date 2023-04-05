import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

database = datasets.load_digits()
predictors = np.asarray(database.data, 'float32')
classe = database.target

normalizer = MinMaxScaler(feature_range = (0,1))
predictors = normalizer.fit_transform(predictors)

train_predictors, test_predictors, train_class, test_class = train_test_split(predictors,
                                                                              classe,
                                                                              test_size = 0.2,
                                                                              random_state = 0)

rbm = BernoulliRBM(random_state = 0)
rbm.n_iter = 25
rbm.n_components = 50

mlp_rbm = MLPClassifier(hidden_layer_sizes = (37, 37),
                        activation = 'relu',
                        solver = 'adam',
                        batch_size = 50,
                        max_iter = 1000,
                        verbose = 1)

classifier_rbm = Pipeline(steps = [('rbm', rbm), ('mlp', mlp_rbm)])
classifier_rbm.fit(train_predictors, train_class)

predictions_rbm = classifier_rbm.predict(test_predictors)
precision_rbm = metrics.accuracy_score(predictions_rbm, test_class)

mlp_simple = MLPClassifier(hidden_layer_sizes = (37, 37),
                           activation = 'relu',
                           solver = 'adam',
                           batch_size = 50,
                           max_iter = 1000,
                           verbose = 1)
mlp_simple.fit(train_predictors, train_class)
previsions_mlp = mlp_simple.predict(test_predictors)
precision_mlp = metrics.accuracy_score(previsions_mlp, test_class)