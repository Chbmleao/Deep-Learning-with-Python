import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

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
naive_rbm = GaussianNB()
classifier_rbm = Pipeline(steps = [('rbm', rbm), ('naive', naive_rbm)])
classifier_rbm.fit(train_predictors, train_class)

plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i+1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()

predictions_rbm = classifier_rbm.predict(test_predictors)
precision_rbm = metrics.accuracy_score(predictions_rbm, test_class)

naive_simple = GaussianNB()
naive_simple.fit(train_predictors, train_class)
predictions_naive = naive_simple.predict(test_predictors)
precision_naive = metrics.accuracy_score(predictions_naive, test_class)