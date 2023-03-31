from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

x = pd.read_csv('input_breast.csv').values
y = pd.read_csv('output_breast.csv').iloc[:, 0].values

normalizer = MinMaxScaler(feature_range = (0, 1))
x = normalizer.fit_transform(x)

som = MiniSom(x = 11, y = 11, input_len = 30, 
              sigma = 2.0, learning_rate = 0.8,
              random_seed = 2)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 1000)

som._weights
som._activation_map
q = som.activation_response(x)

pcolor(som.distance_map().T)
# MID - mean inter neuron distance
colorbar()

w = som.winner(x[1])
markers = ['o', 's']
color = ['g', 'r']

for i, n in enumerate(x):
    # print(i)
    # print(n)
    w = som.winner(n)
    # print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgecolor = color[y[i]],
         markeredgewidth = 2)