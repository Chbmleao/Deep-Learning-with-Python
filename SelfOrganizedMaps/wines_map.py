from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

database = pd.read_csv('wines.csv')
x = database.iloc[:, 1:14].values
y = database.iloc[:, 0].values
y = y - 1

normalizer = MinMaxScaler(feature_range = (0, 1))
x = normalizer.fit_transform(x)

som = MiniSom(x = 8, y = 8, input_len = 13, 
              sigma = 1.5, learning_rate = 0.5,
              random_seed = 2)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 500)

som._weights
som._activation_map
q = som.activation_response(x)

pcolor(som.distance_map().T)
# MID - mean inter neuron distance
colorbar()

w = som.winner(x[1])
markers = ['o', 's', 'D']
color = ['r', 'g', 'b']

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