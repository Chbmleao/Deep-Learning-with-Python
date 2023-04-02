from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

database = pd.read_csv("credit_data.csv")
database = database.dropna()
database.loc[database.age < 0, 'age'] = 40.92

X = database.iloc[:, 0:4].values
y = database.iloc[:, 4].values

normalizer = MinMaxScaler(feature_range = (0,1))
X = normalizer.fit_transform(X)

som = MiniSom(x = 15, y = 15, input_len = 4, random_seed = 0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], 
         markeredgewidth = 2)
    
mapping = som.win_map(X)
susppects = np.concatenate((mapping[(4,5)], mapping[(6,13)]), axis = 0)
susppects = normalizer.inverse_transform(susppects)

classe = []
for i in range(len(database)):
    for j in range(len(susppects)):
        if database.iloc[i, 0] == int(round(susppects[j, 0])):
            classe.append(database.iloc[i, 4])

classe = np.asarray(classe)

final_susppects = np.column_stack((susppects, classe))
final_susppects = final_susppects[final_susppects[:, 4].argsort()]

