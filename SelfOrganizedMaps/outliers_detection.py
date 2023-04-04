from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pylab import pcolor, colorbar, plot

database = pd.read_csv("personagens.csv")
database = database.dropna()

X = database.iloc[:, 0:6].values
y = database.iloc[:, 6].values

labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)

normalizer = MinMaxScaler(feature_range = (0,1))
X = normalizer.fit_transform(X)

som = MiniSom(x = 9, y = 9, input_len = 6, random_seed = 0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 500)

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
susppects = mapping[(7,4)]
susppects = normalizer.inverse_transform(susppects)

classe = []
for i in range(len(database)):
    for j in range(len(susppects)):
        if ((database.iloc[i, 0] == susppects[j,0]) and
           (database.iloc[i, 1] == susppects[j,1]) and
           (database.iloc[i, 2] == susppects[j,2]) and
           (database.iloc[i, 3] == susppects[j,3]) and
           (database.iloc[i, 4] == susppects[j,4]) and
           (database.iloc[i, 5] == susppects[j,5])):
            classe.append(database.iloc[i, 6])

classe = np.asarray(classe)

final_susppects = np.column_stack((susppects, classe))
final_susppects = final_susppects[final_susppects[:, 4].argsort()]

