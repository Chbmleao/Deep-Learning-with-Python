from rbm import RBM
import numpy as np

rbm = RBM(num_visible = 6, num_hidden = 3)

database = np.array([[0,1,1,1,0,1],
                     [1,1,0,1,1,1],
                     [0,1,0,1,0,1],
                     [0,1,1,1,0,1],
                     [1,1,0,1,1,1]])

movies = ["Freddy x Jason", "O Ultimato Bourne", "Star Trek",
          "Exterminador do Futuro", "Norbit", "Star Wars"]

rbm.train(database, max_epochs = 5000)
rbm.weights

leonardo = np.array([[0,1,0,1,0,0]])

hidden_layer = np.array(rbm.run_visible(leonardo))
recommendation = rbm.run_hidden(hidden_layer)

print("Movies for Leonardo: ")
for i in range(len(leonardo[0])):
    # print(user1[0,i])
    if leonardo[0,i] == 0 and recommendation[0,i] == 1:
        print(movies[i])
    