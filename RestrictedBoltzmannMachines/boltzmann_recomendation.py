from rbm import RBM
import numpy as np

rbm = RBM(num_visible = 6, num_hidden = 2)

database = np.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,1],
                     [0,0,1,1,0,1],
                     [0,0,1,1,0,1]])

movies = ["A bruxa", "Invocação do mal", "O chamado",
          "Se beber não case", "Gente grande", "American pie"]

rbm.train(database, max_epochs = 5000)
rbm.weights

user1 = np.array([[1,1,0,1,0,0]])
user2 = np.array([[0,0,0,1,1,0]])

rbm.run_visible(user1)
rbm.run_visible(user2)

hidden_layer = np.array([[0,1]])
recommendation1 = rbm.run_hidden(hidden_layer)

hidden_layer = np.array([[1,0]])
recommendation2 = rbm.run_hidden(hidden_layer)

print("User 1: ")
for i in range(len(user1[0])):
    # print(user1[0,i])
    if user1[0,i] == 0 and recomendation1[0,i] == 1:
        print(movies[i])
        
print("\nUser 2: ")
for i in range(len(user2[0])):
    # print(user1[0,i])
    if user2[0,i] == 0 and recomendation2[0,i] == 1:
        print(movies[i])