# NOME: Leonardo Luis de Vargas
# MATRICULA: 141110047
import numpy as np
from data_read import features, targets, features_test, targets_test

def sigmoid(x): # Calculo do sigmóide
    return 1/(1+np.exp(-x))

def sigmoid_prime(x): # Derivada da função sigmóide
    return sigmoid(x) * (1 - sigmoid(x))

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

weights = np.random.normal(scale=1 / n_features**.5, size=n_features) # Pesos iniciais

epochs = 1000 # Nº de rodadas
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        output = sigmoid(np.dot(x,y))
        error = y - output
        error_term = error * sigmoid_prime(output)
        del_w += learnrate * error_term * x # Atualização dos pesos

    weights += (learnrate*del_w)/n_features
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("perda: ", loss, " [Aumento de perda]")
        else:
            print("perda: ", loss)
        last_loss = loss

# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Precisão: {:.3f}".format(accuracy))
