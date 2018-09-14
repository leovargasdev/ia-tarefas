import numpy as np
from read_data import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x): # Calculo do sigmóide
    return 1/(1+np.exp(-x))

def sigmoid_prime(x): # Derivada da função sigmóide
    return x * (1 - x)

n_hidden = 2  # Nº de nós ocultos
rodadas = 900 # Nº de interações
learnrate = 0.005

n_records, n_features = features.shape
ultima_perda = None
# Initialize weights
pesos_entrada_camada_1 = np.random.normal(scale = 1 / n_features ** .5, size = (n_features, n_hidden))
pesos_saida_camada_2 = np.random.normal(scale = 1 / n_features ** .5, size = n_hidden)

for e in range(rodadas):
    del_w_entrada_camada_1 = np.zeros(pesos_entrada_camada_1.shape)
    del_w_saida_camada_2 = np.zeros(pesos_saida_camada_2.shape)

    for x, y in zip(features.values, targets):
        entrada_camada_1 = np.dot(x, pesos_entrada_camada_1)

        saida_camada_2 = sigmoid(entrada_camada_1)

        saida_camada_1 = sigmoid(np.dot(saida_camada_2, pesos_saida_camada_2))

        error_camada_1 = y - saida_camada_1
        error_term_camada_1 = error_camada_1 * sigmoid_prime(saida_camada_1)

        error_camada_2 = np.dot(error_term_camada_1, pesos_saida_camada_2)
        error_term_camada_2 = error_camada_2 * sigmoid_prime(saida_camada_2)

        del_w_saida_camada_2 += error_term_camada_1 * saida_camada_2
        del_w_entrada_camada_1 += error_term_camada_2 * x[:, None]

    # Atualiza os pesos da camada Nº 1
    pesos_entrada_camada_1 += learnrate * del_w_entrada_camada_1 / n_records
    # Atualiza os pesos da camada Nº 2
    pesos_saida_camada_2 += learnrate * del_w_saida_camada_2 / n_records
    # A cada 10 rodadas tira-se uma amostra
    if e % (rodadas / 10) == 0:
        saida_camada_2 = sigmoid(np.dot(x, pesos_entrada_camada_1))
        saida = sigmoid(np.dot(saida_camada_2, pesos_saida_camada_2))
        perda = np.mean((saida - targets) ** 2)

        if ultima_perda and ultima_perda < perda:
            print("Train loss: ", perda, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", perda)
        ultima_perda = perda

# Calculate accuracy on test data
oculta = sigmoid(np.dot(features_test, pesos_entrada_camada_1))
saida = sigmoid(np.dot(oculta, pesos_saida_camada_2))
predictions = saida > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
