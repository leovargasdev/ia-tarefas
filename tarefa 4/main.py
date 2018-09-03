# NOME: Leonardo Luis de Vargas
# MATRICULA: 141110047
import numpy as np

def sigmoid(x): # Calculo do sigmóide
    return 1/(1+np.exp(-x))

def sigmoid_prime(x): # Derivada da função sigmóide
    return sigmoid(x) * (1 - sigmoid(x))

x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

pesos_camada_1 = np.array([[0.5, -0.6], [0.1, -0.2], [0.1, 0.7]])

pesos_camada_2 = np.array([0.1, -0.3])

in_camada_1 = np.dot(x, pesos_camada_1) # Combinação linear 1ª camada
out_camada_1 = sigmoid(in_camada_1)

in_camada_2 = np.dot(out_camada_1, pesos_camada_2)
out_camada_2 = sigmoid(in_camada_2)

error = (target - out_camada_2)

error_term_fase_1 = error * sigmoid_prime(out_camada_2)

error_term_fase_2 = np.dot(error_term_fase_1, pesos_camada_2) * sigmoid_prime(out_camada_1)

delta_fase_1 = learnrate * error_term_fase_1 * out_camada_1

delta_fase_2 = learnrate * error_term_fase_2 * out_camada_2

print('Change in weights for hidden layer to output layer:')
print(delta_fase_1)
print('Change in weights for input layer to hidden layer:')
print(delta_fase_2)
