# NOME: Leonardo Luis de Vargas
# MATRICULA: 141110047
import numpy as np

def sigmoid(x): # Calculo do sigmóide
    return 1/(1+np.exp(-x))

def sigmoid_prime(x): # Derivada da função sigmóide
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
# Número de entradas
x = np.array([1, 2, 3, 4])
y = np.array(0.5)
del_w = 0 # Passo inicial

w = np.array([0.00041932, 0.00083865, 0.00125797, 0.00167729]) # Pesos iniciais

h = np.dot(x,w) # Combinação linear entre a entrada e os pesos

nn_output = sigmoid(h)

error = (y - nn_output)

error_term = error * sigmoid_prime(h)
# Atualização dos pesos
del_w += learnrate * error_term * x

print('Saída Rede Neural: ', nn_output)
print('Amount of Error:', error)
print('Change in Weights:', del_w)
