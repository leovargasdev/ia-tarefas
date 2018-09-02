# NOME: Leonardo Luis de Vargas
# MATRICULA: 141110047
import numpy as np

def sigmoid(x): # Calculo do sigmóide
    return 1/(1+np.exp(-x))

N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
x = np.random.randn(4)

pesos_camada_1 = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
entrada_camada_1 = np.dot(x, pesos_camada_1) # Combinação linear 1ª camada
saida_camada_1 = sigmoid(entrada_camada_1)
print('Saida camada[1]: ', saida_camada_1)

pesos_camada_2 = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))
entrada_camada_2 = np.dot(saida_camada_1, pesos_camada_2) # Combinação linear 2ª camada
saida_camada_2 = sigmoid(entrada_camada_2)

print('Saida camada[2]: ', saida_camada_2)
