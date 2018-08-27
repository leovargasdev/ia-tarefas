
# coding: utf-8

# Tarefa 1 - Gradiente Descendente
# Implementar o gradiente descendente baseada na Aula do Encontro 07/08/2018
# 

# In[32]:


import numpy as np


# Função de ativação 

# In[49]:


def sigmoid(x):
    #Calculate sigmoid
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    # Derivative of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))


# Entradas e pesos

# In[106]:


learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)
del_w = 0
# Initial weights
w = np.array([-1.0127941, -1.0255882, -1.0383823, -0.0511764])


# combinação linear entre os pesos e as entradas

# In[107]:


h = np.dot(x,w)


# In[108]:


nn_output = sigmoid(h)


# In[109]:


error = (y - nn_output)


# Gradiente

# In[110]:


error_term = error * sigmoid_prime(h)


# atualização dos pesos

# In[111]:


del_w += learnrate * error_term * x


# In[112]:


print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)

