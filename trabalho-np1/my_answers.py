import numpy as np


class NeuralNetwork(object):

    def sigmoid(self, x): # Calculo do sigmóide
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x): # Derivada da função sigmóide
        return x * (1 - x)

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # input_to_hidden: Camada Nº 1
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))
        # hidden_to_output: Camada Nº 2
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))

        self.lr = learning_rate

        self.activation_function = lambda x : 1/(1+np.exp(-x))

    def train(self, features, targets):

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for x, y in zip(features, targets):
            # Gera as saidas das camadas 1 e 2
            final_outputs, hidden_outputs = self.forward_pass_train(x)
            # Recalcula os pesos
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, x, y, delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, x):

        hidden_inputs = np.dot(x, self.weights_input_to_hidden) # Gera entrada da camada Nº 1
        hidden_outputs = self.sigmoid(hidden_inputs) # Sigmóide da camada Nº 1

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # Gera entrada da camada Nº 2
        final_outputs = self.sigmoid(final_inputs) # Sigmóide da camada Nº 2

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        error = y - final_outputs
        output_error_term = error

        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)

        hidden_error_term = hidden_error * self.sigmoid_prime(hidden_outputs)

        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        delta_weights_i_h += hidden_error_term * X[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        # Camada Nº 1
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
        # Camada Nº 2
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records

    def run(self, features):

        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # Gera entrada da camada Nº 1
        hidden_outputs = self.sigmoid(hidden_inputs) # Sigmóide da camada Nº 1

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # Gera entrada da camada Nº 2
        final_outputs = final_inputs # Sigmóide da camada Nº 2

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1000
learning_rate = 0.1
hidden_nodes = 15
output_nodes = 1
