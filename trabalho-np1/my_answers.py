import numpy as np


class NeuralNetwork(object):

    def sigmoid(x): # Calculo do sigmóide
        return 1/(1+np.exp(-x))

    def sigmoid_prime(x): # Derivada da função sigmóide
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


    def forward_pass_train(self, X):

        entrada_camada_1 = np.dot(x, self.weights_input_to_hidden) # Gera entrada da camada Nº 1
        saida_camada_1 = sigmoid(entrada_camada_1) # Sigmóide da camada Nº 1

        entrada_camada_2 = np.dot(x, self.weights_hidden_to_output) # Gera entrada da camada Nº 2
        saida_camada_2 = sigmoid(entrada_camada_2) # Sigmóide da camada Nº 2

        return saida_camada_2, saida_camada_1

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        error_camada_1 = y - final_outputs
        error_term_camada_1 = error_camada_1 * sigmoid_prime(final_outputs)

        error_camada_2 = np.dot(error_term_camada_1, self.weights_hidden_to_output)
        error_term_camada_2 = error_camada_2 * sigmoid_prime(hidden_outputs)

        delta_weights_i_h += error_term_camada_1 * saida_camada_2
        delta_weights_h_o += error_term_camada_2 * x[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        # Camada Nº 1
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
        # Camada Nº 2
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records

    def run(self, features):

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = None # signals into hidden layer
        hidden_outputs = None # signals from hidden layer

        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = None # signals into final output layer
        final_outputs = None # signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
