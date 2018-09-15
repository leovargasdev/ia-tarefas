import numpy as np

class NeuralNetwork(object):

    def sigmoid_prime(self, x): # Derivada da função sigmóide
        return x * (1 - x)

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.


    def train(self, features, targets):

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            # Recalcula os pesos
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):

        inputs = np.array(X, ndmin=2).T
        inputs = inputs.reshape((1, -1))
        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden) # Entrada camada Nº 1
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output) # Entrada camada Nº 2
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        x2 = np.array(X, ndmin=2).reshape((1, -1))
        y2 = np.array(y, ndmin=2).reshape((1, -1))

        error = final_outputs - y2

        output_error_term =  error # Erro term camada Nº 1

        hidden_output_error_term = np.dot(output_error_term, self.weights_hidden_to_output.T) # Erro term camada Nº 2
        hidden_input_error_term = hidden_output_error_term * self.sigmoid_prime(hidden_outputs)
        # Somando os pesos
        delta_weights_i_h += np.dot(x2.T, hidden_input_error_term)
        delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weights_hidden_to_output += -self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += -self.lr * delta_weights_i_h / n_records

    def run(self, features):
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs# signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1000
learning_rate = 0.1
hidden_nodes = 15
output_nodes = 1
