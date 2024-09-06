from ml import *


class TwoLayerNeural(Neural):
    def __init__(self, n_features, n_classes, hidden_layer: int):
        super().__init__(n_features, n_classes)
        self.hidden_layer = hidden_layer
        self.i_to_h_weights = xavier(self.input_neurons, hidden_layer)
        self.h_to_o_weights = xavier(hidden_layer, self.output_neurons)
        self.i_to_h_bias = xavier(1, hidden_layer)
        self.h_to_o_bias = xavier(1, self.output_neurons)

    def forward(self, X):
        first_feed = sigmoid(np.dot(X, self.i_to_h_weights) + self.i_to_h_bias)
        return sigmoid(np.dot(first_feed, self.h_to_o_weights) + self.h_to_o_bias)

    def backprop(self, X, y, alpha):
        # feed-forward : input -> hidden -> output
        hidden_output = sigmoid(np.dot(X, self.i_to_h_weights) + self.i_to_h_bias)
        output = self.forward(X)

        # back propagation
        # 1. output layer
        output_err = d_mse(output, y) * d_sigmoid(np.dot(hidden_output, self.h_to_o_weights) + self.h_to_o_bias)
        h_to_o_dW = np.dot(hidden_output.T, output_err) / X.shape[0]
        h_to_o_dB = np.mean(output_err, axis=0)

        # hidden layer
        error = np.dot(output_err, self.h_to_o_weights.T) * d_sigmoid(np.dot(X, self.i_to_h_weights) + self.i_to_h_bias)
        dW = np.dot(X.T, error) / X.shape[0]
        dB = np.mean(error, axis=0)

        # weights and biases update
        self.h_to_o_weights -= alpha * h_to_o_dW
        self.h_to_o_bias -= alpha * h_to_o_dB
        self.i_to_h_weights -= alpha * dW
        self.i_to_h_bias -= alpha * dB

        return mse(self.forward(X), y)  # loss



