from ml import *


class NLayerNeural(Neural):
    def __init__(self, n_features, hidden, n_classes, layer):
        super().__init__(n_features, n_classes)
        self.layers = layer
        self.hidden_neurons = hidden
        self.w = self.weight_init()
        self.bias = self.bias_init()

    def weight_init(self) -> list:
        """
        This function creates a list of weight matrices, where each matrix represents the connections between two
        adjacent layers in the network.
        The number of neurons in the deeper layers gradually reduces by half of neurons in previous layer.
        For a valid 3 layer NN with neurons of [784, 128, 64, 10]:
            - weights[0].shape == (784, 128)
            - weights[1].shape == (128, 64)
            - weights[2].shape == (64, 10)
        :return: list of 2D numpy arrays, each representing the weights between two adjacent layers in a NN
        """
        weights, start_L, next_L = ([], self.input_neurons, self.hidden_neurons)

        for _ in range(self.layers - 1):
            weights.append(xavier(start_L, next_L))
            start_L = next_L
            next_L //= 2
        weights.append(xavier(start_L, self.output_neurons))  # last hidden neuron connects to output neuron

        return weights

    def bias_init(self) -> list:
        """
        Similar to the logic for weight initialization.Here we initialize bias of each adjacent layer in the NN using
        Xavier initialization.
        :return: list of 2D numpy arrays, each representing the bias between two adjacent layers in a NN
        """
        bias = []
        neurons = self.hidden_neurons

        for _ in range(self.layers - 1):
            bias.append(xavier(1, neurons))
            neurons //= 2
        bias.append(xavier(1, self.output_neurons))

        return bias

    def forward(self, X: np.ndarray) -> list:
        feedforward = []
        temp = X
        for layer in range(self.layers):
            temp_fwd = sigmoid((np.dot(temp, self.w[layer]) + self.bias[layer]))
            feedforward.append(temp_fwd)
            temp = temp_fwd

        return feedforward

    def backprop(self, X, y, alpha):
        pass
