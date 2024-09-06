from ml import *


class OneLayerNeural(Neural):
    def __init__(self, n_features, n_classes):
        super().__init__(n_features, n_classes)
        self.weights = xavier(n_features, n_classes)
        self.bias = xavier(1, n_classes)

    def forward(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def backprop(self, X, y, alpha):
        # forward propagation
        pred = self.forward(X)

        # error calculation
        error = (d_mse(pred, y) * d_sigmoid(np.dot(X, self.weights) + self.bias))

        # gradient descent
        dW = (np.dot(X.T, error)) / X.shape[0]
        dB = np.mean(error, axis=0)

        # weight and bias updates
        self.weights -= alpha * dW
        self.bias -= alpha * dB
        loss = mse(self.forward(X), y)
        return loss




