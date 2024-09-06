import numpy as np
from abc import ABC, abstractmethod


# rescale function for adjusting data size
def scale(x_train, x_test):
    return x_train / np.max(x_train), x_test / np.max(x_train)


# initialization function
def xavier(n_in, n_out):
    low = -np.sqrt(6 / (n_in + n_out))
    high = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(low, high, (n_in, n_out))


# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


# cost / loss function
def mse(y_pred: np.ndarray, y: np.ndarray):
    return np.mean(np.square(y_pred - y))


def d_mse(y_pred: np.ndarray, y: np.ndarray):
    return 2 * (y_pred - y)


# Neural class
class Neural(ABC):
    def __init__(self, n_features, n_classes):
        self.input_neurons = n_features
        self.output_neurons = n_classes

    @abstractmethod
    def forward(self, X: np.ndarray):
        pass

    @abstractmethod
    def backprop(self, X, y, alpha):
        pass

    def train(self, alpha, X_train, y_train, batch_size=100) -> list:
        loss_list = []
        size = X_train.shape[0]
        for i in range(0, size, batch_size):
            loss_list.append(self.backprop(X_train[i:i + batch_size], y_train[i:i + batch_size], alpha))

        return loss_list

    def accuracy(self, X_test, y_test):
        y_pred = np.argmax(self.forward(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        return np.mean(y_pred == y_true)
