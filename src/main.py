import numpy as np
import pandas as pd
import os
import requests
import OneLayerNeural as oln
import TwoLayerNeural as tln
from matplotlib import pyplot as plt


def one_hot(data: np.ndarray) -> np.ndarray:
    yy_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    yy_train[rows, data] = 1
    return yy_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # rescale data for neural network
    X_train, X_test = oln.scale(X_train, X_test)
    
    # xavier_result = xavier(2, 3)  # xavier initialization
    # sigmoid_input = np.array([-1, 0, 1, 2])  # sigmoid
    # sigmoid_result = sigmoid(sigmoid_input)

    # stage 1
    # print([float(X_train[2, 778].flatten()), float(X_test[0, 774].flatten())],
    #       [float(x) for x in xavier_result.flatten()],

    model = oln.OneLayerNeural(784, 10)
    
    # stage 2 -- 28 x 28 pixels in grayscale image, and 10 labels
    # print([float(x) for x in model.forward(X_train[:2]).flatten()])

    # stage 3 -- backprop, MSE, MSE derivative and sigmoid derivative
    # y_true = np.array([4, 3, 2, 1])
    # y_pred = np.array([-1, 0, 1, 2])
    # print([float(x) for x in oln.mse(y_pred, y_true).flatten()],
    #       [int(x) for x in oln.mse_prime(y_pred, y_true).flatten()],
    #       [float(x) for x in oln.sigmoid_prime(y_pred).flatten()],
    #       [float(x) for x in model.backprop(X_train[:2], y_train[:2], alpha=0.1).flatten()])
    
    # stage 4 -- train and carrying out on batch education
    # result_1 = model.accuracy(X_test, y_test).flatten().tolist()
    # result_2 = []
    # for _ in range(20):
    #     model.train(0.5, X_train, y_train, 100)
    #     result_2.append(model.accuracy(X_test, y_test))
    # print(result_1,
    #       [float(x) for x in result_2])
    # plot(result_1, result_2, filename="plot_")

    # stage 5 -- creating a model for 2 neural layer
    model_1 = tln.TwoLayerNeural(784, 10, 64)
    # print([float(x) for x in model_1.forward(X_train[:2]).flatten()])

    # stage 6 -- 2 layer backprop
    # print([float(x) for x in model_1.backprop(X_train[:2], y_train[:2], alpha=0.1).flatten()])

    # stage 7 -- same as 4, but for mode_1
    result = []
    for _ in range(20):
        model_1.train(0.5, X_train, y_train, 100)
        result.append(model_1.accuracy(X_test, y_test))
    print([float(x) for x in result])

