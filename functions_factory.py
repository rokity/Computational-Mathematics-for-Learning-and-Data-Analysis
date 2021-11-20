import numpy as np


class FunctionsFactory:

    @staticmethod
    def build(name):
        if name == 'tanh':
            return Function(name, tanh, tanh_der)
        elif name == 'sigmoid':
            return Function(name, sigmoid, sigmoid_der)
        elif name == 'relu':
            return Function(name, reLU, reLU_der)
        elif name == 'leaky_relu':
            return Function(name, leaky_reLU, leaky_reLU_der)
        elif name == 'linear':
            return Function(name, linear, linear_der)
        elif name == 'mee':
            return Function(name, mee, None)
        elif name == 'mse':
            return Function(name, mse, mse_der)
        elif name == 'accuracy':
            return Function(name, accuracy, None)
        elif name == 'accuracy1-1':
            return Function('accuracy', accuracy1, None)


class Function:
    def __init__(self, name, c_fun, c_der):
        self.name = name
        self.compute_fun = c_fun
        self.compute_der = c_der


# ----------------- Activation functions -----------------

def tanh(x):
    return np.tanh(x)


def tanh_der(x):
    return 1 - tanh(x)**2


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    z = sigmoid(x)
    return z * (1 - z)


def linear(x):
    return x


def linear_der(x):
    return np.ones((x.shape))


def reLU(x):
    y = x.copy()
    y[y <= 0] = 0.0
    return y


def reLU_der(x):
    y = x.copy()
    y[y > 0] = 1.0
    y[y <= 0] = 0.0
    return y


def leaky_reLU(x):
    y = x.copy()
    y[y <= 0] = 0.001 * y[y <= 0]
    return y


def leaky_reLU_der(x):
    y = x.copy()
    y[y > 0] = 1.0
    y[y <= 0] = 0.001
    return y


# ----------------- Loss functions -----------------

def mse(d, y):
    return np.sum(np.square(d-y))


def mse_der(d, y):
    return (-2)*(d - y)

# ----------------- Accuracy functions -----------------


def accuracy(d, y):
    res = 0
    if y >= 0.5:
        res = 1
    return res == d


def accuracy1(d, y):
    res = -1
    if y >= 0:
        res = 1
    return res == d


def mee(d, y):
    return np.sqrt(np.sum(np.square(d-y)))
