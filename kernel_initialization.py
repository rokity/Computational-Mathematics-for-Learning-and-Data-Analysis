import numpy as np


class AbstractKernelInitialization:

    def initialize(self, dim_out, dim_in):
        """

        @param dim_out: output dimension
        @param dim_in: input dimension

        initializes the weights of a single layer
        """
        pass


class RandomInitialization(AbstractKernelInitialization):
    def __init__(self, trsl=1.0):
        self.trsl = trsl

    def initialize(self, dim_out, dim_in):
        return np.random.randn(dim_out, dim_in) * self.trsl


class RandomNormalInitialization(AbstractKernelInitialization):
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def initialize(self, dim_out, dim_in):
        return np.random.normal(self.mean, self.std, (dim_out, dim_in))


class RandomUniformInitialization(AbstractKernelInitialization):
    def __init__(self, low=-0.05, high=0.05):
        self.low = low
        self.high = high

    def initialize(self, dim_out, dim_in):
        return np.random.uniform(self.low, self.high, (dim_out, dim_in))


class HeInitialization(AbstractKernelInitialization):
    def __init__(self):
        pass

    def initialize(self, dim_out, dim_in):
        return np.random.randn(dim_out, dim_in) * np.sqrt(2/dim_in)


class XavierUniformInitialization(AbstractKernelInitialization):
    def __init__(self):
        pass

    def initialize(self, dim_out, dim_in):
        limit = np.sqrt(6/(dim_in + dim_out))
        return np.random.uniform(-limit, limit, (dim_out, dim_in))


class XavierNormalInitialization(AbstractKernelInitialization):
    def __init__(self):
        pass

    def initialize(self, dim_out, dim_in):
        std = np.sqrt(2/(dim_in + dim_out))
        return np.random.normal(0, std, (dim_out, dim_in))


class ZerosInitialization(AbstractKernelInitialization):
    def __init__(self):
        pass

    def initialize(self, dim_out, dim_in):
        return np.zeros((dim_out, dim_in))

