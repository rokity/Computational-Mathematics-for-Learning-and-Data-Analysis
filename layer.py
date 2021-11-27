import numpy as np
from kernel_initialization import *

# Layer class
# COMPUTATION OF FEEDFORWARD

class Layer:

    # Contructor
    # @param dim_in: input dimension
    # @param dim_out: output dimension
    # @param f_act: activation function
    # @param loss: loss fucntion
    # @param kernel_initialization: weights initialization
    # @param name: layer name (NOT IMPORTANT)
    def __init__(self, dim_in, dim_out, f_act, loss, kernel_initialization, name):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.f_act = f_act
        self.loss = loss
        self.name = name
        self.kernel_initialization = kernel_initialization

    # Compile Layer
    def compile(self):
        self.__init_layer(self.kernel_initialization)

    # Initialize layer
    # @param kernel_initialization: weights initialization
    # @return: None
    def __init_layer(self, kernel_initialization):
        self.w = kernel_initialization.initialize(self.dim_out, self.dim_in)
        self.b = np.zeros((self.dim_out, 1)) #w0
        self.cache = None

    # Forward propagation
    # @param x: input sample
    # @return: output prediction
    #
    def feedforward(self, x):
        v = np.dot(self.w, x) + self.b
        y = self.f_act.compute_fun(v)
        self.cache = (x, v, y)
        return y

    # Print Info Layer
    def print_info(self):
        print('name: {}\n'
              'input size: {}\n'
              'output size: {}\n'
              'activation function: {}\n'
              'loss function: {}\n'
              '--------'
              .format(self.name, self.dim_in, self.dim_out, self.f_act.name, self.loss.name)
              )
