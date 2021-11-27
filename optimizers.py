import numpy as np

# This class implements the Stepeest Descent Gradient optimizer
class SGD:

    # Constructor
    #   @param lr: learning rate
    #   @param mom: momentum
    #   @param l2: l2 regularizer
    #   @param decay_lr: dictionary composed by the following keys: "epoch_tau", "epsilon_tau". It implements the linear learning rate decay strategy
    #   @param nesterov: True if you want to use Nesterov momentum, False otherwise
    #
    def __init__(self, lr=1e-3, mom=0.0, l2=0.0, decay_lr=None, nesterov=False):
        self.lr = lr
        self.momentum = mom
        self.l2 = l2
        self.nesterov = nesterov
        self.is_init = False
        if decay_lr is None:
            self.decay_lr = None
        else:
            self.decay_lr = {
                "epoch_tau": decay_lr["epoch_tau"],
                "epsilon_tau": decay_lr["epsilon_tau_perc"] * lr,
                "init_lr": lr
            }
        self.params = dict()

    # Initialize matrix of derivations of weights and bias
    # @param layers: list of layers
    # 
    def initialize(self, layers):
        for layer in layers:
            name = layer.name
            self.params["dw_" + name] = np.zeros(layer.w.shape) #derivation of weight matrix
            self.params["db_" + name] = np.zeros(layer.b.shape) #deriviation of bias matrix (w0)
            if not self.is_init:
                self.params["vw_" + name] = np.zeros(layer.w.shape)
                self.params["vb_" + name] = np.zeros(layer.b.shape)
            if self.nesterov and not self.is_init:
                self.params["vb_prev_" + name] = np.zeros(layer.b.shape)
                self.params["vw_prev_" + name] = np.zeros(layer.w.shape)
        self.is_init = True

    # Compute gradients for loss and activation
    # @param layers: list of layers
    # @param d: target output
    #
    def compute_gradients(self, d, layers):
        # compute the gradient of the output layer
        output_l = layers[len(layers) - 1] #Take last layer alias output layer
        _, _, y = output_l.cache #Take cache of output layer(feedforward only with activation) [x:input,v:feedforward without activation function,y:feedforward with activation function]
        loc_grad = output_l.loss.compute_der(d, y) #calculate derivation of loss function on the last layer
        for layer in reversed(layers): #start from the last layer
            (x, v, y) = layer.cache #[x:input,v:feedforward without activation function,y:feedforward with activation function]
            partial = loc_grad * layer.f_act.compute_der(v) #error derivation * derivation of activation function
            self.params["dw_" + layer.name] += np.dot(partial, x.T) #derivation of weight matrix
            self.params["db_" + layer.name] += partial #derivation of bias matrix (w0)
            loc_grad = np.dot(layer.w.T, partial) #dw output layer

    # Calculate Regularization for loss function (NOT UPDATES PARAMETERS)
    #   @param layers: list of layers
    #   @return: 1/2 * lambda * || w ||^2
    #
    def get_regualarization_for_loss(self, layers):
        sum_w = 0
        for layer in layers:
            w = layer.w
            sum_w += np.sum(w*w)
        return 0.5 * sum_w * self.l2

    # Update Weights and Biases for each layer
    #   @param layers: list of layers
    #   @param batch_size: size of the batch
    #
    def update_parameters(self, layers, batch_size):
        for layer in layers:
            dw = self.params["dw_" + layer.name] / batch_size
            db = self.params["db_" + layer.name] / batch_size

            if self.nesterov:
                self.params["vw_prev_" + layer.name] = self.params["vw_" + layer.name]
                self.params["vb_prev_" + layer.name] = self.params["vb_" + layer.name]

            self.params["vw_" + layer.name] = \
                self.momentum * self.params["vw_" + layer.name] - self.lr * dw
            self.params["vb_" + layer.name] = \
                self.momentum * self.params["vb_" + layer.name] - self.lr * db

            if self.nesterov:
                layer.w += \
                    self.momentum * self.params["vw_prev_" + layer.name] + \
                    (1 - self.momentum) * self.params["vw_" + layer.name]
                layer.b += \
                    self.momentum * self.params["vb_prev_" + layer.name] + \
                    (1 - self.momentum) * self.params["vb_" + layer.name]
            else:
                layer.w += self.params["vw_" + layer.name]
                layer.b += self.params["vb_" + layer.name]
            layer.w -= self.l2 * layer.w
        self.initialize(layers)

    # Update learning rate
    #  @param curr_epoch: current epoch
    #
    def update_hyperparameters(self, curr_epoch):
        if self.decay_lr is not None:
            if curr_epoch < self.decay_lr["epoch_tau"]:
                alpha = curr_epoch / self.decay_lr["epoch_tau"]
                self.lr = (1 - alpha) * self.decay_lr["init_lr"] + alpha * self.decay_lr["epsilon_tau"]


# class RMSprop:
#     """
#         This class implements the RMSprop optimizer
#     """

#     def __init__(self, lr=1e-3, moving_average=0.9, epsilon=1e-7, l2=0.0):
#         """

#         @param lr: learning rate
#         @param moving_average: discounting factor for the history/coming gradient
#         @param epsilon: a small constant for numerical stability
#         @param l2: l2 regularizer
#         """
#         self.lr = lr
#         self.moving_avg = moving_average
#         self.epsilon = epsilon
#         self.l2 = l2
#         self.is_init = False
#         self.params = dict()

#     def initialize(self, layers):
#         """

#         @param layers: list of layers

#         Initialize the optimizer
#         """
#         for layer in layers:
#             name = layer.name
#             self.params["dw_" + name] = np.zeros(layer.w.shape)
#             self.params["db_" + name] = np.zeros(layer.b.shape)
#             if not self.is_init:
#                 self.params["vw_" + name] = np.zeros(layer.w.shape)
#                 self.params["vb_" + name] = np.zeros(layer.b.shape)
#         self.is_init = True

#     def compute_gradients(self, d, layers):
#         """

#         @param d: target output
#         @param layers: list of layers

#         Compute the gradients with the backpropagation method
#         """
#         # compute the gradient of the output layer
#         output_l = layers[-1]
#         _, _, y = output_l.cache
#         loc_grad = output_l.loss.compute_der(d, y)
#         for layer in reversed(layers):
#             (x, v, y) = layer.cache
#             partial = loc_grad * layer.f_act.compute_der(v)
#             self.params["dw_" + layer.name] += np.dot(partial, x.T)
#             self.params["db_" + layer.name] += partial
#             loc_grad = np.dot(layer.w.T, partial)

#     def get_regualarization_for_loss(self, layers):
#         """

#         @param layers: list of layers
#         @return: 1/2 * :lambda: * || w ||^2
#         """
#         sum_w = 0
#         for layer in layers:
#             w = layer.w
#             sum_w += np.sum(w*w)
#         return 0.5 * sum_w * self.l2

#     def update_parameters(self, layers, batch_size):
#         """

#         @param layers: list of layers
#         @param batch_size: size of the batch

#         It updates the parameters of each layer
#         """
#         for layer in layers:
#             dw = self.params["dw_" + layer.name] / batch_size
#             db = self.params["db_" + layer.name] / batch_size
#             self.params["vw_" + layer.name] = \
#                 self.moving_avg * self.params["vw_" + layer.name] + (1 - self.moving_avg) * (dw ** 2)
#             self.params["vb_" + layer.name] = \
#                 self.moving_avg * self.params["vb_" + layer.name] + (1 - self.moving_avg) * (db ** 2)

#             layer.w -= \
#                 self.lr * (self.params["dw_" + layer.name] + self.l2 * layer.w) / (np.sqrt(self.params["vw_" + layer.name]) + self.epsilon)
#             layer.b -= \
#                 self.lr * self.params["db_" + layer.name] / (np.sqrt(self.params["vb_" + layer.name]) + self.epsilon)

#         self.initialize(layers)

#     def update_hyperparameters(self, curr_epoch):
#         pass


# class Adam:
#     """
#         This class implements the Adam optimizer
#     """
#     def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, l2=0.0):
#         """

#         @param lr: learning rate
#         @param beta_1: the exponential decay rate for the 1st moment estimates
#         @param beta_2: the exponential decay rate for the 2nd moment estimates
#         @param epsilon: a small constant for numerical stability
#         @param l2: l2 regularizer
#         """
#         self.lr = lr
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.epsilon = epsilon
#         self.l2 = l2
#         self.is_init = False
#         self.params = dict()

#     def initialize(self, layers):
#         """

#         @param layers: list of layers

#         Initialize the optimizer
#         """
#         for layer in layers:
#             name = layer.name
#             self.params["dw_" + name] = np.zeros(layer.w.shape)
#             self.params["db_" + name] = np.zeros(layer.b.shape)
#             if not self.is_init:
#                 self.params["vw_" + name] = np.zeros(layer.w.shape)
#                 self.params["vb_" + name] = np.zeros(layer.b.shape)
#                 self.params["mw_" + name] = np.zeros(layer.w.shape)
#                 self.params["mb_" + name] = np.zeros(layer.b.shape)
#         self.is_init = True

#     def compute_gradients(self, d, layers):
#         """

#         @param d: target output
#         @param layers: list of layers

#         Compute the gradients with the backpropagation method
#         """
#         # compute the gradient of the output layer
#         output_l = layers[-1]
#         _, _, y = output_l.cache
#         loc_grad = output_l.loss.compute_der(d, y)
#         for layer in reversed(layers):
#             (x, v, y) = layer.cache
#             partial = loc_grad * layer.f_act.compute_der(v)
#             self.params["dw_" + layer.name] += np.dot(partial, x.T)
#             self.params["db_" + layer.name] += partial
#             loc_grad = np.dot(layer.w.T, partial)

#     def get_regualarization_for_loss(self, layers):
#         """

#         @param layers: list of layers
#         @return: 1/2 * :lambda: * || w ||^2
#         """
#         sum_w = 0
#         for layer in layers:
#             w = layer.w
#             sum_w += np.sum(w*w)
#         return 0.5 * sum_w * self.l2

#     def update_parameters(self, layers, batch_size):
#         """

#         @param layers: list of layers
#         @param batch_size: size of the batch

#         It updates the parameters of each layer
#         """
#         for layer in layers:
#             dw = self.params["dw_" + layer.name] / batch_size
#             db = self.params["db_" + layer.name] / batch_size

#             self.params["vw_" + layer.name] = \
#                 self.beta_2 * self.params["vw_" + layer.name] + (1 - self.beta_2) * (dw ** 2)
#             self.params["vb_" + layer.name] = \
#                 self.beta_2 * self.params["vb_" + layer.name] + (1 - self.beta_2) * (db ** 2)

#             self.params["mw_" + layer.name] = \
#                 self.beta_1 * self.params["mw_" + layer.name] + (1 - self.beta_1) * dw
#             self.params["mb_" + layer.name] = \
#                 self.beta_1 * self.params["mb_" + layer.name] + (1 - self.beta_1) * db

#             curr_mw = self.params["mw_" + layer.name] / (1 - self.beta_1)
#             curr_mb = self.params["mb_" + layer.name] / (1 - self.beta_1)
#             curr_vw = self.params["vw_" + layer.name] / (1 - self.beta_2)
#             curr_vb = self.params["vb_" + layer.name] / (1 - self.beta_2)

#             layer.w -= \
#                 self.lr * (curr_mw / ((np.sqrt(curr_vw)) + self.epsilon) + self.l2 * layer.w)
#             layer.b -= \
#                 self.lr * curr_mb / ((np.sqrt(curr_vb)) + self.epsilon)

#         self.initialize(layers)

#     def update_hyperparameters(self, curr_epoch):
#         pass
