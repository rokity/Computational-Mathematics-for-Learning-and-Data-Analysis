from layer import *
from functions_factory import FunctionsFactory
from kernel_initialization import *
import matplotlib.pyplot as plt
from optimizers import *

# This class is used to create a neural network
# Train the model with training ,test and validation dataset
# BATCH & EPOCHS MANAGEMENT
# There aren't loss or backpropagation formulas (NO FORMULAS)
# Plot Functions implemented for Loss and Accuracy
class NeuralNetwork:
    # Constructor
    # You can find the list of loss functions and metrics in the functions_factory.py file
    #   @param loss: string that represents the loss function to use
    #   @param metric: string that represents the metric to use to evaluate the model
    #
    def __init__(self, loss, metric):

        self.layers = []
        self.loss = FunctionsFactory.build(loss)
        self.metric = FunctionsFactory.build(metric)
        self.n_layers = 0
        self.history = dict()

    # This method is used to initialize the optimizer and compile layers(watch layer class)
    #    @param optimizer: optimizer used (@see optimizers.py)
    #
    def compile(self, optimizer=SGD()):

        self.optimizer = optimizer
        for i in range(len(self.layers)):
            self.layers[i].compile()
        self.optimizer.initialize(self.layers)

    # This method is used to add a layer to the neural network
    #   @param dim_out: dimension of the output
    #   @param input_dim: dimension of the input
    #   @param activation: string that represents the activation function to use on this layer
    #   @param kernel_initialization: string that represents weights initialization algorithm
    #
    def add_layer(self, dim_out, input_dim=None, activation='linear', kernel_initialization=RandomUniformInitialization()):
        f_act = FunctionsFactory.build(activation)
        if input_dim is not None:
            layer = Layer(input_dim, dim_out, f_act, self.loss,
                          kernel_initialization, "Dense_" + str(self.n_layers))
        else:
            #Take the same topology of the previous layer
            layer = Layer(self.layers[-1].dim_out, dim_out, f_act, self.loss,
                          kernel_initialization, "Dense_" + str(self.n_layers))
        self.n_layers += 1
        self.layers.append(layer)

    # Predict function without evalute the output of the model
    #   @param X: input features
    #   @return: output predictions
    #
    def predict(self, X: np.ndarray):
        out = np.zeros((X.shape[0], self.layers[-1].dim_out))
        n_samples = X.shape[0]
        for i in range(n_samples):
            x = X[i].reshape((1, X.shape[1]))
            out[i] = self.__feedforward(x).reshape(1, self.layers[-1].dim_out)
        return out

    # Predict and Evalute the output of the model using labels/targets
    #   @param X: input features
    #   @param Y: output targets
    #   @return: loss and metric
    def evaluate(self, X: np.ndarray, Y: np.ndarray):
        err = 0
        metric = 0
        n_samples = X.shape[0]
        for i in range(n_samples):
            x = X[i].reshape((1, X.shape[1]))
            d = Y[i].reshape((1, Y.shape[1]))
            y = self.__feedforward(x)
            err += float(self.loss.compute_fun(d.T, y))
            metric += float(self.metric.compute_fun(d.T, y))
        return err / n_samples, metric / n_samples

    # Call Feedforward function on each layer
    #   @param x: input features
    #   @return: output prediction
    def __feedforward(self, x: np.ndarray):
        x = x.reshape((x.shape[1], x.shape[0]))
        for layer in self.layers:
            x = layer.feedforward(x)
        return x

    # Execute Backpropagation using optimizer algorithm
    #   @param d: targets/labels real output
    def _backpropagation(self, d):
        self.optimizer.compute_gradients(d, self.layers)

    # Update weights and biases of each layer using optmizer
    #   @param batch_size: size of the batch
    def _update_parameters(self, batch_size):
        self.optimizer.update_parameters(self.layers, batch_size)

    # Train the model using the training data and labels
    #    @param X_train: training samples
    #    @param Y_train: training labels
    #    @param epochs: epochs number
    #    @param batch_size: size of the batch
    #    @param vl: pair (validation samples, validation targets)
    #    @param ts: pair (test samples, test targets) (used only for plot)
    #    @param tol: tolerance on the training set, if null => It uses an early stopping method
    #    @param shuffle: True if you want shuffle training data (at each epoch), False otherwise
    #    @param early_stopping: early stopping method
    #    @param verbose: used to print some informations
    #    @return: history
    #
    def fit(self, X_train, Y_train, epochs, batch_size=32, vl=None, ts=None, tol=None, shuffle=False, early_stopping=None, verbose=False):
        #History rappresents loss and accuracy for each epoch
        self.history[self.loss.name] = []
        self.history[self.metric.name] = []

        if vl is not None:
            X_val, Y_val = vl
            self.history["val_" + self.loss.name] = []
            self.history["val_" + self.metric.name] = []

        if ts is not None:
            X_test, Y_test = ts
            self.history["test_" + self.loss.name] = []
            self.history["test_" + self.metric.name] = []

        #Size of training set
        n_samples = X_train.shape[0]
        curr_epoch = 0
        curr_i = 0  # Index of the current sample

        #Calculate the number of batches
        n_batch = int(n_samples / batch_size)
        tr_loss_batch = np.zeros(n_batch)
        tr_lossr_batch = np.zeros(n_batch)
        tr_metric_batch = np.zeros(n_batch)
        end = False  # Used to stop the training

        while curr_epoch < epochs and not end:
            if shuffle:
                idx = np.random.permutation(n_samples)
                X_train = X_train[idx]
                Y_train = Y_train[idx]

            for nb in range(n_batch):
                loc_err = 0
                loc_metric = 0

                for _ in range(batch_size):
                    x = X_train[curr_i].reshape(1, X_train.shape[1])
                    d = Y_train[curr_i].reshape(1, Y_train.shape[1])
                    y = self.__feedforward(x)
                    self._backpropagation(d.T)
                    loc_err += self.loss.compute_fun(d.T, y)
                    loc_metric += self.metric.compute_fun(d.T, y)
                    curr_i = (curr_i + 1) % n_samples

                tr_loss_batch[nb] = loc_err / batch_size
                tr_lossr_batch[nb] = tr_loss_batch[nb] + \
                    self.optimizer.get_regualarization_for_loss(
                        self.layers)  # Loss with Regularization
                tr_metric_batch[nb] = loc_metric / batch_size  # Accuracy

                self._update_parameters(batch_size)
            self.optimizer.update_hyperparameters(curr_epoch)

            # compute average loss/metric in training set
            tr_err = np.mean(tr_loss_batch)
            tr_metric = np.mean(tr_metric_batch)
            tr_err_pen = np.mean(tr_lossr_batch)

            self.history[self.loss.name].append(tr_err)
            self.history[self.metric.name].append(tr_metric)

            # compute error on validation set
            if vl is not None:
                vl_err, vl_metric = self.evaluate(X_val, Y_val)
                self.history["val_" + self.loss.name].append(vl_err)
                self.history["val_" + self.metric.name].append(vl_metric)

            # compute error on test set
            if ts is not None:
                ts_err, ts_metric = self.evaluate(X_test, Y_test)
                self.history["test_" + self.loss.name].append(ts_err)
                self.history["test_" + self.metric.name].append(ts_metric)

            if verbose:
                self.print_epoch_verbose(curr_epoch, tr_err, tr_err, vl)

            curr_epoch += 1

            if early_stopping is not None:
                if early_stopping.early_stopping_check(self.history):
                    end = True

            if tol is not None:
                if tr_err_pen < tol:
                    end = True

        self.history["epochs"] = list(range(curr_epoch))
        if verbose:
            self.print_final_results_model(curr_epoch, vl, ts)

        return self.history

    #   Plot metric of loss ,save images and show if you want
    #   @param val: True if you want plot the validation loss, False otherwise
    #   @param test: True if you want plot the test loss, False otherwise
    #   @param show: True if you want show the plots, False otherwise
    #   @param path: path of file where to save the plots
    #
    def plot_loss(self, val=False, test=False, show=True, path=None):
        epochs = self.history['epochs']
        plt.xlabel('epochs', fontsize=15)
        plt.ylabel(self.loss.name, fontsize=15)
        plt.plot(epochs, self.history[self.loss.name],
                 color='tab:orange', linestyle='-', label="Training")
        if val:
            plt.plot(epochs, self.history["val_" + self.loss.name],
                     color='tab:green', linestyle='--', label="Validation")
        if test:
            plt.plot(epochs, self.history["test_" + self.loss.name],
                     color='tab:blue', linestyle='-.', label="Test")
        plt.legend(fontsize=20)
        if path is not None:
            plt.draw()
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()

    #   Plot metric of accuracy ,save images and show if you want
    #   @param val: True if you want plot the validation metric, False otherwise
    #   @param test: True if you want plot the test metric, False otherwise
    #   @param show: True if you want show the plots, False otherwise
    #   @param path: path of file where to save the plots
    #
    def plot_metric(self, val=False, test=False, show=True, path=None):
        epochs = self.history['epochs']
        plt.xlabel('epochs', fontsize=15)
        plt.ylabel(self.metric.name, fontsize=15)
        plt.plot(epochs, self.history[self.metric.name],
                 color='tab:orange', linestyle='-', label="Training")
        if val:
            plt.plot(epochs, self.history["val_" + self.metric.name],
                     color='tab:green', linestyle='--', label="Validation")
        if test:
            plt.plot(epochs, self.history["test_" + self.metric.name],
                     color='tab:blue', linestyle='-.', label="Test")
        plt.legend(fontsize=20)
        if path is not None:
            plt.draw()
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()

    # Print Status of model during an epoch
    # @curr_epoch: current epoch
    # @tr_err: training error
    # @tr_err_pen: training error with penality
    # @vl: validation set
    #
    def print_epoch_verbose(self, curr_epoch, tr_err_pen, tr_err, vl):
        print("It {:6d}: tr_err (with penality term): {:.6f},"
              "\t tr_err (without penality term): {:.6f}"
              .format(
                  curr_epoch,
                  tr_err_pen,
                  tr_err,
              ),
              end=''
              )
        if vl is not None:
            print("\t vl_err: {:.6f}".format(
                self.history["val_" + self.loss.name][-1]))
        else:
            print()

    # Print Final Results of Training Model Phase
    # @curr_epoch: Exit Epochs
    # @vl: validation set
    # @ts: test set
    #
    def print_final_results_model(self, curr_epoch, vl, ts):
        print()
        print("Exit at epoch: {}".format(curr_epoch))

        print("{} training set: {:.6f}".format(
            self.loss.name, self.history[self.loss.name][-1]))
        if vl is not None:
            print("{} validation set: {:.6f}".format(
                self.loss.name, self.history["val_" + self.loss.name][-1]))
        if ts is not None:
            print("{} test set: {:.6f}".format(self.loss.name,
                                               self.history["test_" + self.loss.name][-1]))
        if self.metric.name == 'accuracy':
            print("% accuracy training set: {}".format(
                self.history[self.metric.name][-1] * 100))
            if vl is not None:
                print("% accuracy validation set: {}".format(
                    self.history["val_" + self.metric.name][-1] * 100))
            if ts is not None:
                print("% accuracy test set: {}".format(
                    self.history["test_" + self.metric.name][-1] * 100))
        else:
            print("{} training set: {:.6f}".format(
                self.metric.name, self.history[self.metric.name][-1]))
            if vl is not None:
                print("{} validation set: {:.6f}".format(
                    self.metric.name, self.history["val_" + self.metric.name][-1]))
            if ts is not None:
                print("{} test set: {:.6f}".format(self.metric.name,
                                                   self.history["test_" + self.metric.name][-1]))
